"""ExtractionAgent: structured property records from paper chunks, plus a Materials Project cross-check."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.domain.chemistry import elements_in, find_formulas
from matsci_agent.domain.schemas import PropertyRecord
from matsci_agent.domain.state import ResearchState
from matsci_agent.domain.units import PROPERTIES, normalise, parse_value
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000


class RawRecord(BaseModel):
    """What the LLM returns for one value; converted to a PropertyRecord afterwards."""

    material_name: str
    material_formula: str | None
    property_name: str
    value_raw: str
    value_numeric: float | None
    unit: str
    conditions: str
    confidence: Literal["high", "medium", "low"]
    excerpt: str


class ExtractionOutput(BaseModel):
    records: list[RawRecord]


# ---- deterministic fallback (no LLM) ---------------------------------------------------

_VALUE = (
    r"(?P<val>[<>~≈≥≤]?\s*\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?"
    r"(?:\s*[×xX·*]\s*10\s*\^?\s*\{?[-−+]?[\d⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+\}?|[eE][-+]?\d+)?)"
)
_UNITS = {
    "ionic_conductivity": r"(?:[mµμu]?S\s?/\s?cm|[mµμu]?S\s?cm(?:\^?-1|⁻¹)|S\s?/\s?m)",
    "bandgap": r"m?eV",
    "formation_energy": r"(?:eV/atom|meV/atom|kJ/mol)",
    "density": r"(?:g\s?/\s?cm(?:3|³)|g\s?cm(?:\^?-3|⁻³)|kg\s?/\s?m(?:3|³))",
    "electrochemical_stability_window": r"V",
    "activation_energy": r"m?eV",
    "critical_temperature": r"K",
    "power_conversion_efficiency": r"%",
    "capacity": r"(?:mAh\s?/\s?g|mAh\s?g(?:\^?-1|⁻¹))",
}
_VALUE_RE = {
    prop: re.compile(_VALUE + r"\s*(?P<unit>" + unit + r")(?![A-Za-z/])")
    for prop, unit in _UNITS.items()
}
_PROXIMITY = 90  # characters after the property name in which its value must appear


def heuristic_extract(chunk: dict) -> list[RawRecord]:
    """Regex extraction of 'property ... number unit' patterns; always low confidence."""
    text = re.sub(r"\s+", " ", chunk["text"])
    records: list[RawRecord] = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        formulas = find_formulas(sentence)
        lowered = sentence.lower()
        for prop, pattern in _VALUE_RE.items():
            for alias in PROPERTIES[prop][1]:
                pos = lowered.find(alias)
                if pos < 0:
                    continue
                match = pattern.search(sentence, pos, pos + len(alias) + _PROXIMITY)
                if match:
                    records.append(RawRecord(
                        material_name=formulas[0] if formulas else "Unknown",
                        material_formula=formulas[0] if formulas else None,
                        property_name=prop,
                        value_raw=f"{match['val'].strip()} {match['unit']}",
                        value_numeric=None,
                        unit=match["unit"],
                        conditions="",
                        confidence="low",
                        excerpt=sentence[:300],
                    ))
                    break
    return records


# ---- agent -------------------------------------------------------------------------------


class ExtractionAgent:
    """Turns relevant chunks into normalised PropertyRecords with provenance."""

    def __init__(
        self,
        llm: LLMClient,
        mp_client,  # MaterialsProjectClient
        *,
        max_chunks: int = 12,
        max_formulas: int = 8,
        concurrency: int = 4,
    ) -> None:
        self.llm = llm
        self.mp_client = mp_client
        self.max_chunks = max_chunks
        self.max_formulas = max_formulas
        self._semaphore = asyncio.Semaphore(concurrency)

    async def run(self, state: ResearchState, config=None) -> dict:
        chunks = state.get("retrieved_papers", [])[: self.max_chunks]
        warnings: list[str] = []
        if not chunks:
            warnings.append("Extraction skipped: no relevant literature chunks were retrieved.")
            return {
                "extracted_properties": [],
                "warnings": warnings,
                "agent_log": [{"step": "EXTRACTION", "details": "No relevant chunks to extract from."}],
            }

        papers = {c["metadata"].get("arxiv_id", c["id"]) for c in chunks}
        await emit(config, "extraction_start", chunks=len(chunks), papers=len(papers))
        if not self.llm.available:
            warnings.append("LLM not configured: property extraction used low-confidence regex heuristics.")

        outcomes = await asyncio.gather(
            *(self._extract_chunk(c, warnings) for c in chunks), return_exceptions=True
        )
        records: list[PropertyRecord] = []
        skipped = 0
        for chunk, outcome in zip(chunks, outcomes):
            if isinstance(outcome, BaseException):
                skipped += 1
                warnings.append(f"Skipped chunk {chunk['id']}: {type(outcome).__name__}: {outcome}")
            else:
                records.extend(outcome)
        records = self._dedupe(records)
        literature_count = len(records)

        mp_records, mp_note = await self._cross_check(records, config)
        records.extend(mp_records)
        warnings.extend(mp_note)

        await emit(config, "properties_extracted", records=len(records), literature=literature_count)
        details = (
            f"Extracted {literature_count} literature records from {len(papers)} papers"
            f"{f' ({skipped} chunks skipped)' if skipped else ''}; "
            f"added {len(mp_records)} database records."
        )
        return {
            "extracted_properties": [r.model_dump() for r in records],
            "warnings": warnings,
            "agent_log": [{"step": "EXTRACTION", "details": details}],
        }

    # ---- per-chunk extraction ---------------------------------------------------------

    async def _extract_chunk(self, chunk: dict, warnings: list[str]) -> list[PropertyRecord]:
        raws = None
        if self.llm.available:
            try:
                meta = chunk["metadata"]
                async with self._semaphore:
                    out = await self.llm.structured(
                        prompts.EXTRACTION_PROMPT.format(
                            title=meta.get("title", "Unknown"),
                            doi=meta.get("doi") or "Unknown",
                            text=chunk["text"][:MAX_CHUNK_CHARS],
                        ),
                        ExtractionOutput,
                        system=prompts.EXTRACTION_SYSTEM,
                        max_tokens=2000,
                    )
                raws = out.records
            except LLMError as exc:
                warnings.append(f"Extraction fell back to heuristics for {chunk['id']}: {exc}")
        if raws is None:
            raws = heuristic_extract(chunk)
        return [self._to_record(raw, chunk) for raw in raws]

    @staticmethod
    def _to_record(raw: RawRecord, chunk: dict) -> PropertyRecord:
        meta = chunk["metadata"]
        numeric = raw.value_numeric if raw.value_numeric is not None else parse_value(raw.value_raw).value
        prop, numeric, unit, _ = normalise(raw.property_name, numeric, raw.unit)
        formula = raw.material_formula if raw.material_formula and elements_in(raw.material_formula) else None
        return PropertyRecord(
            material_name=raw.material_name,
            material_formula=formula,
            property_name=prop,
            value_raw=raw.value_raw,
            value_numeric=numeric,
            unit=unit,
            conditions=raw.conditions,
            source_title=meta.get("title", ""),
            source_doi=meta.get("doi", ""),
            source_id=meta.get("arxiv_id", ""),
            chunk_id=chunk["id"],
            confidence=raw.confidence,
            excerpt=raw.excerpt[:300],
        )

    @staticmethod
    def _dedupe(records: list[PropertyRecord]) -> list[PropertyRecord]:
        seen: dict[tuple, PropertyRecord] = {}
        for r in records:
            key = (
                (r.material_formula or r.material_name).lower(),
                r.property_name,
                None if r.value_numeric is None else round(r.value_numeric, 9),
                r.unit,
                r.source_id,
            )
            seen.setdefault(key, r)
        return list(seen.values())

    # ---- Materials Project cross-check -------------------------------------------------

    async def _cross_check(self, records: list[PropertyRecord], config) -> tuple[list[PropertyRecord], list[str]]:
        counts = Counter(r.material_formula for r in records if r.material_formula)
        formulas = [f for f, _ in counts.most_common(self.max_formulas)]
        notes: list[str] = []
        if not formulas:
            return [], notes
        if not self.mp_client.online:
            notes.append(
                "MATERIALS_PROJECT_API_KEY not set: database cross-check limited to the offline reference table."
            )
        lookups = await asyncio.gather(
            *(asyncio.to_thread(self.mp_client.get_properties, f) for f in formulas)
        )
        method = "Materials Project (DFT)" if self.mp_client.online else "offline reference"
        out: list[PropertyRecord] = []
        for formula, props in zip(formulas, lookups):
            for p in props:
                numeric = float(p.value) if isinstance(p.value, (int, float)) else None
                out.append(PropertyRecord(
                    material_name=formula,
                    material_formula=formula,
                    property_name=p.property_name,
                    value_raw=f"{p.value} {p.unit}".strip(),
                    value_numeric=numeric,
                    unit=p.unit,
                    source_title=p.source,
                    source_doi=p.doi,
                    source_id=p.material_id,
                    confidence="high" if self.mp_client.online else "medium",
                    is_predicted=False,
                    method=method,
                ))
        await emit(
            config, "mp_lookup", formulas=len(formulas), records=len(out), online=self.mp_client.online
        )
        return out, notes
