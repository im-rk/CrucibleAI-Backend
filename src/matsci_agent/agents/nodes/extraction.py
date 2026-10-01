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
from matsci_agent.domain.units import (
    PROPERTIES,
    UNIT_PATTERNS,
    VALUE_PATTERN,
    normalise,
    parse_value,
)
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000
# Passages per LLM call. One call per passage made a run use ~20 requests, a whole free-tier
# daily Gemini quota; 4 per call cuts extraction from up to 12 calls to 3.
BATCH_SIZE = 4


class RawRecord(BaseModel):
    """What the LLM returns for one value; converted to a PropertyRecord afterwards."""

    passage: int  # index of the passage (within the batch) the value was read from
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

_VALUE_RE = {
    prop: re.compile(VALUE_PATTERN + r"\s*(?P<unit>" + unit + r")(?![A-Za-z/])")
    for prop, unit in UNIT_PATTERNS.items()
}
_PROXIMITY = 90  # characters after the property name in which its value must appear


def heuristic_extract(chunk: dict) -> list[RawRecord]:
    """Regex extraction of 'property ... number unit' patterns; always low confidence.

    A sentence with no formula of its own ("Its stability window is 5.5 V.") is attributed to
    the most recent formula earlier in the chunk.
    """
    text = re.sub(r"\s+", " ", chunk["text"])
    records: list[RawRecord] = []
    subject: str | None = None
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        formulas = find_formulas(sentence)
        if formulas:
            subject = formulas[0]
        formulas = formulas or ([subject] if subject else [])
        lowered = sentence.lower()
        for prop, pattern in _VALUE_RE.items():
            for alias in PROPERTIES[prop][1]:
                pos = lowered.find(alias)
                if pos < 0:
                    continue
                match = pattern.search(sentence, pos, pos + len(alias) + _PROXIMITY)
                if match:
                    records.append(RawRecord(
                        passage=0,
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
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.llm = llm
        self.mp_client = mp_client
        self.max_chunks = max_chunks
        self.batch_size = batch_size
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
            warnings.append("LLM unavailable: property extraction used low-confidence regex heuristics.")

        batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        outcomes = await asyncio.gather(
            *(self._extract_batch(b, warnings) for b in batches), return_exceptions=True
        )
        records: list[PropertyRecord] = []
        skipped = 0
        for batch, outcome in zip(batches, outcomes):
            if isinstance(outcome, BaseException):
                skipped += len(batch)
                ids = ", ".join(c["id"] for c in batch)
                warnings.append(f"Skipped chunk {ids}: {type(outcome).__name__}: {outcome}")
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

    # ---- batched extraction -------------------------------------------------------------

    async def _extract_batch(self, batch: list[dict], warnings: list[str]) -> list[PropertyRecord]:
        """One LLM call for several passages; each record names the passage it came from."""
        if self.llm.available:
            try:
                passages = "\n\n".join(
                    f"[{i}] Paper: {c['metadata'].get('title', 'Unknown')} "
                    f"(DOI: {c['metadata'].get('doi') or 'Unknown'})\n{c['text'][:MAX_CHUNK_CHARS]}"
                    for i, c in enumerate(batch)
                )
                async with self._semaphore:
                    out = await self.llm.structured(
                        prompts.EXTRACTION_PROMPT.format(passages=passages),
                        ExtractionOutput,
                        system=prompts.EXTRACTION_SYSTEM,
                        max_tokens=4000,
                    )
                kept = [r for r in out.records if 0 <= r.passage < len(batch)]
                if len(kept) < len(out.records):
                    warnings.append(f"Dropped {len(out.records) - len(kept)} extracted value(s) citing a passage that does not exist.")
                return [self._to_record(r, batch[r.passage]) for r in kept]
            except LLMError as exc:
                ids = ", ".join(c["id"] for c in batch)
                warnings.append(f"Extraction fell back to heuristics for {ids}: {exc}")
        return [self._to_record(raw, chunk) for chunk in batch for raw in heuristic_extract(chunk)]

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
                    is_direct_gap=p.is_direct_gap,
                    method=method,
                ))
        await emit(
            config, "mp_lookup", formulas=len(formulas), records=len(out), online=self.mp_client.online
        )
        return out, notes
