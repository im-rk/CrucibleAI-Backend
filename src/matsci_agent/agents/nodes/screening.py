"""ScreeningAgent: parse criteria from the goal, evaluate candidates against them, rank by evidence."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.domain.chemistry import ELEMENTS, RARE_EARTH, elements_in, looks_generic
from matsci_agent.domain.schemas import Candidate, Criterion, CriterionResult, PropertyRecord
from matsci_agent.domain.state import ResearchState
from matsci_agent.domain.units import PROPERTIES, UNIT_PATTERNS, VALUE_PATTERN, normalise, parse_value
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

# canonical property -> KRR model key, for properties the predictor can estimate
KRR_KEYS = {"bandgap": "band_gap", "formation_energy": "formation_energy_per_atom", "density": "density"}
STATUS_ORDER = {"qualified": 0, "partial": 1, "unscreened": 2, "rejected": 3}
SCORE = {"met": 1.0, "predicted_met": 0.5}
MAX_CANDIDATES = 12


# ---- criteria parsing ------------------------------------------------------------------


class CriterionDraft(BaseModel):
    kind: Literal["numeric", "exclude_elements", "measured_evidence", "qualitative"]
    label: str
    property_name: str | None
    operator: Literal[">", ">=", "<", "<=", "between"] | None
    value: float | None
    value_high: float | None
    unit: str | None
    elements: list[str] | None


class CriteriaOutput(BaseModel):
    criteria: list[CriterionDraft]


_OPS = {
    ">": ">", ">=": ">=", "≥": ">=", "above": ">", "over": ">", "greater than": ">",
    "more than": ">", "exceeding": ">", "at least": ">=",
    "<": "<", "<=": "<=", "≤": "<=", "below": "<", "under": "<", "less than": "<", "at most": "<=",
}
_OP_RE = "|".join(re.escape(op) for op in sorted(_OPS, key=len, reverse=True))
_NAMED_ELEMENTS = {
    "cobalt": "Co", "lead": "Pb", "cadmium": "Cd", "mercury": "Hg", "nickel": "Ni",
    "arsenic": "As", "vanadium": "V", "chromium": "Cr",
}


def parse_criteria_heuristic(goal: str) -> list[Criterion]:
    """Regex parser for the common phrasings: 'ionic conductivity above 1e-3 S/cm', 'no rare earth'."""
    lowered = goal.lower()
    criteria: list[Criterion] = []

    for prop, (_, aliases) in PROPERTIES.items():
        unit_re = UNIT_PATTERNS.get(prop)
        for alias in sorted(aliases, key=len, reverse=True):
            pos = lowered.find(alias)
            if pos < 0:
                continue
            tail = goal[pos + len(alias): pos + len(alias) + 80]
            unit_part = rf"(?:\s*(?P<unit>{unit_re}))?" if unit_re else ""
            match = re.search(rf"(?P<op>{_OP_RE})\s*{VALUE_PATTERN}{unit_part}", tail, re.IGNORECASE)
            if match:
                op = _OPS[match["op"].lower()]
                value = parse_value(match["val"]).value
                unit = match.groupdict().get("unit") or ""
                if value is not None:
                    label = f"{alias.title()} {op} {match['val'].strip()}{(' ' + unit) if unit else ''}"
                    criteria.append(_numeric(prop, op, value, None, unit, label))
                    break
            between = re.search(
                r"(?:between|from)\s*(?P<a>[\d.]+)\s*(?:and|to|–|-)\s*(?P<b>[\d.]+)"
                + (rf"(?:\s*(?P<unit>{unit_re}))?" if unit_re else ""),
                tail, re.IGNORECASE,
            )
            if between:
                label = f"{alias.title()} between {between['a']} and {between['b']}"
                unit = between.groupdict().get("unit") or ""
                criteria.append(_numeric(prop, "between", float(between["a"]), float(between["b"]), unit, label))
                break

    if re.search(r"(?:no|without|excluding|avoid(?:ing)?)\s+(?:any\s+)?rare[\s-]?earth|rare[\s-]?earth[\s-]free", lowered):
        criteria.append(Criterion(kind="exclude_elements", label="No rare earth elements", elements=sorted(RARE_EARTH)))
    for name, symbol in _NAMED_ELEMENTS.items():
        if re.search(rf"(?:no|without|excluding)\s+{name}|{name}[\s-]free", lowered):
            criteria.append(Criterion(kind="exclude_elements", label=f"No {name}", elements=[symbol]))
    if re.search(
        r"experimentally\s+(?:demonstrated|verified|validated|measured)|demonstrated\s+experimentally"
        r"|not\s+(?:just|only)\s+(?:predicted|computed|dft)", lowered,
    ):
        criteria.append(Criterion(kind="measured_evidence", label="Demonstrated experimentally (not only predicted)"))
    qualitative = re.search(r"air[\s-]stab\w*|moisture[\s-]\w+|stable in air", lowered)
    if qualitative:
        criteria.append(Criterion(kind="qualitative", label=qualitative.group().capitalize()))
    return criteria


def _numeric(prop: str, op: str, value: float, high: float | None, unit: str, label: str) -> Criterion:
    _, converted, canon_unit, _ = normalise(prop, value, unit or PROPERTIES[prop][0])
    upper = None
    if high is not None:
        _, upper, _, _ = normalise(prop, high, unit or PROPERTIES[prop][0])
    return Criterion(
        kind="numeric", label=label, property_name=prop, operator=op,
        value=converted, value_high=upper, unit=canon_unit,
    )


def criterion_from_draft(draft: CriterionDraft, warnings: list[str]) -> Criterion | None:
    if draft.kind == "numeric":
        if not (draft.property_name and draft.operator and draft.value is not None):
            return None
        prop = normalise(draft.property_name, None, "")[0]
        if prop not in PROPERTIES:
            warnings.append(f"Criterion '{draft.label}': property '{draft.property_name}' is not one the pipeline can check.")
            return Criterion(kind="qualitative", label=draft.label)
        unit = draft.unit or PROPERTIES[prop][0]
        _, low, canon_unit, ok = normalise(prop, draft.value, unit)
        if not ok:
            warnings.append(f"Criterion '{draft.label}': unit '{unit}' not recognised; comparing the raw value.")
        high = None if draft.value_high is None else normalise(prop, draft.value_high, unit)[1]
        return Criterion(
            kind="numeric", label=draft.label, property_name=prop, operator=draft.operator,
            value=low, value_high=high, unit=canon_unit,
        )
    if draft.kind == "exclude_elements":
        symbols = [e for e in (draft.elements or []) if e in ELEMENTS]
        return Criterion(kind="exclude_elements", label=draft.label, elements=symbols) if symbols else None
    return Criterion(kind=draft.kind, label=draft.label)


# ---- evaluation --------------------------------------------------------------------------


def _passes(criterion: Criterion, value: float) -> bool:
    op, bound, high = criterion.operator, criterion.value, criterion.value_high
    return {
        ">": lambda: value > bound, ">=": lambda: value >= bound, "<": lambda: value < bound,
        "<=": lambda: value <= bound, "between": lambda: bound <= value <= (high if high is not None else bound),
    }[op]()


def _favourable(criterion: Criterion, values: list[float]) -> float:
    """The reported value most favourable to the criterion (we screen for recall, then disclose it)."""
    if criterion.operator in (">", ">="):
        return max(values)
    if criterion.operator in ("<", "<="):
        return min(values)
    inside = [v for v in values if _passes(criterion, v)]
    if inside:
        return inside[0]
    mid = (criterion.value + (criterion.value_high or criterion.value)) / 2
    return min(values, key=lambda v: abs(v - mid))


class ScreeningAgent:
    """Evaluates candidate materials against the goal's criteria, predicting missing properties with KRR."""

    def __init__(self, llm: LLMClient, krr, mp_client, *, max_candidates: int = MAX_CANDIDATES) -> None:
        self.llm = llm
        self.krr = krr
        self.mp_client = mp_client
        self.max_candidates = max_candidates

    async def run(self, state: ResearchState, config=None) -> dict:
        goal = state["research_goal"]
        # Predictions are recomputed on every pass, so a second pass (after the investigator added
        # evidence) never sees the first pass's estimates as if they were data.
        records = [PropertyRecord(**r) for r in state.get("extracted_properties", []) if not r.get("is_predicted")]
        warnings: list[str] = []

        planned = state.get("criteria")
        criteria = [Criterion(**c) for c in planned] if planned else await self._parse_criteria(goal, warnings)
        await emit(config, "screening_start", criteria_count=len(criteria), records=len(records))
        if not criteria:
            warnings.append("No screening criteria could be parsed from the goal; candidates are unscreened.")

        grouped = self._group(records, warnings)
        origins = {
            key: "literature" if any(r.method == "literature" for r in recs) else "database"
            for key, recs in grouped.items()
        }
        if len(grouped) < 3:
            seeded = await self._seed_from_database(criteria, grouped, origins, warnings)
            records.extend(seeded)

        predicted: list[PropertyRecord] = []
        candidates = [
            await self._evaluate(key, recs, criteria, origins[key], predicted, warnings, config)
            for key, recs in grouped.items()
        ]
        candidates.sort(key=lambda c: (STATUS_ORDER[c.status], -c.score, -len(c.sources)))
        candidates = candidates[: self.max_candidates]
        for rank, cand in enumerate(candidates, 1):
            cand.rank = rank
        qualified = sum(c.status == "qualified" for c in candidates)
        await emit(config, "candidates_found", count=len(candidates), qualified=qualified)

        details = (
            f"Parsed {len(criteria)} criteria; screened {len(candidates)} candidates "
            f"({qualified} qualified, {len(predicted)} KRR predictions)."
        )
        return {
            "criteria": [c.model_dump() for c in criteria],
            "screened_candidates": [c.model_dump() for c in candidates],
            "extracted_properties": [r.model_dump() for r in records + predicted],
            "warnings": warnings,
            "agent_log": [{"step": "SCREENING", "details": details}],
        }

    # ---- criteria ----------------------------------------------------------------------

    async def _parse_criteria(self, goal: str, warnings: list[str]) -> list[Criterion]:
        if self.llm.available:
            try:
                out = await self.llm.structured(
                    prompts.CRITERIA_PROMPT.format(goal=goal), CriteriaOutput,
                    system=prompts.CRITERIA_SYSTEM, max_tokens=1500,
                )
                parsed = [c for c in (criterion_from_draft(d, warnings) for d in out.criteria) if c]
                if parsed:
                    return parsed
            except LLMError as exc:
                warnings.append(f"Criteria parsing fell back to regex heuristics: {exc}")
        else:
            warnings.append("LLM unavailable: screening criteria were parsed with regex heuristics and may be incomplete.")
        return parse_criteria_heuristic(goal)

    # ---- candidates --------------------------------------------------------------------

    @staticmethod
    def _group(records: list[PropertyRecord], warnings: list[str]) -> dict[str, list[PropertyRecord]]:
        grouped: dict[str, list[PropertyRecord]] = {}
        unattributed = generic = 0
        for r in records:
            key = r.material_formula or (r.material_name if r.material_name.lower() != "unknown" else "")
            if not key:
                unattributed += 1
                continue
            if not r.material_formula and looks_generic(key):
                generic += 1  # a class of materials ("lithium thiophosphates"), not a candidate to screen
                continue
            grouped.setdefault(key, []).append(r)
        if unattributed:
            warnings.append(f"{unattributed} extracted values could not be attributed to a named material.")
        if generic:
            warnings.append(f"{generic} extracted value(s) described a class of materials rather than a specific one and were not screened.")
        return grouped

    async def _seed_from_database(self, criteria, grouped, origins, warnings) -> list[PropertyRecord]:
        """When the literature yields too few materials, add database matches for band-gap criteria."""
        gap = next((c for c in criteria if c.kind == "numeric" and c.property_name == "bandgap"), None)
        if gap is None:
            return []
        low = gap.value if gap.operator in (">", ">=", "between") else None
        high = gap.value_high if gap.operator == "between" else (gap.value if gap.operator in ("<", "<=") else None)
        rows = await asyncio.to_thread(self.mp_client.screen_by_criteria, low, high, True, None, 8)
        added: list[PropertyRecord] = []
        method = "Materials Project (DFT)" if self.mp_client.online else "offline reference"
        for row in rows:
            formula = row["formula"]
            if formula in grouped or row.get("bandgap_eV") is None:
                continue
            record = PropertyRecord(
                material_name=formula, material_formula=formula, property_name="bandgap",
                value_raw=f"{round(float(row['bandgap_eV']), 4)} eV", value_numeric=round(float(row["bandgap_eV"]), 4), unit="eV",
                source_title=row.get("source", "Materials Project"), source_id=row["material_id"],
                confidence="high" if self.mp_client.online else "medium", method=method,
            )
            grouped[formula] = [record]
            origins[formula] = "database"
            added.append(record)
        if added:
            warnings.append(
                f"{len(added)} candidate(s) were added from a database band-gap screen and have no supporting literature: "
                + ", ".join(r.material_formula for r in added)
            )
        return added

    async def _evaluate(self, key, records, criteria, origin, predicted, warnings, config) -> Candidate:
        formula = next((r.material_formula for r in records if r.material_formula), None)
        name = Counter(r.material_name for r in records).most_common(1)[0][0]
        results: list[CriterionResult] = []
        flags: list[str] = []

        for criterion in criteria:
            if criterion.kind == "numeric":
                result = await self._numeric(criterion, records, formula, predicted, warnings, config)
                if result.status.startswith("predicted"):
                    flags.append(f"predicted_only:{criterion.property_name}")
                elif result.status == "missing":
                    flags.append(f"missing:{criterion.property_name}")
            elif criterion.kind == "exclude_elements":
                result = self._composition(criterion, formula)
            elif criterion.kind == "measured_evidence":
                lit = [r for r in records if r.method == "literature"]
                result = CriterionResult(
                    label=criterion.label, kind=criterion.kind, status="met" if lit else "not_met",
                    detail=f"{len(lit)} literature record(s)" if lit else "only database or predicted values",
                )
            else:
                result = CriterionResult(
                    label=criterion.label, kind=criterion.kind, status="unverified",
                    detail="Cannot be checked from extracted numeric data.",
                )
            results.append(result)

        checked = [r for r in results if r.kind != "qualitative"]
        score = (
            sum(SCORE.get(r.status, 0.0) for r in checked) / len(checked) if checked
            else min(1.0, len(records) / 5)
        )
        statuses = {r.status for r in checked}
        if not checked:
            status = "unscreened"
        elif "not_met" in statuses:
            status = "rejected"
        elif statuses == {"met"}:
            status = "qualified"
        else:
            status = "partial"

        sources = {
            (r.source_id or r.source_title): {"title": r.source_title, "doi": r.source_doi, "source_id": r.source_id}
            for r in records if r.method == "literature" and (r.source_id or r.source_title)
        }
        return Candidate(
            formula=formula, name=name, status=status, score=round(score, 3), origin=origin,
            criteria=results, flags=flags, sources=list(sources.values()), record_count=len(records),
        )

    async def _numeric(self, criterion, records, formula, predicted, warnings, config) -> CriterionResult:
        prop, canon_unit = criterion.property_name, criterion.unit or ""
        usable = [
            r for r in records
            if r.property_name == prop and r.value_numeric is not None and not r.is_predicted and r.unit == canon_unit
        ]
        literature = [r for r in usable if r.method == "literature"]
        pool = literature or usable
        if pool:
            best = _favourable(criterion, [r.value_numeric for r in pool])
            chosen = next(r for r in pool if r.value_numeric == best)
            return CriterionResult(
                label=criterion.label, kind="numeric",
                status="met" if _passes(criterion, best) else "not_met",
                value=best, unit=canon_unit, source=chosen.source_title or chosen.method,
                detail=f"best of {len(pool)} reported value(s); {chosen.method}",
            )

        krr_key = KRR_KEYS.get(prop or "")
        if formula and krr_key and elements_in(formula):
            pred = await asyncio.to_thread(self.krr.predict, formula, krr_key)
            if "error" in pred:
                warnings.append(f"KRR prediction skipped for {formula} ({prop}): {pred['error']}")
            else:
                value = float(pred["predicted_value"])
                predicted.append(PropertyRecord(
                    material_name=formula, material_formula=formula, property_name=prop,
                    value_raw=f"{value} {pred['unit']}", value_numeric=value, unit=pred["unit"],
                    source_title="KRR prediction", confidence=pred["confidence"], is_predicted=True,
                    uncertainty=pred.get("uncertainty"),
                    method=pred["method"],
                ))
                await emit(config, "krr_prediction", formula=formula, property=prop, value=value)
                unc = pred.get("uncertainty")
                return CriterionResult(
                    label=criterion.label, kind="numeric", is_predicted=True,
                    status="predicted_met" if _passes(criterion, value) else "predicted_not_met",
                    value=value, unit=pred["unit"], source="KRR prediction",
                    detail=f"ML estimate{'' if unc is None else f' ±{unc}'}; not a measurement",
                )
        return CriterionResult(
            label=criterion.label, kind="numeric", status="missing", unit=canon_unit,
            detail="no measured value found and no predictor for this property",
        )

    @staticmethod
    def _composition(criterion: Criterion, formula: str | None) -> CriterionResult:
        present = elements_in(formula) if formula else None
        if present is None:
            return CriterionResult(
                label=criterion.label, kind=criterion.kind, status="missing",
                detail="no chemical formula available to check",
            )
        clash = sorted(present & set(criterion.elements))
        return CriterionResult(
            label=criterion.label, kind=criterion.kind, status="not_met" if clash else "met",
            detail=f"contains {', '.join(clash)}" if clash else "none of the excluded elements present",
        )
