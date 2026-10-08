"""Extraction benchmark: do extractors recover the right (material, property, value) triples?

Scoring is record-level. A predicted record is a true positive when a gold record of the same
material and property has the same value in canonical units (2% tolerance); anything else is a false
positive, and unmatched gold records are false negatives. Precision penalises invented or mis-united
values (and "traps" such as thermal conductivity read as ionic conductivity); recall penalises misses.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

REL_TOL = 0.02


def canon_formula(formula: str | None) -> str:
    """Compare formulas ignoring case, spaces, dashes and phase prefixes like 'beta-'."""
    if not formula:
        return ""
    return re.sub(r"[^a-z0-9().]", "", re.sub(r"^[^A-Za-z0-9]*(?:alpha|beta|gamma|[αβγ])[-\s]*", "", formula, flags=re.I).lower())


def _names(gold_record: dict) -> set[str]:
    """The names a gold record answers to: its formula plus accepted aliases (an abbreviation and what it stands for)."""
    return {canon_formula(gold_record["formula"]), *(canon_formula(a) for a in gold_record.get("also", []))}


def values_match(pred: float | None, gold: float) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) <= REL_TOL * max(abs(gold), 1e-12)


def load_gold(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["items"]


def score(gold_items: list[dict], predictions: dict[str, list[dict]]) -> dict:
    """`predictions`: item id -> list of {formula, property, value}. Returns overall and per-property scores."""
    per_prop: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    value_checked = value_correct = 0
    trap_items = [i for i in gold_items if not i["records"]]
    trap_false_alarms = 0

    for item in gold_items:
        remaining = list(item["records"])
        for pred in predictions.get(item["id"], []):
            hit = next(
                (g for g in remaining if canon_formula(pred["formula"]) in _names(g) and g["property"] == pred["property"]),
                None,
            )
            if hit is not None:
                value_checked += 1
                if values_match(pred["value"], hit["value"]):
                    per_prop[hit["property"]]["tp"] += 1
                    value_correct += 1
                else:
                    per_prop[pred["property"]]["fp"] += 1
                    per_prop[hit["property"]]["fn"] += 1  # right material/property, wrong value
                remaining.remove(hit)
            else:
                per_prop[pred["property"]]["fp"] += 1
        for g in remaining:
            per_prop[g["property"]]["fn"] += 1
        if not item["records"] and predictions.get(item["id"]):
            trap_false_alarms += 1

    def prf(c: dict[str, int]) -> dict[str, float]:
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0, **c}

    total = {k: sum(c[k] for c in per_prop.values()) for k in ("tp", "fp", "fn")}
    return {
        "overall": prf(total),
        "value_accuracy": round(value_correct / value_checked, 3) if value_checked else None,
        "negative_items": len(trap_items),
        "negative_items_with_false_alarms": trap_false_alarms,
        "per_property": {p: prf(c) for p, c in sorted(per_prop.items())},
    }


def records_by_item(records) -> dict[str, list[dict]]:
    """Group PropertyRecords by the gold item they were extracted from (chunk id == item id).

    Records are keyed the way the pipeline keys them (formula, else material name), and properties the
    pipeline does not screen on (e.g. thermal conductivity) are out of scope, not errors.
    """
    from matsci_agent.domain.units import PROPERTIES

    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.property_name in PROPERTIES:
            out[r.chunk_id].append({"formula": r.material_formula or r.material_name, "property": r.property_name, "value": r.value_numeric})
    return out


def as_chunks(gold_items: list[dict]) -> list[dict]:
    """Gold passages in the shape the extraction agent reads; a distinct paper id per item stops de-duplication."""
    return [
        {"id": i["id"], "text": i["text"], "relevance_score": 1.0,
         "metadata": {"arxiv_id": f"gold-{i['id']}", "title": f"Gold passage {i['id']}", "doi": ""}}
        for i in gold_items
    ]
