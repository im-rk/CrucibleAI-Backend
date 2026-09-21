"""Assemble the API response from the final graph state.

Keeps the fields the first frontend already consumes (`candidate_formulas`, `property_records`,
`krr_predictions`, `search_results`, `hypothesis`, `agent_logs`) and adds the full structured
report (candidates, criteria, hypotheses, knowledge gaps, citations, grounding, warnings).
"""

from __future__ import annotations

from typing import Any

from matsci_agent.domain.state import ResearchState

CHART_PROPERTIES = {"bandgap", "formation_energy", "density"}


def _search_result(chunk: dict) -> dict:
    meta = chunk["metadata"]
    similarity = chunk.get("similarity")
    return {
        "id": chunk["id"],
        "text": chunk["text"],
        "similarity": round(float(similarity if similarity is not None else chunk["relevance_score"]), 4),
        "relevance_score": chunk.get("relevance_score"),
        "rrf_score": chunk.get("rrf_score"),
        "metadata": {
            "arxiv_id": meta.get("arxiv_id", ""),
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("year"),
            "section": meta.get("section", ""),
            "doi": meta.get("doi", ""),
        },
    }


def _material_property(rec: dict) -> dict:
    return {
        "material_id": rec["source_id"] or "literature",
        "formula": rec["material_formula"] or rec["material_name"],
        "property_name": rec["property_name"],
        "value": rec["value_numeric"],
        "unit": rec["unit"],
        "source": rec["source_title"] or rec["method"],
        "doi": rec["source_doi"],
        "is_direct_gap": rec.get("is_direct_gap"),
        "is_predicted": rec["is_predicted"],
        "confidence": rec["confidence"],
        "method": rec["method"],
    }


def _headline(state: ResearchState) -> str:
    hypotheses = state.get("hypotheses", [])
    if hypotheses:
        return hypotheses[0]["statement"]
    candidates = state.get("screened_candidates", [])
    qualified = sum(c["status"] == "qualified" for c in candidates)
    return (
        f"{len(candidates)} candidate(s) were screened against {len(state.get('criteria', []))} criteria; "
        f"{qualified} fully qualified. No hypotheses could be generated from the available evidence."
    )


def build_report(
    state: ResearchState,
    *,
    session_id: str,
    query: str,
    max_results: int,
    mode: dict[str, Any],
    duration_ms: int,
) -> dict[str, Any]:
    candidates = state.get("screened_candidates", [])
    records = state.get("extracted_properties", [])
    formulas = [c["formula"] or c["name"] for c in candidates]

    property_records: dict[str, list[dict]] = {f: [] for f in formulas}
    krr_predictions: dict[str, dict] = {}
    for rec in records:
        key = rec["material_formula"] or rec["material_name"]
        if key not in property_records or rec["value_numeric"] is None:
            continue
        if rec["is_predicted"]:
            if rec["property_name"] == "bandgap":
                unc = rec.get("uncertainty")
                krr_predictions[key] = {
                    "formula": key, "property": "band_gap", "predicted_value": rec["value_numeric"],
                    "unit": rec["unit"], "method": rec["method"], "is_predicted": True,
                    "confidence": rec["confidence"], "uncertainty": unc,
                    "note": "Machine-learning estimate" + (f" (cross-validated RMSE ±{unc} {rec['unit']})" if unc else "")
                    + "; not a measurement.",
                }
        elif rec["property_name"] in CHART_PROPERTIES:
            property_records[key].append(rec)

    # MaterialProperty-shaped rows for the charts: database rows first, then literature.
    property_rows = {
        f: [_material_property(r) for r in sorted(rows, key=lambda r: r["method"] == "literature")]
        for f, rows in property_records.items()
    }

    seen: set[str] = set()
    warnings = [w for w in state.get("warnings", []) if not (w in seen or seen.add(w))]

    return {
        "session_id": session_id,
        "status": "COMPLETED",
        "query": query,
        "max_results": max_results,
        "search_results": [_search_result(c) for c in state.get("retrieved_papers", [])[:max_results]],
        "candidate_formulas": formulas,
        "property_records": property_rows,
        "krr_predictions": krr_predictions,
        "hypothesis": _headline(state),
        "agent_logs": state.get("agent_log", []),
        # full structured report
        "criteria": state.get("criteria", []),
        "candidates": candidates,
        "extracted_properties": records,
        "hypotheses": state.get("hypotheses", []),
        "knowledge_gaps": state.get("knowledge_gaps", []),
        "citations": state.get("citations", []),
        "grounding": state.get("grounding", {}),
        "warnings": warnings,
        "search_queries": state.get("search_queries", []),
        "retrieval_attempts": state.get("retry_count", 0) + 1,
        "mode": mode,
        "duration_ms": duration_ms,
    }
