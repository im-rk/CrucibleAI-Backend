"""End-to-end evaluation: run whole goals through the pipeline and check what must always be true.

Two kinds of measurement:

* **invariants**: properties no correct run may violate, whatever the goal or the model. They are checked
  from the report alone (a qualified candidate really meets its numeric criteria; excluded elements never
  appear in a qualified material; a machine-learning estimate never decides a verdict; every cited
  evidence id resolves to a real source; extracted values carry provenance).
* **task success**: whether the materials a domain expert would expect appear among the qualified
  candidates, plus efficiency from the run's own trace (LLM calls, tokens, duration).

Invariants are the reliable signal: they hold for any model, and a violation is always a bug.
"""

from __future__ import annotations

from statistics import mean

from matsci_agent.agents.nodes.screening import _passes
from matsci_agent.domain.chemistry import elements_in
from matsci_agent.domain.schemas import Criterion

# Goals the pipeline should be able to answer. `expect_qualified` marks goals whose criterion the offline
# reference table (experimental band gaps) can satisfy, so a run that returns no qualified candidate failed.
CASES: list[dict] = [
    {"goal": "Find wide band gap materials with a band gap above 5 eV", "expect_qualified": True},
    {"goal": "Find semiconductors with a band gap between 1.2 and 1.6 eV for solar cells", "expect_qualified": True},
    {"goal": "Find UV LED materials with bandgap above 3.4 eV", "expect_qualified": True},
    {"goal": "Find solid-state electrolytes with ionic conductivity above 1e-3 S/cm and no rare earth elements", "expect_qualified": None},
    {"goal": "Find superconductors with a critical temperature above 77 K", "expect_qualified": None},
]


def _check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def check_report(report: dict) -> list[dict]:
    """The invariants, each as {name, passed, detail}."""
    criteria = {c["label"]: Criterion(**c) for c in report.get("criteria", [])}
    candidates = report.get("candidates", [])
    qualified = [c for c in candidates if c["status"] == "qualified"]

    numeric_violations, predicted_deciders, excluded_hits = [], [], []
    for cand in qualified:
        who = cand.get("formula") or cand.get("name")
        for res in cand.get("criteria", []):
            crit = criteria.get(res["label"])
            if crit is None:
                continue
            if crit.kind == "numeric" and res["status"] == "met":
                if res.get("value") is None or not _passes(crit, res["value"]):
                    numeric_violations.append(f"{who}: {res['label']} (value {res.get('value')})")
                if res.get("is_predicted"):
                    predicted_deciders.append(f"{who}: {res['label']}")
            if crit.kind == "exclude_elements" and cand.get("formula"):
                bad = (elements_in(cand["formula"]) or set()) & set(crit.elements)
                if bad:
                    excluded_hits.append(f"{who}: {sorted(bad)}")
        # a candidate marked qualified must have no criterion that is not met
        if any(r["status"] not in ("met",) and criteria.get(r["label"], Criterion(kind="x", label="")).kind != "qualitative" for r in cand.get("criteria", [])):
            numeric_violations.append(f"{who}: qualified with an unmet or unverified criterion")

    hypotheses = report.get("hypotheses", [])
    cited_sources = {c["source_id"] for c in report.get("citations", [])}
    unresolved = [
        i + 1 for i, h in enumerate(hypotheses)
        if h.get("unsupported_ids") or any("id" not in e for e in h.get("evidence", []))
    ]
    uncited = [
        i + 1 for i, h in enumerate(hypotheses)
        if any(e.get("source_id") and "offline" not in e["source_id"] and e["source_id"] not in cited_sources for e in h.get("evidence", []))
    ]
    literature = [r for r in report.get("extracted_properties", []) if r.get("method") == "literature"]
    no_provenance = [r for r in literature if not (r.get("source_id") and r.get("excerpt"))]

    return [
        _check("qualified_candidates_meet_their_criteria", not numeric_violations, "; ".join(numeric_violations[:3])),
        _check("excluded_elements_never_in_qualified_candidates", not excluded_hits, "; ".join(excluded_hits[:3])),
        _check("predictions_never_decide_a_verdict", not predicted_deciders, "; ".join(predicted_deciders[:3])),
        _check("hypothesis_evidence_ids_resolve", not unresolved, f"hypotheses {unresolved}" if unresolved else ""),
        _check("hypothesis_sources_are_in_the_citation_list", not uncited, f"hypotheses {uncited}" if uncited else ""),
        _check("extracted_values_carry_provenance", not no_provenance, f"{len(no_provenance)} record(s) without source/excerpt" if no_provenance else ""),
        _check("report_states_how_it_was_produced", bool(report.get("mode")) and report.get("trace") is not None and "grounding" in report),
    ]


def evaluate_case(report: dict, case: dict) -> dict:
    checks = check_report(report)
    qualified = {c.get("formula") or c.get("name") for c in report.get("candidates", []) if c["status"] == "qualified"}
    expect = case.get("expect_qualified")
    llm = ((report.get("trace") or {}).get("summary") or {}).get("llm", {})
    return {
        "goal": case["goal"],
        "invariants_passed": sum(c["passed"] for c in checks), "invariants_total": len(checks),
        "failed_invariants": [c for c in checks if not c["passed"]],
        "expected_hit": (bool(qualified) if expect else None),
        "qualified": sorted(q for q in qualified if q)[:8],
        "candidates": len(report.get("candidates", [])),
        "grounding_score": (report.get("grounding") or {}).get("score"),
        "llm_calls": llm.get("calls"), "tokens": llm.get("total_tokens"), "duration_s": round(report.get("duration_ms", 0) / 1000, 1),
        "mode": "llm" if (report.get("mode") or {}).get("llm") else "heuristic",
    }


async def run_e2e(service, cases: list[dict] | None = None) -> dict:
    from matsci_agent.domain.schemas import ResearchRequest

    rows = []
    for case in cases or CASES:
        report = await service.run(ResearchRequest(query=case["goal"], max_results=10))
        rows.append(evaluate_case(report, case))
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    expected = [r["expected_hit"] for r in rows if r["expected_hit"] is not None]
    scores = [r["grounding_score"] for r in rows if r["grounding_score"] is not None]
    return {
        "cases": rows,
        "invariant_pass_rate": round(sum(r["invariants_passed"] for r in rows) / max(sum(r["invariants_total"] for r in rows), 1), 4),
        "task_success_rate": round(sum(expected) / len(expected), 4) if expected else None,
        "mean_grounding_score": round(mean(scores), 3) if scores else None,
        "mean_duration_s": round(mean(r["duration_s"] for r in rows), 1) if rows else None,
    }
