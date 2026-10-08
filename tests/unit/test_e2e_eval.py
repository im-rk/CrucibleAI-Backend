"""The end-to-end invariants: they must pass on a sound report and catch each kind of violation."""

from __future__ import annotations

import copy

from matsci_agent.evaluation.e2e_eval import check_report, evaluate_case, summarise

GOOD = {
    "mode": {"llm": None}, "trace": {"summary": {"llm": {"calls": 3, "total_tokens": 900}}}, "duration_ms": 4200,
    "grounding": {"score": 1.0, "is_grounded": True},
    "criteria": [
        {"kind": "numeric", "label": "Bandgap > 5 eV", "property_name": "bandgap", "operator": ">", "value": 5.0, "unit": "eV"},
        {"kind": "exclude_elements", "label": "No rare earths", "elements": ["La", "Y"]},
    ],
    "candidates": [
        {"formula": "AlN", "name": "AlN", "status": "qualified", "criteria": [
            {"label": "Bandgap > 5 eV", "status": "met", "value": 6.2, "is_predicted": False},
            {"label": "No rare earths", "status": "met"}]},
        {"formula": "GaN", "name": "GaN", "status": "rejected", "criteria": [{"label": "Bandgap > 5 eV", "status": "not_met", "value": 3.4}]},
    ],
    "hypotheses": [{"statement": "x", "evidence": [{"id": "R1", "source_id": "mp-661"}], "unsupported_ids": []}],
    "citations": [{"source_id": "mp-661"}],
    "extracted_properties": [{"method": "literature", "source_id": "2301.1", "excerpt": "AlN gap 6.2 eV"}],
}


def failed(report):
    return {c["name"] for c in check_report(report) if not c["passed"]}


def mutate(**changes):
    r = copy.deepcopy(GOOD)
    for path, value in changes.items():
        node = r
        *head, last = path.split("__")
        for key in head:
            node = node[int(key)] if key.isdigit() else node[key]
        node[int(last) if last.isdigit() else last] = value
    return r


def test_a_sound_report_passes_every_invariant() -> None:
    assert failed(GOOD) == set()


def test_a_qualified_candidate_whose_value_fails_the_criterion_is_caught() -> None:
    assert "qualified_candidates_meet_their_criteria" in failed(mutate(candidates__0__criteria__0__value=4.0))


def test_a_qualified_candidate_with_an_unmet_criterion_is_caught() -> None:
    assert "qualified_candidates_meet_their_criteria" in failed(mutate(candidates__0__criteria__1__status="not_met"))


def test_an_excluded_element_in_a_qualified_candidate_is_caught() -> None:
    bad = mutate(candidates__0__formula="LaN", candidates__0__name="LaN")
    assert "excluded_elements_never_in_qualified_candidates" in failed(bad)


def test_a_prediction_deciding_a_verdict_is_caught() -> None:
    assert "predictions_never_decide_a_verdict" in failed(mutate(candidates__0__criteria__0__is_predicted=True))


def test_hypotheses_citing_unknown_evidence_or_unlisted_sources_are_caught() -> None:
    assert "hypothesis_evidence_ids_resolve" in failed(mutate(hypotheses__0__unsupported_ids=["E99"]))
    assert "hypothesis_sources_are_in_the_citation_list" in failed(mutate(citations=[]))


def test_literature_values_without_provenance_are_caught() -> None:
    assert "extracted_values_carry_provenance" in failed(mutate(extracted_properties=[{"method": "literature", "source_id": "", "excerpt": ""}]))


def test_a_report_must_say_how_it_was_produced() -> None:
    assert "report_states_how_it_was_produced" in failed(mutate(trace=None))


def test_case_evaluation_scores_expectations_and_efficiency() -> None:
    row = evaluate_case(GOOD, {"goal": "g", "expect_any": ["AlN", "MgO"]})
    assert row["expected_hit"] is True and row["qualified"] == ["AlN"] and row["llm_calls"] == 3 and row["duration_s"] == 4.2
    assert evaluate_case(GOOD, {"goal": "g", "expect_any": ["Si"]})["expected_hit"] is False
    assert evaluate_case(GOOD, {"goal": "g", "expect_any": []})["expected_hit"] is None


def test_the_summary_averages_over_cases() -> None:
    rows = [evaluate_case(GOOD, {"goal": "a", "expect_any": ["AlN"]}), evaluate_case(GOOD, {"goal": "b", "expect_any": ["Si"]})]
    s = summarise(rows)
    assert s["invariant_pass_rate"] == 1.0 and s["task_success_rate"] == 0.5 and s["mean_grounding_score"] == 1.0
