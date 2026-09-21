"""Unit tests for criteria parsing and ScreeningAgent (fake LLM / KRR / Materials Project)."""

from __future__ import annotations

import pytest

from matsci_agent.agents.nodes.screening import (
    CriteriaOutput,
    CriterionDraft,
    ScreeningAgent,
    parse_criteria_heuristic,
)
from matsci_agent.domain.chemistry import RARE_EARTH
from matsci_agent.domain.schemas import PropertyRecord
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.llm.anthropic_client import LLMError

GOAL = (
    "Find solid-state electrolyte materials with ionic conductivity above 10⁻³ S/cm, "
    "electrochemical stability above 5V, no rare earth elements, and demonstrated experimentally. "
    "Must be air stable."
)


class FakeLLM:
    def __init__(self, available=False, criteria=None, error=None):
        self.available, self.criteria, self.error = available, criteria, error

    async def structured(self, prompt, schema, **kwargs):
        if self.error:
            raise self.error
        return CriteriaOutput(criteria=self.criteria)


class FakeKRR:
    def __init__(self, value=3.3, error=None):
        self.value, self.error, self.calls = value, error, []

    def predict(self, formula, property_name):
        self.calls.append((formula, property_name))
        if self.error:
            return {"error": self.error}
        return {
            "predicted_value": self.value, "unit": "eV", "method": "Kernel Ridge Regression (RBF kernel)",
            "confidence": "low", "uncertainty": 0.5,
        }


class FakeMP:
    def __init__(self, online=False, rows=None):
        self.online, self.rows, self.asked = online, rows or [], None

    def screen_by_criteria(self, low, high, stable, system, limit):
        self.asked = (low, high)
        return self.rows


def rec(formula, prop, value, unit, method="literature", name=None, source="p1", **kw) -> dict:
    return PropertyRecord(
        material_name=name or formula, material_formula=formula, property_name=prop,
        value_raw=f"{value} {unit}", value_numeric=value, unit=unit, source_title=f"Paper {source}",
        source_id=source, method=method, **kw,
    ).model_dump()


def run_state(goal, records):
    state = initial_state(goal)
    state["extracted_properties"] = records
    return state


# ---- criteria parsing ----------------------------------------------------------------------


def test_heuristic_parses_the_headline_example_goal() -> None:
    criteria = {c.label: c for c in parse_criteria_heuristic(GOAL)}
    conductivity = next(c for c in criteria.values() if c.property_name == "ionic_conductivity")
    assert (conductivity.operator, conductivity.value, conductivity.unit) == (">", pytest.approx(1e-3), "S/cm")
    window = next(c for c in criteria.values() if c.property_name == "electrochemical_stability_window")
    assert (window.operator, window.value, window.unit) == (">", 5.0, "V")
    excluded = next(c for c in criteria.values() if c.kind == "exclude_elements")
    assert set(excluded.elements) == RARE_EARTH
    assert any(c.kind == "measured_evidence" for c in criteria.values())
    assert any(c.kind == "qualitative" for c in criteria.values())


def test_heuristic_parses_bandgap_and_between() -> None:
    (gap,) = [c for c in parse_criteria_heuristic("UV LED materials with bandgap above 3.4 eV") if c.kind == "numeric"]
    assert (gap.property_name, gap.operator, gap.value) == ("bandgap", ">", 3.4)
    (window,) = [c for c in parse_criteria_heuristic("band gap between 1.0 and 1.6 eV") if c.kind == "numeric"]
    assert (window.operator, window.value, window.value_high) == ("between", 1.0, 1.6)


def test_heuristic_handles_properties_without_a_unit_pattern() -> None:
    (crit,) = parse_criteria_heuristic("bulk modulus above 100")
    assert crit.property_name == "bulk_modulus" and crit.value == 100.0


async def test_llm_criteria_are_normalised_to_canonical_units() -> None:
    draft = CriterionDraft(
        kind="numeric", label="Conductivity", property_name="ionic conductivity", operator=">",
        value=1.0, value_high=None, unit="mS/cm", elements=None,
    )
    out = await ScreeningAgent(FakeLLM(True, [draft]), FakeKRR(), FakeMP()).run(run_state("g", []))
    assert out["criteria"][0]["value"] == pytest.approx(1e-3) and out["criteria"][0]["unit"] == "S/cm"


async def test_llm_failure_falls_back_to_regex_with_a_warning() -> None:
    out = await ScreeningAgent(FakeLLM(True, error=LLMError("down")), FakeKRR(), FakeMP()).run(
        run_state("bandgap above 3 eV", [])
    )
    assert out["criteria"][0]["property_name"] == "bandgap"
    assert any("regex heuristics" in w for w in out["warnings"])


# ---- screening & ranking ---------------------------------------------------------------------


async def test_qualified_candidates_rank_above_partial_and_rejected() -> None:
    records = [
        # LGPS-like: passes both numbers, no rare earth, measured in literature
        rec("Li10GeP2S12", "ionic_conductivity", 1.2e-2, "S/cm", source="a"),
        rec("Li10GeP2S12", "electrochemical_stability_window", 5.5, "V", source="a"),
        # LLZO: numbers pass but contains lanthanum
        rec("Li7La3Zr2O12", "ionic_conductivity", 2e-3, "S/cm", source="b"),
        rec("Li7La3Zr2O12", "electrochemical_stability_window", 6.0, "V", source="b"),
        # Li3PS4: conductivity too low
        rec("Li3PS4", "ionic_conductivity", 1e-4, "S/cm", source="c"),
        rec("Li3PS4", "electrochemical_stability_window", 5.2, "V", source="c"),
    ]
    goal = "ionic conductivity above 1e-3 S/cm, electrochemical stability above 5 V, no rare earth"
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state(goal, records))

    ranked = [(c["rank"], c["formula"], c["status"]) for c in out["screened_candidates"]]
    assert ranked[0] == (1, "Li10GeP2S12", "qualified")
    assert {f: s for _, f, s in ranked} == {
        "Li10GeP2S12": "qualified", "Li7La3Zr2O12": "rejected", "Li3PS4": "rejected",
    }
    llzo = next(c for c in out["screened_candidates"] if c["formula"] == "Li7La3Zr2O12")
    assert "contains La" in next(r for r in llzo["criteria"] if r["kind"] == "exclude_elements")["detail"]


async def test_conflicting_reports_use_the_favourable_value_and_disclose_it() -> None:
    records = [
        rec("GaN", "bandgap", 3.2, "eV", source="a"),
        rec("GaN", "bandgap", 3.5, "eV", source="b"),
    ]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state("bandgap above 3.4 eV", records))
    result = out["screened_candidates"][0]["criteria"][0]
    assert result["status"] == "met" and result["value"] == 3.5
    assert "best of 2" in result["detail"]


async def test_literature_values_take_priority_over_database_values() -> None:
    records = [
        rec("ZnO", "bandgap", 3.37, "eV", source="paper"),
        rec("ZnO", "bandgap", 5.0, "eV", method="Materials Project (DFT)", source="mp-1"),
    ]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state("bandgap above 4 eV", records))
    assert out["screened_candidates"][0]["criteria"][0]["status"] == "not_met"  # 3.37 measured, not 5.0 DFT


async def test_missing_property_is_predicted_by_krr_and_never_counts_as_qualified() -> None:
    krr = FakeKRR(value=3.3)
    records = [rec("ZnO", "ionic_conductivity", 1e-9, "S/cm")]
    out = await ScreeningAgent(FakeLLM(), krr, FakeMP()).run(run_state("bandgap above 3.0 eV", records))

    cand = out["screened_candidates"][0]
    assert krr.calls == [("ZnO", "band_gap")]
    assert cand["criteria"][0]["status"] == "predicted_met" and cand["criteria"][0]["is_predicted"]
    assert cand["status"] == "partial" and "predicted_only:bandgap" in cand["flags"]
    assert cand["score"] == 0.5
    predicted = [r for r in out["extracted_properties"] if r["is_predicted"]]
    assert predicted and predicted[0]["property_name"] == "bandgap" and predicted[0]["method"].startswith("Kernel Ridge")


async def test_property_without_a_predictor_is_reported_missing() -> None:
    records = [rec("Li3PS4", "bandgap", 3.0, "eV")]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state("ionic conductivity above 1e-3 S/cm", records))
    cand = out["screened_candidates"][0]
    assert cand["criteria"][0]["status"] == "missing" and "missing:ionic_conductivity" in cand["flags"]
    assert cand["status"] == "partial"


async def test_krr_errors_become_warnings_not_crashes() -> None:
    records = [rec("ZnO", "ionic_conductivity", 1e-9, "S/cm")]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(error="no model"), FakeMP()).run(run_state("bandgap above 3 eV", records))
    assert out["screened_candidates"][0]["criteria"][0]["status"] == "missing"
    assert any("KRR prediction skipped" in w for w in out["warnings"])


async def test_measured_evidence_requires_a_literature_record() -> None:
    records = [
        rec("GaN", "bandgap", 3.4, "eV", method="offline reference"),
        rec("ZnO", "bandgap", 3.37, "eV", method="literature"),
    ]
    goal = "materials that are experimentally demonstrated"
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state(goal, records))
    by_formula = {c["formula"]: c["criteria"][0]["status"] for c in out["screened_candidates"]}
    assert by_formula == {"GaN": "not_met", "ZnO": "met"}


async def test_database_screen_seeds_candidates_when_literature_is_thin() -> None:
    mp = FakeMP(rows=[
        {"formula": "AlN", "bandgap_eV": 6.2, "material_id": "offline-reference", "source": "Offline reference"},
        {"formula": "ZnO", "bandgap_eV": 3.37, "material_id": "offline-reference"},
    ])
    records = [rec("ZnO", "bandgap", 3.37, "eV")]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), mp).run(run_state("bandgap above 3.0 eV", records))
    assert mp.asked == (3.0, None)
    origins = {c["formula"]: c["origin"] for c in out["screened_candidates"]}
    assert origins == {"ZnO": "literature", "AlN": "database"}  # ZnO not duplicated
    assert any("database band-gap screen" in w for w in out["warnings"])


async def test_no_criteria_yields_unscreened_candidates_and_a_warning() -> None:
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state("something vague", [rec("GaN", "bandgap", 3.4, "eV")]))
    assert out["screened_candidates"][0]["status"] == "unscreened"
    assert any("No screening criteria" in w for w in out["warnings"])


async def test_unattributed_values_are_counted_not_silently_dropped() -> None:
    records = [rec(None, "bandgap", 3.0, "eV", name="Unknown"), rec("GaN", "bandgap", 3.4, "eV")]
    out = await ScreeningAgent(FakeLLM(), FakeKRR(), FakeMP()).run(run_state("bandgap above 3 eV", records))
    assert any("could not be attributed" in w for w in out["warnings"])
    assert [c["formula"] for c in out["screened_candidates"]] == ["GaN"]
