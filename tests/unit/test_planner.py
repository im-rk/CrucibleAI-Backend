"""PlannerAgent: LLM plan, heuristic fallback, user-edited override, and its place in the graph."""

from __future__ import annotations

import asyncio

from matsci_agent.agents.nodes.planner import PlanOutput, PlannerAgent
from matsci_agent.agents.nodes.screening import CriterionDraft
from matsci_agent.agents.orchestrator import build_research_graph, initial_state, run_config
from matsci_agent.infrastructure.llm.anthropic_client import LLMError

GOAL = "Find solid-state electrolytes with ionic conductivity above 1e-3 S/cm and no rare earth elements"


def draft(**kw) -> CriterionDraft:
    base = dict(kind="numeric", label="x", property_name=None, operator=None, value=None, value_high=None, unit=None, elements=None)
    return CriterionDraft(**{**base, **kw})


class FakeLLM:
    def __init__(self, output=None, available=True, error=None):
        self.output, self.available, self.error, self.prompts = output, available, error, []

    async def structured(self, prompt, schema, **kw):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.output


def llm_plan() -> PlanOutput:
    return PlanOutput(
        summary="Screen solid electrolytes for conductivity without rare earths",
        criteria=[
            draft(label="Ionic conductivity above 1 mS/cm", property_name="ionic_conductivity", operator=">", value=1.0, unit="mS/cm"),
            draft(kind="exclude_elements", label="No rare earths", elements=["La", "Y", "Xx"]),
        ],
        queries=["argyrodite sulfide electrolyte conductivity", "  ", "argyrodite sulfide electrolyte conductivity", "LGPS lithium thiophosphate"],
        material_families=["argyrodite sulfides", "thiophosphates"],
    )


def plan_of(agent, goal=GOAL, override=None):
    warnings: list[str] = []
    return asyncio.run(agent.plan(goal, warnings, override)), warnings


def test_the_llm_plan_is_converted_to_validated_criteria_and_unique_queries() -> None:
    plan, warnings = plan_of(PlannerAgent(FakeLLM(llm_plan())))

    assert plan["source"] == "llm" and plan["summary"].startswith("Screen solid electrolytes")
    numeric, exclude = plan["criteria"]
    assert numeric["property_name"] == "ionic_conductivity" and numeric["value"] == 1e-3 and numeric["unit"] == "S/cm"  # mS/cm normalised
    assert exclude["elements"] == ["La", "Y"]  # "Xx" is not an element
    assert plan["queries"] == ["argyrodite sulfide electrolyte conductivity", "LGPS lithium thiophosphate"]
    assert plan["material_families"] == ["argyrodite sulfides", "thiophosphates"] and warnings == []


def test_without_an_llm_the_plan_comes_from_the_regex_parser_and_says_so() -> None:
    plan, warnings = plan_of(PlannerAgent(FakeLLM(available=False)))

    assert plan["source"] == "heuristic"
    labels = " ".join(c["label"] for c in plan["criteria"]).lower()
    assert "conductivity" in labels and "rare earth" in labels
    assert plan["queries"][0] == GOAL
    assert any("LLM unavailable" in w for w in warnings)


def test_an_llm_error_falls_back_to_heuristics_with_a_warning() -> None:
    plan, warnings = plan_of(PlannerAgent(FakeLLM(error=LLMError("quota"))))
    assert plan["source"] == "heuristic" and any("Planning fell back" in w and "quota" in w for w in warnings)


def test_a_user_edited_plan_overrides_the_model_and_is_validated() -> None:
    llm = FakeLLM(llm_plan())
    override = {
        "summary": "Mine",
        "criteria": [{"kind": "numeric", "label": "Bandgap > 5 eV", "property_name": "bandgap", "operator": ">", "value": 5.0, "unit": "eV"}],
        "queries": ["wide gap oxide"],
    }
    plan, warnings = plan_of(PlannerAgent(llm), override=override)

    assert plan["source"] == "user" and plan["summary"] == "Mine" and plan["queries"] == ["wide gap oxide"]
    assert plan["criteria"][0]["value"] == 5.0 and llm.prompts == [] and warnings == []  # the model was not consulted


def test_an_invalid_override_is_ignored_with_a_warning() -> None:
    plan, warnings = plan_of(PlannerAgent(FakeLLM(llm_plan())), override={"criteria": [{"label": "no kind"}]})
    assert plan["source"] == "llm"
    assert any("invalid criteria" in w for w in warnings)


def test_the_plan_node_runs_first_and_its_output_reaches_the_other_agents() -> None:
    seen: dict = {}

    class Probe:
        def __init__(self, name):
            self.name = name

        async def run(self, state, config=None):
            seen[self.name] = {"criteria": state.get("criteria"), "queries": state.get("planned_queries"), "plan": state.get("plan")}
            return {"agent_log": [{"step": self.name, "details": "ran"}], "retrieved_papers": [{"id": str(i)} for i in range(5)]}

        def should_retry(self, state):
            return False

    events: list[dict] = []

    async def emit(event):
        events.append(event)

    planner = PlannerAgent(FakeLLM(llm_plan()))
    graph = build_research_graph(Probe("search"), Probe("extract"), Probe("screen"), Probe("hypothesis"), Probe("report"), planner=planner)
    final = asyncio.run(graph.ainvoke(initial_state(GOAL), run_config(emit)))

    assert events[0] == {"event": "agent_start", "agent": "PlannerAgent"}
    assert seen["search"]["queries"] == ["argyrodite sulfide electrolyte conductivity", "LGPS lithium thiophosphate"]
    assert seen["screen"]["criteria"][0]["property_name"] == "ionic_conductivity"
    assert final["plan"]["source"] == "llm"
    plan_event = next(e for e in events if e["event"] == "plan_ready")
    assert plan_event["source"] == "llm" and len(plan_event["queries"]) == 2


def test_a_criterion_about_experimental_evidence_is_dropped_unless_the_goal_asks_for_it() -> None:
    sneaky = llm_plan()
    sneaky.criteria.append(draft(kind="measured_evidence", label="Experimentally demonstrated"))

    plan, warnings = plan_of(PlannerAgent(FakeLLM(sneaky)))
    assert "measured_evidence" not in {c["kind"] for c in plan["criteria"]}
    assert any("does not ask for it" in w for w in warnings)

    asked, none = plan_of(PlannerAgent(FakeLLM(sneaky)), goal=GOAL + " that are experimentally demonstrated")
    assert "measured_evidence" in {c["kind"] for c in asked["criteria"]} and none == []
