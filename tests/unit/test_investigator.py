"""Tool registry, the bounded tool loop, and the InvestigatorAgent (evidence gap filling)."""

from __future__ import annotations

import asyncio

from matsci_agent.agents.nodes.investigator import InvestigatorAgent
from matsci_agent.agents.orchestrator import build_research_graph, initial_state, run_config
from matsci_agent.agents.toolloop import AgentStep, Argument, run_tool_loop
from matsci_agent.agents.tools import Tool, ToolRegistry
from matsci_agent.domain.schemas import PropertyRecord
from matsci_agent.infrastructure.llm.anthropic_client import LLMError


def run(coro):
    return asyncio.run(coro)


# ---- tools ---------------------------------------------------------------------------------------


def registry() -> ToolRegistry:
    reg = ToolRegistry()

    async def slow(query: str) -> str:
        await asyncio.sleep(1)
        return "late"

    def boom(query: str) -> str:
        raise ValueError("bad thing")

    reg.register(Tool("echo", "Echo.", {"text": "what to echo"}, lambda text: f"echo:{text}"))
    reg.register(Tool("slow", "Slow.", {"query": "q"}, slow, timeout_s=0.05))
    reg.register(Tool("boom", "Fails.", {"query": "q"}, boom))
    reg.register(Tool("long", "Long output.", {"query": "q"}, lambda query: "x" * 5000))
    return reg


def test_a_tool_call_returns_its_result() -> None:
    assert run(registry().call("echo", {"text": "hi"})) == ("echo:hi", True)


def test_tool_failures_become_observations_not_exceptions() -> None:
    reg = registry()
    assert "no tool named 'nope'" in run(reg.call("nope", {}))[0]
    assert "needs the argument(s): text" in run(reg.call("echo", {}))[0]
    assert "timed out" in run(reg.call("slow", {"query": "q"}))[0]
    text, ok = run(reg.call("boom", {"query": "q"}))
    assert not ok and "ValueError: bad thing" in text


def test_long_observations_are_truncated() -> None:
    text, ok = run(registry().call("long", {"query": "q"}))
    assert ok and len(text) < 1000 and text.endswith("...")


def test_the_registry_describes_its_tools_for_the_prompt() -> None:
    assert "- echo: Echo. Arguments: text (what to echo)." in registry().describe()


# ---- the loop ---------------------------------------------------------------------------------------


class Scripted:
    """An LLM that plays back prepared steps and records the prompts it was given."""

    available = True

    def __init__(self, steps):
        self.steps, self.prompts = list(steps), []

    async def structured(self, prompt, schema, **kw):
        self.prompts.append(prompt)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def step(action, thought="t", answer=None, **args):
    return AgentStep(thought=thought, action=action, arguments=[Argument(name=k, value=v) for k, v in args.items()], answer=answer)


def test_the_loop_calls_tools_feeds_back_observations_and_finishes() -> None:
    llm = Scripted([step("echo", text="one"), step("echo", text="two"), step("finish", answer="done")])
    result = run(run_tool_loop(llm, registry(), "task", max_steps=5))

    assert result.stopped == "finish" and result.answer == "done"
    assert [s["action"] for s in result.steps] == ["echo", "echo", "finish"]
    assert "echo:one" in llm.prompts[1] and "<document>" in llm.prompts[1] and "3 step(s) left" in llm.prompts[2]


def test_the_loop_stops_at_the_step_cap() -> None:
    llm = Scripted([step("echo", text=str(i)) for i in range(10)])
    result = run(run_tool_loop(llm, registry(), "task", max_steps=3))
    assert result.stopped == "max_steps" and len(result.steps) == 3


def test_a_repeated_identical_call_is_refused_and_a_second_repeat_stops_the_loop() -> None:
    llm = Scripted([step("echo", text="a"), step("echo", text="a"), step("echo", text="a"), step("finish")])
    result = run(run_tool_loop(llm, registry(), "task", max_steps=6))
    assert result.stopped == "repeat"
    assert "already made this exact call" in result.steps[1]["observation"]


def test_an_unknown_tool_is_an_observation_the_agent_can_react_to() -> None:
    llm = Scripted([step("teleport"), step("finish", answer="ok")])
    result = run(run_tool_loop(llm, registry(), "task"))
    assert "no tool named 'teleport'" in result.steps[0]["observation"] and result.stopped == "finish"


def test_an_llm_error_ends_the_loop_with_the_steps_so_far() -> None:
    llm = Scripted([step("echo", text="a"), LLMError("quota")])
    result = run(run_tool_loop(llm, registry(), "task"))
    assert result.stopped == "llm_error" and len(result.steps) == 1


def test_tool_arguments_may_be_a_plain_object_or_a_list_of_pairs() -> None:
    as_object = AgentStep(thought="t", action="search_literature", arguments={"query": "AlN band gap"}, answer=None)
    as_pairs = AgentStep(thought="t", action="search_literature", arguments=[Argument(name="query", value="AlN band gap")], answer=None)
    assert [(a.name, a.value) for a in as_object.arguments] == [(a.name, a.value) for a in as_pairs.arguments] == [("query", "AlN band gap")]


# ---- the investigator ------------------------------------------------------------------------------------


def candidate(formula, origin="database", status="qualified", criteria_status="met"):
    return {
        "formula": formula, "name": formula, "status": status, "origin": origin, "rank": 1,
        "criteria": [{"label": "Bandgap > 3.4 eV", "kind": "numeric", "status": criteria_status}],
    }


def chunk(cid, text):
    return {"id": cid, "text": text, "metadata": {"arxiv_id": cid, "title": f"Paper {cid}", "year": 2024, "doi": ""}}


class FakeSearchAgent:
    def __init__(self, chunks):
        self.chunks, self.queries = chunks, []

    async def targeted_search(self, query, goal, *, n, warnings):
        self.queries.append(query)
        return [dict(c) for c in self.chunks]


class FakeExtraction:
    def __init__(self, records):
        self.records, self.seen = records, []

    async def extract_chunks(self, chunks, warnings):
        self.seen.extend(c["id"] for c in chunks)
        return self.records


class FakeMP:
    online = True

    def get_properties(self, formula):
        return []


class FakeKRR:
    def predict(self, formula, prop):
        return {"error": "none"}


def record(formula, value=6.2, source="2301.00001"):
    return PropertyRecord(
        material_name=formula, material_formula=formula, property_name="bandgap", value_raw=f"{value} eV",
        value_numeric=value, unit="eV", source_title="T", source_id=source, method="literature",
    )


def investigator(llm, chunks, records, **kw):
    search, extraction = FakeSearchAgent(chunks), FakeExtraction(records)
    return InvestigatorAgent(llm, search, extraction, FakeMP(), FakeKRR(), **kw), search, extraction


def state_with(*candidates):
    state = initial_state("goal")
    state["screened_candidates"] = list(candidates)
    state["criteria"] = [{"label": "Bandgap > 3.4 eV"}]
    state["retrieved_papers"] = [chunk("old", "unrelated")]
    return state


def test_an_agent_run_searches_extracts_and_requests_a_rescreen() -> None:
    llm = Scripted([step("search_literature", query="AlN measured band gap"), step("finish", answer="found a paper")])
    agent, search, extraction = investigator(llm, [chunk("new1", "AlN has a measured band gap of 6.2 eV."), chunk("noise", "nothing relevant")], [record("AlN")])
    events: list[dict] = []

    async def emit(e):
        events.append(e)

    out = run(agent.run(state_with(candidate("AlN"), candidate("GaN", origin="literature")), {"configurable": {"emit": emit}}))
    inv = out["investigation"]

    assert inv["mode"] == "agent" and inv["targets"] == ["AlN"] and inv["answer"] == "found a paper"
    assert [s["action"] for s in inv["steps"]] == ["search_literature", "finish"]
    assert extraction.seen == ["new1"]  # the chunk that never mentions the target is not extracted
    assert inv["new_passages"] == 1 and inv["supported"] == ["AlN"] and inv["rescreen"] is True
    assert out["rescreen_count"] == 1 and any(r["material_formula"] == "AlN" for r in out["extracted_properties"])
    assert out["retrieved_papers"][0]["id"] == "new1"  # the new evidence is ranked ahead of the unrelated chunk
    assert [e["event"] for e in events] == ["investigation_start", "tool_call", "tool_call", "investigation_done"]
    assert agent.should_rescreen({**state_with(), **out})


def test_without_an_llm_it_runs_one_targeted_search_per_candidate() -> None:
    class NoLLM:
        available = False

    agent, search, _ = investigator(NoLLM(), [chunk("n", "AlN data")], [record("AlN")])
    out = run(agent.run(state_with(candidate("AlN"), candidate("MgO"))))

    assert out["investigation"]["mode"] == "workflow" and len(search.queries) == 2
    assert all(q.startswith(("AlN", "MgO")) for q in search.queries)


def test_nothing_is_investigated_when_there_are_no_gaps() -> None:
    agent, search, _ = investigator(Scripted([]), [], [])
    out = run(agent.run(state_with(candidate("GaN", origin="literature"), candidate("Si", status="rejected"))))
    assert out["investigation"] == {"targets": [], "rescreen": False} and search.queries == []


def test_a_candidate_with_a_predicted_value_is_a_target_and_targets_are_capped() -> None:
    agent, _, _ = investigator(Scripted([]), [], [], max_targets=2)
    targets = agent._targets([candidate(f"X{i}", origin="literature", criteria_status="predicted_met") for i in range(5)])
    assert len(targets) == 2


def test_no_new_support_means_no_rescreen() -> None:
    class NoLLM:
        available = False

    agent, _, _ = investigator(NoLLM(), [chunk("n", "AlN mentioned")], [])  # extraction finds no values
    out = run(agent.run(state_with(candidate("AlN"))))
    assert out["investigation"]["rescreen"] is False and out["rescreen_count"] == 0


def test_a_second_visit_is_skipped() -> None:
    agent, search, _ = investigator(Scripted([]), [], [])
    state = {**state_with(candidate("AlN")), "rescreen_count": 1}
    assert run(agent.run(state))["investigation"]["skipped"] and search.queries == []


def test_the_graph_rescreens_exactly_once_after_new_evidence() -> None:
    calls: list[str] = []

    class Probe:
        def __init__(self, name, update=None):
            self.name, self.update = name, update or {}

        async def run(self, state, config=None):
            calls.append(self.name)
            return {"agent_log": [{"step": self.name, "details": "ran"}], **self.update}

        def should_retry(self, state):
            return False

    class Finds(InvestigatorAgent):
        def __init__(self):
            self.n = 0

        async def run(self, state, config=None):
            calls.append("investigate")
            self.n += 1
            found = state.get("rescreen_count", 0) == 0
            return {"investigation": {"rescreen": found}, "rescreen_count": state.get("rescreen_count", 0) + (1 if found else 0)}

    graph = build_research_graph(
        Probe("search", {"retrieved_papers": [{"id": "1"}] * 5}), Probe("extract"), Probe("screen"),
        Probe("hypothesis"), Probe("report"), investigator=Finds(),
    )
    run(graph.ainvoke(initial_state("goal"), run_config()))
    assert calls == ["search", "extract", "screen", "investigate", "screen", "investigate", "hypothesis", "report"]
