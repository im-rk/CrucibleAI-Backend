"""Unit tests for SearchAgent using a fake LLM and fake retrieval (no network, no models)."""

from __future__ import annotations

import pytest

from matsci_agent.agents.nodes.search import (
    ChunkScore,
    ReformulatedQuery,
    RelevanceScores,
    SearchAgent,
    SearchQueries,
    heuristic_relevance,
    keywords,
)
from matsci_agent.domain.schemas import PaperMetadata
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError


class FakeLLM:
    """Returns canned objects by schema type; can be told to fail."""

    def __init__(self, responses=None, available=True, fail=False):
        self.available = available
        self.responses = responses or {}
        self.fail = fail
        self.calls: list[type] = []

    async def structured(self, prompt, schema, **kwargs):
        self.calls.append(schema)
        if self.fail:
            raise LLMError("boom")
        return self.responses[schema]


class FakeSearch:
    def __init__(self, by_query: dict[str, list[dict]] | None = None, default=None):
        self.by_query = by_query or {}
        self.default = default or []
        self.queries: list[str] = []

    def search(self, query, n_results):
        self.queries.append(query)
        return [dict(c) for c in self.by_query.get(query, self.default)]


def chunk(cid: str, text: str = "ionic conductivity of LLZO", sim: float = 0.6) -> dict:
    return {"id": cid, "text": text, "similarity": sim, "metadata": {"arxiv_id": cid}}


def scores(*values: float) -> RelevanceScores:
    return RelevanceScores(scores=[ChunkScore(passage=i, score=v) for i, v in enumerate(values)])


class Events:
    def __init__(self):
        self.seen: list[dict] = []

    async def __call__(self, event):
        self.seen.append(event)

    @property
    def config(self):
        return {"configurable": {"emit": self}}

    def names(self):
        return [e["event"] for e in self.seen]


async def test_first_pass_generates_queries_and_keeps_only_relevant_chunks() -> None:
    llm = FakeLLM({
        SearchQueries: SearchQueries(queries=["llzo conductivity", "garnet electrolyte"]),
        RelevanceScores: scores(0.9, 0.3),
    })
    search = FakeSearch(default=[chunk("a"), chunk("b")])
    events = Events()
    agent = SearchAgent(llm, search)

    out = await agent.run(initial_state("solid electrolytes"), events.config)

    assert out["search_queries"] == ["llzo conductivity", "garnet electrolyte"]
    assert [c["id"] for c in out["retrieved_papers"]] == ["a"]  # b scored 0.3 < 0.65
    assert out["retry_count"] == 0
    assert events.names() == ["queries_generated", "papers_found", "relevance_check"]
    assert events.seen[-1]["relevant"] == 1 and events.seen[-1]["weak"] == 1


async def test_relevant_chunks_are_sorted_best_first() -> None:
    llm = FakeLLM({SearchQueries: SearchQueries(queries=["q"]), RelevanceScores: scores(0.7, 0.95)})
    out = await SearchAgent(llm, FakeSearch(default=[chunk("a"), chunk("b")])).run(initial_state("g"))
    assert [c["id"] for c in out["retrieved_papers"]] == ["b", "a"]


async def test_retry_reformulates_scores_only_new_chunks_and_counts_attempt() -> None:
    llm = FakeLLM({
        ReformulatedQuery: ReformulatedQuery(query="better query"),
        RelevanceScores: scores(0.9),  # only the single new chunk is scored
    })
    search = FakeSearch(by_query={"better query": [chunk("a"), chunk("new")]})
    state = initial_state("goal")
    state.update(
        search_queries=["first"],
        retrieved_chunks=[{**chunk("a"), "relevance_score": 0.7, "relevance_method": "llm"}],
        retrieved_papers=[{**chunk("a"), "relevance_score": 0.7}],
    )
    events = Events()

    out = await SearchAgent(llm, search, live_arxiv=False).run(state, events.config)

    assert out["retry_count"] == 1
    assert out["search_queries"] == ["first", "better query"]
    assert {c["id"] for c in out["retrieved_papers"]} == {"a", "new"}
    assert "re_retrieving" in events.names()
    assert llm.calls == [ReformulatedQuery, RelevanceScores]


def test_should_retry_only_when_weak_and_attempts_remain() -> None:
    agent = SearchAgent(FakeLLM(), FakeSearch(), min_relevant=2, max_retries=3)
    weak = {"retrieved_papers": [chunk("a")], "retry_count": 1}
    assert agent.should_retry(weak)
    assert not agent.should_retry({**weak, "retry_count": 3})  # retries exhausted
    assert not agent.should_retry({"retrieved_papers": [chunk("a"), chunk("b")], "retry_count": 0})


async def test_llm_failure_falls_back_to_heuristics_with_a_warning() -> None:
    llm = FakeLLM(fail=True)
    strong = chunk("a", "solid state electrolyte ionic conductivity measured", sim=0.7)
    weak = chunk("b", "unrelated text about banana bread", sim=0.15)
    out = await SearchAgent(llm, FakeSearch(default=[strong, weak])).run(
        initial_state("solid state electrolyte ionic conductivity")
    )
    assert [c["id"] for c in out["retrieved_papers"]] == ["a"]
    assert any("heuristics" in w for w in out["warnings"])


async def test_one_failed_batch_falls_back_only_for_that_batch() -> None:
    class FlakyLLM(FakeLLM):
        async def structured(self, prompt, schema, **kwargs):
            if schema is SearchQueries:
                return SearchQueries(queries=["q"])
            if "[0] first-batch" in prompt:
                return scores(*([0.9] * 8))
            raise LLMError("second batch down")

    first = [chunk(f"a{i}", "first-batch text") for i in range(8)]
    second = [chunk("b0", "ionic conductivity solid electrolyte", sim=0.7)]
    events = Events()
    out = await SearchAgent(FlakyLLM(), FakeSearch(default=first + second)).run(
        initial_state("ionic conductivity solid electrolyte"), events.config
    )
    methods = {c["id"]: c["relevance_method"] for c in out["retrieved_chunks"]}
    assert {methods[f"a{i}"] for i in range(8)} == {"llm"}
    assert methods["b0"] == "heuristic"
    assert events.seen[-1]["method"] == "mixed"
    assert any("heuristics" in w for w in out["warnings"])


async def test_no_llm_configured_warns_and_still_searches() -> None:
    llm = LLMClient(api_key=None, model="m")
    assert not llm.available
    out = await SearchAgent(llm, FakeSearch(default=[chunk("a", sim=0.7)])).run(
        initial_state("ionic conductivity")
    )
    assert out["search_queries"] == ["ionic conductivity"]
    assert any("LLM not configured" in w for w in out["warnings"])
    with pytest.raises(LLMError):
        await llm.structured("x", SearchQueries)


async def test_live_arxiv_abstracts_join_the_pool_on_retry() -> None:
    class FakeArxiv:
        def search_abstracts(self, query, n):
            return [PaperMetadata(arxiv_id="2401.1", title="Live", authors=["A"], abstract="abs", year=2024)]

    llm = FakeLLM({ReformulatedQuery: ReformulatedQuery(query="q2"), RelevanceScores: scores(0.9)})
    state = initial_state("goal")
    state["search_queries"] = ["q1"]
    out = await SearchAgent(llm, FakeSearch(), FakeArxiv()).run(state)
    assert [c["id"] for c in out["retrieved_papers"]] == ["2401.1_abstract"]
    assert out["retrieved_papers"][0]["metadata"]["section"].startswith("Abstract")


async def test_live_arxiv_failure_is_a_warning_not_a_crash() -> None:
    class BrokenArxiv:
        def search_abstracts(self, query, n):
            raise OSError("no network")

    llm = FakeLLM({ReformulatedQuery: ReformulatedQuery(query="q2")})
    state = initial_state("goal")
    state["search_queries"] = ["q1"]
    out = await SearchAgent(llm, FakeSearch(), BrokenArxiv()).run(state)
    assert any("Live arXiv search unavailable" in w for w in out["warnings"])


def test_heuristic_relevance_orders_on_topic_above_off_topic() -> None:
    goal = "solid state electrolyte ionic conductivity"
    on = heuristic_relevance(goal, chunk("a", "the ionic conductivity of this solid electrolyte", 0.6))
    off = heuristic_relevance(goal, chunk("b", "sourdough bread recipe", 0.16))
    assert on > 0.8 and off < 0.1


def test_keywords_drops_stopwords_duplicates_and_bare_numbers_but_keeps_short_terms() -> None:
    assert keywords("Find the materials with high high conductivity") == ["conductivity"]
    assert keywords("UV LED bandgap above 3.4 eV") == ["uv", "led", "bandgap", "ev"]
