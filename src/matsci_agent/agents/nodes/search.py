"""SearchAgent: query generation, hybrid retrieval, ISREL relevance scoring, re-retrieval."""

from __future__ import annotations

import asyncio
import logging
import re

from pydantic import BaseModel

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.domain.state import ResearchState
from matsci_agent.observability.metrics import METRICS
from matsci_agent.observability.trace import trace_span
from matsci_agent.security.guardrails import scan, wrap_untrusted
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "with find materials material which using use high low above below over under no not".split()
)
# Extra terms appended by the deterministic reformulation used when no LLM is available.
FALLBACK_SUFFIXES = ("experimental measurement", "density functional theory", "synthesis properties")

LLM_THRESHOLD = 0.65  # overridden by settings.relevance_threshold
# ponytail: the no-LLM scorer blends cosine and keyword overlap; it needs a lower bar than an
# LLM judge because MiniLM cosine for on-topic passages rarely exceeds ~0.6. Swap for a
# calibrated classifier if the LLM path is ever unavailable in production.
HEURISTIC_THRESHOLD = 0.45
BATCH_SIZE = 8


class SearchQueries(BaseModel):
    queries: list[str]


class ChunkScore(BaseModel):
    passage: int
    score: float


class RelevanceScores(BaseModel):
    scores: list[ChunkScore]


class ReformulatedQuery(BaseModel):
    query: str


def keywords(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if len(token) >= 2 and not token.isdigit() and token not in STOPWORDS:
            seen.setdefault(token)
    return list(seen)


def heuristic_relevance(goal: str, chunk: dict) -> float:
    """Blend of scaled cosine similarity and the share of goal keywords present in the text."""
    goal_terms = keywords(goal)
    text = chunk["text"].lower()
    coverage = sum(t in text for t in goal_terms) / len(goal_terms) if goal_terms else 0.0
    cosine = chunk.get("similarity")
    if cosine is None:
        return coverage
    scaled = min(1.0, max(0.0, (cosine - 0.15) / 0.35))
    return 0.7 * scaled + 0.3 * coverage


class SearchAgent:
    """Finds relevant paper chunks for a research goal, re-retrieving when results are weak."""

    def __init__(
        self,
        llm: LLMClient,
        search,  # HybridSearch
        arxiv=None,  # ArxivClient, used only for the live fallback on retries
        *,
        relevance_threshold: float = LLM_THRESHOLD,
        min_relevant: int = 5,
        max_retries: int = 3,
        live_arxiv: bool = True,
        concurrency: int = 4,
    ) -> None:
        self.llm = llm
        self.search = search
        self.arxiv = arxiv
        self.relevance_threshold = relevance_threshold
        self.min_relevant = min_relevant
        self.max_retries = max_retries
        self.live_arxiv = live_arxiv
        self._semaphore = asyncio.Semaphore(concurrency)

    # ---- graph interface -------------------------------------------------------------

    async def run(self, state: ResearchState, config=None) -> dict:
        goal = state["research_goal"]
        pool_size = max(state.get("max_results", 10), 10)
        tried = list(state.get("search_queries", []))
        attempt = state.get("retry_count", 0)
        warnings: list[str] = []

        if not tried:
            queries = list(state.get("planned_queries") or []) or await self._generate_queries(goal, warnings)
            await emit(config, "queries_generated", queries=queries)
        else:
            attempt += 1
            query = await self._reformulate(goal, tried, state.get("retrieved_papers", []), attempt, warnings)
            queries = [query]
            await emit(config, "re_retrieving", new_query=query, attempt=attempt)

        found: list[dict] = []
        for query in queries:
            with trace_span("retrieval", "hybrid_search", query=query[:100]) as span:
                hits = await asyncio.to_thread(self.search.search, query, pool_size)
                span["results"] = len(hits)
            found.extend(hits)
        if tried and self.live_arxiv and self.arxiv is not None:
            found.extend(await self._live_abstracts(queries[-1], warnings))

        pool = {c["id"]: c for c in state.get("retrieved_chunks", [])}
        fresh = list({c["id"]: c for c in found if c["id"] not in pool}.values())
        flagged = [c for c in fresh if scan(c["text"])]
        if flagged:
            METRICS.inc("injection_flags_total", len(flagged), source="retrieval")
            warnings.append(
                f"{len(flagged)} retrieved passage(s) contained instruction-like text (possible prompt injection); "
                "it was neutralised before reaching the model."
            )
        await emit(config, "papers_found", count=len(pool) + len(fresh), new=len(fresh))

        method, threshold = await self._score(goal, fresh, warnings)
        pool.update({c["id"]: c for c in fresh})
        relevant = sorted(
            (c for c in pool.values() if c["relevance_score"] >= self._threshold_for(c)),
            key=lambda c: c["relevance_score"],
            reverse=True,
        )
        await emit(
            config,
            "relevance_check",
            relevant=len(relevant),
            weak=len(pool) - len(relevant),
            threshold=threshold,
            method=method,
        )

        detail = (
            f"{'Retry ' + str(attempt) + ': ' if tried else ''}retrieved {len(pool)} unique chunks, "
            f"{len(relevant)} relevant ({method} scoring, threshold {threshold})"
        )
        return {
            "search_queries": tried + queries,
            "retrieved_chunks": list(pool.values()),
            "retrieved_papers": relevant,
            "retry_count": attempt,
            "warnings": warnings,
            "agent_log": [{"step": "LITERATURE_SEARCH", "details": detail}],
        }

    async def targeted_search(self, query: str, goal: str, *, n: int = 5, warnings: list[str]) -> list[dict]:
        """One-off retrieval for a specific question: hybrid search plus live arXiv abstracts, best first."""
        with trace_span("retrieval", "targeted_search", query=query[:100]) as span:
            hits = await asyncio.to_thread(self.search.search, query, n)
            if self.live_arxiv and self.arxiv is not None:
                hits = hits + await self._live_abstracts(query, warnings, timeout_s=12)
            span["results"] = len(hits)
        unique = {c["id"]: c for c in hits}.values()
        for chunk in unique:
            chunk["relevance_score"] = round(heuristic_relevance(goal, chunk), 4)
            chunk.setdefault("relevance_method", "heuristic")
        return sorted(unique, key=lambda c: c["relevance_score"], reverse=True)[: n + 3]

    def should_retry(self, state: ResearchState) -> bool:
        """Conditional-edge predicate: too few relevant chunks and retries remain."""
        return (
            len(state.get("retrieved_papers", [])) < self.min_relevant
            and state.get("retry_count", 0) < self.max_retries
        )

    # ---- query generation ------------------------------------------------------------

    async def _generate_queries(self, goal: str, warnings: list[str]) -> list[str]:
        if self.llm.available:
            try:
                out = await self.llm.structured(
                    prompts.GENERATE_QUERIES_PROMPT.format(goal=goal),
                    SearchQueries,
                    system=prompts.SEARCH_SYSTEM,
                    max_tokens=500,
                )
                queries = [q.strip() for q in out.queries if q.strip()][:5]
                if queries:
                    return queries
            except LLMError as exc:
                warnings.append(f"Query generation fell back to heuristics: {exc}")
        else:
            warnings.append("LLM unavailable: using the research goal as the search query.")
        terms = " ".join(keywords(goal))
        return [goal] if terms == goal.lower() or not terms else [goal, terms]

    async def _reformulate(
        self, goal: str, tried: list[str], weak: list[dict], attempt: int, warnings: list[str]
    ) -> str:
        if self.llm.available:
            try:
                found = "\n".join(f"- {c['text'][:100]}" for c in weak[:3]) or "- (nothing relevant)"
                out = await self.llm.structured(
                    prompts.REFORMULATE_PROMPT.format(
                        goal=goal, tried="\n".join(f"- {q}" for q in tried), found=found
                    ),
                    ReformulatedQuery,
                    system=prompts.SEARCH_SYSTEM,
                    max_tokens=200,
                )
                if out.query.strip() and out.query.strip() not in tried:
                    return out.query.strip()
            except LLMError as exc:
                warnings.append(f"Query reformulation fell back to heuristics: {exc}")
        base = " ".join(keywords(goal)) or goal
        suffix = FALLBACK_SUFFIXES[(attempt - 1) % len(FALLBACK_SUFFIXES)]
        return f"{base} {suffix}"

    # ---- relevance (ISREL) -----------------------------------------------------------

    def _threshold_for(self, chunk: dict) -> float:
        return HEURISTIC_THRESHOLD if chunk.get("relevance_method") == "heuristic" else self.relevance_threshold

    async def _score(self, goal: str, chunks: list[dict], warnings: list[str]) -> tuple[str, float]:
        """Set `relevance_score` / `relevance_method` on each chunk; return (method, threshold).

        Batches the LLM couldn't score fall back to the heuristic individually, so one failed
        batch never discards the rest.
        """
        if not chunks:
            return ("llm" if self.llm.available else "heuristic"), self._method_threshold(self.llm.available)

        failed: list[dict] = list(chunks)
        if self.llm.available:
            batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
            results = await asyncio.gather(
                *(self._score_batch(goal, b) for b in batches), return_exceptions=True
            )
            failed = []
            for batch, result in zip(batches, results):
                if isinstance(result, LLMError):
                    failed.extend(batch)
                    warnings.append(f"Relevance scoring fell back to heuristics: {result}")
                elif isinstance(result, BaseException):
                    raise result  # a bug, not an LLM outage: don't mask it
        for chunk in failed:
            chunk["relevance_score"] = round(heuristic_relevance(goal, chunk), 4)
            chunk["relevance_method"] = "heuristic"

        if not failed:
            return "llm", self.relevance_threshold
        if len(failed) == len(chunks):
            return "heuristic", HEURISTIC_THRESHOLD
        return "mixed", self.relevance_threshold

    def _method_threshold(self, llm: bool) -> float:
        return self.relevance_threshold if llm else HEURISTIC_THRESHOLD

    async def _score_batch(self, goal: str, batch: list[dict]) -> None:
        passages = "\n\n".join(f"[{i}] {wrap_untrusted(c['text'][:500])}" for i, c in enumerate(batch))
        async with self._semaphore:
            out = await self.llm.structured(
                prompts.RELEVANCE_PROMPT.format(goal=goal, passages=passages),
                RelevanceScores,
                system=prompts.SEARCH_SYSTEM,
                max_tokens=600,
            )
        scores = {s.passage: min(1.0, max(0.0, s.score)) for s in out.scores}
        for i, chunk in enumerate(batch):
            # A passage the model skipped is treated as irrelevant rather than guessed at.
            chunk["relevance_score"] = round(scores.get(i, 0.0), 4)
            chunk["relevance_method"] = "llm"

    # ---- live arXiv fallback ---------------------------------------------------------

    async def _live_abstracts(self, query: str, warnings: list[str], timeout_s: float = 30) -> list[dict]:
        try:
            papers = await asyncio.wait_for(
                asyncio.to_thread(self.arxiv.search_abstracts, query, 5), timeout=timeout_s
            )
        except Exception as exc:
            warnings.append(f"Live arXiv search unavailable: {exc}")
            return []
        return [
            {
                "id": f"{p.arxiv_id}_abstract",
                "text": f"{p.title}. {p.abstract}",
                "metadata": {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": ", ".join(p.authors[:3]),
                    "year": p.year,
                    "doi": p.doi or "",
                    "section": "Abstract (live arXiv search)",
                },
            }
            for p in papers
        ]
