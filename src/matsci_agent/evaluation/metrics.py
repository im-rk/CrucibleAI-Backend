"""RAGAS-style RAG metrics, judged by the project's LLM client.

These follow the definitions of the four RAGAS metrics (Es et al., 2023) but are implemented
directly rather than with the `ragas` library, so scores are comparable between the systems
evaluated here and NOT directly comparable to numbers published with the library.

Each metric returns a float in [0, 1], or None when it cannot be computed (nothing to judge).
"""

from __future__ import annotations

from pydantic import BaseModel

JUDGE_SYSTEM = "You are a strict, literal evaluator. Judge only from the text provided; use no outside knowledge."


class ClaimCheck(BaseModel):
    claim: str
    supported: bool


class Claims(BaseModel):
    claims: list[ClaimCheck]


class Score(BaseModel):
    score: float


class Relevance(BaseModel):
    relevant: list[bool]


class StatementCheck(BaseModel):
    statement: str
    attributable: bool


class Statements(BaseModel):
    statements: list[StatementCheck]


class Answer(BaseModel):
    answer: str


def _numbered(items: list[str], limit: int = 700) -> str:
    return "\n\n".join(f"[{i + 1}] {text[:limit]}" for i, text in enumerate(items))


async def faithfulness(llm, answer: str, contexts: list[str]) -> float | None:
    """Share of the answer's factual claims that the contexts support."""
    if not contexts:
        return None
    out = await llm.structured(
        "Split the answer into its individual factual claims, then mark each claim supported only if "
        f"the context states or directly implies it.\n\nContext:\n{_numbered(contexts)}\n\nAnswer:\n{answer}",
        Claims, system=JUDGE_SYSTEM, max_tokens=1500,
    )
    return sum(c.supported for c in out.claims) / len(out.claims) if out.claims else None


async def answer_relevancy(llm, question: str, answer: str) -> float | None:
    """How directly the answer addresses the question (1 = fully and only, 0 = not at all)."""
    out = await llm.structured(
        f"Question: {question}\nAnswer: {answer}\n\nScore from 0.0 to 1.0 how directly and completely the answer "
        "addresses the question. Penalise evasion, missing parts, and irrelevant content.",
        Score, system=JUDGE_SYSTEM, max_tokens=100,
    )
    return min(1.0, max(0.0, out.score))


def average_precision(relevant: list[bool]) -> float | None:
    """Rank-weighted precision: relevant contexts near the top score higher. None if none are relevant."""
    hits, total = 0, 0.0
    for rank, is_relevant in enumerate(relevant, 1):
        if is_relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0 if relevant else None


async def context_precision(llm, question: str, ground_truth: str, contexts: list[str]) -> float | None:
    """Were the retrieved contexts the right ones, with the useful ones ranked first?"""
    if not contexts:
        return None
    out = await llm.structured(
        f"Question: {question}\nReference answer: {ground_truth}\n\nFor each numbered context, say whether it "
        f"contains information useful for arriving at the reference answer. One boolean per context, in order.\n\n"
        f"{_numbered(contexts)}",
        Relevance, system=JUDGE_SYSTEM, max_tokens=300,
    )
    flags = (out.relevant + [False] * len(contexts))[: len(contexts)]  # pad/truncate a misaligned reply
    return average_precision(flags)


async def context_recall(llm, ground_truth: str, contexts: list[str]) -> float | None:
    """Share of the reference answer's statements that the retrieved contexts can support."""
    if not contexts:
        return 0.0
    out = await llm.structured(
        "Split the reference answer into its individual statements and mark each attributable only if the "
        f"context contains the information to support it.\n\nContext:\n{_numbered(contexts)}\n\nReference answer:\n{ground_truth}",
        Statements, system=JUDGE_SYSTEM, max_tokens=800,
    )
    return sum(s.attributable for s in out.statements) / len(out.statements) if out.statements else None


async def generate_answer(llm, question: str, contexts: list[str]) -> str:
    """Answer strictly from the contexts; the same prompt is used for every system compared."""
    if not contexts:
        return "I cannot determine the answer: no relevant context was retrieved."
    out = await llm.structured(
        f"Answer the question using ONLY the context below. If the context does not contain the answer, say "
        f"you cannot determine it.\n\nContext:\n{_numbered(contexts)}\n\nQuestion: {question}",
        Answer, max_tokens=600,
    )
    return out.answer
