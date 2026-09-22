"""Compare retrieval strategies on the evaluation set with identical answer generation and judging."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from matsci_agent.evaluation import metrics

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# A retrieval strategy: question -> ordered context passages (best first).
ContextProvider = Callable[[str], Awaitable[list[str]]]


def load_questions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in the evaluation dataset.")
    for q in questions:
        if not (q.get("question") and q.get("ground_truth")):
            raise ValueError(f"Question {q.get('id')} needs both 'question' and 'ground_truth'.")
    return questions


async def evaluate_question(llm, question: dict, contexts: list[str]) -> dict[str, Any]:
    answer = await metrics.generate_answer(llm, question["question"], contexts)
    scores = await asyncio.gather(
        metrics.faithfulness(llm, answer, contexts),
        metrics.answer_relevancy(llm, question["question"], answer),
        metrics.context_precision(llm, question["question"], question["ground_truth"], contexts),
        metrics.context_recall(llm, question["ground_truth"], contexts),
    )
    return {
        "id": question["id"], "answer": answer, "n_contexts": len(contexts),
        **dict(zip(METRICS, scores)),
    }


def summarise(rows: list[dict]) -> dict[str, float | None]:
    """Mean of each metric over the questions where it could be computed."""
    out: dict[str, float | None] = {}
    for name in METRICS:
        values = [r[name] for r in rows if r[name] is not None]
        out[name] = round(mean(values), 4) if values else None
    return out


async def compare(
    llm,
    questions: list[dict],
    systems: dict[str, ContextProvider],
    *,
    concurrency: int = 3,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run every system over every question; return per-question rows and per-system means."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(system: str, provider: ContextProvider, question: dict) -> dict:
        async with semaphore:
            row = await evaluate_question(llm, question, await provider(question["question"]))
        if on_progress:
            on_progress(system, question["id"])
        return row

    results: dict[str, Any] = {}
    for name, provider in systems.items():
        rows = await asyncio.gather(*(one(name, provider, q) for q in questions))
        results[name] = {"summary": summarise(rows), "questions": rows}

    names = list(systems)
    delta = None
    if len(names) == 2:
        base, other = results[names[0]]["summary"], results[names[1]]["summary"]
        delta = {
            m: None if base[m] is None or other[m] is None else round(other[m] - base[m], 4) for m in METRICS
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "answer_and_judge_model": getattr(llm, "model", None),
        "systems": results,
        "delta_vs_first": delta,
        "note": (
            "RAGAS-style metrics judged by an LLM, not the ragas library; comparable between the systems "
            "here, not to published RAGAS numbers. Reference answers are general knowledge, not derived "
            "from the corpus, so context recall reflects corpus coverage."
        ),
    }


def markdown_table(report: dict[str, Any]) -> str:
    names = list(report["systems"])
    header = "| Metric | " + " | ".join(names) + (" | Change |" if report["delta_vs_first"] else " |")
    lines = [header, "|" + "---|" * (len(names) + 1 + (1 if report["delta_vs_first"] else 0))]
    for m in METRICS:
        cells = [("n/a" if report["systems"][n]["summary"][m] is None else f"{report['systems'][n]['summary'][m]:.3f}") for n in names]
        row = f"| {m} | " + " | ".join(cells)
        if report["delta_vs_first"]:
            d = report["delta_vs_first"][m]
            row += " | " + ("n/a" if d is None else f"{d:+.3f}")
        lines.append(row + " |")
    return "\n".join(lines)
