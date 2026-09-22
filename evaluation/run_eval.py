"""Compare plain RAG with the agentic retrieval (ISREL filtering + re-retrieval) on the eval set.

  python evaluation/run_eval.py                 # all 20 questions, k=5
  python evaluation/run_eval.py --limit 5       # quick, cheap trial run

Needs ANTHROPIC_API_KEY: the answers and the metric judgements are LLM calls, so without a key
there is nothing meaningful to measure. Roughly 6-8 LLM calls per question per system, so a full
run makes several hundred calls; try --limit first. Results are written to evaluation/results.json.

Both systems retrieve with the same hybrid search, answer with the same prompt, and are judged by
the same judge, so the comparison isolates what the agent adds: relevance filtering and re-retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.append("src")

from matsci_agent.api.dependencies import get_arxiv_client, get_hybrid_search, get_llm_client  # noqa: E402
from matsci_agent.agents.nodes.search import SearchAgent  # noqa: E402
from matsci_agent.core.config import settings  # noqa: E402
from matsci_agent.core.logging import configure_logging  # noqa: E402
from matsci_agent.domain.state import initial_state  # noqa: E402
from matsci_agent.evaluation.runner import compare, load_questions, markdown_table  # noqa: E402

HERE = Path(__file__).parent


async def main(limit: int | None, k: int, out: Path) -> None:
    llm = get_llm_client()
    if not llm.available:
        sys.exit("ANTHROPIC_API_KEY is not set. The evaluation judges answers with the LLM, so it cannot run without one.")

    questions = load_questions(HERE / "eval_dataset.json")[:limit]
    search = get_hybrid_search()
    agent = SearchAgent(
        llm, search, get_arxiv_client(),
        relevance_threshold=settings.relevance_threshold, min_relevant=settings.min_relevant_chunks,
        max_retries=settings.max_retries, live_arxiv=False,  # a fixed corpus keeps both systems comparable
        concurrency=settings.llm_concurrency,
    )

    async def plain_rag(question: str) -> list[str]:
        chunks = await asyncio.to_thread(search.search, question, k)
        return [c["text"] for c in chunks]

    async def agentic(question: str) -> list[str]:
        state = initial_state(question, k)
        while True:  # the same retrieval loop the graph runs
            state.update(await agent.run(state))
            if not agent.should_retry(state):
                break
        return [c["text"] for c in state["retrieved_papers"][:k]]

    done = 0

    def progress(system: str, qid: str) -> None:
        nonlocal done
        done += 1
        print(f"  [{done}/{len(questions) * 2}] {system}: {qid}", flush=True)

    print(f"Evaluating {len(questions)} questions x 2 systems with {llm.model} (k={k})...")
    report = await compare(llm, questions, {"plain_rag": plain_rag, "crucible_agentic": agentic}, on_progress=progress)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + markdown_table(report))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--k", type=int, default=5, help="contexts given to the answerer")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    configure_logging()
    asyncio.run(main(args.limit, args.k, args.out))
