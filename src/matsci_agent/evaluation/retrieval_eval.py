"""Retrieval benchmark: how well do BM25, dense, hybrid and reranked hybrid find the right papers?

Two tasks, both with ground truth that needs no human labelling and no LLM:

* **sentence -> paper** (self-retrieval): the query is one abstract sentence, the relevant paper is the
  one it came from. Exact labels; measures whether a retriever can find a paper from its own content
  (Recall@k, MRR).
* **topic -> papers**: the query is a topic phrase, relevant papers are those arXiv returned for that
  topic when the corpus was built (`topics` in the manifest). Noisier labels, many relevant papers;
  measures topical ranking quality (Precision@10, nDCG@10).

Absolute numbers depend on the label quality, so the useful reading is the *comparison* between
methods on identical queries.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean

Search = Callable[[str, int], list[dict]]  # (query, n) -> ranked chunk dicts, best first


# ---- metrics -------------------------------------------------------------------------------------


def paper_of(chunk: dict) -> str:
    return (chunk.get("metadata") or {}).get("arxiv_id") or chunk["id"].rsplit("_", 1)[0]


def ranked_papers(hits: list[dict]) -> list[str]:
    """Paper ids in rank order, each paper counted once (its best-ranked chunk)."""
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(paper_of(h))
    return list(seen)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(p in relevant for p in ranked[:k]) / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, p in enumerate(ranked, 1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1) if p in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---- tasks ---------------------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentence_tasks(docs: list[dict], n: int, seed: int = 7) -> list[dict]:
    """One mid-abstract sentence per sampled paper (the first sentence is the title, so it is skipped)."""
    rng = random.Random(seed)
    candidates = []
    for d in docs:
        sentences = _SENTENCE.split(d["text"])
        usable = [s for s in sentences[1:] if 60 <= len(s) <= 260]
        if usable:
            candidates.append((paper_of(d), usable[len(usable) // 2]))
    rng.shuffle(candidates)
    return [{"query": q, "relevant": {pid}} for pid, q in candidates[:n]]


def topic_tasks(manifest: dict, min_relevant: int = 8) -> list[dict]:
    by_topic: dict[str, set[str]] = {}
    for p in manifest["papers"]:
        for t in p["topics"]:
            by_topic.setdefault(t, set()).add(p["arxiv_id"])
    return [{"query": t, "relevant": ids} for t, ids in by_topic.items() if len(ids) >= min_relevant]


# ---- running -------------------------------------------------------------------------------------


def evaluate(search: Search, tasks: list[dict], *, topic: bool, depth: int = 30) -> dict:
    """Mean metrics for one method over `tasks`."""
    rows, started = [], time.perf_counter()
    for t in tasks:
        ranked = ranked_papers(search(t["query"], depth))
        if topic:
            rows.append({"p10": precision_at_k(ranked, t["relevant"], 10), "ndcg10": ndcg_at_k(ranked, t["relevant"], 10)})
        else:
            rows.append({
                "recall1": recall_at_k(ranked, t["relevant"], 1), "recall5": recall_at_k(ranked, t["relevant"], 5),
                "recall10": recall_at_k(ranked, t["relevant"], 10), "mrr": reciprocal_rank(ranked, t["relevant"]),
            })
    out = {k: round(mean(r[k] for r in rows), 4) for k in rows[0]} if rows else {}
    out["queries"] = len(tasks)
    out["ms_per_query"] = round((time.perf_counter() - started) * 1000 / max(len(tasks), 1), 1)
    return out


def run_retrieval_eval(systems: dict[str, Search], docs: list[dict], manifest: dict, *, sentences: int = 150) -> dict:
    s_tasks, t_tasks = sentence_tasks(docs, sentences), topic_tasks(manifest)
    return {
        "corpus_papers": len({paper_of(d) for d in docs}),
        "tasks": {"sentence_to_paper": len(s_tasks), "topic_to_papers": len(t_tasks)},
        "sentence_to_paper": {name: evaluate(fn, s_tasks, topic=False) for name, fn in systems.items()},
        "topic_to_papers": {name: evaluate(fn, t_tasks, topic=True) for name, fn in systems.items()},
    }


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
