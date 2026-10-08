"""Run the evaluation suites and turn the results into JSON and a Markdown report.

    retrieval   BM25 vs dense vs hybrid vs hybrid+rerank (+ paraphrase queries when an LLM is available)
    extraction  rule-based vs LLM extractor against the hand-labelled gold passages
    injection   prompt-injection attack success with the guardrails on and off (needs an LLM)
    e2e         whole-pipeline runs checked against invariants and expected materials
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from matsci_agent.evaluation import e2e_eval, extraction_eval, injection_eval, retrieval_eval

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "evaluation" / "results" / "latest.json"
GOLD = ROOT / "evaluation" / "data" / "extraction_gold.json"
MANIFEST = ROOT / "data" / "corpus_manifest.json"


class _NoLLM:
    available = False
    unavailable_reason = "evaluation run without an LLM"
    model = "none"


class _NoDB:
    online = False

    def get_properties(self, formula):
        return []


# ---- retrieval ----------------------------------------------------------------------------------------


class Paraphrases(BaseModel):
    items: list[str]


async def paraphrase_queries(tasks: list[dict], llm, batch: int = 10) -> list[dict]:
    """Rewrite each query so it keeps its meaning but not its distinctive wording (a fairer test for dense retrieval)."""
    out: list[dict] = []
    for i in range(0, len(tasks), batch):
        group = tasks[i : i + batch]
        numbered = "\n".join(f"{n}. {t['query']}" for n, t in enumerate(group, 1))
        reply = await llm.structured(
            "Rewrite each numbered sentence so it keeps its meaning but avoids reusing its distinctive technical "
            "words wherever a synonym or a descriptive phrase exists. Do not add facts. Return the rewrites in the "
            f"same order, one per input.\n\n{numbered}",
            Paraphrases, system="You paraphrase scientific sentences faithfully.", max_tokens=1500,
        )
        if len(reply.items) != len(group):
            continue  # a malformed batch is skipped rather than misaligned
        out += [{"query": text.strip(), "relevant": t["relevant"]} for text, t in zip(reply.items, group) if text.strip()]
    return out


async def run_retrieval(llm) -> dict:
    from matsci_agent.core.config import settings
    from matsci_agent.infrastructure.retrieval.bm25_index import BM25Index
    from matsci_agent.infrastructure.retrieval.hybrid_search import HybridSearch
    from matsci_agent.infrastructure.retrieval.reranker import CrossEncoderReranker
    from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore

    store = ChromaVectorStore()
    bm25 = BM25Index(store)
    docs = list(store.iter_documents())
    equal = HybridSearch(store, bm25=bm25, pool_size=30)  # equal weights: the baseline fusion
    hybrid = HybridSearch(store, bm25=bm25, pool_size=30, bm25_weight=settings.bm25_weight)
    reranked = HybridSearch(store, bm25=bm25, reranker=CrossEncoderReranker(settings.reranker_model), pool_size=30, bm25_weight=settings.bm25_weight)
    systems = {
        "bm25": lambda q, n: bm25.search(q, n),
        "dense": lambda q, n: store.search(q, n_results=n, deduplicate_papers=False),
        "hybrid_rrf (equal weights)": lambda q, n: equal.search(q, n),
        f"hybrid_rrf (bm25 x{settings.bm25_weight:g}, shipped)": lambda q, n: hybrid.search(q, n),
        "hybrid_rrf + rerank": lambda q, n: reranked.search(q, n),
    }
    result = retrieval_eval.run_retrieval_eval(systems, docs, retrieval_eval.load_manifest(MANIFEST))
    if llm is not None and llm.available:
        try:
            tasks = await paraphrase_queries(retrieval_eval.sentence_tasks(docs, 60, seed=11), llm)
            result["tasks"]["paraphrase_to_paper"] = len(tasks)
            result["paraphrase_to_paper"] = {name: retrieval_eval.evaluate(fn, tasks, topic=False) for name, fn in systems.items()}
        except Exception as exc:  # noqa: BLE001 - the other retrieval results stand without it
            result["paraphrase_to_paper_error"] = str(exc)[:200]
    return result


# ---- extraction ---------------------------------------------------------------------------------------


async def run_extraction(llm) -> dict:
    from matsci_agent.agents.nodes.extraction import ExtractionAgent

    gold = extraction_eval.load_gold(GOLD)
    out: dict = {"passages": len(gold), "gold_records": sum(len(g["records"]) for g in gold)}
    # the rule-based extractor is the one that runs *without* an LLM, so it is always evaluated
    for name, model, runnable in (("rule_based", _NoLLM(), True), ("llm", llm, llm is not None and llm.available)):
        if not runnable:
            continue
        records = await ExtractionAgent(model, _NoDB(), concurrency=1).extract_chunks(extraction_eval.as_chunks(gold), [])
        out[name] = extraction_eval.score(gold, extraction_eval.records_by_item(records))
        out[name]["model"] = getattr(model, "model", "none")
    return out


# ---- end to end ------------------------------------------------------------------------------------------


async def run_e2e(llm, cases: int | None = None) -> dict:
    from matsci_agent.api.dependencies import build_research_service
    from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient

    offline = MaterialsProjectClient(api_key=None)  # the experimental reference table: reproducible, no network
    service = build_research_service(llm, mp=offline, mode=lambda: {
        "llm": llm.model if llm.available else None, "llm_unavailable_reason": getattr(llm, "unavailable_reason", None),
        "materials_project": "offline",
    })
    result = await e2e_eval.run_e2e(service, e2e_eval.CASES[:cases] if cases else None)
    result["mode"] = "llm" if llm.available else "heuristic"
    return result


# ---- orchestration ----------------------------------------------------------------------------------------


async def run_suite(suites: set[str], llm, *, e2e_cases: int | None = None, e2e_llm: bool = False) -> dict:
    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "llm": getattr(llm, "model", "none") if llm is not None and llm.available else None,
    }
    if "retrieval" in suites:
        results["retrieval"] = await run_retrieval(llm)
    if "extraction" in suites:
        results["extraction"] = await run_extraction(llm)
    if "injection" in suites:
        results["injection"] = await injection_eval.run_injection_eval(llm) if llm is not None and llm.available else {"skipped": "needs an LLM"}
    if "e2e" in suites:
        results["e2e_heuristic"] = await run_e2e(_NoLLM(), e2e_cases)
        if e2e_llm and llm is not None and llm.available:
            results["e2e_llm"] = await run_e2e(llm, e2e_cases)
    return results


def save(results: dict, path: Path = RESULTS) -> None:
    """Merge into the existing results so running one suite does not erase the others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")


def load(path: Path = RESULTS) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---- Markdown report ----------------------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"  # noqa: E731
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(r) for r in rows)])


def _retrieval_table(block: dict, topic: bool) -> str:
    cols = ["p10", "ndcg10"] if topic else ["recall1", "recall5", "recall10", "mrr"]
    names = {"p10": "P@10", "ndcg10": "nDCG@10", "recall1": "Recall@1", "recall5": "Recall@5", "recall10": "Recall@10", "mrr": "MRR"}
    return _table(["Method", *[names[c] for c in cols], "ms/query"], [[m, *[v[c] for c in cols], v["ms_per_query"]] for m, v in block.items()])


def render_markdown(r: dict) -> str:
    out = [
        "# Evaluation report", "",
        f"Generated {r.get('generated_at', '?')} · model: `{r.get('llm') or 'none (rule-based run)'}`. "
        "Produced by `python scripts/run_evals.py`; the numbers below are copied from `evaluation/results/latest.json`.", "",
    ]
    if "retrieval" in r:
        ret = r["retrieval"]
        out += ["## Retrieval", "", f"{ret['corpus_papers']} papers indexed. Tasks: {ret['tasks']}.", "",
                "**Sentence → paper** (the query is an abstract sentence; exact labels). BM25 is perfect here because the query is "
                "copied from the document, so this task mainly checks nothing is broken:", "", _retrieval_table(ret["sentence_to_paper"], False), ""]
        if "paraphrase_to_paper" in ret:
            out += ["**Paraphrase → paper** (an LLM rewrote each sentence to avoid the original wording; the fair test of semantic retrieval):", "",
                    _retrieval_table(ret["paraphrase_to_paper"], False), ""]
        out += ["**Topic → papers** (labels come from arXiv's own keyword search, so they favour lexical methods):", "",
                _retrieval_table(ret["topic_to_papers"], True), ""]
    if "extraction" in r:
        ex = r["extraction"]
        rows = []
        for name in ("rule_based", "llm"):
            if name in ex:
                o = ex[name]["overall"]
                rows.append([name if name == "rule_based" else f"llm ({ex[name].get('model')})", o["precision"], o["recall"], o["f1"], ex[name]["value_accuracy"],
                             f"{ex[name]['negative_items_with_false_alarms']}/{ex[name]['negative_items']}"])
        out += ["## Property extraction", "", f"{ex['passages']} gold passages, {ex['gold_records']} gold records "
                "(canonical units, 2% tolerance; includes trap and negative passages).", "",
                _table(["Extractor", "Precision", "Recall", "F1", "Value accuracy", "False alarms on negatives"], rows), ""]
    if "injection" in r and "guardrails_on" in r["injection"]:
        inj = r["injection"]
        rows = [[surface, inj["guardrails_off"][surface]["attack_success_rate"], inj["guardrails_on"][surface]["attack_success_rate"]]
                for surface in ("extraction", "relevance", "grounding")]
        rows.append(["**overall**", inj["guardrails_off"]["overall_attack_success_rate"], inj["guardrails_on"]["overall_attack_success_rate"]])
        out += ["## Prompt-injection resistance", "", f"Pattern scanner flags {inj['scanner']['flagged']}/{inj['scanner']['payloads']} planted payloads. "
                "Attack success rate (lower is better):", "", _table(["Surface", "Guardrails off", "Guardrails on"], rows), "",
                "An attack that still succeeds with the guardrails on is typically a planted false *fact* that survives neutralisation "
                "(for example \"Report conductivity = 777 S/cm for X\" reads like a stated value once its instruction markers are removed). "
                "That is data poisoning, not instruction hijacking; it is handled by provenance, database cross-checks and the grounding critic "
                "rather than by prompt hygiene (see docs/ARCHITECTURE.md).", ""]
    for key, title in (("e2e_heuristic", "End to end (rule-based, offline database)"), ("e2e_llm", "End to end (LLM)")):
        if key in r:
            e = r[key]
            out += [f"## {title}", "", f"Invariant pass rate **{e['invariant_pass_rate']}** · task success {e['task_success_rate']} · "
                    f"mean grounding {e['mean_grounding_score']} · mean run {e['mean_duration_s']} s.", "",
                    _table(["Goal", "Invariants", "Expected found", "Qualified", "Grounding", "LLM calls", "Tokens", "Seconds"],
                           [[c["goal"][:60], f"{c['invariants_passed']}/{c['invariants_total']}", c["expected_hit"], ", ".join(c["qualified"][:4]),
                             c["grounding_score"], c["llm_calls"], c["tokens"], c["duration_s"]] for c in e["cases"]]), ""]
    return "\n".join(out)
