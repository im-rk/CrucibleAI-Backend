"""ReportQA: answer follow-up questions about a finished run, using only that run's own evidence.

Retrieval-augmented generation over the report: the report is split into small, id-labelled context
items (papers, hypotheses, candidates, extracted values, gaps), the most relevant are shown to the model
inside fenced blocks, and the answer must cite those ids. Citations are checked afterwards: an answer
that cites nothing real is flagged as ungrounded. Recent turns are passed back in (conversation
memory), so "and what about its stability?" works. Without an LLM it returns the best-matching context
items verbatim instead of generating an answer.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from matsci_agent.agents.nodes.search import keywords
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.security.guardrails import UNTRUSTED_NOTICE, wrap_untrusted

logger = logging.getLogger(__name__)

MAX_CONTEXT_ITEMS = 14
MAX_HISTORY_TURNS = 6

QA_SYSTEM = (
    "You answer follow-up questions about a materials-research report. Use ONLY the numbered context "
    "items provided; if they do not contain the answer, say so plainly. Cite the ids of the items you "
    "used. Never invent values." + UNTRUSTED_NOTICE
)

QA_PROMPT = """{history}Question: {question}

Context items (cite by id):
{context}

Return:
- answer: a direct answer in one to four sentences, citing ids in square brackets like [P2] or [H1]
- citations: the ids of the context items you used
- insufficient_evidence: true if the context does not contain what is needed to answer"""


class QAAnswer(BaseModel):
    answer: str
    citations: list[str]
    insufficient_evidence: bool


def build_context(report: dict) -> dict[str, str]:
    """Every fact in the report a question might need, as short id-labelled text items."""
    items: dict[str, str] = {}
    for i, r in enumerate(report.get("search_results", [])[:10], 1):
        meta = r.get("metadata", {})
        items[f"P{i}"] = f"Paper '{meta.get('title', '')}' ({meta.get('year', '?')}, arXiv {meta.get('arxiv_id', '')}): {r['text'][:420]}"
    for i, h in enumerate(report.get("hypotheses", []), 1):
        cited = ", ".join(e["id"] for e in h.get("evidence", []))
        items[f"H{i}"] = f"Hypothesis ({h.get('confidence', '?')} confidence, cites {cited or 'nothing'}): {h['statement']} Suggested experiment: {h.get('suggested_experiment', '')}"
    for i, c in enumerate(report.get("candidates", [])[:10], 1):
        checks = "; ".join(
            f"{r['label']}: {r['status']}" + (f" ({r['value']} {r['unit']})" if r.get("value") is not None else "")
            for r in c.get("criteria", [])
        )
        items[f"C{i}"] = f"Candidate {c.get('formula') or c.get('name')} [{c['status']}, from {c.get('origin', '?')}]: {checks}"
    literature = [r for r in report.get("extracted_properties", []) if r.get("method") == "literature"]
    database = [r for r in report.get("extracted_properties", []) if r.get("method") != "literature" and not r.get("is_predicted")]
    for i, r in enumerate((literature[:12] + database[:8]), 1):
        items[f"R{i}"] = (
            f"{r.get('material_formula') or r.get('material_name')}: {r['property_name']} = {r['value_raw']} "
            f"({r.get('method')}; source: {r.get('source_title') or r.get('source_id')})"
        )
    if report.get("knowledge_gaps"):
        items["G1"] = "Knowledge gaps: " + " ".join(report["knowledge_gaps"][:6])
    if report.get("criteria"):
        items["K1"] = "Criteria screened against: " + "; ".join(c["label"] for c in report["criteria"])
    items["Q1"] = f"The research goal was: {report.get('query', '')}"
    return items


def rank_context(items: dict[str, str], question: str, limit: int = MAX_CONTEXT_ITEMS) -> dict[str, str]:
    """Keyword-overlap ranking; cheap and good enough for a few dozen short items."""
    terms = set(keywords(question))
    scored = sorted(items.items(), key=lambda kv: -sum(t in kv[1].lower() for t in terms))
    return dict(scored[:limit])


class ReportQA:
    def __init__(self, llm) -> None:
        self.llm = llm

    async def ask(self, report: dict, question: str, history: list[dict] | None = None) -> dict:
        context = rank_context(build_context(report), question)
        history = [h for h in (history or []) if h.get("role") in ("user", "assistant") and h.get("content")][-MAX_HISTORY_TURNS:]

        if self.llm.available:
            try:
                turns = "".join(f"{h['role'].capitalize()}: {h['content'][:400]}\n" for h in history)
                out = await self.llm.structured(
                    QA_PROMPT.format(
                        history=f"Conversation so far:\n{turns}\n" if turns else "",
                        question=question,
                        context="\n".join(f"{k}: {wrap_untrusted(v)}" for k, v in context.items()),
                    ),
                    QAAnswer, system=QA_SYSTEM, max_tokens=900,
                )
                valid = [c for c in dict.fromkeys(out.citations) if c in context]
                return {
                    "answer": out.answer.strip(), "citations": [{"id": c, "text": context[c]} for c in valid],
                    "grounded": bool(valid) and not out.insufficient_evidence,
                    "insufficient_evidence": out.insufficient_evidence or not valid, "mode": "llm",
                }
            except LLMError as exc:
                logger.warning(f"Report Q&A fell back to extractive mode: {exc}")

        top = list(context.items())[:3]
        return {
            "answer": "The language model is unavailable, so here are the report passages that best match your question.",
            "citations": [{"id": k, "text": v} for k, v in top], "grounded": False, "insufficient_evidence": False, "mode": "extractive",
        }
