"""HypothesisAgent: evidence-grounded hypotheses, suggested experiments, and knowledge gaps."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.domain.chemistry import elements_in
from matsci_agent.domain.state import ResearchState
from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError
from matsci_agent.security.guardrails import neutralize

logger = logging.getLogger(__name__)

ANION_CLASSES = {"O": "oxides", "N": "nitrides", "S": "sulfides", "F": "fluorides", "P": "phosphides"}
# How each property is normally measured; used to suggest experiments in the no-LLM fallback.
EXPERIMENTS = {
    "ionic_conductivity": "electrochemical impedance spectroscopy on dense pellets at room temperature",
    "bandgap": "UV-Vis diffuse-reflectance spectroscopy with a Tauc analysis",
    "formation_energy": "DFT total-energy calculations against the competing phases, or calorimetry",
    "electrochemical_stability_window": "linear-sweep voltammetry against a lithium reference",
    "density": "Archimedes-method density measurement cross-checked with XRD lattice parameters",
    "critical_temperature": "temperature-dependent resistivity and magnetisation measurements",
}


class HypothesisDraft(BaseModel):
    statement: str
    evidence_ids: list[str]
    suggested_experiment: str
    confidence: Literal["high", "medium", "low"]


class HypothesisOutput(BaseModel):
    hypotheses: list[HypothesisDraft]
    knowledge_gaps: list[str]


class HypothesisAgent:
    """Finds patterns across screened candidates and proposes falsifiable, cited hypotheses."""

    def __init__(self, llm: LLMClient, *, top_n: int = 6, max_chunks: int = 8, max_records: int = 15) -> None:
        self.llm = llm
        self.top_n = top_n
        self.max_chunks = max_chunks
        self.max_records = max_records

    async def run(self, state: ResearchState, config=None) -> dict:
        candidates = state.get("screened_candidates", [])
        criteria = state.get("criteria", [])
        top = [c for c in candidates if c["status"] != "rejected"][: self.top_n] or candidates[: self.top_n]
        evidence = self._build_evidence(candidates[: self.top_n * 2], state.get("extracted_properties", []), state.get("retrieved_papers", []))
        warnings: list[str] = []

        gaps = self._knowledge_gaps(top, criteria)
        hypotheses: list[dict] | None = None
        if self.llm.available and top:
            try:
                out = await self.llm.structured(
                    prompts.HYPOTHESIS_PROMPT.format(
                        goal=state["research_goal"],
                        n_candidates=len(candidates),
                        candidates=self._describe(top),
                        evidence="\n".join(f"{i}: {e['snippet']}" for i, e in evidence.items()) or "(none)",
                        scarcity_instruction=(
                            prompts.SCARCITY_INSTRUCTION
                            if sum(c["status"] == "qualified" for c in candidates) < 3 else ""
                        ),
                    ),
                    HypothesisOutput,
                    system=prompts.HYPOTHESIS_SYSTEM,
                    max_tokens=3000,
                )
                hypotheses = [self._resolve(d, evidence) for d in out.hypotheses]
                gaps = list(dict.fromkeys(gaps + [g.strip() for g in out.knowledge_gaps if g.strip()]))
            except LLMError as exc:
                warnings.append(f"Hypothesis generation fell back to templates: {exc}")
        elif not self.llm.available:
            warnings.append(
                "LLM unavailable: hypotheses are template-based pattern summaries, not model reasoning."
            )
        if hypotheses is None:
            hypotheses = self._template_hypotheses(top, candidates, criteria, evidence)

        await emit(config, "hypotheses_generated", count=len(hypotheses), gaps=len(gaps))
        return {
            "hypotheses": hypotheses,
            "evidence_index": evidence,
            "knowledge_gaps": gaps[:12],
            "warnings": warnings,
            "agent_log": [{
                "step": "HYPOTHESIS",
                "details": f"Generated {len(hypotheses)} hypotheses and {len(gaps[:12])} knowledge gaps.",
            }],
        }

    # ---- evidence ----------------------------------------------------------------------

    def _build_evidence(self, top: list[dict], records: list[dict], chunks: list[dict]) -> dict[str, dict]:
        """Numbered evidence the hypotheses may cite: E* = literature chunks, R* = property records."""
        evidence: dict[str, dict] = {}
        for i, chunk in enumerate(chunks[: self.max_chunks], 1):
            meta = chunk["metadata"]
            evidence[f"E{i}"] = {
                "id": f"E{i}", "kind": "chunk", "ref": chunk["id"], "title": meta.get("title", ""),
                "doi": meta.get("doi", ""), "source_id": meta.get("arxiv_id", ""),
                "snippet": neutralize(f"[{meta.get('title', 'paper')}] {chunk['text'][:300]}"),
            }
        wanted = {c["formula"] or c["name"] for c in top}
        picked = [r for r in records if (r["material_formula"] or r["material_name"]) in wanted]
        picked.sort(key=lambda r: (r["method"] != "literature", r["is_predicted"]))
        for i, r in enumerate(picked[: self.max_records], 1):
            evidence[f"R{i}"] = {
                "id": f"R{i}", "kind": "record", "ref": r.get("chunk_id") or r["source_id"],
                "title": r["source_title"], "doi": r["source_doi"], "source_id": r["source_id"],
                "snippet": (
                    f"{r['material_name']}: {r['property_name']} = {r['value_raw']} "
                    f"({'predicted by KRR' if r['is_predicted'] else r['method']})"
                ),
            }
        return evidence

    @staticmethod
    def _describe(top: list[dict]) -> str:
        lines = []
        for c in top:
            checks = "; ".join(
                f"{r['label']}: {r['status']}" + (f" ({r['value']} {r['unit']})" if r["value"] is not None else "")
                for r in c["criteria"]
            )
            lines.append(f"{c['rank']}. {c['name']} ({c['formula'] or 'no formula'}) [{c['status']}] {checks}")
        return "\n".join(lines) or "(no candidates)"

    @staticmethod
    def _resolve(draft: HypothesisDraft, evidence: dict[str, dict]) -> dict:
        """Attach the cited evidence; unknown ids are dropped and the hypothesis is marked."""
        cited = [evidence[i] for i in dict.fromkeys(draft.evidence_ids) if i in evidence]
        unknown = [i for i in draft.evidence_ids if i not in evidence]
        return {
            "statement": draft.statement,
            "suggested_experiment": draft.suggested_experiment,
            "confidence": draft.confidence if cited else "low",
            "evidence": cited,
            "unsupported_ids": unknown,
            "ungrounded": not cited,
            "origin": "llm",
        }

    # ---- knowledge gaps (deterministic) ------------------------------------------------

    @staticmethod
    def _knowledge_gaps(top: list[dict], criteria: list[dict]) -> list[str]:
        gaps: list[str] = []
        if not top:
            gaps.append("No materials with reported values for the requested properties were found in the retrieved literature.")
        for cand in top:
            who = cand["formula"] or cand["name"]
            for res in cand["criteria"]:
                if res["status"] == "missing":
                    gaps.append(f"No measured value for '{res['label']}' was found for {who}.")
                elif res["status"].startswith("predicted"):
                    gaps.append(f"'{res['label']}' for {who} rests on a KRR estimate; no experimental value was found.")
        for cand in top:
            if cand.get("origin") == "database":
                gaps.append(
                    f"{cand['formula'] or cand['name']} came from a database screen only; no retrieved paper reports it."
                )
        for crit in criteria:
            if crit["kind"] == "qualitative":
                gaps.append(f"'{crit['label']}' cannot be verified from the extracted data and needs dedicated experimental evidence.")
        return list(dict.fromkeys(gaps))

    # ---- template fallback (no LLM) ----------------------------------------------------

    def _template_hypotheses(self, top, candidates, criteria, evidence) -> list[dict]:
        numeric = [c for c in criteria if c["kind"] == "numeric"]
        target = numeric[0]["label"] if numeric else "the target property"
        method = EXPERIMENTS.get(numeric[0]["property_name"], "a targeted measurement") if numeric else "a targeted measurement"

        def cite(who: set[str]) -> list[dict]:
            return [
                e for e in evidence.values()
                if e["kind"] == "record" and any(w and w in e["snippet"] for w in who)
            ][:4]

        out: list[dict] = []
        formulas = [c["formula"] for c in top if c["formula"]]
        elem_sets = [elements_in(f) or set() for f in formulas]
        if len(formulas) >= 2 and elem_sets:
            klass = next((label for a, label in ANION_CLASSES.items() if all(a in s for s in elem_sets)), None)
            shared = sorted(set.intersection(*elem_sets))
            if klass or shared:
                shared_txt = klass or f"compounds sharing {', '.join(shared)}"
                out.append(self._template(
                    f"The leading candidates ({', '.join(formulas)}) are all {shared_txt}, which suggests this "
                    f"chemistry favours {target.lower()}.",
                    f"Measure {target.lower()} for further {shared_txt} outside the current set using {method}, to "
                    "test whether the trend holds.",
                    cite(set(formulas)),
                ))
        for cand in top[:2]:
            if cand.get("origin") == "database":
                who = cand["formula"] or cand["name"]
                out.append(self._template(
                    f"{who} appears to satisfy the requirements on database values alone: no retrieved paper reports "
                    "it, so its suitability is unconfirmed.",
                    f"Confirm {target.lower()} for {who} experimentally using {method}, and check that it can be "
                    "synthesised phase-pure.",
                    cite({who}),
                ))
                continue
            weak = next((r for r in cand["criteria"] if r["status"] in ("missing", "predicted_met", "predicted_not_met")), None)
            if weak:
                who = cand["formula"] or cand["name"]
                out.append(self._template(
                    f"{who} is a plausible candidate, but '{weak['label']}' is "
                    f"{'only estimated by machine learning' if weak['is_predicted'] else 'unmeasured in the retrieved literature'}, "
                    "so it may not actually satisfy the requirement.",
                    f"Measure '{weak['label']}' for {who} experimentally ({EXPERIMENTS.get(next((c['property_name'] for c in numeric if c['label'] == weak['label']), ''), 'a targeted measurement')}).",
                    cite({who}),
                ))
        if sum(c["status"] == "qualified" for c in candidates) < 3:
            failed = Counter(r["label"] for c in candidates for r in c["criteria"] if r["status"] in ("not_met", "missing"))
            if failed:
                label, count = failed.most_common(1)[0]
                blocked = {
                    c["formula"] or c["name"] for c in candidates
                    if any(r["label"] == label and r["status"] in ("not_met", "missing") for r in c["criteria"])
                }
                out.append(self._template(
                    f"Few materials satisfy every requirement because '{label}' is the criterion most often unmet "
                    f"or undocumented ({count} of {len(candidates)} candidates).",
                    f"Broaden the search around '{label}' or relax that threshold to see which materials become viable.",
                    cite(blocked),
                ))
        return out[:5]

    @staticmethod
    def _template(statement: str, experiment: str, cited: list[dict]) -> dict:
        return {
            "statement": statement, "suggested_experiment": experiment, "confidence": "low",
            "evidence": cited, "unsupported_ids": [], "ungrounded": not cited, "origin": "template",
        }
