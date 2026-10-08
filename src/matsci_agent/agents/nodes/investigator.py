"""InvestigatorAgent: closes evidence gaps with a bounded tool-using loop.

After screening, some candidates have no literature behind them (they came from a database screen) or
rest on a missing or predicted value. The investigator takes the most promising of those and, as an
agent, decides which tools to use: it searches the literature with queries it writes itself, can look a
formula up in Materials Project or ask the ML model, and stops when it has what it needs. Whatever it
finds is passed through the normal extraction step, merged into the state, and screening runs again so
a candidate that now has measured values is judged on them. Without an LLM the same tools run as a fixed
workflow (one targeted search per candidate), so the behaviour degrades instead of disappearing.
"""

from __future__ import annotations

import logging

from matsci_agent.agents.events import emit
from matsci_agent.agents.nodes.search import heuristic_relevance
from matsci_agent.agents.tools import Tool, ToolRegistry
from matsci_agent.agents.toolloop import run_tool_loop
from matsci_agent.domain.schemas import PropertyRecord
from matsci_agent.domain.state import ResearchState

logger = logging.getLogger(__name__)

GAP_STATUSES = {"missing", "predicted_met", "predicted_not_met"}


def _mentions(chunk: dict, names: set[str]) -> bool:
    text = chunk["text"]
    return any(n and n in text for n in names)


class InvestigatorAgent:
    def __init__(
        self, llm, search_agent, extraction_agent, mp_client, krr, *,
        max_targets: int = 3, max_steps: int = 6, results_per_search: int = 5,
    ) -> None:
        self.llm = llm
        self.search_agent = search_agent
        self.extraction = extraction_agent
        self.mp = mp_client
        self.krr = krr
        self.max_targets = max_targets
        self.max_steps = max_steps
        self.results_per_search = results_per_search

    # ---- graph interface ----------------------------------------------------------------

    def should_rescreen(self, state: ResearchState) -> bool:
        return bool(state.get("investigation", {}).get("rescreen")) and state.get("rescreen_count", 0) <= 1

    async def run(self, state: ResearchState, config=None) -> dict:
        if state.get("rescreen_count", 0) >= 1:  # already investigated and re-screened once
            return {"investigation": {"targets": [], "rescreen": False, "skipped": "already investigated"}}

        targets = self._targets(state.get("screened_candidates", []))
        if not targets:
            return {
                "investigation": {"targets": [], "rescreen": False},
                "agent_log": [{"step": "INVESTIGATION", "details": "No evidence gaps to investigate."}],
            }

        goal = state["research_goal"]
        warnings: list[str] = []
        names = {n for t in targets for n in (t["formula"], t["name"]) if n}
        await emit(config, "investigation_start", targets=[t["formula"] or t["name"] for t in targets])

        collected: dict[str, dict] = {}
        registry = self._registry(goal, collected, warnings)

        async def on_step(step: dict) -> None:
            await emit(config, "tool_call", action=step["action"], arguments=step["arguments"], ok=step.get("ok", True))

        steps: list[dict] = []
        mode = "workflow"
        answer = ""
        if self.llm.available:
            loop = await run_tool_loop(
                self.llm, registry, self._task(goal, state.get("criteria", []), targets),
                max_steps=self.max_steps, on_step=on_step,
            )
            steps, answer = loop.steps, loop.answer
            mode = "agent"
            if loop.stopped == "llm_error" and not steps:
                warnings.append("The investigator could not use the LLM and ran as a fixed workflow instead.")
                mode = "workflow"
        if not steps:  # no LLM, or it failed immediately: one targeted search per candidate
            for t in targets:
                call = {"query": f"{t['formula'] or t['name']} {self._property_words(t)}".strip()}
                observation, ok = await registry.call("search_literature", call)
                step = {"n": len(steps) + 1, "thought": "rule-based gap filling", "action": "search_literature",
                        "arguments": call, "observation": observation, "ok": ok}
                steps.append(step)
                await on_step(step)

        known = {c["id"] for c in state.get("retrieved_papers", [])}
        fresh = [c for c in collected.values() if c["id"] not in known and _mentions(c, names)]
        new_records = await self._extract(fresh, names | {c["formula"] or c["name"] for c in state.get("screened_candidates", [])}, state, warnings)
        target_formulas = {t["formula"] or t["name"] for t in targets}
        supported = sorted({r.material_formula or r.material_name for r in new_records} & target_formulas)

        papers = list(state.get("retrieved_papers", []))
        for chunk in fresh:
            chunk.setdefault("relevance_score", max(heuristic_relevance(goal, chunk), 0.7))
            papers.append(chunk)
        papers.sort(key=lambda c: c.get("relevance_score", 0), reverse=True)

        rescreen = bool(supported)
        await emit(config, "investigation_done", new_passages=len(fresh), new_records=len(new_records), supported=supported)
        summary = {
            "targets": sorted(target_formulas), "mode": mode, "steps": steps, "answer": answer,
            "new_passages": len(fresh), "new_records": len(new_records), "supported": supported, "rescreen": rescreen,
        }
        return {
            "investigation": summary,
            "retrieved_papers": papers,
            "extracted_properties": [*state.get("extracted_properties", []), *(r.model_dump() for r in new_records)],
            "rescreen_count": state.get("rescreen_count", 0) + (1 if rescreen else 0),
            "warnings": warnings,
            "agent_log": [{
                "step": "INVESTIGATION",
                "details": (
                    f"Investigated {len(targets)} candidate(s) with {len(steps)} tool call(s) ({mode}); "
                    f"{len(fresh)} new passage(s), {len(new_records)} new value(s)"
                    + (f"; literature support found for {', '.join(supported)}" if supported else "; nothing new for the candidates")
                    + "."
                ),
            }],
        }

    # ---- targets and task ------------------------------------------------------------------

    def _targets(self, candidates: list[dict]) -> list[dict]:
        """Non-rejected candidates with no literature behind them, or with a missing/predicted value."""
        gaps = [
            c for c in candidates
            if c["status"] != "rejected"
            and (c.get("origin") == "database" or any(r["status"] in GAP_STATUSES for r in c["criteria"]))
        ]
        return gaps[: self.max_targets]

    @staticmethod
    def _property_words(target: dict) -> str:
        unmet = [r["label"] for r in target["criteria"] if r["status"] in GAP_STATUSES or r["kind"] == "numeric"]
        return " ".join(dict.fromkeys(unmet[:1])) or "properties"

    def _task(self, goal: str, criteria: list[dict], targets: list[dict]) -> str:
        lines = [
            f"- {t['formula'] or t['name']}: "
            + ("came from a database screen, no paper reports it; " if t.get("origin") == "database" else "")
            + "; ".join(f"{r['label']} is {r['status']}" for r in t["criteria"] if r["status"] in GAP_STATUSES | {"not_met"})
            for t in targets
        ]
        return (
            f"Research goal: {goal}\nCriteria: {'; '.join(c['label'] for c in criteria) or 'none'}\n\n"
            "These candidates lack literature evidence for the goal:\n" + "\n".join(lines) + "\n\n"
            "For each, search the literature for measured values of the relevant property, writing specific "
            "queries that contain the formula. Use lookup_material or predict_property only to cross-check. "
            "Finish when each candidate has been searched or nothing more can be found."
        )

    # ---- tools ----------------------------------------------------------------------------------

    def _registry(self, goal: str, collected: dict[str, dict], warnings: list[str]) -> ToolRegistry:
        reg = ToolRegistry()

        async def search_literature(query: str) -> str:
            chunks = await self.search_agent.targeted_search(query, goal, n=self.results_per_search, warnings=warnings)
            for c in chunks:
                collected.setdefault(c["id"], c)
            if not chunks:
                return "No passages found for that query."
            return "\n".join(
                f"[{c['id']}] ({c['metadata'].get('year', '?')}) {c['metadata'].get('title', '')[:90]} - {c['text'][:170]}"
                for c in chunks
            )

        def lookup_material(formula: str) -> str:
            props = self.mp.get_properties(formula)
            if not props:
                return f"No database entry for {formula}."
            src = "Materials Project (DFT)" if self.mp.online else "offline reference table"
            return f"{formula} from {src}: " + "; ".join(f"{p.property_name} = {p.value} {p.unit}".strip() for p in props)

        def predict_property(formula: str, property_name: str) -> str:
            out = self.krr.predict(formula, property_name)
            if "error" in out:
                return f"Prediction failed: {out['error']}"
            return f"ML estimate for {formula}: {out['property']} = {out['predicted_value']} {out['unit']} (estimate, +/-{out.get('uncertainty')})"

        reg.register(Tool("search_literature", "Search the paper corpus and live arXiv abstracts for passages.", {"query": "short search phrase"}, search_literature, timeout_s=45))
        reg.register(Tool("lookup_material", "Look a formula up in the materials database.", {"formula": "chemical formula, e.g. AlN"}, lookup_material))
        reg.register(Tool("predict_property", "Machine-learning estimate of a property from the formula alone.",
                          {"formula": "chemical formula", "property_name": "band_gap, formation_energy_per_atom or density"}, predict_property))
        return reg

    # ---- extraction of what was found ------------------------------------------------------------

    async def _extract(self, chunks: list[dict], allowed: set[str], state: ResearchState, warnings: list[str]) -> list[PropertyRecord]:
        if not chunks:
            return []
        records = await self.extraction.extract_chunks(chunks, warnings)
        existing = {
            (r["material_formula"] or r["material_name"], r["property_name"], r["value_numeric"], r["source_id"])
            for r in state.get("extracted_properties", [])
        }
        kept = []
        for r in records:
            key = (r.material_formula or r.material_name, r.property_name, r.value_numeric, r.source_id)
            if key not in existing and (r.material_formula or r.material_name) in allowed:
                kept.append(r)
                existing.add(key)
        return kept
