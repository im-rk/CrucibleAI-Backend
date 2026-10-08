"""Critic and reviser: an evaluator-optimizer loop that makes hypotheses earn their citations.

    hypothesise -> verify --(unsupported or partial claims, rounds left)--> revise -> verify -> ... -> report

`verify` checks every hypothesis against the evidence it cites (rule checks plus an LLM judge, the
stricter verdict wins). `revise` hands the failing ones back to the model together with the critic's
reason and the full evidence list, asking for a narrower claim that the evidence really states. The
grounding score after each round is recorded, so the report can show what the loop achieved.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from matsci_agent.agents import prompts
from matsci_agent.agents.events import emit
from matsci_agent.agents.nodes.hypothesis import HypothesisDraft, HypothesisAgent
from matsci_agent.domain.state import ResearchState
from matsci_agent.infrastructure.llm.anthropic_client import LLMError
from matsci_agent.services.grounding import GroundingChecker

logger = logging.getLogger(__name__)

MAX_REVISIONS = 2
WEAK = {"partial", "unsupported"}


class RevisedHypothesis(BaseModel):
    index: int
    statement: str
    evidence_ids: list[str]
    suggested_experiment: str
    confidence: str


class RevisionOutput(BaseModel):
    revisions: list[RevisedHypothesis]


def weak_indices(state: ResearchState) -> list[int]:
    """Hypotheses the critic judged only partly supported or unsupported (and the model wrote)."""
    hypotheses = state.get("hypotheses", [])
    return [
        c["index"] for c in state.get("grounding", {}).get("claims", [])
        if c["verdict"] in WEAK and c["index"] < len(hypotheses) and hypotheses[c["index"]].get("origin") != "template"
    ]


class VerifyAgent:
    """The critic: grounding check, with the score history of the loop."""

    def __init__(self, checker: GroundingChecker, reviser_available) -> None:
        self.checker = checker
        self._can_revise = reviser_available  # callable -> bool

    def will_revise(self, state: ResearchState) -> bool:
        return bool(weak_indices(state)) and state.get("revision_count", 0) < MAX_REVISIONS and self._can_revise()

    async def run(self, state: ResearchState, config=None) -> dict:
        hypotheses, grounding, warnings = await self.checker.check(state.get("hypotheses", []))
        history = [*state.get("grounding_history", []), {
            "round": state.get("revision_count", 0), "score": grounding["score"],
            "unsupported": grounding["unsupported"], "weak": sum(c["verdict"] in WEAK for c in grounding["claims"]),
        }]
        grounding = {**grounding, "history": history}
        merged = {**state, "hypotheses": hypotheses, "grounding": grounding}
        revising = self.will_revise(merged)
        await emit(config, "grounding_check", is_grounded=grounding["is_grounded"], score=grounding["score"],
                   round=state.get("revision_count", 0), revising=revising)
        return {
            "hypotheses": hypotheses,
            "grounding": grounding,
            "grounding_history": history,
            # a hypothesis that is about to be rewritten should not leave a warning behind
            "warnings": [] if revising else warnings,
            "agent_log": [{
                "step": "VERIFICATION",
                "details": (
                    f"Round {state.get('revision_count', 0)}: grounding {grounding['score']:.2f} "
                    f"({grounding['unsupported']} unsupported of {grounding['checked']}, {grounding['method']})"
                    + ("; sending the weak ones back for revision." if revising else ".")
                ),
            }],
        }


class ReviseAgent:
    """The optimizer: rewrites weakly grounded hypotheses so the cited evidence really supports them."""

    def __init__(self, llm) -> None:
        self.llm = llm

    async def run(self, state: ResearchState, config=None) -> dict:
        rounds = state.get("revision_count", 0) + 1
        indices = weak_indices(state)
        hypotheses = [dict(h) for h in state.get("hypotheses", [])]
        evidence = state.get("evidence_index", {})
        claims = {c["index"]: c for c in state.get("grounding", {}).get("claims", [])}
        warnings: list[str] = []
        revised: list[int] = []

        if self.llm.available and indices and evidence:
            items = "\n\n".join(
                f"[{i}] Hypothesis: {hypotheses[i]['statement']}\n"
                f"Critic's verdict: {claims[i]['verdict']} - {claims[i]['reason']}\n"
                f"Currently cites: {', '.join(e['id'] for e in hypotheses[i].get('evidence', [])) or '(nothing)'}"
                for i in indices
            )
            try:
                out = await self.llm.structured(
                    prompts.REVISE_PROMPT.format(
                        goal=state["research_goal"], items=items,
                        evidence="\n".join(f"{k}: {e['snippet']}" for k, e in evidence.items()),
                    ),
                    RevisionOutput, system=prompts.REVISE_SYSTEM, max_tokens=2500,
                )
                for rev in out.revisions:
                    if rev.index in indices and rev.statement.strip():
                        conf = rev.confidence if rev.confidence in ("high", "medium", "low") else "low"
                        draft = HypothesisDraft(
                            statement=rev.statement.strip(), evidence_ids=rev.evidence_ids,
                            suggested_experiment=rev.suggested_experiment, confidence=conf,
                        )
                        hypotheses[rev.index] = {
                            **HypothesisAgent._resolve(draft, evidence),
                            "origin": "llm-revised", "revisions": hypotheses[rev.index].get("revisions", 0) + 1,
                        }
                        revised.append(rev.index)
            except LLMError as exc:
                warnings.append(f"Hypothesis revision skipped: {exc}")

        await emit(config, "hypotheses_revised", round=rounds, revised=len(revised))
        return {
            "hypotheses": hypotheses,
            "revision_count": rounds,
            "warnings": warnings,
            "agent_log": [{"step": "REVISION", "details": f"Round {rounds}: rewrote {len(revised)} of {len(indices)} weak hypotheses."}],
        }
