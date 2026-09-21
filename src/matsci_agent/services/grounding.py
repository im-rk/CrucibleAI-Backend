"""Grounding check (ISSUP): is each hypothesis actually supported by the evidence it cites?"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel

from matsci_agent.infrastructure.llm.anthropic_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

# A figure is only checked when it carries a unit or a decimal point, so counts ("3 candidates")
# and years are not mistaken for claims about data.
_FIGURE = re.compile(
    r"(\d+(?:\.\d+)?)\s?(?:meV|eV|mV|V|K|GPa|MPa|mS/cm|S/cm|%|mAh/g|g/cm3)\b|(\d+\.\d+)"
)

GROUNDING_SYSTEM = (
    "You are a strict fact-checker for materials science claims. A claim is supported only if the "
    "cited evidence itself states or directly implies it."
)
GROUNDING_PROMPT = """For each numbered hypothesis, decide whether the cited evidence supports its factual claims.
- supported: the evidence states or directly implies the claims
- partial: some claims are supported, others go beyond the evidence
- unsupported: the evidence does not support the claims
Give a one-sentence reason.

{items}"""


class Verdict(BaseModel):
    index: int
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


def _rule_check(hypothesis: dict) -> tuple[str, str]:
    """Deterministic checks: it cites evidence, and every quoted figure appears in that evidence."""
    evidence = hypothesis.get("evidence", [])
    if not evidence:
        return "unsupported", "cites no evidence"
    text = " ".join(e["snippet"] for e in evidence)
    figures = {a or b for a, b in _FIGURE.findall(hypothesis["statement"])}
    missing = sorted(f for f in figures if f not in text)
    if missing:
        return "partial", f"figure(s) {', '.join(missing)} not found in the cited evidence"
    return "supported", "cited evidence contains every quoted figure"


class GroundingChecker:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def check(self, hypotheses: list[dict]) -> tuple[list[dict], dict, list[str]]:
        """Return (hypotheses with `grounded`/`grounding_note`, summary, warnings)."""
        warnings: list[str] = []
        verdicts = [_rule_check(h) for h in hypotheses]
        method = "rules"

        if self.llm.available and hypotheses:
            try:
                items = "\n\n".join(
                    f"[{i}] Hypothesis: {h['statement']}\nCited evidence:\n"
                    + ("\n".join(f"- {e['snippet']}" for e in h.get("evidence", [])) or "- (none)")
                    for i, h in enumerate(hypotheses)
                )
                out = await self.llm.structured(
                    GROUNDING_PROMPT.format(items=items), Verdicts, system=GROUNDING_SYSTEM, max_tokens=1500
                )
                rank = {"supported": 2, "partial": 1, "unsupported": 0}
                for v in out.verdicts:
                    if 0 <= v.index < len(verdicts):
                        rules_verdict, rules_reason = verdicts[v.index]
                        # The stricter of the two judgements wins; the reason comes from that judge.
                        verdicts[v.index] = (
                            (v.verdict, v.reason) if rank[v.verdict] < rank[rules_verdict] else (rules_verdict, rules_reason)
                        )
                method = "llm+rules"
            except LLMError as exc:
                warnings.append(f"LLM grounding check unavailable, used rule checks only: {exc}")

        checked = []
        for i, (h, (verdict, reason)) in enumerate(zip(hypotheses, verdicts)):
            checked.append({**h, "grounded": verdict != "unsupported", "grounding_note": f"{verdict}: {reason}"})
            if verdict == "unsupported":
                warnings.append(f"Hypothesis {i + 1} is not supported by its cited evidence ({reason}).")
        points = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
        score = sum(points[v] for v, _ in verdicts) / len(verdicts) if verdicts else 1.0
        summary = {
            "is_grounded": all(v != "unsupported" for v, _ in verdicts),
            "score": round(score, 3),
            "method": method,
            "checked": len(verdicts),
            "unsupported": sum(v == "unsupported" for v, _ in verdicts),
            "claims": [
                {"index": i, "statement": h["statement"], "verdict": v, "reason": r}
                for i, (h, (v, r)) in enumerate(zip(hypotheses, verdicts))
            ],
        }
        return checked, summary, warnings
