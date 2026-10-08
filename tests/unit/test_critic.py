"""Critic (verify) and reviser: the evaluator-optimizer loop over hypotheses."""

from __future__ import annotations

import asyncio

from matsci_agent.agents.nodes.critic import MAX_REVISIONS, ReviseAgent, RevisedHypothesis, RevisionOutput, VerifyAgent, weak_indices
from matsci_agent.agents.orchestrator import build_research_graph, initial_state, run_config
from matsci_agent.infrastructure.llm.anthropic_client import LLMError


def run(coro):
    return asyncio.run(coro)


def hyp(statement, origin="llm", cited=("E1",)):
    return {
        "statement": statement, "origin": origin, "confidence": "medium", "suggested_experiment": "x",
        "evidence": [{"id": i, "snippet": f"snippet {i}"} for i in cited], "unsupported_ids": [], "ungrounded": False,
    }


class FakeChecker:
    """A hypothesis is supported once its statement starts with REVISED."""

    calls = 0

    async def check(self, hypotheses):
        FakeChecker.calls += 1
        claims = [
            {"index": i, "statement": h["statement"], "verdict": "supported" if h["statement"].startswith("REVISED") else "unsupported",
             "reason": "goes beyond the evidence"}
            for i, h in enumerate(hypotheses)
        ]
        bad = sum(c["verdict"] != "supported" for c in claims)
        summary = {"is_grounded": bad == 0, "score": round(1 - bad / len(claims), 3) if claims else 1.0, "method": "fake",
                   "checked": len(claims), "unsupported": bad, "claims": claims}
        return hypotheses, summary, [f"Hypothesis {c['index'] + 1} is not supported." for c in claims if c["verdict"] != "supported"]


class ScriptedLLM:
    available = True

    def __init__(self, revisions=None, error=None):
        self.revisions, self.error, self.prompts = revisions, error, []

    async def structured(self, prompt, schema, **kw):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.revisions


def revision(index, statement="REVISED claim", ids=("E1",), confidence="medium"):
    return RevisedHypothesis(index=index, statement=statement, evidence_ids=list(ids), suggested_experiment="measure it", confidence=confidence)


EVIDENCE = {"E1": {"id": "E1", "snippet": "AlN band gap 6.2 eV"}, "R1": {"id": "R1", "snippet": "AlN: bandgap = 6.2 eV"}}


def graded(*hypotheses, revision_count=0):
    state = initial_state("goal")
    state.update(hypotheses=list(hypotheses), evidence_index=EVIDENCE, revision_count=revision_count)
    claims = [
        {"index": i, "statement": h["statement"], "verdict": v, "reason": "reason"}
        for i, (h, v) in enumerate(zip(hypotheses, ["unsupported", "partial", "supported", "unsupported"]))
    ]
    state["grounding"] = {"claims": claims}
    return state


# ---- which hypotheses are revisable --------------------------------------------------------------------


def test_only_weak_model_written_hypotheses_are_revised() -> None:
    state = graded(hyp("a"), hyp("b"), hyp("c"), hyp("d", origin="template"))
    assert weak_indices(state) == [0, 1]  # c is supported, d is a rule-built template


# ---- the critic -------------------------------------------------------------------------------------------


def test_the_critic_records_the_score_history_and_holds_back_warnings_while_a_revision_is_coming() -> None:
    critic = VerifyAgent(FakeChecker(), lambda: True)
    state = initial_state("goal")
    state["hypotheses"] = [hyp("a"), hyp("b")]
    out = run(critic.run(state))

    assert out["grounding"]["history"] == [{"round": 0, "score": 0.0, "unsupported": 2, "weak": 2}]
    assert out["warnings"] == []  # the weak ones are about to be rewritten
    assert "sending the weak ones back" in out["agent_log"][0]["details"]


def test_the_critic_reports_warnings_when_no_revision_will_follow() -> None:
    state = initial_state("goal")
    state["hypotheses"] = [hyp("a")]
    no_llm = run(VerifyAgent(FakeChecker(), lambda: False).run(state))
    assert no_llm["warnings"] == ["Hypothesis 1 is not supported."]

    state["revision_count"] = MAX_REVISIONS
    exhausted = run(VerifyAgent(FakeChecker(), lambda: True).run(state))
    assert exhausted["warnings"] == ["Hypothesis 1 is not supported."]


# ---- the reviser ---------------------------------------------------------------------------------------------


def test_weak_hypotheses_are_rewritten_against_the_evidence_and_supported_ones_are_untouched() -> None:
    llm = ScriptedLLM(RevisionOutput(revisions=[
        revision(0, "REVISED a", ids=("E1", "R1", "E99")),  # E99 does not exist
        revision(1, "REVISED b", confidence="certain"),  # not a valid confidence
        revision(2, "REVISED c"),  # index 2 was supported: ignored
    ]))
    state = graded(hyp("a"), hyp("b"), hyp("c"))
    out = run(ReviseAgent(llm).run(state))
    a, b, c = out["hypotheses"]

    assert a["statement"] == "REVISED a" and a["origin"] == "llm-revised" and a["revisions"] == 1
    assert [e["id"] for e in a["evidence"]] == ["E1", "R1"] and a["unsupported_ids"] == ["E99"]
    assert b["confidence"] == "low"  # invalid confidence normalised
    assert c["statement"] == "c" and c["origin"] == "llm"  # untouched
    assert out["revision_count"] == 1 and "rewrote 2 of 2" in out["agent_log"][0]["details"]
    assert "goes beyond the evidence" not in llm.prompts[0] and "unsupported - reason" in llm.prompts[0]  # critic's reason is passed
    assert "E1: AlN band gap 6.2 eV" in llm.prompts[0]  # the full evidence list is passed


def test_a_failing_llm_leaves_the_hypotheses_alone_but_still_counts_the_round() -> None:
    out = run(ReviseAgent(ScriptedLLM(error=LLMError("quota"))).run(graded(hyp("a"))))
    assert out["hypotheses"][0]["statement"] == "a" and out["revision_count"] == 1
    assert any("revision skipped" in w and "quota" in w for w in out["warnings"])


# ---- the loop in the graph ---------------------------------------------------------------------------------------


class Probe:
    def __init__(self, name, update=None):
        self.name, self.update = name, update or {}

    async def run(self, state, config=None):
        return {"agent_log": [{"step": self.name, "details": "ran"}], **self.update}

    def should_retry(self, state):
        return False


def build(reviser, hypotheses):
    llm_available = True
    verifier = VerifyAgent(FakeChecker(), lambda: llm_available)
    return build_research_graph(
        Probe("search", {"retrieved_papers": [{"id": "1"}] * 5}), Probe("extract"), Probe("screen"),
        Probe("hypothesis", {"hypotheses": hypotheses, "evidence_index": EVIDENCE}), Probe("report"),
        verifier=verifier, reviser=reviser,
    )


def test_the_loop_revises_until_the_critic_is_satisfied_and_records_the_improvement() -> None:
    llm = ScriptedLLM(RevisionOutput(revisions=[revision(0), revision(1)]))
    final = run(build(ReviseAgent(llm), [hyp("a"), hyp("b")]).ainvoke(initial_state("goal"), run_config()))

    assert [h["statement"] for h in final["hypotheses"]] == ["REVISED claim", "REVISED claim"]
    assert [(h["round"], h["score"]) for h in final["grounding"]["history"]] == [(0, 0.0), (1, 1.0)]
    assert final["revision_count"] == 1 and len(llm.prompts) == 1  # one revision was enough
    assert final["warnings"] == []


def test_the_loop_stops_after_the_revision_limit_when_the_model_cannot_fix_the_hypotheses() -> None:
    llm = ScriptedLLM(RevisionOutput(revisions=[]))  # the model changes nothing
    final = run(build(ReviseAgent(llm), [hyp("a")]).ainvoke(initial_state("goal"), run_config()))

    assert final["revision_count"] == MAX_REVISIONS and len(llm.prompts) == MAX_REVISIONS
    assert [h["round"] for h in final["grounding"]["history"]] == [0, 1, 2]
    assert final["warnings"] == ["Hypothesis 1 is not supported."]  # reported once, at the end
    assert final["grounding"]["is_grounded"] is False
