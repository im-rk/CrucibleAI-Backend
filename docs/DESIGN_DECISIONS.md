# Design decisions

Each entry records what was decided, why, and the evidence. Where a decision was reversed or tuned after measuring, that is
said plainly. Numbers come from `docs/EVALUATION.md` and `docs/MODEL_CARD.md`.

## 1. A workflow with bounded agents, not one free-running agent

**Decision.** A fixed LangGraph pipeline; autonomy only in the Investigator (a model-chosen tool loop capped at 6 steps) and
in the Critic/Reviser loop (capped at 2 rounds).

**Why.** Anthropic's guidance is to prefer the simplest pattern that works and add autonomy only where the path cannot be
known in advance. Planning, extraction and screening are predictable; "which papers to look for to support this candidate"
is not. Bounding the loops keeps cost and latency predictable and makes runs comparable in the evaluation suite.

## 2. Every agent has a rule-based path

**Decision.** With no LLM (no key, no credits, provider down), each agent falls back to rules and the report says so.

**Why.** Free tiers fail often: during development Gemini's daily quota ran out mid-run and an Anthropic account had no
credits. A system that stops is useless in a demo; a system that silently degrades is dangerous. This one degrades *and
says so* in the banner, the warnings and the health endpoint. It also gives the evaluation a baseline: the LLM extractor
reaches F1 1.00 against 0.71 for the rule-based one on the 64 labelled values, so the model's contribution is measured.

## 3. Provider-agnostic structured output, not native function calling

**Decision.** Every model call returns a schema-validated object; the tool loop is "thought, action, arguments" as one
structured call per step.

**Why.** It works unchanged on Anthropic, Gemini and Groq, is unit-testable with a scripted fake, and keeps the full
trajectory as data for the report and the evaluation. **Found by running it live:** one model returned `arguments` as an object
instead of the list of pairs the schema asked for, so the loop now accepts both.

## 4. An evaluator-optimizer loop on hypotheses

**Decision.** After hypotheses are written, a critic checks each against its cited evidence; weak ones go back to a reviser
with the critic's reasons, at most twice; the score per round is recorded.

**Why.** On a real run the grounding check judged 4 of 5 model-written hypotheses unsupported (score 0.10). With the loop, the same
goal reached 0.70 to 0.88 across runs. Capping at two rounds stops the loop from chasing a score; what remains unsupported
is reported rather than hidden.

## 5. Cache the model, then schedule around its limits

**Decision.** A local response cache, a client-side token-per-minute limiter, and a pause after a temporary provider failure.

**Why.** With the free tier's 8,000 tokens per minute, a run that used every provider retry died at the 300 s limit
because each step repeated the same waits. Scheduling around the limit (short deliberate waits) replaced failing and
retrying. The cache turned a 6-minute rate-limited run into a 43-second replay (3 real calls, 21 from cache), which is also
what makes the evaluation repeatable.

## 6. Hybrid retrieval, with the keyword list weighted

**Decision.** BM25 and dense search fused with Reciprocal Rank Fusion; the BM25 list counts double (`BM25_WEIGHT=2`).

**Why.** Measured, and surprising: on this abstract corpus plain BM25 beat the MiniLM dense retriever, even on
LLM-paraphrased queries, and equal-weight fusion let the weaker list dilute it. Weighting BM25 improved every task (sentence-to-paper
Recall@1 0.83 to 0.91, topic P@10 0.97 to 0.996). Higher weights drift towards pure BM25, but the benchmark's queries are partly
lexical by construction, so the default is a conservative 2, not the maximum. A cross-encoder reranker helps further but costs
about 2.6 s per query on CPU, so it is off by default.

## 7. Abstract-only corpus, rebuilt from topic queries

**Decision.** One chunk per paper (title and abstract) from 26 targeted arXiv queries; a manifest makes the build reproducible.

**Why.** The original corpus was 19 unrelated recent arXiv papers with placeholder titles, so a materials question retrieved
astrophysics. Abstracts are fast to fetch, carry exact metadata, and state the headline numbers. The cost is real: tables and
supplementary data are not read. Deleting and re-creating a Chroma collection in place corrupted the embedded index, so a reset
now removes the directory.

## 8. Untrusted text is fenced, neutralised and never trusted with actions

**Decision.** Instruction-like phrases are removed, passages are fenced in `<document>` blocks the system prompt names as
data, outputs are schema-constrained, and tools are read-only.

**Why.** Retrieved papers are attacker-controllable input. On the attack benchmark the overall success rate was 0.50 without the
guardrails and 0.10 with them (relevance and grounding attacks went to 0). The benchmark found a real gap along the way
("ignore the research goal" was not matched by the first pattern). The remaining success is a planted false *fact*, which is
data poisoning and is left to provenance and the grounding critic.

## 9. A calibrated band-gap model, and the Laplacian kernel

**Decision.** Keep kernel ridge regression (it is the project's specified method and interpretable), but use the Laplacian
kernel, and report a conformal 90% interval.

**Why.** Benchmarked against six baselines on 1,500 Materials Project materials: the original RBF kernel scored R² 0.683,
below random forests (0.720) and boosted trees (0.733); tuning the RBF's alpha and gamma with nested cross-validation changed
nothing (0.683), but the Laplacian kernel reached 0.728, level with the best model. The old "plus or minus one RMSE" contained the truth
75% of the time; the conformal interval contains it 90% of the time. The intervals are about 2.8 eV wide: an estimate is a
triage signal, and it still cannot make a candidate qualify.

## 10. Plan first, and let the user edit it

**Decision.** The planner's output (criteria, queries, material families) is an explicit object the UI can show, and an edited plan
is sent back and trusted over the model's.

**Why.** It is the cheapest human-in-the-loop checkpoint: a wrong criterion costs a whole run, and removing it costs one click.
Planner models also invent requirements; one added "experimentally demonstrated", which the user never asked for, so unrequested
criteria of that kind are dropped with a warning.

## 11. MCP server, hand-written

**Decision.** A small JSON-RPC 2.0 stdio server (initialize, tools/list, tools/call, ping) exposing the same capabilities as the API.

**Why.** It lets any MCP client use literature search, lookup, prediction, screening and the full pipeline as tools, with no extra
dependency. It is tested both in-process and by spawning the real process and holding a conversation over its pipes.
