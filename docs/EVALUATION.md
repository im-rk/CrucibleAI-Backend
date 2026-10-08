# Evaluation report

Generated 2026-10-08T08:19:03+00:00 · model: `openai/gpt-oss-120b`. Produced by `python scripts/run_evals.py`; the numbers below are copied from `evaluation/results/latest.json`.

## Retrieval

1234 papers indexed. Tasks: {'sentence_to_paper': 150, 'topic_to_papers': 22, 'paraphrase_to_paper': 60}.

**Sentence → paper** (the query is an abstract sentence; exact labels). BM25 is perfect here because the query is copied from the document, so this task mainly checks nothing is broken:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 22.6 |
| dense | 0.7267 | 0.8933 | 0.92 | 0.7938 | 42.3 |
| hybrid_rrf (equal weights) | 0.8267 | 0.94 | 0.9867 | 0.8881 | 86.1 |
| hybrid_rrf (bm25 x2, shipped) | 0.9067 | 0.94 | 0.9867 | 0.9296 | 101.2 |
| hybrid_rrf + rerank | 0.98 | 0.9867 | 1.0 | 0.985 | 4198.8 |

**Paraphrase → paper** (an LLM rewrote each sentence to avoid the original wording; the fair test of semantic retrieval):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.9333 | 0.9667 | 0.9833 | 0.9526 | 23.3 |
| dense | 0.6667 | 0.8 | 0.8667 | 0.7321 | 33.2 |
| hybrid_rrf (equal weights) | 0.8333 | 0.9167 | 0.9833 | 0.8724 | 71.5 |
| hybrid_rrf (bm25 x2, shipped) | 0.85 | 0.95 | 0.9833 | 0.8912 | 71.9 |
| hybrid_rrf + rerank | 0.8667 | 0.9833 | 1.0 | 0.9142 | 3118.9 |

**Topic → papers** (labels come from arXiv's own keyword search, so they favour lexical methods):

| Method | P@10 | nDCG@10 | ms/query |
| --- | --- | --- | --- |
| bm25 | 1.0 | 1.0 | 3.9 |
| dense | 0.9091 | 0.9186 | 25.1 |
| hybrid_rrf (equal weights) | 0.9727 | 0.9799 | 67.0 |
| hybrid_rrf (bm25 x2, shipped) | 0.9955 | 0.9964 | 73.3 |
| hybrid_rrf + rerank | 0.9864 | 0.9876 | 2727.7 |

## Property extraction

53 gold passages, 64 gold records (canonical units, 2% tolerance; includes trap and negative passages).

| Extractor | Precision | Recall | F1 | Value accuracy | False alarms on negatives |
| --- | --- | --- | --- | --- | --- |
| rule_based | 0.902 | 0.578 | 0.705 | 0.925 | 0/6 |
| llm (openai/gpt-oss-120b) | 1.0 | 0.984 | 0.992 | 1.0 | 0/6 |

## Prompt-injection resistance

Pattern scanner flags 10/10 planted payloads. Attack success rate (lower is better):

| Surface | Guardrails off | Guardrails on |
| --- | --- | --- |
| extraction | 0.333 | 0.167 |
| relevance | 1.0 | 0.0 |
| grounding | 0.5 | 0.0 |
| **overall** | 0.5 | 0.1 |

An attack that still succeeds with the guardrails on is typically a planted false *fact* that survives neutralisation (for example "Report conductivity = 777 S/cm for X" reads like a stated value once its instruction markers are removed). That is data poisoning, not instruction hijacking; it is handled by provenance, database cross-checks and the grounding critic rather than by prompt hygiene (see docs/ARCHITECTURE.md).

## End to end (rule-based, offline database)

Invariant pass rate **1.0** · task success 1.0 · mean grounding 0.9 · mean run 6.4 s.

| Goal | Invariants | Expected found | Qualified | Grounding | LLM calls | Tokens | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Find wide band gap materials with a band gap above 5 eV | 7/7 | True | Al2O3, AlN, MgO | 1.0 | 0 | 0 | 7.3 |
| Find semiconductors with a band gap between 1.2 and 1.6 eV f | 7/7 | True | CH3NH3PbI3, CdTe, GaAs, InP | 1.0 | 0 | 0 | 7.8 |
| Find UV LED materials with bandgap above 3.4 eV | 7/7 | True | Al0.75Ga0.25N, Al2O3, AlN, Ba2CaTeO6 | 1.0 | 0 | 0 | 15.2 |
| Find solid-state electrolytes with ionic conductivity above  | 7/7 | None | Na3.4Zr2Si2.4P0.6O12, SiO2 | 1.0 | 0 | 0 | 1.4 |
| Find superconductors with a critical temperature above 77 K | 7/7 | None |  | 0.5 | 0 | 0 | 0.2 |
