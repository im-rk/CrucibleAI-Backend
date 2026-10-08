# Evaluation report

Generated 2026-10-08T02:47:57+00:00 · model: `openai/gpt-oss-120b`. Produced by `python scripts/run_evals.py`; the numbers below are copied from `evaluation/results/latest.json`.

## Retrieval

1234 papers indexed. Tasks: {'sentence_to_paper': 150, 'topic_to_papers': 22, 'paraphrase_to_paper': 60}.

**Sentence → paper** (the query is an abstract sentence; exact labels). BM25 is perfect here because the query is copied from the document, so this task mainly checks nothing is broken:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 12.9 |
| dense | 0.7267 | 0.8933 | 0.92 | 0.7938 | 17.4 |
| hybrid_rrf | 0.8267 | 0.94 | 0.9867 | 0.8882 | 45.9 |
| hybrid_rrf_rerank | 0.98 | 0.9933 | 1.0 | 0.9853 | 2602.9 |

**Paraphrase → paper** (an LLM rewrote each sentence to avoid the original wording; the fair test of semantic retrieval):

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
| --- | --- | --- | --- | --- | --- |
| bm25 | 0.9333 | 0.9667 | 0.9833 | 0.9526 | 22.6 |
| dense | 0.6667 | 0.8 | 0.8667 | 0.7321 | 34.4 |
| hybrid_rrf | 0.8333 | 0.9167 | 0.9833 | 0.8724 | 79.3 |
| hybrid_rrf_rerank | 0.8667 | 0.9667 | 0.9833 | 0.9072 | 3108.3 |

**Topic → papers** (labels come from arXiv's own keyword search, so they favour lexical methods):

| Method | P@10 | nDCG@10 | ms/query |
| --- | --- | --- | --- |
| bm25 | 1.0 | 1.0 | 3.4 |
| dense | 0.9091 | 0.9186 | 25.3 |
| hybrid_rrf | 0.9727 | 0.9799 | 49.5 |
| hybrid_rrf_rerank | 0.9773 | 0.9837 | 2697.1 |

## Property extraction

53 gold passages, 64 gold records (canonical units, 2% tolerance; includes trap and negative passages).

| Extractor | Precision | Recall | F1 | Value accuracy | False alarms on negatives |
| --- | --- | --- | --- | --- | --- |
| rule_based | 0.902 | 0.578 | 0.705 | 0.925 | 0/6 |
| llm (openai/gpt-oss-120b) | 1.0 | 1.0 | 1.0 | 1.0 | 0/6 |
