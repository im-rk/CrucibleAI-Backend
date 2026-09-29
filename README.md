# CrucibleAI Backend

Autonomous multi-agent research for materials discovery. Give it a goal in plain English
("find solid-state electrolytes with ionic conductivity above 10⁻³ S/cm, electrochemical
stability above 5 V, and no rare earth elements") and it searches a corpus of papers, extracts
property values with provenance, screens candidates against your criteria, predicts missing
properties, proposes testable hypotheses, and checks that each hypothesis is actually supported
by the evidence it cites.

The web UI lives in [CrucibleAI-Frontend](https://github.com/im-rk/CrucibleAI-Frontend).
Design documents are in [`01_README.md`](01_README.md), [`02_SETUP_GUIDE.md`](02_SETUP_GUIDE.md),
[`03_IMPLEMENTATION_PLAN.md`](03_IMPLEMENTATION_PLAN.md) and
[`MaterialsResearchAgent_PRD.md`](MaterialsResearchAgent_PRD.md).

## How it works

```
goal ──► SearchAgent ──(too few relevant chunks, retries left)──┐
            ▲   hybrid BM25 + semantic (RRF), LLM relevance     │ re-retrieve with a
            └───────────────────────────────────────────────────┘ reformulated query
            │
            ▼
        ExtractionAgent   structured property records + units normalised + Materials Project cross-check
            ▼
        ScreeningAgent    criteria parsed from the goal; per-criterion verdicts; KRR fills gaps
            ▼
        HypothesisAgent   hypotheses that must cite numbered evidence; knowledge gaps; experiments
            ▼
        ReportBuilder     grounding check (every claim vs its evidence) + citations
```

Orchestrated with LangGraph (`agents/orchestrator.py`). Each agent is timed, failure-isolated (a
crash becomes a warning and the run continues), and streams progress events.

## Runs with or without API keys

Every key is optional. Without them the pipeline still runs, in a **degraded mode that is stated
in `/api/health`, in every report (`mode`, `warnings`), and in the UI**:

| Missing | What happens instead |
|---|---|
| `ANTHROPIC_API_KEY` | Query generation, relevance scoring, extraction, criteria parsing and hypotheses fall back to rules and regexes. Results are lower confidence and hypotheses are labelled as templates. |
| `MATERIALS_PROJECT_API_KEY` | A small table of well-known experimental band gaps is used, labelled *"not Materials Project"*. Unknown formulas return nothing; values are never invented. |
| PostgreSQL | Runs still complete; they just are not saved. |

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env                      # add keys if you have them
python scripts/verify_setup.py            # checks corpus, embedding model, keys, database, arXiv
uvicorn matsci_agent.api.server:app --reload --port 8000     # docs at /docs
```

The repository ships with a small indexed corpus (`chroma_db/`). To build a bigger one:

```bash
python scripts/ingest_papers.py --count 200      # downloads + parses + chunks + embeds (slow)
python scripts/train_krr.py --source mp          # train KRR on real Materials Project data (needs a key)
python scripts/run_research.py --query "Find UV LED materials with bandgap above 3.4 eV"
```

### PostgreSQL (optional audit trail)

```bash
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=matsci_agent postgres:15-alpine
alembic upgrade head
```

Stores each session (status, timing, full report), the per-agent audit log, and every extracted
property record. `GET /api/sessions/{id}` reads it back.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/research/stream` | Run the pipeline, streaming progress as Server-Sent Events; the last event carries the report |
| `POST /api/research` | Same, returning the finished report as JSON |
| `GET /api/papers`, `/api/papers/{arxiv_id}` | Browse the corpus |
| `GET /api/materials/lookup?formula=`, `/api/materials/screen` | Materials Project (or offline table) |
| `POST /api/predict` | KRR property estimate with uncertainty |
| `GET /api/sessions`, `/api/sessions/{id}` | Saved runs |
| `GET /api/health`, `/api/corpus` | What is actually running, and corpus statistics |

Stream events, in order: `session_started` → per agent `agent_start` … `agent_complete` (with fine
grained events such as `queries_generated`, `re_retrieving`, `relevance_check`, `mp_lookup`,
`candidates_found`, `hypotheses_generated`, `grounding_check`) → `complete` (or `error`).

```bash
curl -N -X POST localhost:8000/api/research/stream -H 'Content-Type: application/json' \
  -d '{"query": "perovskite solar cell bandgap"}'
```

## Evaluation

```bash
python evaluation/run_eval.py --limit 5     # cheap trial
python evaluation/run_eval.py               # all 20 questions
```

Compares plain RAG with the agentic retrieval (relevance filtering + re-retrieval) on
`evaluation/eval_dataset.json`. Both use the same hybrid search, answer prompt and judge, so the
difference isolates what the agent adds. Reports faithfulness, answer relevancy, context precision
and context recall to `evaluation/results.json`. It needs `ANTHROPIC_API_KEY` (answers and judgements
are LLM calls, roughly 6-8 per question per system) and refuses to run without one.

Read the numbers with these caveats: the metrics follow the RAGAS definitions but are judged by the
project's own LLM client rather than the `ragas` library, so they are comparable between the two
systems here but not to published RAGAS scores; and the 20 reference answers are standard textbook
values, not derived from your corpus, so context recall mostly measures corpus coverage. No results
are checked in: run it to get your own numbers.

## Pinecone (production vector store)

Set `VECTOR_STORE=pinecone` and `PINECONE_API_KEY`, `pip install pinecone`, and copy the local
corpus over with `python scripts/migrate_to_pinecone.py`. With Pinecone the pipeline uses semantic
search only (no BM25) and `/api/papers` answers 501. Verified against the live service: the migrated
1,558-chunk corpus returns the same top results and similarity scores as Chroma (to 3 decimals).

## Layout

```
src/matsci_agent/
  agents/            LangGraph graph + the five nodes (search, extraction, screening, hypothesis, report)
  domain/            schemas, units/value parsing, chemistry helpers, offline reference data
  infrastructure/    LLM client, ChromaDB, BM25/hybrid/rerank, arXiv + Materials Project clients, KRR, PostgreSQL
  services/          research service (streaming), grounding, report builder, persistence, paper catalogue
  api/               FastAPI app, routers, CORS
alembic/             database migrations
scripts/             ingest_papers, train_krr, run_research, verify_setup, migrate_to_pinecone
evaluation/          eval dataset and the plain-RAG vs agentic comparison
tests/               unit + integration (incl. an end-to-end run of the real graph)
```

## Testing

```bash
uv run pytest -q
# also exercise the persistence layer against a real PostgreSQL:
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/matsci_agent uv run pytest -q
```

CI runs lint, the migrations (`alembic upgrade head` + `alembic check`), and the full suite against a
real Postgres service.

## Docker and deployment

```bash
docker compose up --build                        # API + PostgreSQL
docker compose --profile full up --build         # also the web UI on :8080 (needs ../CrucibleAI-Frontend)
```

The image installs the locked dependencies (PyTorch is among them, so it is several GB). The corpus,
KRR models, papers and the embedding-model cache are volumes, not baked in.

**AWS EC2** (the plan's target): use an instance with enough RAM for PyTorch (a `t2.micro` with 1 GB is
likely too small for PyTorch plus the embedding model; I have not measured it, so start with
`t3.medium` and size down if memory allows). Install Docker, clone both
repos side by side, create `.env` from `.env.example`, set `CORS_ORIGINS` to the UI's public origin,
and run `docker compose --profile full up -d --build`. Put a TLS-terminating proxy or load balancer
in front, and give it a long read timeout (research requests stream for tens of seconds); the
bundled nginx already disables response buffering for the SSE endpoint.

## Notes and limits

- KRR predictions are estimates. Each reports its cross-validated uncertainty; only a model trained
  on Materials Project data (`--source mp`) can reach "medium" confidence, and a predicted value can
  never make a candidate "qualified".
- The shipped band-gap model (`models/krr_band_gap.*`) was trained on 1,500 experimentally known,
  near-stable Materials Project materials with Magpie features: cross-validated R² 0.68, RMSE ±0.93 eV.
  It predicts the **PBE-DFT** band gap, which underestimates measured gaps (GaN: 1.97 eV predicted,
  1.74 eV DFT, 3.4 eV measured); each prediction's note says so. Retrain with
  `python scripts/train_krr.py --source mp --samples 1500`.
- `claude-sonnet-4-6` is the model named in the project spec; set `ANTHROPIC_MODEL` to use another.
- The Materials Project citation DOI shown for database values is Jain et al., *APL Materials* 2013
  (`10.1063/1.4812323`).
- Network note: HTTPS is verified against the operating system's certificate store (`USE_SYSTEM_CERTS=true`),
  which fixes `CERTIFICATE_VERIFY_FAILED` behind TLS-inspecting proxies or antivirus without disabling
  verification.
- Materials Project lookups use the **ground state** (lowest energy above hull) of each formula; the
  database seed for thin literature only considers experimentally known structures (`theoretical=False`)
  without radioactive elements. Band gaps from Materials Project are PBE DFT values and are typically
  underestimated (GaN: 1.74 eV vs 3.4 eV measured), which is why literature values take priority.
- If Anthropic rejects the account (no credits, invalid key), the pipeline makes one failing call, switches
  to heuristic mode for the rest of the process, and reports the reason in `/api/health` and the report `mode`.
