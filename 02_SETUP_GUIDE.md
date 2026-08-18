# Setup Guide — MaterialsResearchAgent
## Everything you need to run this project from zero

---

## Prerequisites

| Tool | Version | How to Install |
|------|---------|---------------|
| Python | 3.11+ | python.org or `pyenv install 3.11` |
| Node.js | 20+ | nodejs.org or `nvm install 20` |
| PostgreSQL | 15+ | `brew install postgresql` (Mac) or `apt install postgresql` (Linux) |
| Git | Any | pre-installed on most systems |

---

## API Keys You Need (All Free)

### 1. Anthropic API Key (for Claude)
- Go to: console.anthropic.com
- Sign up → API Keys → Create Key
- Free credits given on signup (~$5 — enough for all development)
- Save as: `ANTHROPIC_API_KEY`

### 2. Materials Project API Key (for property data)
- Go to: next.materialsproject.org
- Sign up with email (free)
- Go to Dashboard → API Key
- Save as: `MATERIALS_PROJECT_API_KEY`

### 3. Pinecone API Key (for vector database — production)
- Go to: pinecone.io
- Sign up → API Keys → Create Key
- Free tier: 1 index, 100k vectors (enough for 500 papers)
- Save as: `PINECONE_API_KEY`
- **For development only:** ChromaDB is used locally — no key needed

---

## Project Structure (Create This First)

```
matsci-agent/
├── agents/
│   ├── __init__.py
│   ├── search_agent.py
│   ├── extraction_agent.py
│   ├── screening_agent.py
│   ├── hypothesis_agent.py
│   └── orchestrator.py
├── tools/
│   ├── __init__.py
│   ├── materials_project.py
│   ├── arxiv_search.py
│   ├── krr_predictor.py
│   ├── unit_converter.py
│   └── grounding_checker.py
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── hybrid_search.py
│   └── reranker.py
├── data_pipeline/
│   ├── __init__.py
│   ├── arxiv_fetcher.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   └── embedder.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── research.py
│       ├── papers.py
│       └── materials.py
├── evaluation/
│   ├── eval_dataset.json
│   └── run_ragas_eval.py
├── scripts/
│   ├── ingest_papers.py
│   └── train_krr.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── App.tsx
│   └── package.json
├── .env
├── .env.example
├── requirements.txt
└── docker-compose.yml
```

---

## Step 1 — Clone and Python Setup

```bash
# Create project
mkdir matsci-agent && cd matsci-agent
git init

# Create virtual environment (ALWAYS use a venv for Python projects)
python -m venv venv

# Activate it
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Verify Python version
python --version   # Should show 3.11.x
```

---

## Step 2 — Install Python Dependencies

Create `requirements.txt`:

```txt
# API Framework
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
sse-starlette==2.1.0

# Agent Orchestration
langgraph==0.1.19
langchain==0.2.6
langchain-anthropic==0.1.19
langchain-community==0.2.6

# LLM
anthropic==0.29.0

# Vector Database
chromadb==0.5.3
pinecone-client==4.1.0

# Embeddings (runs locally, no API key)
sentence-transformers==3.0.1
torch==2.3.1

# Materials Science
mp-api==0.41.2
pymatgen==2024.6.10
matminer==0.9.2

# Paper Fetching
arxiv==2.1.0
PyMuPDF==1.24.5

# Search
rank-bm25==0.2.2

# Property Prediction
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2

# Evaluation
ragas==0.1.14
datasets==2.19.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.30

# Utilities
pydantic==2.7.1
python-dotenv==1.0.1
httpx==0.27.0
aiofiles==23.2.1
```

```bash
pip install -r requirements.txt

# This will take 5-10 minutes — sentence-transformers downloads the model
# You will see it download "all-MiniLM-L6-v2" — that is correct
```

---

## Step 3 — Environment Variables

Create `.env` file in project root:

```bash
# .env — DO NOT commit this file to git

# LLM
ANTHROPIC_API_KEY=sk-ant-xxxx

# Materials Science Database
MATERIALS_PROJECT_API_KEY=your_mp_key_here

# Vector Database (leave blank during development — uses local ChromaDB)
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=

# PostgreSQL
DATABASE_URL=postgresql://localhost:5432/matsci_agent

# App Settings
DEBUG=true
LOG_LEVEL=INFO
CORPUS_SIZE=200         # Start with 200 papers
MAX_RETRIES=3           # Max re-retrieval attempts per query
RELEVANCE_THRESHOLD=0.65  # ISREL cutoff
```

Create `.env.example` (safe to commit):

```bash
ANTHROPIC_API_KEY=your_anthropic_key
MATERIALS_PROJECT_API_KEY=your_mp_key
PINECONE_API_KEY=optional_for_production
DATABASE_URL=postgresql://localhost:5432/matsci_agent
```

Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "chroma_db/" >> .gitignore   # vector DB data
echo "papers/" >> .gitignore      # downloaded PDFs
```

---

## Step 4 — PostgreSQL Database Setup

```bash
# Create the database
createdb matsci_agent

# Connect and create tables
psql matsci_agent
```

```sql
-- Paste this into psql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Research sessions
CREATE TABLE research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    research_goal TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    -- PENDING | RUNNING | COMPLETE | FAILED
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Papers ingested into the corpus
CREATE TABLE papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    doi VARCHAR(200),
    pdf_path TEXT,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chunk_count INTEGER DEFAULT 0
);

-- Extracted property records
CREATE TABLE property_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES research_sessions(id),
    material_name VARCHAR(200) NOT NULL,
    material_formula VARCHAR(100),
    property_name VARCHAR(100) NOT NULL,
    value_raw TEXT NOT NULL,
    value_numeric FLOAT,
    unit VARCHAR(50),
    conditions TEXT,
    source_paper_id UUID REFERENCES papers(id),
    source_doi VARCHAR(200),
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_predicted BOOLEAN DEFAULT FALSE,
    prediction_confidence FLOAT
);

-- Agent decision log (every decision recorded)
CREATE TABLE agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES research_sessions(id),
    agent_name VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    duration_ms INTEGER,
    logged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Evaluation results
CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES research_sessions(id),
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Verify tables created
\dt
```

---

## Step 5 — Verify Everything Works

Run these checks before starting Week 1 build:

```python
# save as: scripts/verify_setup.py

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def check_all():
    errors = []

    # Check 1: Anthropic API
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        print("✅ Anthropic API: working")
    except Exception as e:
        errors.append(f"❌ Anthropic API: {e}")

    # Check 2: Materials Project API
    try:
        from mp_api.client import MPRester
        with MPRester(os.getenv("MATERIALS_PROJECT_API_KEY")) as mpr:
            docs = mpr.summary.search(formula="GaN", fields=["band_gap"])
            print(f"✅ Materials Project API: working (GaN bandgap: {docs[0].band_gap} eV)")
    except Exception as e:
        errors.append(f"❌ Materials Project API: {e}")

    # Check 3: ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        col = client.get_or_create_collection("test")
        print("✅ ChromaDB: working")
    except Exception as e:
        errors.append(f"❌ ChromaDB: {e}")

    # Check 4: Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(["test"])
        print(f"✅ Embedding model: working (dim={len(embedding[0])})")
    except Exception as e:
        errors.append(f"❌ Embedding model: {e}")

    # Check 5: PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        conn.close()
        print("✅ PostgreSQL: connected")
    except Exception as e:
        errors.append(f"❌ PostgreSQL: {e}")

    # Check 6: arXiv
    try:
        import arxiv
        search = arxiv.Search(query="materials science", max_results=1)
        results = list(search.results())
        print(f"✅ arXiv API: working (found: {results[0].title[:50]}...)")
    except Exception as e:
        errors.append(f"❌ arXiv API: {e}")

    # Summary
    print("\n" + "="*50)
    if errors:
        print("SETUP INCOMPLETE. Fix these:")
        for e in errors:
            print(f"  {e}")
    else:
        print("ALL CHECKS PASSED. You are ready to build.")

asyncio.run(check_all())
```

```bash
python scripts/verify_setup.py
```

**Expected output:**
```
✅ Anthropic API: working
✅ Materials Project API: working (GaN bandgap: 3.4 eV)
✅ ChromaDB: working
✅ Embedding model: working (dim=384)
✅ PostgreSQL: connected
✅ arXiv API: working (found: Machine learning for materials...)

==================================================
ALL CHECKS PASSED. You are ready to build.
```

---

## Step 6 — Frontend Setup

```bash
cd frontend
npx create-react-app . --template typescript
npm install tailwindcss @tailwindcss/typography axios
npx tailwindcss init

# Start frontend dev server
npm start
# Opens http://localhost:3000
```

---

## Step 7 — Start the Backend

```bash
# From project root
cd matsci-agent
source venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Visit: http://localhost:8000/docs
# You should see the Swagger UI with your API endpoints
```

---

## Docker Compose (Alternative — runs everything at once)

```yaml
# docker-compose.yml
version: "3.8"

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - MATERIALS_PROJECT_API_KEY=${MATERIALS_PROJECT_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/matsci_agent
    depends_on:
      - db
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./papers:/app/papers

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: matsci_agent
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker-compose up
# Everything starts together
```

---

## Common Problems and Fixes

| Problem | Fix |
|---------|-----|
| `torch` install fails | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| ChromaDB import error | `pip install chromadb --upgrade` |
| MP API returns empty | Check your API key is correct at next.materialsproject.org |
| arXiv search slow | Normal — arXiv API has rate limits, add `time.sleep(3)` between requests |
| Anthropic 401 error | Check API key starts with `sk-ant-` |
| PostgreSQL connection refused | `brew services start postgresql` or `pg_ctl start` |
