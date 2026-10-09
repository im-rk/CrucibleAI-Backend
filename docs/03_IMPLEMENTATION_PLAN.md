# Implementation Plan — MaterialsResearchAgent
## Week by Week. Day by Day. File by File.

---

## Overview

| Week | What You Build | End-of-Week Goal |
|------|---------------|-----------------|
| 1 | Data pipeline — fetch papers, parse, chunk, embed, store | 200 papers searchable in ChromaDB |
| 2 | Materials Project client + KRR property predictor | Property lookup + prediction working |
| 3 | SearchAgent — hybrid retrieval + ISREL + re-retrieval | Agent finds and evaluates relevant papers |
| 4 | ExtractionAgent — structured property extraction from papers | Agent extracts material properties as typed records |
| 5 | ScreeningAgent + HypothesisAgent | End-to-end pipeline: goal → candidates → hypotheses |
| 6 | LangGraph orchestration + FastAPI streaming backend | Full pipeline runs, streams progress to caller |
| 7 | React frontend — query UI + streaming display + report view | Complete UI showing real-time agent progress |
| 8 | Evaluation + polish + AWS deployment | RAGAS scores in README, live URL |

---

## Week 1 — Data Pipeline

### What you are building

The knowledge base. You download materials science papers from arXiv,
parse their text, break them into chunks, convert chunks to embedding
vectors, and store everything in ChromaDB so agents can search it.

Without this, agents have nothing to search.

---

### Day 1 — arXiv Fetcher

```python
# data_pipeline/arxiv_fetcher.py

import arxiv
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

@dataclass
class FetchedPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    doi: str | None
    pdf_path: str | None = None

class ArxivFetcher:
    """Fetches materials science papers from arXiv"""

    # These categories cover your subject syllabus
    MATERIALS_CATEGORIES = [
        "cond-mat.mtrl-sci",   # Materials science
        "cond-mat.supr-con",   # Superconductivity
        "physics.chem-ph",     # Chemical physics
    ]

    MATERIALS_QUERIES = [
        "solid state electrolyte ionic conductivity",
        "perovskite solar cell efficiency",
        "battery electrode materials machine learning",
        "semiconductor bandgap density functional theory",
        "kernel ridge regression materials property",
        "graph neural network crystal structure",
        "gaussian approximation potential molecular dynamics",
        "machine learning interatomic potential",
        "high entropy alloy properties prediction",
        "materials discovery autonomous",
    ]

    def __init__(self, papers_dir: str = "./papers"):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(exist_ok=True)

    def fetch_papers(
            self,
            max_per_query: int = 20) -> Iterator[FetchedPaper]:
        """Fetch papers for all predefined queries"""

        for query in self.MATERIALS_QUERIES:
            print(f"Fetching papers for: '{query}'")

            search = arxiv.Search(
                query=query,
                max_results=max_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            for result in search.results():
                yield FetchedPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    year=result.published.year,
                    doi=result.doi,
                )
                # Respect arXiv rate limits
                time.sleep(0.5)

    def download_pdf(self, paper: FetchedPaper) -> str | None:
        """Download PDF for a paper — returns path or None if failed"""
        pdf_path = self.papers_dir / f"{paper.arxiv_id}.pdf"

        if pdf_path.exists():
            return str(pdf_path)

        try:
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(search.results())
            result.download_pdf(dirpath=str(self.papers_dir),
                               filename=f"{paper.arxiv_id}.pdf")
            return str(pdf_path)
        except Exception as e:
            print(f"  Could not download {paper.arxiv_id}: {e}")
            return None
```

**Test it:**
```python
# Run in Python REPL
from data_pipeline.arxiv_fetcher import ArxivFetcher
fetcher = ArxivFetcher()
papers = list(fetcher.fetch_papers(max_per_query=5))
print(f"Fetched {len(papers)} papers")
print(papers[0].title)
```

---

### Day 2 — PDF Parser

```python
# data_pipeline/pdf_parser.py

import fitz  # PyMuPDF
import re
from dataclasses import dataclass

@dataclass
class ParsedSection:
    section_name: str
    text: str
    page_numbers: list[int]

@dataclass
class ParsedPaper:
    arxiv_id: str
    full_text: str
    sections: list[ParsedSection]
    tables_text: list[str]  # Text extracted from tables

# Common section headers in materials science papers
SECTION_PATTERNS = [
    r"^\s*(abstract)\s*$",
    r"^\s*(1\.?\s*introduction)\s*",
    r"^\s*(2\.?\s*experimental|2\.?\s*methods|2\.?\s*computational)\s*",
    r"^\s*(3\.?\s*results|3\.?\s*results and discussion)\s*",
    r"^\s*(4\.?\s*discussion)\s*",
    r"^\s*(5\.?\s*conclusion|5\.?\s*conclusions)\s*",
    r"^\s*(references)\s*$",
]

class PDFParser:

    def parse(self, pdf_path: str, arxiv_id: str) -> ParsedPaper | None:
        """Parse a PDF into structured sections"""
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Cannot open {pdf_path}: {e}")
            return None

        all_text = []
        tables_text = []

        for page in doc:
            # Extract text — preserves reading order
            text = page.get_text("text")
            all_text.append(text)

            # Look for table-like content (rows with numbers)
            if self._looks_like_table(text):
                tables_text.append(text)

        full_text = "\n".join(all_text)
        sections = self._split_into_sections(full_text)

        doc.close()

        return ParsedPaper(
            arxiv_id=arxiv_id,
            full_text=full_text,
            sections=sections,
            tables_text=tables_text,
        )

    def _split_into_sections(self, text: str) -> list[ParsedSection]:
        """Split paper text into sections by header detection"""
        lines = text.split("\n")
        sections = []
        current_section = "preamble"
        current_lines = []

        for line in lines:
            # Check if this line is a section header
            is_header = False
            for pattern in SECTION_PATTERNS:
                if re.match(pattern, line.lower()):
                    # Save current section
                    if current_lines:
                        sections.append(ParsedSection(
                            section_name=current_section,
                            text="\n".join(current_lines),
                            page_numbers=[],
                        ))
                    current_section = line.strip()
                    current_lines = []
                    is_header = True
                    break

            if not is_header:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections.append(ParsedSection(
                section_name=current_section,
                text="\n".join(current_lines),
                page_numbers=[],
            ))

        return sections

    def _looks_like_table(self, text: str) -> bool:
        """Heuristic: does this text contain a data table?"""
        # Tables often have multiple numbers separated by spaces
        number_pattern = r'\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*'
        return bool(re.search(number_pattern, text))
```

---

### Day 3 — Chunker

```python
# data_pipeline/chunker.py

from dataclasses import dataclass
from .pdf_parser import ParsedPaper, ParsedSection

@dataclass
class Chunk:
    text: str
    section_name: str
    chunk_index: int
    arxiv_id: str
    char_count: int

class SectionAwareChunker:
    """
    Chunks paper text respecting section boundaries.
    Never splits a sentence across chunks.
    Uses overlapping windows to preserve context.
    """

    def __init__(self, max_chars: int = 1500, overlap_chars: int = 200):
        # 1500 chars ≈ 300-400 tokens — fits in LLM context with room
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_paper(self, paper: ParsedPaper) -> list[Chunk]:
        """Chunk a full parsed paper into searchable chunks"""
        all_chunks = []

        for section in paper.sections:
            # Skip very short sections (like "References\n" with no content)
            if len(section.text.strip()) < 100:
                continue

            # Skip reference section — not useful for retrieval
            if "reference" in section.section_name.lower():
                continue

            section_chunks = self._chunk_section(
                section, paper.arxiv_id, len(all_chunks))
            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_section(
            self,
            section: ParsedSection,
            arxiv_id: str,
            start_index: int) -> list[Chunk]:

        text = section.text.strip()
        chunks = []
        chunk_index = start_index
        pos = 0

        while pos < len(text):
            # Find end of this chunk
            end = min(pos + self.max_chars, len(text))

            # Don't cut mid-sentence — find the last period before end
            if end < len(text):
                last_period = text.rfind(".", pos, end)
                if last_period > pos + self.max_chars // 2:
                    end = last_period + 1

            chunk_text = text[pos:end].strip()

            if len(chunk_text) > 50:  # Skip tiny fragments
                chunks.append(Chunk(
                    text=chunk_text,
                    section_name=section.section_name,
                    chunk_index=chunk_index,
                    arxiv_id=arxiv_id,
                    char_count=len(chunk_text),
                ))
                chunk_index += 1

            # Move forward, but overlap with previous chunk
            pos = end - self.overlap_chars if end < len(text) else len(text)

        return chunks
```

---

### Day 4–5 — Embedder + ChromaDB Store

```python
# data_pipeline/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class MaterialsEmbedder:
    """
    Converts text chunks to embedding vectors.
    Uses all-MiniLM-L6-v2 — runs locally, no API cost.
    384-dimensional vectors.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("Model loaded.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts — returns shape (n, 384)"""
        return self.model.encode(
            texts,
            batch_size=32,        # Process 32 chunks at a time
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalise for cosine similarity
        )

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text — used for query embedding"""
        return self.model.encode([text], normalize_embeddings=True)[0]
```

```python
# retrieval/vector_store.py

import chromadb
from dataclasses import asdict
from data_pipeline.chunker import Chunk
from data_pipeline.embedder import MaterialsEmbedder

class MaterialsVectorStore:
    """ChromaDB wrapper for materials science paper chunks"""

    COLLECTION_NAME = "materials_papers"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        self.embedder = MaterialsEmbedder()

    def add_chunks(
            self,
            chunks: list[Chunk],
            paper_metadata: dict) -> None:
        """Add paper chunks to the vector store"""

        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts).tolist()

        ids = [f"{c.arxiv_id}_{c.chunk_index}" for c in chunks]

        metadatas = [{
            "arxiv_id": c.arxiv_id,
            "section": c.section_name,
            "chunk_index": c.chunk_index,
            "title": paper_metadata.get("title", ""),
            "authors": paper_metadata.get("authors", ""),
            "year": paper_metadata.get("year", 0),
            "doi": paper_metadata.get("doi", ""),
        } for c in chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
            self,
            query: str,
            n_results: int = 10,
            where: dict | None = None) -> list[dict]:
        """Search for relevant chunks"""

        query_embedding = self.embedder.embed_single(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

        # Format results as list of dicts
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })

        return chunks

    def count(self) -> int:
        return self.collection.count()
```

### Day 5 — Ingestion Script

```python
# scripts/ingest_papers.py

import argparse
import sys
sys.path.append(".")

from data_pipeline.arxiv_fetcher import ArxivFetcher
from data_pipeline.pdf_parser import PDFParser
from data_pipeline.chunker import SectionAwareChunker
from retrieval.vector_store import MaterialsVectorStore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    fetcher = ArxivFetcher()
    pdf_parser = PDFParser()
    chunker = SectionAwareChunker()
    vector_store = MaterialsVectorStore()

    print(f"Starting ingestion. Target: {args.count} papers")
    print(f"Current corpus size: {vector_store.count()} chunks")

    papers_processed = 0
    total_chunks = 0

    for paper in fetcher.fetch_papers(max_per_query=args.count // 10):
        if papers_processed >= args.count:
            break

        print(f"\n[{papers_processed+1}] {paper.title[:60]}...")

        # Download PDF
        pdf_path = fetcher.download_pdf(paper)
        if not pdf_path:
            print("  Skipping — PDF download failed")
            continue

        # Parse PDF
        parsed = pdf_parser.parse(pdf_path, paper.arxiv_id)
        if not parsed:
            print("  Skipping — PDF parse failed")
            continue

        # Chunk
        chunks = chunker.chunk_paper(parsed)
        if not chunks:
            print("  Skipping — no usable text extracted")
            continue

        # Store in vector DB
        paper_meta = {
            "title": paper.title,
            "authors": ", ".join(paper.authors[:3]),
            "year": paper.year,
            "doi": paper.doi or "",
        }
        vector_store.add_chunks(chunks, paper_meta)

        papers_processed += 1
        total_chunks += len(chunks)
        print(f"  ✓ Added {len(chunks)} chunks")

    print(f"\nIngestion complete!")
    print(f"Papers processed: {papers_processed}")
    print(f"Total chunks: {total_chunks}")
    print(f"Corpus size: {vector_store.count()} chunks")

if __name__ == "__main__":
    main()
```

**Run it:**
```bash
python scripts/ingest_papers.py --count 200
# Takes 20-40 minutes for 200 papers
# Go do something else while it runs
```

**Week 1 verification:**
```python
from retrieval.vector_store import MaterialsVectorStore
store = MaterialsVectorStore()
print(f"Corpus: {store.count()} chunks")

results = store.search("ionic conductivity solid electrolyte", n_results=3)
for r in results:
    print(f"\n[{r['similarity']:.3f}] {r['metadata']['title'][:60]}")
    print(f"  {r['text'][:100]}...")
```

---

## Week 2 — Materials Project Client + KRR Predictor

### Day 1–2 — Materials Project Client

```python
# tools/materials_project.py

import os
from mp_api.client import MPRester
from pydantic import BaseModel
from typing import Optional

class MaterialProperty(BaseModel):
    material_id: str
    formula: str
    property_name: str
    value: float | str
    unit: str
    source: str = "Materials Project"
    doi: str = "https://doi.org/10.1038/s41524-020-00448-7"
    is_direct_gap: Optional[bool] = None

class MaterialsProjectClient:
    """
    Wraps the Materials Project API.
    Provides clean, typed access to material properties.
    """

    def __init__(self):
        self.api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
        if not self.api_key:
            raise ValueError("MATERIALS_PROJECT_API_KEY not set in .env")

    def get_properties(
            self,
            formula: str,
            properties: list[str] = None) -> list[MaterialProperty]:
        """
        Get properties for a material.
        formula: chemical formula, e.g. "GaN", "Li7La3Zr2O12"
        properties: list of property names, or None for all
        """

        default_properties = [
            "band_gap", "formation_energy_per_atom",
            "energy_above_hull", "density",
            "crystal_system", "is_gap_direct"
        ]
        requested = properties or default_properties

        with MPRester(self.api_key) as mpr:
            docs = mpr.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty"] + requested
            )

        if not docs:
            return []

        results = []
        for doc in docs:
            # Bandgap
            if "band_gap" in requested and doc.band_gap is not None:
                results.append(MaterialProperty(
                    material_id=doc.material_id,
                    formula=doc.formula_pretty,
                    property_name="bandgap",
                    value=round(doc.band_gap, 4),
                    unit="eV",
                    is_direct_gap=doc.is_gap_direct,
                ))
            # Formation energy
            if ("formation_energy_per_atom" in requested
                    and doc.formation_energy_per_atom is not None):
                results.append(MaterialProperty(
                    material_id=doc.material_id,
                    formula=doc.formula_pretty,
                    property_name="formation_energy",
                    value=round(doc.formation_energy_per_atom, 4),
                    unit="eV/atom",
                ))
            # Density
            if "density" in requested and doc.density is not None:
                results.append(MaterialProperty(
                    material_id=doc.material_id,
                    formula=doc.formula_pretty,
                    property_name="density",
                    value=round(doc.density, 4),
                    unit="g/cm³",
                ))

        return results

    def screen_by_criteria(
            self,
            bandgap_min: float = None,
            bandgap_max: float = None,
            is_stable: bool = True,
            crystal_system: str = None,
            max_results: int = 20) -> list[dict]:
        """Screen the entire Materials Project by property criteria"""

        with MPRester(self.api_key) as mpr:
            query_params = {
                "fields": ["material_id", "formula_pretty",
                           "band_gap", "formation_energy_per_atom",
                           "energy_above_hull", "crystal_system"],
            }

            if bandgap_min is not None:
                query_params["band_gap"] = (bandgap_min, bandgap_max or 20)
            if is_stable:
                query_params["energy_above_hull"] = (0, 0.1)
            if crystal_system:
                query_params["crystal_system"] = crystal_system

            docs = mpr.summary.search(**query_params)

        return [
            {
                "material_id": d.material_id,
                "formula": d.formula_pretty,
                "bandgap_eV": d.band_gap,
                "formation_energy_eV_per_atom": d.formation_energy_per_atom,
                "stability_eV": d.energy_above_hull,
                "crystal_system": str(d.crystal_system),
            }
            for d in docs[:max_results]
        ]
```

### Day 3–5 — KRR Predictor (Syllabus requirement)

```python
# tools/krr_predictor.py

"""
Kernel Ridge Regression for materials property prediction.

This implements the technique from your syllabus:
"Kernel Ridge Regression for materials property prediction"
Based on: Pilania et al. Scientific Reports 2013

Used by ScreeningAgent when a candidate material has missing properties.
Instead of skipping the material, we predict the missing property.
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os
from mp_api.client import MPRester

class KRRPropertyPredictor:
    """
    Predicts material properties using Kernel Ridge Regression.
    Features: Magpie compositional descriptors (132 features per material)
    Kernel: RBF (Radial Basis Function)
    """

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}      # property_name → trained KRR model
        self.scalers = {}     # property_name → fitted StandardScaler

    def train(self, property_name: str = "band_gap", n_samples: int = 1000):
        """
        Train KRR on Materials Project data.
        Run this once: python scripts/train_krr.py
        """
        print(f"Training KRR for: {property_name}")
        print(f"Downloading {n_samples} training samples from Materials Project...")

        # Get training data from Materials Project
        X, y, formulas = self._get_training_data(property_name, n_samples)

        print(f"Got {len(y)} training samples")
        print(f"Property range: {y.min():.3f} to {y.max():.3f}")

        # Normalise features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train KRR with RBF kernel
        # alpha: regularisation strength
        # gamma: RBF kernel width
        model = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.01)
        model.fit(X_scaled, y)

        # Cross-validation to check quality
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Save model and scaler
        model_path = os.path.join(self.model_dir, f"krr_{property_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"scaler_{property_name}.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        self.models[property_name] = model
        self.scalers[property_name] = scaler
        print(f"Model saved to {model_path}")

    def predict(
            self,
            formula: str,
            property_name: str = "band_gap") -> dict:
        """
        Predict a property for a material given its formula.
        Returns prediction with uncertainty estimate.
        """
        # Load model if not in memory
        if property_name not in self.models:
            model_path = os.path.join(
                self.model_dir, f"krr_{property_name}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No trained model for {property_name}. "
                    f"Run: python scripts/train_krr.py first.")
            self.models[property_name] = joblib.load(model_path)
            self.scalers[property_name] = joblib.load(
                os.path.join(self.model_dir, f"scaler_{property_name}.joblib"))

        # Compute features for the input formula
        features = self._compute_magpie_features(formula)
        if features is None:
            return {"error": f"Cannot compute features for formula: {formula}"}

        features_scaled = self.scalers[property_name].transform([features])
        prediction = self.models[property_name].predict(features_scaled)[0]

        return {
            "formula": formula,
            "property": property_name,
            "predicted_value": round(float(prediction), 4),
            "unit": self._get_unit(property_name),
            "method": "Kernel Ridge Regression (RBF kernel)",
            "is_predicted": True,
            "confidence": "medium",  # Could improve with proper uncertainty
            "note": "This is a ML prediction, not an experimental measurement. "
                    "Verify with DFT calculation or experiment before use.",
        }

    def _compute_magpie_features(self, formula: str) -> np.ndarray | None:
        """
        Compute Magpie compositional features for a chemical formula.
        Magpie: 132 element-derived features (Ward et al. 2016)
        """
        try:
            from matminer.featurizers.composition import ElementProperty
            from pymatgen.core import Composition

            comp = Composition(formula)
            featurizer = ElementProperty.from_preset("magpie")
            featurizer.set_n_jobs(1)

            # Featurizer returns a list of 132 numbers
            features = featurizer.featurize(comp)
            return np.array(features)

        except Exception as e:
            print(f"Feature computation failed for {formula}: {e}")
            return None

    def _get_training_data(
            self, property_name: str, n_samples: int):
        """Download training data from Materials Project"""
        api_key = os.getenv("MATERIALS_PROJECT_API_KEY")

        with MPRester(api_key) as mpr:
            if property_name == "band_gap":
                docs = mpr.summary.search(
                    band_gap=(0.1, 10.0),  # Only materials with a bandgap
                    fields=["formula_pretty", "band_gap"],
                    num_chunks=n_samples // 100,
                )
                formulas = [d.formula_pretty for d in docs]
                y = np.array([d.band_gap for d in docs])

        # Compute features for all formulas
        X_list = []
        y_list = []
        valid_formulas = []

        for formula, target in zip(formulas, y):
            features = self._compute_magpie_features(formula)
            if features is not None and not np.any(np.isnan(features)):
                X_list.append(features)
                y_list.append(target)
                valid_formulas.append(formula)

        return np.array(X_list), np.array(y_list), valid_formulas

    def _get_unit(self, property_name: str) -> str:
        units = {
            "band_gap": "eV",
            "formation_energy_per_atom": "eV/atom",
            "density": "g/cm³",
        }
        return units.get(property_name, "unknown")
```

**Training script:**
```python
# scripts/train_krr.py
import sys
sys.path.append(".")
from tools.krr_predictor import KRRPropertyPredictor

predictor = KRRPropertyPredictor()
predictor.train(property_name="band_gap", n_samples=500)
print("\nKRR training complete.")
print("Test prediction:")
result = predictor.predict("GaN", "band_gap")
print(f"  Predicted GaN bandgap: {result['predicted_value']} eV")
print(f"  (Actual: 3.4 eV — KRR should be close)")
```

---

## Week 3 — SearchAgent

```python
# agents/search_agent.py

from anthropic import Anthropic
from retrieval.vector_store import MaterialsVectorStore
from retrieval.bm25_index import BM25Index
import json

class SearchAgent:
    """
    Finds relevant papers for a research goal.
    Implements Self-RAG retrieval pattern:
    1. Decide what to search for
    2. Hybrid search (semantic + BM25)
    3. Evaluate relevance (ISREL)
    4. Re-retrieve if results are weak
    """

    MAX_RETRIES = 3
    RELEVANCE_THRESHOLD = 0.65

    def __init__(self):
        self.llm = Anthropic()
        self.vector_store = MaterialsVectorStore()
        self.bm25 = BM25Index()

    async def run(self, research_goal: str, state: dict) -> dict:
        """Main entry point — returns updated state"""

        state["agent_log"].append("SearchAgent: Analysing research goal...")

        # Step 1: Generate search queries from the research goal
        queries = await self._generate_queries(research_goal)
        state["search_queries"] = queries
        state["agent_log"].append(
            f"SearchAgent: Generated {len(queries)} search queries")

        # Step 2: Hybrid search for each query
        all_chunks = []
        for query in queries:
            chunks = await self._hybrid_search(query)
            all_chunks.extend(chunks)

        # Deduplicate by chunk ID
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                unique_chunks.append(chunk)

        state["agent_log"].append(
            f"SearchAgent: Retrieved {len(unique_chunks)} unique chunks")

        # Step 3: Evaluate relevance (ISREL)
        relevant = await self._evaluate_relevance(
            research_goal, unique_chunks)

        state["agent_log"].append(
            f"SearchAgent: {len(relevant)} chunks are relevant "
            f"(threshold: {self.RELEVANCE_THRESHOLD})")

        # Step 4: Re-retrieve if too few relevant results
        retry_count = 0
        while len(relevant) < 5 and retry_count < self.MAX_RETRIES:
            retry_count += 1
            state["agent_log"].append(
                f"SearchAgent: Too few results ({len(relevant)}). "
                f"Reformulating query (attempt {retry_count})...")

            better_query = await self._reformulate_query(research_goal, relevant)
            new_chunks = await self._hybrid_search(better_query)

            # Add new unique chunks
            for chunk in new_chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    unique_chunks.append(chunk)

            # Re-evaluate relevance of all chunks
            relevant = await self._evaluate_relevance(
                research_goal, unique_chunks)

        state["retrieved_papers"] = relevant
        state["retrieval_attempts"] = retry_count + 1

        return state

    async def _generate_queries(self, research_goal: str) -> list[str]:
        """Generate 3-5 search queries from a research goal"""

        prompt = f"""You are a materials science search expert.
        
Research goal: {research_goal}

Generate 3 to 5 search queries that would find the most relevant 
materials science papers for this goal.

Think about:
- Specific material names/formulas that might be relevant
- The key property being searched for
- The application domain
- The experimental techniques used to measure this property

Return a JSON array of query strings only. Example:
["query 1", "query 2", "query 3"]"""

        response = self.llm.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            queries = json.loads(response.content[0].text)
            return queries[:5]  # Max 5 queries
        except:
            # Fallback: use the research goal directly
            return [research_goal]

    async def _hybrid_search(self, query: str) -> list[dict]:
        """Combine semantic search + BM25 keyword search"""

        # Semantic search
        semantic_results = self.vector_store.search(query, n_results=15)

        # BM25 keyword search
        bm25_results = self.bm25.search(query, n_results=15)

        # Merge using Reciprocal Rank Fusion
        merged = self._rrf_merge(semantic_results, bm25_results)
        return merged[:12]  # Return top 12

    def _rrf_merge(
            self,
            list1: list[dict],
            list2: list[dict],
            k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion — merges two ranked lists"""
        scores = {}
        chunk_data = {}

        for rank, item in enumerate(list1):
            id_ = item["id"]
            scores[id_] = scores.get(id_, 0) + 1.0 / (k + rank + 1)
            chunk_data[id_] = item

        for rank, item in enumerate(list2):
            id_ = item["id"]
            scores[id_] = scores.get(id_, 0) + 1.0 / (k + rank + 1)
            chunk_data[id_] = item

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunk_data[id_] for id_ in sorted_ids]

    async def _evaluate_relevance(
            self,
            research_goal: str,
            chunks: list[dict]) -> list[dict]:
        """Score each chunk for relevance to the research goal (ISREL)"""

        relevant = []

        for chunk in chunks:
            prompt = f"""Research goal: {research_goal}

Retrieved text: {chunk['text'][:400]}

Is this text relevant to the research goal?
Score: 0.0 (not relevant) to 1.0 (highly relevant)
Return only the number, nothing else."""

            response = self.llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                score = float(response.content[0].text.strip())
                if score >= self.RELEVANCE_THRESHOLD:
                    chunk["relevance_score"] = score
                    relevant.append(chunk)
            except:
                pass  # Skip chunks we cannot score

        # Sort by relevance score
        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return relevant

    async def _reformulate_query(
            self, original_goal: str, weak_results: list[dict]) -> str:
        """Generate a better search query when initial results are weak"""

        prompt = f"""Research goal: {original_goal}

The initial search returned weak results. The few relevant results found were about:
{[r['text'][:100] for r in weak_results[:3]]}

Generate ONE improved search query that might find more relevant materials science papers.
Return only the query string."""

        response = self.llm.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()
```

---

## Week 4 — ExtractionAgent

```python
# agents/extraction_agent.py

from anthropic import Anthropic
from pydantic import BaseModel
import json

class PropertyRecord(BaseModel):
    """One extracted property from one paper"""
    material_name: str
    material_formula: str | None
    property_name: str
    value_raw: str         # exactly as written in the paper
    value_numeric: float | None
    unit: str
    conditions: str        # temperature, pressure, synthesis method
    source_title: str
    source_doi: str
    confidence: str        # high | medium | low
    excerpt: str           # the exact text this was extracted from

class ExtractionResult(BaseModel):
    properties: list[PropertyRecord]
    papers_processed: int
    papers_failed: int

class ExtractionAgent:
    """
    Reads retrieved paper chunks and extracts structured property records.
    
    The hard part: making the LLM return consistent, typed JSON — not freeform text.
    Uses Pydantic for validation.
    """

    def __init__(self):
        self.llm = Anthropic()

    async def run(self, chunks: list[dict], state: dict) -> dict:
        state["agent_log"].append(
            f"ExtractionAgent: Processing {len(chunks)} chunks...")

        all_properties = []
        papers_seen = set()
        papers_failed = 0

        for chunk in chunks:
            try:
                properties = await self._extract_from_chunk(chunk)
                all_properties.extend(properties)
                papers_seen.add(chunk["metadata"].get("arxiv_id", "unknown"))
            except Exception as e:
                papers_failed += 1
                state["agent_log"].append(
                    f"ExtractionAgent: Failed on chunk "
                    f"{chunk.get('id', '?')}: {e}")

        # Normalise units across all extracted records
        normalised = self._normalise_units(all_properties)

        state["extracted_properties"] = normalised
        state["agent_log"].append(
            f"ExtractionAgent: Extracted {len(normalised)} property records "
            f"from {len(papers_seen)} papers ({papers_failed} failed)")

        return state

    async def _extract_from_chunk(
            self, chunk: dict) -> list[PropertyRecord]:
        """Extract property records from a single text chunk"""

        meta = chunk["metadata"]

        prompt = f"""You are extracting materials science data from a paper.

Paper: {meta.get('title', 'Unknown')}
Authors: {meta.get('authors', 'Unknown')}
DOI: {meta.get('doi', 'Unknown')}

Text excerpt:
{chunk['text']}

Extract ALL material property measurements mentioned in this text.
For each one, return a JSON object with these exact fields:
- material_name: name as written in text
- material_formula: chemical formula if mentioned (null if not)
- property_name: e.g. "ionic_conductivity", "bandgap", "formation_energy"
- value_raw: the value exactly as written (e.g. "3.4 × 10⁻³")
- value_numeric: as a float (e.g. 0.0034) — null if cannot parse
- unit: the unit (e.g. "S/cm", "eV", "GPa")
- conditions: measurement conditions (temperature, pressure, method)
- confidence: "high" if clearly stated, "medium" if inferred, "low" if uncertain
- excerpt: the exact sentence this came from

Return a JSON array. If no properties found, return [].
Return ONLY the JSON array, no other text."""

        response = self.llm.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return []

        records = []
        for item in data:
            try:
                record = PropertyRecord(
                    material_name=item.get("material_name", "Unknown"),
                    material_formula=item.get("material_formula"),
                    property_name=item.get("property_name", "unknown"),
                    value_raw=item.get("value_raw", ""),
                    value_numeric=item.get("value_numeric"),
                    unit=item.get("unit", ""),
                    conditions=item.get("conditions", ""),
                    source_title=meta.get("title", ""),
                    source_doi=meta.get("doi", ""),
                    confidence=item.get("confidence", "medium"),
                    excerpt=item.get("excerpt", "")[:300],
                )
                records.append(record)
            except Exception:
                pass  # Skip malformed records

        return records

    def _normalise_units(
            self, records: list[PropertyRecord]) -> list[PropertyRecord]:
        """Convert all conductivity values to S/cm, all energies to eV"""

        CONDUCTIVITY_TO_SCM = {
            "S/m": 0.01,
            "mS/cm": 0.001,
            "μS/cm": 1e-6,
            "S cm⁻¹": 1.0,
        }

        for record in records:
            if "conductivity" in record.property_name.lower():
                if record.unit in CONDUCTIVITY_TO_SCM and record.value_numeric:
                    record.value_numeric *= CONDUCTIVITY_TO_SCM[record.unit]
                    record.unit = "S/cm"

        return records
```

---

## Week 5 — ScreeningAgent + HypothesisAgent

These build directly on ExtractionAgent's output.
See `04_ARCHITECTURE.md` for full implementation.
The pattern is the same as ExtractionAgent — LLM with structured Pydantic output.

**ScreeningAgent does:**
- Apply researcher criteria to extracted property records
- Call KRR predictor for materials with missing properties
- Rank candidates by evidence strength
- Flag materials with only predicted (not measured) properties

**HypothesisAgent does:**
- Find patterns across top candidates
- Generate 3 hypotheses grounded in specific evidence
- Identify knowledge gaps (what we looked for but could not find)
- Suggest follow-up experiments for each hypothesis

---

## Week 6 — LangGraph Orchestration + FastAPI

```python
# agents/orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ResearchState(TypedDict):
    research_goal: str
    search_queries: list[str]
    retrieved_papers: list[dict]
    extracted_properties: list[dict]
    screened_candidates: list[dict]
    hypotheses: list[dict]
    knowledge_gaps: list[str]
    citations: list[dict]
    agent_log: Annotated[list[str], operator.add]
    retry_count: int
    warnings: list[str]

def build_graph(search_agent, extraction_agent,
                screening_agent, hypothesis_agent):
    """Build the LangGraph agent workflow"""

    workflow = StateGraph(ResearchState)

    workflow.add_node("search", search_agent.run)
    workflow.add_node("extract", extraction_agent.run)
    workflow.add_node("screen", screening_agent.run)
    workflow.add_node("hypothesise", hypothesis_agent.run)

    # Linear flow: search → extract → screen → hypothesise → END
    workflow.set_entry_point("search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "screen")
    workflow.add_edge("screen", "hypothesise")
    workflow.add_edge("hypothesise", END)

    return workflow.compile()
```

```python
# api/main.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio, json

app = FastAPI(title="MaterialsResearchAgent API")

@app.post("/api/research")
async def run_research(request: ResearchRequest):
    """
    Stream the research pipeline progress via SSE.
    Frontend receives events as agents complete each step.
    """

    async def event_generator():
        initial_state = {
            "research_goal": request.goal,
            "search_queries": [],
            "retrieved_papers": [],
            "extracted_properties": [],
            "screened_candidates": [],
            "hypotheses": [],
            "knowledge_gaps": [],
            "citations": [],
            "agent_log": [],
            "retry_count": 0,
            "warnings": [],
        }

        # Stream each agent's progress
        async for event in agent_graph.astream(initial_state):
            node_name = list(event.keys())[0]
            node_state = event[node_name]

            yield {
                "event": "agent_update",
                "data": json.dumps({
                    "node": node_name,
                    "log": node_state.get("agent_log", []),
                    "papers_found": len(node_state.get("retrieved_papers", [])),
                    "properties_extracted": len(node_state.get("extracted_properties", [])),
                    "candidates": len(node_state.get("screened_candidates", [])),
                })
            }

        # Final complete event
        yield {
            "event": "complete",
            "data": json.dumps({
                "candidates": final_state["screened_candidates"],
                "hypotheses": final_state["hypotheses"],
                "knowledge_gaps": final_state["knowledge_gaps"],
                "citations": final_state["citations"],
                "warnings": final_state["warnings"],
                "agent_log": final_state["agent_log"],
            })
        }

    return EventSourceResponse(event_generator())
```

---

## Week 7 — React Frontend

Key components (see `05_FRONTEND_GUIDE.md` for full code):
- `ResearchInput.tsx` — text area for research goal + submit button
- `AgentProgress.tsx` — real-time SSE stream display
- `CandidatesTable.tsx` — sortable table of screened materials
- `HypothesisCard.tsx` — hypothesis with evidence and experiment suggestion
- `CitationsPanel.tsx` — all papers used with clickable DOIs

---

## Week 8 — Evaluation + Deploy

```bash
# Evaluation
python evaluation/run_ragas_eval.py
# → Outputs RAGAS scores, comparison table, saves to evaluation/results.json

# AWS Deployment
# 1. Launch EC2 t2.micro (Amazon Linux 2023)
# 2. SSH in and install Python, Node, Nginx, PostgreSQL
# 3. Clone your repo
# 4. Set environment variables
# 5. Start with systemd services
# 6. Configure Nginx as reverse proxy
# 7. Get your public IP for the README
```

---

## Daily Progress Tracking

Use this checklist:

```
Week 1:
  [ ] Day 1: arxiv_fetcher.py — fetch 10 papers as test
  [ ] Day 2: pdf_parser.py — parse one paper, see sections
  [ ] Day 3: chunker.py — chunk one paper, check sizes
  [ ] Day 4: embedder.py + vector_store.py
  [ ] Day 5: ingest_papers.py — ingest 50 papers end to end
  Weekend:   ingest_papers.py --count 200 (let it run)

Week 2:
  [ ] Day 1-2: materials_project.py — test with GaN, Si, LiCoO2
  [ ] Day 3-5: krr_predictor.py — train and test
  Weekend:   train_krr.py with 500 samples

Week 3:
  [ ] Day 1-2: bm25_index.py (keyword search)
  [ ] Day 3-5: search_agent.py — test end to end
  Weekend:   Tune RELEVANCE_THRESHOLD, test with 5 research goals

Week 4:
  [ ] Day 1-3: extraction_agent.py — test on 10 papers
  [ ] Day 4-5: unit normalisation — verify all conductivities in S/cm

Week 5:
  [ ] Day 1-3: screening_agent.py
  [ ] Day 4-5: hypothesis_agent.py

Week 6:
  [ ] Day 1-2: orchestrator.py (LangGraph)
  [ ] Day 3-5: api/main.py (FastAPI + SSE streaming)

Week 7:
  [ ] Day 1-5: React frontend (all 5 components)

Week 8:
  [ ] Day 1-2: Evaluation dataset + RAGAS runner
  [ ] Day 3-5: AWS deployment + README polish
```
