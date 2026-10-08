"""Build the materials-science corpus from arXiv abstracts.

Abstracts rather than PDFs: they are quick to fetch, carry exact metadata (title, authors, year, DOI)
and state the headline numbers, so every paper becomes one clean, citable chunk. Full-text ingestion
(`scripts/ingest_papers.py`) can be layered on top for selected papers.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import arxiv

from matsci_agent.corpus.topics import TOPICS, build_query
from matsci_agent.domain.schemas import PaperChunk, PaperMetadata

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_chunk(result: "arxiv.Result") -> PaperChunk:
    """One chunk per paper: 'Title. Abstract', with full metadata attached."""
    arxiv_id = result.entry_id.split("/")[-1]
    title, abstract = _clean(result.title), _clean(result.summary)
    meta = PaperMetadata(
        arxiv_id=arxiv_id, title=title, authors=[a.name for a in result.authors],
        abstract=abstract, year=result.published.year, doi=result.doi,
    )
    text = f"{title}. {abstract}"
    return PaperChunk(
        chunk_id=f"{arxiv_id}_abs", arxiv_id=arxiv_id, text=text, section_name="abstract",
        chunk_index=0, char_count=len(text), metadata=meta,
    )


def fetch_topic(topic: str, per_topic: int, client: "arxiv.Client | None" = None) -> list[PaperChunk]:
    client = client or arxiv.Client(page_size=min(per_topic, 100), delay_seconds=3.0, num_retries=4)
    search = arxiv.Search(query=build_query(topic), max_results=per_topic, sort_by=arxiv.SortCriterion.Relevance)
    return [to_chunk(r) for r in client.results(search)]


def _slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def _load_cached(cache_dir: Path | None, topic: str, per_topic: int) -> list[PaperChunk] | None:
    path = cache_dir / f"{_slug(topic)}.json" if cache_dir else None
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("per_topic", 0) >= per_topic:
            return [PaperChunk.model_validate(c) for c in data["chunks"][:per_topic]]
    except (ValueError, KeyError):
        logger.warning(f"Ignoring unreadable cache for '{topic}'")
    return None


def _save_cached(cache_dir: Path | None, topic: str, per_topic: int, chunks: list[PaperChunk]) -> None:
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"per_topic": per_topic, "chunks": [c.model_dump(mode="json") for c in chunks]}
        (cache_dir / f"{_slug(topic)}.json").write_text(json.dumps(payload), encoding="utf-8")


def _fetch_with_backoff(fetch, topic: str, per_topic: int, retries: int, pause_s: float) -> list[PaperChunk] | None:
    """arXiv answers a burst of queries with HTTP 429; wait longer after each failed attempt."""
    for attempt in range(retries):
        try:
            return fetch(topic, per_topic)
        except Exception as exc:  # noqa: BLE001 - one failing topic must not lose the others
            wait = max(pause_s, 1.0) * 6 * (attempt + 1) if pause_s else 0
            logger.warning(f"Topic '{topic}' failed (attempt {attempt + 1}/{retries}): {exc}; waiting {wait:.0f}s")
            time.sleep(wait)
    logger.error(f"Topic '{topic}' skipped after {retries} attempts")
    return None


def build_corpus(
    store,
    *,
    per_topic: int = 40,
    topics: list[str] | None = None,
    min_abstract_chars: int = 400,
    fetch: Callable[[str, int], list[PaperChunk]] = fetch_topic,
    pause_s: float = 5.0,
    retries: int = 3,
    cache_dir: Path | None = None,
) -> dict:
    """Fetch every topic, de-duplicate across topics, index into `store`, and return the manifest."""
    papers: dict[str, dict] = {}
    chunks: dict[str, PaperChunk] = {}
    for topic in topics or TOPICS:
        found = _load_cached(cache_dir, topic, per_topic)
        if found is None:
            found = _fetch_with_backoff(fetch, topic, per_topic, retries, pause_s)
            if found is None:
                continue
            _save_cached(cache_dir, topic, per_topic, found)
            time.sleep(pause_s)  # arXiv rate-limits bursts of queries
        new = 0
        for chunk in found:
            if len(chunk.metadata.abstract) < min_abstract_chars:
                continue  # too short to carry a result
            entry = papers.setdefault(chunk.arxiv_id, {
                "arxiv_id": chunk.arxiv_id, "title": chunk.metadata.title, "year": chunk.metadata.year,
                "doi": chunk.metadata.doi, "topics": [],
            })
            if topic not in entry["topics"]:
                entry["topics"].append(topic)
            if chunk.arxiv_id not in chunks:
                chunks[chunk.arxiv_id] = chunk
                new += 1
        logger.info(f"{topic}: {len(found)} found, {new} new ({len(chunks)} papers so far)")

    batch = list(chunks.values())
    for i in range(0, len(batch), 128):
        store.add_chunks(batch[i : i + 128])
    return {
        "version": MANIFEST_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "arXiv API, category cond-mat.mtrl-sci, abstracts",
        "per_topic": per_topic,
        "topics": topics or TOPICS,
        "paper_count": len(papers),
        "papers": sorted(papers.values(), key=lambda p: p["arxiv_id"]),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
