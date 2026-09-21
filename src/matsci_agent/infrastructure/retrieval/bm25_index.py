"""BM25 keyword index over the chunks stored in the vector store."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator
from typing import Protocol

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; keeps formulas like 'li7la3zr2o12' as one term."""
    return _TOKEN_RE.findall(text.lower())


class DocumentSource(Protocol):
    def count(self) -> int: ...

    def iter_documents(self) -> Iterator[dict]: ...


class BM25Index:
    """Lazily built BM25 index that rebuilds itself when the corpus size changes."""

    def __init__(self, source: DocumentSource) -> None:
        self._source = source
        self._lock = threading.Lock()
        self._docs: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._built_count = -1

    def _ensure_built(self) -> None:
        count = self._source.count()
        with self._lock:
            if count == self._built_count:
                return
            self._docs = list(self._source.iter_documents())
            corpus = [tokenize(d["text"]) for d in self._docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._built_count = count
            logger.info(f"BM25 index built over {len(self._docs)} chunks.")

    def search(self, query: str, n_results: int = 15) -> list[dict]:
        """Return chunks ranked by BM25; chunks sharing no term with the query are dropped."""
        self._ensure_built()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [
            {**self._docs[i], "bm25_score": float(scores[i])}
            for i in ranked[:n_results]
            if scores[i] > 0
        ]
