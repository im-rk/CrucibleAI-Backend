"""Optional cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs with a cross-encoder; degrades to a no-op if unavailable."""

    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._load_failed = False

    def _load(self):
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                self._load_failed = True
                logger.warning(f"Reranker '{self.model_name}' unavailable ({exc}); skipping rerank.")
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks or not self.enabled:
            return chunks[:top_k]
        model = self._load()
        if model is None:
            return chunks[:top_k]
        scores = model.predict([(query, c["text"]) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [{**chunk, "rerank_score": float(score)} for chunk, score in ranked[:top_k]]
