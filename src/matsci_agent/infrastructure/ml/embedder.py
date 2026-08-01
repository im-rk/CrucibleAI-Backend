"""Embedding generation for paper chunks and retrieved records."""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MaterialsEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
    