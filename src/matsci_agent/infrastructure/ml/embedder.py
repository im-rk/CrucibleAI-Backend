import logging
import os
import warnings
import math
import hashlib

# Suppress HF Hub unauthenticated warnings and symlink warnings on Windows
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = logging.getLogger(__name__)


def _generate_fallback_vector(text: str, dim: int = 384) -> list[float]:
    """Generate a deterministic normalized vector for fallback embedding."""
    vector = []
    for i in range(dim):
        h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
        val = (int.from_bytes(h[:4], "big") / 4294967295.0) * 2.0 - 1.0
        vector.append(val)
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class MaterialsEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(f"Could not load SentenceTransformer ({exc}). Using deterministic fallback embedder.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.model is not None:
            try:
                return self.model.encode(
                    texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                ).tolist()
            except Exception as exc:
                logger.warning(f"SentenceTransformer encoding failed: {exc}")
        return [_generate_fallback_vector(t) for t in texts]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string — used for query embedding."""
        if self.model is not None:
            try:
                return self.model.encode([text], normalize_embeddings=True)[0].tolist()
            except Exception as exc:
                logger.warning(f"SentenceTransformer single encoding failed: {exc}")
        return _generate_fallback_vector(text)


    