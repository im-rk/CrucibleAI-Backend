import logging
import os
import warnings

# Suppress HF Hub unauthenticated warnings and symlink warnings on Windows
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string — used for query embedding."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    