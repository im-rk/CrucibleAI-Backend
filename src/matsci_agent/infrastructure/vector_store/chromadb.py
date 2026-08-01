"""Local ChromaDB implementation used during development."""

import logging
import chromadb
from matsci_agent.domain.schemas import PaperChunk
from matsci_agent.infrastructure.ml.embedder import MaterialsEmbedder

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    COLLECTION_NAME = "materials_papers"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.embedder = MaterialsEmbedder()

    def add_chunks(self, chunks: list[PaperChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "arxiv_id": c.arxiv_id,
                "section": c.section_name,
                "chunk_index": c.chunk_index,
                "title": c.metadata.title,
                "authors": ", ".join(c.metadata.authors[:3]),
                "year": c.metadata.year,
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB.")

    def count(self) -> int:
        return self.collection.count()