"""Copy the local ChromaDB corpus into Pinecone (one-off, when moving to production).

  pip install pinecone            # or: uv sync --extra pinecone
  PINECONE_API_KEY=... python scripts/migrate_to_pinecone.py

Re-running is safe: vectors are upserted by chunk id. Existing embeddings are copied as they are,
so nothing is re-embedded.
"""

from __future__ import annotations

import sys

sys.path.append("src")

from matsci_agent.core.config import settings  # noqa: E402
from matsci_agent.infrastructure.vector_store.chromadb import DEFAULT_PERSIST_DIR  # noqa: E402
from matsci_agent.infrastructure.vector_store.pinecone import PineconeVectorStore  # noqa: E402

BATCH = 200


def main() -> None:
    import chromadb

    if not settings.pinecone_api_key:
        sys.exit("PINECONE_API_KEY is not set.")
    source = chromadb.PersistentClient(path=DEFAULT_PERSIST_DIR).get_collection("materials_papers")
    target = PineconeVectorStore(
        settings.pinecone_api_key, settings.pinecone_index,
        cloud=settings.pinecone_cloud, region=settings.pinecone_region, embedder=_NoEmbedder(),
    )
    total, copied = source.count(), 0
    while copied < total:
        batch = source.get(limit=BATCH, offset=copied, include=["documents", "metadatas", "embeddings"])
        if not batch["ids"]:
            break
        target.upsert_records([
            {"id": i, "values": [float(x) for x in vec], "metadata": {**meta, "text": doc}}
            for i, vec, meta, doc in zip(batch["ids"], batch["embeddings"], batch["metadatas"], batch["documents"])
        ])
        copied += len(batch["ids"])
        print(f"  copied {copied}/{total}", flush=True)
    print(f"Done. Pinecone index '{settings.pinecone_index}' now holds {target.count()} vectors.")


class _NoEmbedder:
    """The migration copies stored embeddings, so loading the embedding model would be wasted work."""


if __name__ == "__main__":
    main()
