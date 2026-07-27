"""Pinecone implementation intended for production deployments."""

from __future__ import annotations

from matsci_agent.domain.interfaces import IVectorStore


class PineconeVectorStore(IVectorStore):
    """Placeholder integration for Pinecone-backed vector search."""

    async def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, object]]) -> None:
        _ = ids, embeddings, metadatas
