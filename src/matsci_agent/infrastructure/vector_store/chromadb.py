"""Local ChromaDB implementation used during development."""

from __future__ import annotations

from matsci_agent.domain.interfaces import IVectorStore


class ChromaVectorStore(IVectorStore):
    """Thin wrapper around a Chroma collection."""

    async def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, object]]) -> None:
        _ = ids, embeddings, metadatas
