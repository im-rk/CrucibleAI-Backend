"""Abstract interfaces for external systems used by the domain layer."""

# from __future__ import annotations

# from abc import ABC, abstractmethod


# class IVectorStore(ABC):
#     """Contract for vector store implementations."""

#     @abstractmethod
#     async def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict[str, object]]) -> None:
#         raise NotImplementedError


# class IDatabase(ABC):
#     """Contract for persistence implementations."""

#     @abstractmethod
#     async def healthcheck(self) -> bool:
#         raise NotImplementedError
