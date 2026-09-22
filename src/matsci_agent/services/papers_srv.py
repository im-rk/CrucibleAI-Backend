"""Read-only catalogue of the papers in the vector store."""

from __future__ import annotations

import threading


class ListingUnsupported(RuntimeError):
    """The vector store cannot enumerate its documents (e.g. Pinecone)."""


class PaperCatalog:
    """Groups stored chunks by paper; rebuilds itself when the corpus size changes."""

    def __init__(self, store) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._papers: dict[str, dict] = {}
        self._built_count = -1
        self.supported = True

    def _ensure(self) -> None:
        count = self._store.count()
        with self._lock:
            if count == self._built_count:
                return
            papers: dict[str, dict] = {}
            try:
                for doc in self._store.iter_documents():
                    meta = doc["metadata"]
                    key = meta.get("arxiv_id") or doc["id"]
                    paper = papers.setdefault(key, {
                        "arxiv_id": key, "title": meta.get("title", ""), "authors": meta.get("authors", ""),
                        "year": meta.get("year"), "chunk_count": 0, "sections": set(), "snippet": "",
                        "_first": None,
                    })
                    paper["chunk_count"] += 1
                    paper["sections"].add(meta.get("section", ""))
                    index = meta.get("chunk_index", 0)
                    if paper["_first"] is None or index < paper["_first"]:
                        paper["_first"], paper["snippet"] = index, doc["text"][:240]
            except NotImplementedError:
                self.supported = False
            self._papers, self._built_count = papers, count

    @staticmethod
    def _public(paper: dict) -> dict:
        return {**{k: v for k, v in paper.items() if k != "_first"}, "sections": sorted(paper["sections"])}

    def _require_listing(self) -> None:
        self._ensure()
        if not self.supported:
            raise ListingUnsupported("This vector store cannot list its papers.")

    def list(self, query: str | None = None, limit: int = 50, offset: int = 0) -> dict:
        self._require_listing()
        needle = (query or "").lower()
        rows = [
            p for p in self._papers.values()
            if not needle or needle in p["title"].lower() or needle in p["arxiv_id"].lower() or needle in p["authors"].lower()
        ]
        rows.sort(key=lambda p: (-(p["year"] or 0), p["arxiv_id"]))
        return {"total": len(rows), "papers": [self._public(p) for p in rows[offset: offset + limit]]}

    def get(self, arxiv_id: str) -> dict | None:
        self._require_listing()
        paper = self._papers.get(arxiv_id)
        return self._public(paper) if paper else None

    def stats(self) -> dict:
        """Chunk count always; paper count only when the store can enumerate its documents."""
        self._ensure()
        return {"total_papers": len(self._papers) if self.supported else None, "total_chunks": self._built_count}
