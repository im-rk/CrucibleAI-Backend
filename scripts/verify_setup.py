"""Check that everything the backend needs is in place, and say what is optional.

  python scripts/verify_setup.py

Exit code is non-zero only if a REQUIRED component fails. Missing API keys are reported as
"skipped": the pipeline still runs without them, in a labelled degraded mode.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.append("src")

from matsci_agent.core.config import settings  # noqa: E402

OK, SKIP, FAIL = "[ok]  ", "[skip]", "[FAIL]"
results: list[tuple[str, str, str, bool]] = []  # (mark, name, detail, required)


def record(mark: str, name: str, detail: str, required: bool = False) -> None:
    results.append((mark, name, detail, required))
    print(f"{mark} {name}: {detail}", flush=True)


def check_chroma() -> None:
    try:
        import chromadb

        from matsci_agent.infrastructure.vector_store.chromadb import DEFAULT_PERSIST_DIR

        collection = chromadb.PersistentClient(path=DEFAULT_PERSIST_DIR).get_or_create_collection("materials_papers")
        n = collection.count()
        if n:
            record(OK, "Vector store (ChromaDB)", f"{n} chunks indexed", True)
        else:
            record(FAIL, "Vector store (ChromaDB)", "empty; run: python scripts/ingest_papers.py --count 200", True)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Vector store (ChromaDB)", str(exc), True)


def check_embedder() -> None:
    try:
        from matsci_agent.infrastructure.ml.embedder import MaterialsEmbedder

        embedder = MaterialsEmbedder()
        if embedder.model is None:
            record(FAIL, "Embedding model", "all-MiniLM-L6-v2 could not be loaded (the hash fallback makes search meaningless)", True)
        else:
            record(OK, "Embedding model", f"all-MiniLM-L6-v2 (dim={len(embedder.embed_single('test'))})", True)
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Embedding model", str(exc), True)


async def check_anthropic() -> None:
    if not settings.anthropic_api_key:
        record(SKIP, "Anthropic API", "ANTHROPIC_API_KEY not set: the pipeline runs in heuristic mode")
        return
    try:
        from pydantic import BaseModel

        from matsci_agent.infrastructure.llm.anthropic_client import LLMClient

        class Ping(BaseModel):
            ok: bool

        result = await LLMClient(settings.anthropic_api_key, settings.anthropic_model).structured(
            "Reply with ok=true.", Ping, max_tokens=50
        )
        record(OK, "Anthropic API", f"{settings.anthropic_model} responded (ok={result.ok})")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Anthropic API", str(exc))


def check_materials_project() -> None:
    if not settings.materials_project_api_key:
        record(SKIP, "Materials Project API", "MATERIALS_PROJECT_API_KEY not set: only the offline reference table is used")
        return
    try:
        from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient

        props = MaterialsProjectClient().get_properties("GaN")
        gap = next((p.value for p in props if p.property_name == "bandgap"), None)
        if props:
            record(OK, "Materials Project API", f"GaN band gap {gap} eV (DFT)")
        else:
            record(FAIL, "Materials Project API", "no data returned; check the key")
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "Materials Project API", str(exc))


async def check_database() -> None:
    from matsci_agent.infrastructure.db.session import create_session_factory
    from matsci_agent.services.persistence import PersistenceService

    if not settings.persistence_enabled:
        record(SKIP, "PostgreSQL", "disabled (PERSISTENCE_ENABLED=false)")
        return
    state = await PersistenceService(create_session_factory(settings.database_url)).ping()
    if state == "connected":
        record(OK, "PostgreSQL", f"connected ({settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db})")
    else:
        record(SKIP, "PostgreSQL", "unreachable: runs still work but are not saved (then: alembic upgrade head)")


def check_arxiv() -> None:
    try:
        from matsci_agent.infrastructure.clients.arxiv import ArxivClient

        papers = ArxivClient().search_abstracts("materials science", 1)
        record(OK if papers else SKIP, "arXiv API", f"found: {papers[0].title[:50]}..." if papers else "no results")
    except Exception as exc:  # noqa: BLE001
        record(SKIP, "arXiv API", f"unreachable ({type(exc).__name__}): the live-search fallback and ingestion need it")


async def main() -> int:
    check_chroma()
    check_embedder()
    await check_anthropic()
    check_materials_project()
    await check_database()
    check_arxiv()

    failed = [name for mark, name, _, required in results if mark == FAIL and required]
    print("\n" + "=" * 60)
    if failed:
        print("SETUP INCOMPLETE. Fix: " + ", ".join(failed))
        return 1
    optional = [name for mark, name, _, _ in results if mark in (SKIP, FAIL)]
    print("Ready." + (f" Optional items not available: {', '.join(optional)}." if optional else " All checks passed."))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
