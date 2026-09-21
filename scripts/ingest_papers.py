"""CLI entry point for ingesting arXiv papers into the vector store."""

import argparse
import asyncio
import logging
import sys
sys.path.append("src")

from matsci_agent.core.config import settings
from matsci_agent.infrastructure.clients.arxiv import ArxivClient
from matsci_agent.infrastructure.db.session import create_session_factory
from matsci_agent.services.persistence import PersistenceService
from matsci_agent.services.pipeline_srv import PaperPipelineService
from matsci_agent.infrastructure.vector_store.chromadb import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_ingestion(target_count: int) -> None:
    arxiv_client = ArxivClient()
    pipeline_service = PaperPipelineService()
    vector_store = ChromaVectorStore()

    logger.info(f"Current corpus size: {vector_store.count()} chunks")
    
    processed_papers = 0
    total_chunks = 0
    ingested: list[tuple[dict, int, str]] = []

    for paper_meta in arxiv_client.fetch_metadata(max_per_query=target_count // 5 + 1):
        if processed_papers >= target_count:
            break

        pdf_path = arxiv_client.download_pdf(paper_meta)
        if not pdf_path:
            continue

        chunks = pipeline_service.process_pdf(pdf_path, paper_meta)
        if not chunks:
            continue

        vector_store.add_chunks(chunks)
        processed_papers += 1
        total_chunks += len(chunks)
        ingested.append((paper_meta.model_dump(mode="json"), len(chunks), str(pdf_path)))

    logger.info(f"Complete! Papers: {processed_papers} | Chunks: {total_chunks}")
    asyncio.run(record_papers(ingested))


async def record_papers(ingested: list[tuple[dict, int, str]]) -> None:
    """Best-effort: list the ingested papers in PostgreSQL (skipped if it is unreachable)."""
    if not settings.persistence_enabled or not ingested:
        return
    store = PersistenceService(create_session_factory(settings.database_url))
    for meta, chunk_count, pdf_path in ingested:
        await store.record_paper(meta, chunk_count, pdf_path)
    logger.info(f"Recorded {len(ingested)} papers in the database" if store.available else "Database unreachable; papers not recorded")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Number of papers")
    args = parser.parse_args()
    run_ingestion(args.count)
        