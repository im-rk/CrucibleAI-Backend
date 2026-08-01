"""CLI entry point for ingesting arXiv papers into the vector store."""

import argparse
import logging
import sys
sys.path.append("src")

from matsci_agent.infrastructure.clients.arxiv import ArxivClient
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

    logger.info(f"Complete! Papers: {processed_papers} | Chunks: {total_chunks}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Number of papers")
    args = parser.parse_args()
    run_ingestion(args.count)
        