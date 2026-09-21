"""Client for fetching metadata and papers from arXiv."""

import logging
import time
from pathlib import Path
from typing import Iterator, Optional
import arxiv

from matsci_agent.core.config import settings
from matsci_agent.domain.schemas import PaperMetadata

logger=logging.getLogger(__name__)

class ArxivClient:
    """Production wrapper for arXiv API with rate limiting and local caching."""

    MATERIALS_QUERIES = [
        "solid state electrolyte ionic conductivity",
        "perovskite solar cell efficiency",
        "battery electrode materials machine learning",
        "semiconductor bandgap density functional theory",
        "high entropy alloy properties prediction",
    ]

    def __init__(self,download_dir:str="./papers"):
        self.download_dir=Path(download_dir)
        self.download_dir.mkdir(parents=True,exist_ok=True)

    def fetch_metadata(self,max_per_query: int=10)->Iterator[PaperMetadata]:
        """Queries arXiv across materials domains and yields typed metadata objects."""
        for query in self.MATERIALS_QUERIES:
            logger.info(f"Querying arXiv for: '{query}'")
            search=arxiv.Search(
                query=query,
                max_results=max_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            try:
                client = arxiv.Client()
                for result in client.results(search):
                    arxiv_id = result.entry_id.split("/")[-1]
                    yield PaperMetadata(
                        arxiv_id=arxiv_id,
                        title=result.title.replace("\n", " "),
                        authors=[author.name for author in result.authors],
                        abstract=result.summary.replace("\n", " "),
                        year=result.published.year,
                        doi=result.doi,
                    )
                    time.sleep(0.5)  # Respect arXiv API rate limits
            except Exception as exc:
                logger.error(f"Error fetching query '{query}': {exc}")

    def search_abstracts(self, query: str, max_results: int = 5) -> list[PaperMetadata]:
        """Real-time relevance search returning abstracts only (no PDF download)."""
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        return [self._to_metadata(r) for r in arxiv.Client().results(search)]

    @staticmethod
    def _to_metadata(result: "arxiv.Result") -> PaperMetadata:
        return PaperMetadata(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.replace("
", " "),
            authors=[author.name for author in result.authors],
            abstract=result.summary.replace("
", " "),
            year=result.published.year,
            doi=result.doi,
        )

    def download_pdf(self, paper: PaperMetadata) -> Optional[Path]:
        """Downloads PDF to disk if not already cached. Returns Path or None if failed."""
        pdf_path = self.download_dir / f"{paper.arxiv_id}.pdf"

        if pdf_path.exists():
            logger.debug(f"Cache hit for PDF: {paper.arxiv_id}")
            return pdf_path

        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(client.results(search))

            pdf_url = result.pdf_url
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"

            import httpx
            with httpx.Client(follow_redirects=True, verify=settings.arxiv_verify_ssl, timeout=30.0) as http_client:
                resp = http_client.get(pdf_url)
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)

            logger.info(f"Successfully downloaded PDF: {paper.arxiv_id}")
            return pdf_path
        except Exception as exc:
            logger.error(f"Failed to download PDF for {paper.arxiv_id}: {exc}")
            return None

    
