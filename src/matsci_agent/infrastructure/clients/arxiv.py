"""Client for fetching metadata and papers from arXiv."""

import logging
import time
from pathlib import Path
from typing import Iterator, Optional
import arxiv

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
            logger.info(f"Querying arXiv for: '{query}")
            search=arxiv.Search(
                query=query,
                max_result=max_per_query,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            try:
                for result in search.results():
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

    def download_pdf(self, paper: PaperMetadata) -> Optional[Path]:
        """Downloads PDF to disk if not already cached. Returns Path or None if failed."""
        pdf_path = self.download_dir / f"{paper.arxiv_id}.pdf"

        if pdf_path.exists():
            logger.debug(f"Cache hit for PDF: {paper.arxiv_id}")
            return pdf_path

        try:
            search = arxiv.Search(id_list=[paper.arxiv_id])
            result = next(search.results())
            result.download_pdf(dirpath=str(self.download_dir), filename=f"{paper.arxiv_id}.pdf")
            logger.info(f"Successfully downloaded PDF: {paper.arxiv_id}")
            return pdf_path
        except Exception as exc:
            logger.error(f"Failed to download PDF for {paper.arxiv_id}: {exc}")
            return None
    
