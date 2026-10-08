"""Build the materials-science corpus (arXiv abstracts) into the vector store.

    python scripts/build_corpus.py --reset            # replace the corpus with a fresh build
    python scripts/build_corpus.py --per-topic 60     # larger corpus
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.append("src")

from matsci_agent.corpus.builder import build_corpus, write_manifest  # noqa: E402
from matsci_agent.infrastructure.vector_store.chromadb import DEFAULT_PERSIST_DIR, ChromaVectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_corpus")

MANIFEST = Path("data/corpus_manifest.json")
CACHE = Path("data/cache/arxiv")  # fetched abstracts, so a rebuild never re-hits arXiv rate limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-topic", type=int, default=40, help="papers requested per topic (default 40)")
    parser.add_argument("--reset", action="store_true", help="delete the existing collection first")
    args = parser.parse_args()

    if args.reset:
        # Remove the whole directory: deleting and re-creating a collection in place can leave the
        # embedded Chroma index in a state it cannot recover from.
        persist = Path(DEFAULT_PERSIST_DIR)
        logger.info(f"Resetting the vector store at {persist}")
        shutil.rmtree(persist, ignore_errors=True)
    store = ChromaVectorStore()
    manifest = build_corpus(store, per_topic=args.per_topic, cache_dir=CACHE)
    write_manifest(manifest, MANIFEST)
    logger.info(f"Done: {manifest['paper_count']} papers, {store.count()} chunks indexed; manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
