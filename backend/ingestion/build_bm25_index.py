"""
BM25 Keyword Index Builder for Verity.

Loads processed SEC 10-K text chunks from `backend/data/processed/chunks.json`,
tokenizes texts using whitespace + lowercase splitting, builds a BM25 index using rank_bm25 (BM25Okapi),
and pickles the fitted BM25 index alongside parallel chunk metadata into `backend/data/processed/bm25_index.pkl`.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = DATA_PROCESSED_DIR / "chunks.json"
BM25_FILE = DATA_PROCESSED_DIR / "bm25_index.pkl"


def tokenize(text: str) -> list:
    """
    Tokenizes text using simple lowercase whitespace splitting.

    NOTE: This simple whitespace + lowercase tokenization could be upgraded later to a more advanced
    tokenizer (e.g., regex word boundary extraction `re.findall(r'\\w+', text.lower())`, NLTK, or spaCy lemmatization).
    """
    return text.lower().split()


def main():
    print("=" * 70)
    print("Starting Verity BM25 Keyword Index Construction")
    print("=" * 70)

    if not CHUNKS_FILE.exists():
        logger.error(f"Processed chunks file not found at {CHUNKS_FILE}")
        logger.error("Please run chunker.py first to generate chunks.json.")
        sys.exit(1)

    logger.info(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        logger.warning("Chunks file is empty. Nothing to index.")
        return

    logger.info(f"Tokenizing {len(chunks)} text chunks...")
    corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]

    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "bm25": bm25,
        "chunks": chunks,
    }

    logger.info(f"Saving pickled BM25 index and chunk metadata to {BM25_FILE}...")
    with open(BM25_FILE, "wb") as f:
        pickle.dump(payload, f)

    print("=" * 70)
    print("BM25 Indexing Complete!")
    print(f"Total documents indexed: {len(chunks)}")
    print(f"Saved index file to: {BM25_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
