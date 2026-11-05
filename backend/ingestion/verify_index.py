"""
Verification Script for Verity Ingestion and Qdrant Index.

Prints:
1. Total chunks in backend/data/processed/chunks.json
2. Total points in Qdrant collection ('verity_chunks')
3. Number of duplicate texts in Qdrant
4. Chunk counts per company/year/section
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = DATA_PROCESSED_DIR / "chunks.json"
COLLECTION_NAME = "verity_chunks"


def main():
    print("=" * 70)
    print("VERITY INGESTION & QDRANT INDEX VERIFICATION")
    print("=" * 70)

    # 1. Total chunks in chunks.json
    total_chunks_json = 0
    if CHUNKS_FILE.exists():
        try:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                total_chunks_json = len(chunks_data)
        except Exception as e:
            logger.error(f"Failed to load chunks.json: {e}")
    else:
        logger.warning(f"chunks.json not found at {CHUNKS_FILE}")

    # 2. Connect to Qdrant
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("QDRANT_URL environment variable is not set in .env file.")
        print(f"Total chunks in chunks.json: {total_chunks_json}")
        print("Total points in Qdrant:      [QDRANT_URL not configured]")
        return

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=30.0)

        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Total chunks in chunks.json:         {total_chunks_json}")
            print(f"Total points in Qdrant:              0 (Collection '{COLLECTION_NAME}' does not exist)")
            print("Number of duplicate texts in Qdrant: 0")
            print("\nChunk counts per company/year/section:")
            print("  [Collection does not exist]")
            print("=" * 70)
            return

        collection_info = client.get_collection(COLLECTION_NAME)
        total_qdrant_points = collection_info.points_count

        # 3. Page through all points to examine payloads
        logger.info(f"Scrolling points from Qdrant collection '{COLLECTION_NAME}'...")
        all_points = []
        next_offset = None
        while True:
            scroll_res, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(scroll_res)
            if next_offset is None or not scroll_res:
                break

        # 4. Check for duplicate texts in Qdrant
        text_counter = Counter()
        section_counter = Counter()

        for pt in all_points:
            payload = pt.payload or {}
            text = (payload.get("text") or "").strip()
            if text:
                text_counter[text] += 1

            company = payload.get("company", "Unknown")
            year = str(payload.get("fiscal_year", "Unknown"))
            section = payload.get("item_section", "Unknown")
            section_counter[(company, year, section)] += 1

        # Number of redundant duplicate instances (total occurrences beyond the first)
        duplicate_text_count = sum(count - 1 for count in text_counter.values() if count > 1)
        unique_duplicate_texts = sum(1 for count in text_counter.values() if count > 1)

        print(f"Total chunks in chunks.json:         {total_chunks_json}")
        print(f"Total points in Qdrant:              {total_qdrant_points} (retrieved {len(all_points)} via scroll)")
        print(f"Number of duplicate texts in Qdrant: {duplicate_text_count} (across {unique_duplicate_texts} duplicate texts)")
        print("-" * 70)
        print("Chunk counts per company/year/section (in Qdrant):")

        for (company, year, section), count in sorted(section_counter.items()):
            print(f"  - {company:<10} | FY {year} | {section:<8} : {count:>4} chunks")

    except Exception as e:
        logger.warning(f"Could not connect or query Qdrant ({e}). Falling back to chunks.json stats.")
        print(f"Total chunks in chunks.json:         {total_chunks_json}")
        print(f"Total points in Qdrant:              [Unable to connect to Qdrant: {e}]")

        if CHUNKS_FILE.exists():
            text_counter = Counter()
            section_counter = Counter()
            for chunk in chunks_data:
                text = (chunk.get("text") or "").strip()
                if text:
                    text_counter[text] += 1
                company = chunk.get("company", "Unknown")
                year = str(chunk.get("fiscal_year", "Unknown"))
                section = chunk.get("item_section", "Unknown")
                section_counter[(company, year, section)] += 1

            duplicate_text_count = sum(count - 1 for count in text_counter.values() if count > 1)
            unique_duplicate_texts = sum(1 for count in text_counter.values() if count > 1)
            print(f"Number of duplicate texts in chunks: {duplicate_text_count} (across {unique_duplicate_texts} duplicate texts)")
            print("-" * 70)
            print("Chunk counts per company/year/section (from chunks.json):")
            for (company, year, section), count in sorted(section_counter.items()):
                print(f"  - {company:<10} | FY {year} | {section:<8} : {count:>4} chunks")

    print("=" * 70)


if __name__ == "__main__":
    main()
