"""
Embeddings & Vector Indexing Script for Verity.

Loads processed SEC 10-K text chunks from `backend/data/processed/chunks.json`,
generates vector embeddings locally using sentence-transformers (BAAI/bge-small-en-v1.5),
and indexes them into a Qdrant vector database collection ('verity_chunks').
"""

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = DATA_PROCESSED_DIR / "chunks.json"

# Vector DB & Model Config
COLLECTION_NAME = "verity_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 50


def main():
    print("=" * 70)
    print("Starting Verity Embeddings & Qdrant Vector Indexing")
    print("=" * 70)

    # 1. Load environment variables from .env
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()  # Fallback to current working directory .env if any

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("QDRANT_URL environment variable is not set in .env file.")
        sys.exit(1)

    logger.info(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=60.0)

    # 2. Check if chunks file exists and load chunks
    if not CHUNKS_FILE.exists():
        logger.error(f"Processed chunks file not found at {CHUNKS_FILE}")
        logger.error("Please run chunker.py first to generate chunks.json.")
        sys.exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        logger.warning("Loaded chunks.json is empty. No vectors to index.")
        return

    logger.info(f"Successfully loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # 3. Create or verify Qdrant collection
    existing_point_ids = set()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating new collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created with size {VECTOR_SIZE} and COSINE distance.")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists. Checking existing indexed points...")
        try:
            next_offset = None
            while True:
                scroll_res, next_offset = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,
                    offset=next_offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for pt in scroll_res:
                    existing_point_ids.add(str(pt.id))
                if next_offset is None or not scroll_res:
                    break
            logger.info(f"Found {len(existing_point_ids)} existing point IDs in collection '{COLLECTION_NAME}'.")
        except Exception as e:
            logger.warning(f"Could not retrieve existing point IDs: {e}")

    # Ensure payload index on company and fiscal_year (keyword type)
    for field_name in ["company", "fiscal_year"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created/verified payload index on '{field_name}' (keyword) in collection '{COLLECTION_NAME}'.")
        except Exception as e:
            logger.info(f"Payload index on '{field_name}': {e}")

    chunks_to_embed = [c for c in chunks if str(c.get("chunk_id")) not in existing_point_ids]
    if not chunks_to_embed:
        logger.info("All chunks are already indexed in Qdrant. No new vectors to generate.")
        collection_info = client.get_collection(COLLECTION_NAME)
        indexed_count = collection_info.points_count
        print("=" * 70)
        print("Indexing Status: Up to date!")
        print(f"Total vectors in Qdrant collection '{COLLECTION_NAME}': {indexed_count}")
        print("=" * 70)
        return

    logger.info(f"Found {len(chunks_to_embed)} new chunks to embed (out of {len(chunks)} total).")

    # 4. Load local SentenceTransformer model (offline/free)
    logger.info(f"Loading embedding model '{MODEL_NAME}' locally...")
    model = SentenceTransformer(MODEL_NAME)

    # 5. Generate embeddings for new chunks only
    logger.info(f"Generating embeddings for {len(chunks_to_embed)} new chunks...")
    texts = [chunk["text"] for chunk in chunks_to_embed]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # 6. Prepare Qdrant point objects with payload metadata
    points = []
    for chunk, embedding in zip(chunks_to_embed, embeddings):
        point_id = chunk.get("chunk_id") or str(uuid.uuid4())
        payload = {
            "company": chunk.get("company"),
            "fiscal_year": chunk.get("fiscal_year"),
            "item_section": chunk.get("item_section"),
            "item_title": chunk.get("item_title"),
            "text": chunk.get("text"),
            "source_path": chunk.get("source_path"),
        }
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload,
            )
        )

    # 7. Batch upsert into Qdrant
    total_points = len(points)
    logger.info(f"Upserting {total_points} new vectors into Qdrant collection '{COLLECTION_NAME}' in batches of {BATCH_SIZE}...")

    for i in range(0, total_points, BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                )
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    logger.warning(f"Upsert batch attempt {attempt + 1} failed: {exc}. Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.error(f"Upsert batch failed after {max_retries} attempts.")
                    raise exc

        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_points + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Progress: Upserted batch {batch_num}/{total_batches} ({min(i + BATCH_SIZE, total_points)}/{total_points} vectors)")

    # 8. Verify and print final vector count
    collection_info = client.get_collection(COLLECTION_NAME)
    indexed_count = collection_info.points_count
    print("=" * 70)
    print("Indexing Complete!")
    print(f"Total vectors indexed in Qdrant collection '{COLLECTION_NAME}': {indexed_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
