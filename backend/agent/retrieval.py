"""
Hybrid Dense (Qdrant) + Sparse (BM25) Retrieval Engine for Verity.

Combines semantic vector search (Qdrant with BAAI/bge-small-en-v1.5 embeddings) and
keyword search (BM25Okapi) using Reciprocal Rank Fusion (RRF, k=60).
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths & Environment Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "verity_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BM25_INDEX_FILE = BASE_DIR / "data" / "processed" / "bm25_index.pkl"

# Module-level initialization (Loaded ONCE, not per-call)
logger.info(f"Loading embedding model '{MODEL_NAME}' at module level...")
embedding_model = SentenceTransformer(MODEL_NAME)

logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
qdrant_client = None
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")

bm25_index = None
bm25_chunks = []
if BM25_INDEX_FILE.exists():
    try:
        logger.info(f"Loading BM25 index from {BM25_INDEX_FILE}...")
        with open(BM25_INDEX_FILE, "rb") as f:
            bm25_data = pickle.load(f)
            bm25_index = bm25_data.get("bm25")
            bm25_chunks = bm25_data.get("chunks", [])
        logger.info(f"Loaded BM25 index with {len(bm25_chunks)} chunks.")
    except Exception as e:
        logger.warning(f"Could not load BM25 index: {e}")
else:
    logger.warning(f"BM25 index file not found at {BM25_INDEX_FILE}")


def build_qdrant_filter(filters: Optional[Dict]) -> Optional[Filter]:
    """Constructs a Qdrant Filter from a dictionary of metadata constraints."""
    if not filters:
        return None
    must_conditions = []
    for key, value in filters.items():
        if value is not None:
            must_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
    if must_conditions:
        return Filter(must=must_conditions)
    return None


def search_qdrant(query: str, filters: Optional[Dict] = None, limit: int = 20) -> List[Dict]:
    """Searches Qdrant vector database for top nearest neighbors."""
    if not qdrant_client:
        logger.error("Qdrant client is not available.")
        return []

    query_vector = embedding_model.encode(query).tolist()
    qdrant_filter = build_qdrant_filter(filters)

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
    )
    results = response.points

    dense_hits = []
    for hit in results:
        payload = hit.payload or {}
        dense_hits.append({
            "id": str(hit.id),
            "text": payload.get("text", ""),
            "company": payload.get("company"),
            "fiscal_year": payload.get("fiscal_year"),
            "item_section": payload.get("item_section"),
            "item_title": payload.get("item_title"),
            "source_path": payload.get("source_path"),
            "raw_score": float(hit.score),
        })
    return dense_hits


def search_bm25(query: str, filters: Optional[Dict] = None, limit: int = 20) -> List[Dict]:
    """Searches local BM25 keyword index for top scoring chunks."""
    if not bm25_index or not bm25_chunks:
        logger.error("BM25 index is not loaded.")
        return []

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

    scored_chunks = []
    for idx, (chunk, score) in enumerate(zip(bm25_chunks, scores)):
        # Apply metadata filtering if specified
        if filters:
            match = True
            for k, v in filters.items():
                if v is not None and chunk.get(k) != v:
                    match = False
                    break
            if not match:
                continue

        chunk_id = chunk.get("chunk_id") or str(idx)
        scored_chunks.append({
            "id": str(chunk_id),
            "text": chunk.get("text", ""),
            "company": chunk.get("company"),
            "fiscal_year": chunk.get("fiscal_year"),
            "item_section": chunk.get("item_section"),
            "item_title": chunk.get("item_title"),
            "source_path": chunk.get("source_path"),
            "raw_score": float(score),
        })

    scored_chunks.sort(key=lambda x: x["raw_score"], reverse=True)
    return scored_chunks[:limit]


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    top_k: int = 8
) -> List[Dict]:
    """
    Merges dense and sparse search results using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_{m in {dense, sparse}} (1 / (k + rank_m(d)))
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    # Rank 1: Dense (Qdrant)
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Rank 2: Sparse (BM25)
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort doc_ids by RRF score descending
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)

    fused_results = []
    for doc_id in sorted_doc_ids[:top_k]:
        doc_item = doc_map[doc_id].copy()
        rrf_score = rrf_scores[doc_id]
        doc_item["score"] = rrf_score
        doc_item["metadata"] = {
            "company": doc_item.get("company"),
            "fiscal_year": doc_item.get("fiscal_year"),
            "item_section": doc_item.get("item_section"),
            "item_title": doc_item.get("item_title"),
            "source_path": doc_item.get("source_path"),
        }
        fused_results.append(doc_item)

    return fused_results


def retrieve(query: str, top_k: int = 8, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Retrieves top_k text chunks using hybrid Qdrant (dense) + BM25 (sparse) retrieval
    fused with Reciprocal Rank Fusion (RRF).
    """
    dense_results = search_qdrant(query=query, filters=filters, limit=20)
    sparse_results = search_bm25(query=query, filters=filters, limit=20)

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        top_k=top_k
    )
    return fused_results


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Verity Hybrid Retrieval Engine (RRF)")
    print("=" * 70)

    test_query = "What was Apple's revenue in the most recent fiscal year?"
    print(f"Query: '{test_query}'\n")

    results = retrieve(query=test_query, top_k=5)

    print(f"Top {len(results)} Hybrid Retrieval Results (RRF fused):")
    print("-" * 70)

    for i, res in enumerate(results, start=1):
        print(f"Result #{i} | RRF Score: {res['score']:.4f}")
        print(f"  Company:     {res['company']} | Fiscal Year: {res['fiscal_year']}")
        print(f"  Section:     {res['item_section']} ({res['item_title']})")
        print(f"  Source Path: {res['source_path']}")
        snippet = res['text'][:220].replace("\n", " ") + "..." if len(res['text']) > 220 else res['text'].replace("\n", " ")
        print(f"  Text:        {snippet}")
        print("-" * 70)
