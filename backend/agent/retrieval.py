"""
Hybrid Dense (Qdrant) + Sparse (BM25) Retrieval Engine for Verity.

Combines semantic vector search (Qdrant with BAAI/bge-small-en-v1.5 embeddings) and
keyword search (BM25Okapi) using Reciprocal Rank Fusion (RRF, k=60).

Production Features:
- Lazy-loads embedding model on first call (prevents PyTorch memory overhead at boot).
- Swappable embedding backend via EMBEDDING_BACKEND env var:
  - "fastembed" (default for Render free tier: ONNX-based, uses ~40-60MB RAM instead of ~400MB PyTorch).
  - "local" (uses SentenceTransformer / PyTorch).
- Both backends produce identical 384-dimensional vectors using BAAI/bge-small-en-v1.5.
- Loads BM25 index inputs from committed backend/artifacts/ (bm25_inputs.json or bm25_inputs.npz),
  rebuilding BM25Okapi in ~0.5s at startup without needing insecure pickle files.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi

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

# Artifact and data paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BM25_INPUTS_JSON = ARTIFACTS_DIR / "bm25_inputs.json"
BM25_INPUTS_NPZ = ARTIFACTS_DIR / "bm25_inputs.npz"
BM25_LEGACY_PKL = ARTIFACTS_DIR / "bm25_index.pkl"
BM25_DATA_PKL = BASE_DIR / "data" / "processed" / "bm25_index.pkl"

# -----------------------------------------------------------------------------
# Qdrant Client Initialization
# -----------------------------------------------------------------------------
logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
qdrant_client = None
if QDRANT_URL:
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            prefer_grpc=False,
            timeout=2.0,
            check_compatibility=False,
        )
    except Exception as e:
        logger.warning(f"Could not initialize Qdrant client: {e}")


_qdrant_status_cache: Dict[str, Any] = {"connected": False, "last_checked": 0.0}


def is_qdrant_connected() -> bool:
    """Checks if Qdrant client is connected and reachable (cached for 10s to prevent request latency)."""
    global _qdrant_status_cache
    now = time.time()
    if now - _qdrant_status_cache["last_checked"] < 10.0:
        return bool(_qdrant_status_cache["connected"])

    if not qdrant_client:
        _qdrant_status_cache = {"connected": False, "last_checked": now}
        return False
    try:
        qdrant_client.get_collections()
        _qdrant_status_cache = {"connected": True, "last_checked": now}
        return True
    except Exception as e:
        logger.warning(f"Qdrant connection check failed: {e}")
        _qdrant_status_cache = {"connected": False, "last_checked": now}
        return False


# -----------------------------------------------------------------------------
# Lazy-Loaded Embedding Model (Swappable: fastembed vs local)
# -----------------------------------------------------------------------------
_embedding_model: Any = None
_embedding_backend_name: Optional[str] = None


def get_embedding_model() -> Any:
    """
    Lazy-loads the embedding model on first call rather than at module import.
    Swappable via EMBEDDING_BACKEND environment variable:
      - 'fastembed': ONNX Runtime model (~50MB RAM, ideal for Render 512MB free tier)
      - 'local': SentenceTransformer / PyTorch (~400MB RAM)
    Both use 'BAAI/bge-small-en-v1.5' and output 384-dimensional normalized vectors.
    """
    global _embedding_model, _embedding_backend_name
    if _embedding_model is not None:
        return _embedding_model

    backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower().strip()
    logger.info(f"Initializing lazy-loaded embedding model '{MODEL_NAME}' with EMBEDDING_BACKEND='{backend}'...")

    if backend == "local":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(MODEL_NAME)
        _embedding_backend_name = "local"
    else:
        # Default: fastembed (ONNX runtime)
        try:
            from fastembed import TextEmbedding

            _embedding_model = TextEmbedding(model_name=MODEL_NAME)
            _embedding_backend_name = "fastembed"
        except Exception as e:
            logger.warning(f"Failed to initialize FastEmbed ({e}). Falling back to SentenceTransformer...")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            _embedding_model = SentenceTransformer(MODEL_NAME)
            _embedding_backend_name = "local"

    logger.info(f"Embedding model initialized successfully using '{_embedding_backend_name}'.")
    return _embedding_model


def encode_query(query: str) -> List[float]:
    """Encodes a text query into a vector using the active embedding model."""
    model = get_embedding_model()
    if _embedding_backend_name == "local":
        return model.encode(query).tolist()
    else:
        # fastembed
        return list(model.embed([query]))[0].tolist()


# -----------------------------------------------------------------------------
# BM25 Keyword Index Initialization (Rebuilt from artifacts)
# -----------------------------------------------------------------------------
bm25_index: Optional[BM25Okapi] = None
bm25_chunks: List[Dict[str, Any]] = []


def _load_bm25_index() -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
    """
    Loads BM25 inputs from artifacts/bm25_inputs.json (or npz/pkl) and rebuilds BM25Okapi.
    Rebuilding at startup from JSON takes ~0.5s for 5,915 chunks and avoids unsafe pickle.
    """
    # 1. Try bm25_inputs.json (safe JSON format in artifacts/)
    if BM25_INPUTS_JSON.exists():
        try:
            logger.info(f"Loading BM25 corpus from JSON artifact: {BM25_INPUTS_JSON}...")
            with open(BM25_INPUTS_JSON, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            tokenized = [c["text"].lower().split() for c in chunks]
            index = BM25Okapi(tokenized)
            logger.info(f"BM25 index successfully built from JSON artifact ({len(chunks)} chunks).")
            return index, chunks
        except Exception as e:
            logger.error(f"Failed to load BM25 from {BM25_INPUTS_JSON}: {e}")

    # 2. Try bm25_inputs.npz (compact binary format in artifacts/)
    if BM25_INPUTS_NPZ.exists():
        try:
            import numpy as np

            logger.info(f"Loading BM25 corpus from NPZ artifact: {BM25_INPUTS_NPZ}...")
            data = np.load(BM25_INPUTS_NPZ, allow_pickle=False)
            ids = data["ids"]
            texts = data["texts"]
            comps = data["comps"]
            yrs = data["yrs"]
            sects = data["sects"]
            titles = data["titles"]
            paths = data["paths"]

            chunks = [
                {
                    "id": str(ids[i]),
                    "chunk_id": str(ids[i]),
                    "text": str(texts[i]),
                    "company": str(comps[i]),
                    "fiscal_year": str(yrs[i]),
                    "item_section": str(sects[i]),
                    "item_title": str(titles[i]),
                    "source_path": str(paths[i]),
                }
                for i in range(len(ids))
            ]
            tokenized = [c["text"].lower().split() for c in chunks]
            index = BM25Okapi(tokenized)
            logger.info(f"BM25 index successfully built from NPZ artifact ({len(chunks)} chunks).")
            return index, chunks
        except Exception as e:
            logger.error(f"Failed to load BM25 from {BM25_INPUTS_NPZ}: {e}")

    # 3. Fallback: pickle file in artifacts/ or data/processed/
    pkl_file = BM25_LEGACY_PKL if BM25_LEGACY_PKL.exists() else BM25_DATA_PKL
    if pkl_file.exists():
        try:
            logger.info(f"Loading legacy BM25 index from pickle: {pkl_file}...")
            with open(pkl_file, "rb") as f:
                payload = pickle.load(f)
                index = payload.get("bm25")
                chunks = payload.get("chunks", [])
            logger.info(f"Loaded legacy BM25 index ({len(chunks)} chunks).")
            return index, chunks
        except Exception as e:
            logger.error(f"Failed to load legacy BM25 pickle: {e}")

    logger.warning("BM25 index artifacts not found; hybrid sparse search will be unavailable.")
    return None, []


# Initialize BM25 at startup
bm25_index, bm25_chunks = _load_bm25_index()


def is_bm25_loaded() -> bool:
    """Returns True if the BM25 index is active and loaded."""
    return bm25_index is not None and len(bm25_chunks) > 0


def get_bm25_chunks_count() -> int:
    """Returns the total number of chunks indexed in BM25."""
    return len(bm25_chunks)


# -----------------------------------------------------------------------------
# Search Functions
# -----------------------------------------------------------------------------
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

    try:
        query_vector = encode_query(query)
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
    except Exception as e:
        logger.error(f"Qdrant query failed: {e}")
        return []


def search_bm25(query: str, filters: Optional[Dict] = None, limit: int = 20) -> List[Dict]:
    """Searches local BM25 keyword index for top scoring chunks."""
    if not bm25_index or not bm25_chunks:
        logger.warning("BM25 index is not loaded; skipping sparse search.")
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

        chunk_id = chunk.get("chunk_id") or chunk.get("id") or str(idx)
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
    print(f"BM25 Loaded: {is_bm25_loaded()} ({get_bm25_chunks_count()} chunks)")
    print(f"Qdrant Connected: {is_qdrant_connected()}")
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
