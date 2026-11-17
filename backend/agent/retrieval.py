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
import math
import os
import re
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

# Artifact paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BM25_INPUTS_JSON = ARTIFACTS_DIR / "bm25_inputs.json"
CORPUS_MANIFEST_JSON = ARTIFACTS_DIR / "corpus_manifest.json"

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
# BM25 Regex Tokenizer & Keyword Index Initialization (Rebuilt from artifacts)
# -----------------------------------------------------------------------------
def tokenize_bm25(text: str) -> List[str]:
    """
    Regex-based BM25 tokenizer that preserves numbers like '383,285' and '$1.2'
    while stripping surrounding punctuation. Used identically at index build time
    and at query time.
    """
    if not text:
        return []
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


bm25_index: Optional[BM25Okapi] = None
bm25_chunks: List[Dict[str, Any]] = []


def _load_bm25_index() -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
    """
    Loads BM25 inputs from artifacts/bm25_inputs.json and rebuilds BM25Okapi using tokenize_bm25.
    Rebuilding at startup from JSON takes ~0.2s for 2,921 chunks and avoids unsafe pickle files.
    """
    if BM25_INPUTS_JSON.exists():
        try:
            logger.info(f"Loading BM25 corpus from JSON artifact: {BM25_INPUTS_JSON}...")
            with open(BM25_INPUTS_JSON, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            tokenized = [tokenize_bm25(c.get("text", "")) for c in chunks]
            index = BM25Okapi(tokenized)
            logger.info(f"BM25 index successfully built from JSON artifact ({len(chunks)} chunks).")
            return index, chunks
        except Exception as e:
            logger.error(f"Failed to load BM25 from {BM25_INPUTS_JSON}: {e}")

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
# Index Alignment Verification (BM25 vs. Qdrant)
# -----------------------------------------------------------------------------
_index_match_cache: Dict[str, Any] = {"matches": False, "last_checked": 0.0}


def does_bm25_match_index() -> bool:
    """
    Checks if the BM25 artifact chunk count matches Qdrant collection points_count.
    Returns True if counts match, False if diverged or if Qdrant is unreachable.
    Cached for 10 seconds.
    """
    global _index_match_cache
    now = time.time()
    if now - _index_match_cache["last_checked"] < 10.0:
        return bool(_index_match_cache["matches"])

    if not qdrant_client or not is_qdrant_connected() or not is_bm25_loaded():
        _index_match_cache = {"matches": False, "last_checked": now}
        return False

    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        points_count = info.points_count
        matches = bool(points_count == len(bm25_chunks))
        _index_match_cache = {"matches": matches, "last_checked": now}
        return matches
    except Exception as e:
        logger.warning(f"Error checking Qdrant points_count for index match: {e}")
        _index_match_cache = {"matches": False, "last_checked": now}
        return False


def check_index_alignment() -> Optional[bool]:
    """
    Compares the BM25 artifact chunk count with Qdrant collection points_count at startup.
    Logs a loud WARNING if they differ.
    """
    if not is_bm25_loaded():
        logger.warning("BM25 index is not loaded; cannot check index alignment.")
        return False

    bm25_count = len(bm25_chunks)
    if not is_qdrant_connected():
        logger.info(
            f"Qdrant vector DB is not connected at startup; skipping index alignment check (BM25: {bm25_count} chunks)."
        )
        return None

    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        points_count = info.points_count
        if bm25_count != points_count:
            logger.warning(
                "\n"
                + "!" * 80
                + "\n"
                + f"⚠️  LOUD WARNING: BM25 artifact chunk count ({bm25_count}) DOES NOT MATCH "
                + f"Qdrant collection '{COLLECTION_NAME}' points_count ({points_count})!\n"
                + "Sparse keyword hits and dense semantic vectors are out of sync.\n"
                + "Please re-index Qdrant or rebuild artifacts to ensure identical retrieval corpus.\n"
                + "!" * 80
            )
            _index_match_cache["matches"] = False
            _index_match_cache["last_checked"] = time.time()
            return False
        else:
            logger.info(
                f"Index alignment verified: BM25 artifact count ({bm25_count}) matches "
                f"Qdrant collection points_count ({points_count})."
            )
            _index_match_cache["matches"] = True
            _index_match_cache["last_checked"] = time.time()
            return True
    except Exception as e:
        logger.warning(f"Failed to verify index alignment with Qdrant: {e}")
        return False


# Run alignment check at startup
check_index_alignment()


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


SECTION_TITLES = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "Item 8": "Financial Statements and Supplementary Data",
}


def search_bm25(query: str, filters: Optional[Dict] = None, limit: int = 20) -> List[Dict]:
    """Searches local BM25 keyword index for top scoring chunks using regex tokenizer."""
    if not bm25_index or not bm25_chunks:
        logger.warning("BM25 index is not loaded; skipping sparse search.")
        return []

    tokenized_query = tokenize_bm25(query)
    if not tokenized_query:
        return []

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
        comp = chunk.get("company", "")
        fy = str(chunk.get("fiscal_year", ""))
        sec = chunk.get("item_section", "")
        title = chunk.get("item_title") or SECTION_TITLES.get(sec, sec)
        source_path = chunk.get("source_path") or f"backend/data/raw/{comp.lower()}_{fy}.html"

        scored_chunks.append({
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "text": chunk.get("text", ""),
            "company": comp,
            "fiscal_year": fy,
            "item_section": sec,
            "item_title": title,
            "source_path": source_path,
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
    fused with Reciprocal Rank Fusion.

    Controlled by RETRIEVAL_STRATEGY env var:
      - 'clean_v3_rrf10' (new default): clean_v3 query cleaning, dense top-50 + BM25 top-50
        filtered by extractor's company/year, RRF with k=10.
      - 'legacy': exact current behaviour unchanged (clean_v2 query cleaning, dense top-20 +
        BM25 top-20, RRF with k=60).
    """
    from backend.agent.nodes.classifier import (
        extract_all_fiscal_years,
        extract_company_and_year,
        strip_query_metadata_v2,
        strip_query_metadata_v3,
    )

    strategy = os.getenv("RETRIEVAL_STRATEGY", "clean_v3_rrf10").strip().lower()

    # -------------------------------------------------------------------------
    # Legacy Strategy: clean_v2, k=60, top-20 pools (exact current behaviour)
    # -------------------------------------------------------------------------
    if strategy == "legacy":
        cleaning_enabled = os.getenv("RETRIEVAL_QUERY_CLEANING", "true").lower() in ("true", "1", "yes")

        company = (filters or {}).get("company")

        # Determine fiscal years: check filters first, fallback to extract_all_fiscal_years from query
        years: List[str] = []
        if filters and "fiscal_years" in filters and filters["fiscal_years"]:
            years = [str(y) for y in filters["fiscal_years"]]
        elif filters and "fiscal_year" in filters and filters["fiscal_year"]:
            years = [str(filters["fiscal_year"])]
        else:
            years = extract_all_fiscal_years(query)

        # 1. Query cleaning (clean_v2) - applies ONLY to retrieval query
        if cleaning_enabled and company:
            # Strip fiscal_year only when a single-year filter is active
            active_single_year = (
                str(filters.get("fiscal_year"))
                if (filters and filters.get("fiscal_year"))
                else (years[0] if len(years) == 1 else None)
            )
            retrieval_query = strip_query_metadata_v2(query, company=company, fiscal_year=active_single_year)
        else:
            # With no company found or cleaning disabled, do not clean
            retrieval_query = query

        # 2. Branching: Per-Year Retrieval vs Normal Path
        if company and len(years) >= 2:
            n_years = len(years)
            k_per_year = max(math.ceil(8 / n_years), math.ceil(top_k / n_years))
            per_year_results: List[List[Dict]] = []

            for yr in years:
                yr_filters = dict(filters) if filters else {}
                yr_filters["company"] = company
                yr_filters["fiscal_year"] = str(yr)
                yr_filters.pop("fiscal_years", None)

                # In per-year retrieval, clean the query specifically for this target year
                yr_query = (
                    strip_query_metadata_v2(query, company=company, fiscal_year=str(yr))
                    if (cleaning_enabled and company)
                    else query
                )

                dense_results = search_qdrant(query=yr_query, filters=yr_filters, limit=max(20, k_per_year))
                sparse_results = search_bm25(query=yr_query, filters=yr_filters, limit=max(20, k_per_year))
                fused = reciprocal_rank_fusion(
                    dense_results=dense_results,
                    sparse_results=sparse_results,
                    k=60,
                    top_k=k_per_year,
                )
                per_year_results.append(fused)

            merged: List[Dict] = []
            seen_ids = set()
            for idx in range(k_per_year):
                for yr_chunks in per_year_results:
                    if idx < len(yr_chunks):
                        chunk = yr_chunks[idx]
                        cid = chunk.get("id") or chunk.get("chunk_id")
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            merged.append(chunk)
                            if len(merged) == top_k:
                                break
                if len(merged) == top_k:
                    break
            return merged
        else:
            # Single-year and zero-year queries use the normal path
            norm_filters = dict(filters) if filters else None
            if norm_filters and "fiscal_years" in norm_filters:
                norm_filters = dict(norm_filters)
                norm_filters.pop("fiscal_years", None)

            dense_results = search_qdrant(query=retrieval_query, filters=norm_filters, limit=max(20, top_k))
            sparse_results = search_bm25(query=retrieval_query, filters=norm_filters, limit=max(20, top_k))
            fused_results = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=60,
                top_k=top_k,
            )
            return fused_results

    # -------------------------------------------------------------------------
    # New Default Strategy: clean_v3_rrf10
    # -------------------------------------------------------------------------
    company = (filters or {}).get("company")
    years: List[str] = []
    if filters and "fiscal_years" in filters and filters["fiscal_years"]:
        years = [str(y) for y in filters["fiscal_years"]]
    elif filters and "fiscal_year" in filters and filters["fiscal_year"]:
        years = [str(filters["fiscal_year"])]
    else:
        years = extract_all_fiscal_years(query)

    # 1. Multi-Year Per-Year Path
    # NOTE: The multi-year per-year retrieval path stays EXACTLY as it is today
    # (legacy query cleaning v2, k=60). This path is untested on held-out data.
    if company and len(years) >= 2:
        cleaning_enabled = os.getenv("RETRIEVAL_QUERY_CLEANING", "true").lower() in ("true", "1", "yes")
        n_years = len(years)
        k_per_year = max(math.ceil(8 / n_years), math.ceil(top_k / n_years))
        per_year_results: List[List[Dict]] = []

        for yr in years:
            yr_filters = dict(filters) if filters else {}
            yr_filters["company"] = company
            yr_filters["fiscal_year"] = str(yr)
            yr_filters.pop("fiscal_years", None)

            # In per-year retrieval, clean the query specifically for this target year
            yr_query = (
                strip_query_metadata_v2(query, company=company, fiscal_year=str(yr))
                if (cleaning_enabled and company)
                else query
            )

            dense_results = search_qdrant(query=yr_query, filters=yr_filters, limit=max(20, k_per_year))
            sparse_results = search_bm25(query=yr_query, filters=yr_filters, limit=max(20, k_per_year))
            fused = reciprocal_rank_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=60,
                top_k=k_per_year,
            )
            per_year_results.append(fused)

        merged: List[Dict] = []
        seen_ids = set()
        for idx in range(k_per_year):
            for yr_chunks in per_year_results:
                if idx < len(yr_chunks):
                    chunk = yr_chunks[idx]
                    cid = chunk.get("id") or chunk.get("chunk_id")
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        merged.append(chunk)
                        if len(merged) == top_k:
                            break
            if len(merged) == top_k:
                break
        return merged

    # 2. Single-Year / Unfiltered Path under clean_v3_rrf10
    # For clean_v3_rrf10: query = strip_query_metadata_v3(query),
    # dense top-50 + BM25 top-50 using the extractor's company/year filters,
    # RRF with k=10, return top_k. If no filters are detected, use the same recipe unfiltered.
    cleaned_query = strip_query_metadata_v3(query)

    active_filters: Dict[str, Any] = {}
    if filters:
        if "company" in filters and filters["company"]:
            active_filters["company"] = filters["company"]
        if "fiscal_year" in filters and filters["fiscal_year"]:
            active_filters["fiscal_year"] = str(filters["fiscal_year"])

    # Extract company/year using the extractor if not already provided in filters
    if not active_filters.get("company") or not active_filters.get("fiscal_year"):
        ext_comp, ext_fy = extract_company_and_year(query)
        if not active_filters.get("company") and ext_comp:
            active_filters["company"] = ext_comp
        if not active_filters.get("fiscal_year") and ext_fy:
            active_filters["fiscal_year"] = str(ext_fy)

    target_filters = active_filters if active_filters else None

    dense_results = search_qdrant(query=cleaned_query, filters=target_filters, limit=max(50, top_k))
    sparse_results = search_bm25(query=cleaned_query, filters=target_filters, limit=max(50, top_k))
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=10,
        top_k=top_k,
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
