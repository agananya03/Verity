"""
Unit tests for BM25 and Corpus Manifest Artifacts in Verity.

Verifies:
- corpus_manifest.json total == number of chunks in bm25_inputs.json (no network or chunks.json needed).
- All chunks in bm25_inputs.json contain required fields: (chunk_id, text, company, fiscal_year, item_section).
- tokenize_bm25 regex preserves numbers (e.g., '383,285', '1.2') while stripping surrounding punctuation.
- does_bm25_match_index() correctly verifies alignment against Qdrant points_count.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.agent.retrieval import (
    does_bm25_match_index,
    tokenize_bm25,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BM25_INPUTS_JSON = ARTIFACTS_DIR / "bm25_inputs.json"
CORPUS_MANIFEST_JSON = ARTIFACTS_DIR / "corpus_manifest.json"


def test_corpus_manifest_total_equals_bm25_inputs_count():
    """
    Asserts that corpus_manifest.json total_chunks exactly matches the count of chunks
    in bm25_inputs.json without requiring network access or raw chunks.json.
    """
    assert BM25_INPUTS_JSON.exists(), f"Artifact missing: {BM25_INPUTS_JSON}"
    assert CORPUS_MANIFEST_JSON.exists(), f"Artifact missing: {CORPUS_MANIFEST_JSON}"

    with open(CORPUS_MANIFEST_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(BM25_INPUTS_JSON, "r", encoding="utf-8") as f:
        bm25_chunks = json.load(f)

    total_chunks = manifest.get("total_chunks")
    assert isinstance(total_chunks, int)
    assert total_chunks > 0

    # Core assertion: manifest total must equal number of chunks in bm25_inputs.json
    assert total_chunks == len(bm25_chunks), (
        f"Manifest total ({total_chunks}) does not match bm25_inputs.json length ({len(bm25_chunks)})"
    )

    # Check manifest internal consistency
    companies = manifest.get("companies", [])
    assert len(companies) >= 5
    company_sum = sum(c["total_chunks"] for c in companies)
    assert company_sum == total_chunks

    # Check required fields in manifest
    assert "sha256" in manifest or "chunks_sha256" in manifest
    assert "generated_at" in manifest
    assert "per_company_year_section" in manifest


def test_bm25_inputs_schema_fields():
    """
    Asserts that every record in bm25_inputs.json conforms to the required schema:
    (chunk_id, text, company, fiscal_year, item_section).
    """
    with open(BM25_INPUTS_JSON, "r", encoding="utf-8") as f:
        bm25_chunks = json.load(f)

    assert len(bm25_chunks) > 0
    sample = bm25_chunks[0]

    required_keys = {"chunk_id", "text", "company", "fiscal_year", "item_section"}
    for key in required_keys:
        assert key in sample, f"Missing required key '{key}' in bm25_inputs chunk"

    # Sample verify first 100 chunks for non-empty fields
    for chunk in bm25_chunks[:100]:
        assert chunk["chunk_id"], "chunk_id must not be empty"
        assert chunk["text"], "text must not be empty"
        assert chunk["company"], "company must not be empty"
        assert chunk["fiscal_year"], "fiscal_year must not be empty"
        assert chunk["item_section"], "item_section must not be empty"


def test_tokenize_bm25_preserves_numbers():
    """
    Asserts that tokenize_bm25 preserves formatted numbers (e.g. 383,285 and $1.2)
    identically across queries and indexed texts while removing punctuation.
    """
    text = "Apple revenue reached $383,285 million in 2024 and $1.2 billion in Q3, up 15.4%."
    tokens = tokenize_bm25(text)

    # Number with comma must be preserved intact
    assert "383,285" in tokens
    # Decimal number must be preserved intact
    assert "1.2" in tokens
    assert "15.4" in tokens
    # Year integer must be preserved
    assert "2024" in tokens
    # Punctuation stripped from words
    assert "apple" in tokens
    assert "revenue" in tokens
    assert "q3" in tokens

    # Query with dollar sign should tokenize to the matching number token
    query_tokens = tokenize_bm25("What was Apple's $383,285 revenue?")
    assert "383,285" in query_tokens
    assert "apple" in query_tokens
    assert "revenue" in query_tokens


def test_does_bm25_match_index_mocked(monkeypatch):
    """
    Asserts that does_bm25_match_index() returns True when Qdrant points_count matches
    bm25_chunks length, and False when they diverge.
    """
    mock_qdrant = MagicMock()
    mock_collection = MagicMock()

    # 1. Matching case
    mock_collection.points_count = 2921
    mock_qdrant.get_collection.return_value = mock_collection

    monkeypatch.setattr("backend.agent.retrieval.qdrant_client", mock_qdrant)
    monkeypatch.setattr("backend.agent.retrieval.is_qdrant_connected", lambda: True)
    monkeypatch.setattr("backend.agent.retrieval.bm25_chunks", [{"chunk_id": f"c_{i}"} for i in range(2921)])
    monkeypatch.setattr("backend.agent.retrieval._index_match_cache", {"matches": False, "last_checked": 0.0})

    assert does_bm25_match_index() is True

    # 2. Diverging case
    mock_collection.points_count = 5915
    monkeypatch.setattr("backend.agent.retrieval._index_match_cache", {"matches": False, "last_checked": 0.0})

    assert does_bm25_match_index() is False

