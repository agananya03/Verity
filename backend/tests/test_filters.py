"""
Unit tests for Qdrant and BM25 metadata filtering in backend/agent/retrieval.py.

Tests:
- build_qdrant_filter with None, empty dict, single condition, and multiple conditions.
- build_qdrant_filter skips keys with None values.
- BM25 metadata filtering logic matching company and fiscal_year constraints.
"""

import pytest
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from backend.agent.retrieval import build_qdrant_filter, search_bm25


def test_build_qdrant_filter_empty():
    """None or empty dict should return None."""
    assert build_qdrant_filter(None) is None
    assert build_qdrant_filter({}) is None


def test_build_qdrant_filter_single_field():
    """Single metadata constraint produces a Filter with one FieldCondition."""
    f = build_qdrant_filter({"company": "Apple"})
    assert isinstance(f, Filter)
    assert len(f.must) == 1
    assert isinstance(f.must[0], FieldCondition)
    assert f.must[0].key == "company"
    assert f.must[0].match.value == "Apple"


def test_build_qdrant_filter_multiple_fields():
    """Multiple metadata constraints produce a Filter with multiple FieldConditions."""
    f = build_qdrant_filter({"company": "Microsoft", "fiscal_year": "2024"})
    assert isinstance(f, Filter)
    assert len(f.must) == 2
    keys = {cond.key: cond.match.value for cond in f.must}
    assert keys == {"company": "Microsoft", "fiscal_year": "2024"}


def test_build_qdrant_filter_none_values():
    """Filters with None values are excluded; if all values are None, return None."""
    f = build_qdrant_filter({"company": "NVIDIA", "fiscal_year": None})
    assert isinstance(f, Filter)
    assert len(f.must) == 1
    assert f.must[0].key == "company"
    assert f.must[0].match.value == "NVIDIA"

    # All None values
    f_empty = build_qdrant_filter({"company": None, "fiscal_year": None})
    assert f_empty is None


def test_bm25_metadata_filtering(monkeypatch):
    """
    Verifies that search_bm25 respects company and fiscal_year metadata filters
    when filtering candidate chunks.
    """
    sample_chunks = [
        {"id": "c1", "text": "Apple revenue grew in 2024", "company": "Apple", "fiscal_year": "2024", "item_section": "Item 7"},
        {"id": "c2", "text": "Apple iPhone sales in 2023", "company": "Apple", "fiscal_year": "2023", "item_section": "Item 7"},
        {"id": "c3", "text": "Microsoft cloud revenue grew in 2024", "company": "Microsoft", "fiscal_year": "2024", "item_section": "Item 7"},
    ]

    class MockBM25:
        def get_scores(self, tokenized_query):
            # Return arbitrary positive scores
            return [10.0, 8.0, 12.0]

    monkeypatch.setattr("backend.agent.retrieval.bm25_index", MockBM25())
    monkeypatch.setattr("backend.agent.retrieval.bm25_chunks", sample_chunks)

    # Filter for Apple only
    apple_results = search_bm25("revenue", filters={"company": "Apple"}, limit=10)
    assert len(apple_results) == 2
    for r in apple_results:
        assert r["company"] == "Apple"

    # Filter for Apple + 2024 only
    apple_2024_results = search_bm25("revenue", filters={"company": "Apple", "fiscal_year": "2024"}, limit=10)
    assert len(apple_2024_results) == 1
    assert apple_2024_results[0]["id"] == "c1"
    assert apple_2024_results[0]["fiscal_year"] == "2024"

    # Filter for non-existent match returns empty list
    none_results = search_bm25("revenue", filters={"company": "Amazon"}, limit=10)
    assert len(none_results) == 0

