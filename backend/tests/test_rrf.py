"""
Unit tests for Reciprocal Rank Fusion (RRF) in backend/agent/retrieval.py.

Tests:
- RRF ranking when documents appear in one or both ranked candidate lists.
- Tie-breaking behavior.
- Edge cases with empty dense, empty sparse, or both empty inputs.
- Top-k limit truncation.
- Mathematical precision of RRF scoring formula (1 / (k + rank)).
"""

import pytest
from backend.agent.retrieval import reciprocal_rank_fusion


def test_rrf_empty_inputs():
    """RRF should return an empty list when both candidate lists are empty."""
    assert reciprocal_rank_fusion([], [], k=60, top_k=5) == []


def test_rrf_only_dense():
    """RRF should correctly score results when only dense search yields results."""
    dense = [
        {"id": "doc1", "text": "Apple revenue 2024", "company": "Apple"},
        {"id": "doc2", "text": "Apple R&D 2024", "company": "Apple"},
    ]
    fused = reciprocal_rank_fusion(dense, [], k=60, top_k=5)
    assert len(fused) == 2
    assert fused[0]["id"] == "doc1"
    assert fused[1]["id"] == "doc2"
    # rank 1: 1 / (60 + 1)
    assert pytest.approx(fused[0]["score"], rel=1e-5) == 1.0 / 61.0
    # rank 2: 1 / (60 + 2)
    assert pytest.approx(fused[1]["score"], rel=1e-5) == 1.0 / 62.0


def test_rrf_only_sparse():
    """RRF should correctly score results when only sparse search yields results."""
    sparse = [
        {"id": "docA", "text": "Microsoft cloud segment", "company": "Microsoft"},
        {"id": "docB", "text": "Microsoft server products", "company": "Microsoft"},
    ]
    fused = reciprocal_rank_fusion([], sparse, k=60, top_k=5)
    assert len(fused) == 2
    assert fused[0]["id"] == "docA"
    assert fused[1]["id"] == "docB"
    assert pytest.approx(fused[0]["score"], rel=1e-5) == 1.0 / 61.0
    assert pytest.approx(fused[1]["score"], rel=1e-5) == 1.0 / 62.0


def test_rrf_ranking_overlap_boost():
    """
    A document appearing in both dense and sparse lists should rank higher
    than documents appearing at the same rank in only one list.
    """
    dense = [
        {"id": "doc_both", "text": "Overlapping chunk"},
        {"id": "doc_dense_only", "text": "Dense only chunk"},
    ]
    sparse = [
        {"id": "doc_both", "text": "Overlapping chunk"},
        {"id": "doc_sparse_only", "text": "Sparse only chunk"},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)
    assert len(fused) == 3
    assert fused[0]["id"] == "doc_both"

    # doc_both score = 1/(60+1) + 1/(60+1) = 2/61
    expected_both_score = (1.0 / 61.0) + (1.0 / 61.0)
    assert pytest.approx(fused[0]["score"], rel=1e-5) == expected_both_score

    # doc_dense_only rank 2 in dense = 1/(60+2)
    # doc_sparse_only rank 2 in sparse = 1/(60+2)
    expected_single_score = 1.0 / 62.0
    assert pytest.approx(fused[1]["score"], rel=1e-5) == expected_single_score
    assert pytest.approx(fused[2]["score"], rel=1e-5) == expected_single_score


def test_rrf_ties():
    """
    When two documents have identical RRF scores, both are preserved and present in the output.
    """
    dense = [
        {"id": "doc1", "text": "First"},
    ]
    sparse = [
        {"id": "doc2", "text": "Second"},
    ]
    # doc1 has rank 1 in dense, doc2 has rank 1 in sparse -> both score 1/61
    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=2)
    assert len(fused) == 2
    scores = [item["score"] for item in fused]
    assert pytest.approx(scores[0], rel=1e-5) == scores[1]
    assert {fused[0]["id"], fused[1]["id"]} == {"doc1", "doc2"}


def test_rrf_top_k():
    """
    RRF must strictly respect the top_k parameter limit even when more candidates exist.
    """
    dense = [{"id": f"dense_{i}", "text": f"text {i}"} for i in range(10)]
    sparse = [{"id": f"sparse_{i}", "text": f"text {i}"} for i in range(10)]

    for limit in [1, 3, 5, 8]:
        fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=limit)
        assert len(fused) == limit

    # When top_k exceeds total unique candidates, all candidates are returned
    fused_all = reciprocal_rank_fusion(dense[:2], sparse[:2], k=60, top_k=100)
    assert len(fused_all) == 4

