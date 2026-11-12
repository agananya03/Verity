"""
Unit tests for production retrieval integration:
1. Draft node receives the original query (unaltered by retrieval query cleaning).
2. Query cleaning is skipped when no company is found.
3. Per-year merge returns chunks from each named year alternatingly.
4. Baseline config in run_ragas_eval.py uses the raw query with no cleaning or filters.
"""

import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest

from backend.agent.nodes.draft import draft_answer
from backend.agent.nodes.retriever import retrieve_node
from backend.agent.retrieval import retrieve
from backend.agent.state import AgentState
from backend.eval.run_ragas_eval import run_baseline_pipeline


def test_draft_node_receives_original_query(monkeypatch):
    """
    Verifies that retrieval query cleaning does NOT overwrite state['query'].
    The draft node and its LLM prompt must receive the user's original query.
    """
    original_q = "According to Apple's fiscal year 2024 Form 10-K, what were total net sales?"
    state: AgentState = {
        "query": original_q,
        "extracted_filters": {"company": "Apple", "fiscal_year": "2024"},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "first_critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
        "classification": "simple_lookup",
        "original_query": original_q,
        "rewritten_query": original_q,
        "conversation_history": [],
    }

    # Mock retrieve to return dummy chunks without making network calls
    dummy_chunks = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "text": "Total net sales were $391,035 million.",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 7",
        },
        {
            "id": "c2",
            "chunk_id": "c2",
            "text": "Products net sales were $294,866 million.",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 7",
        },
        {
            "id": "c3",
            "chunk_id": "c3",
            "text": "Services net sales were $96,169 million.",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 7",
        },
    ]

    import backend.agent.nodes.retriever
    monkeypatch.setattr(backend.agent.nodes.retriever, "retrieve", lambda query, top_k, filters: dummy_chunks)

    # 1. Execute retriever node
    state = retrieve_node(state)

    # State query MUST remain the exact original query
    assert state["query"] == original_q
    assert len(state["retrieved_chunks"]) >= 3

    # 2. Mock draft LLM and verify received prompt
    captured_messages = []

    class MockDraftLLM:
        def invoke(self, messages, **kwargs):
            captured_messages.extend(messages)
            mock_resp = MagicMock()
            mock_resp.content = "Apple's total net sales were $391,035 million [1]."
            return mock_resp

    import backend.agent.nodes.draft
    monkeypatch.setattr(backend.agent.nodes.draft, "get_llm_with_fallback", lambda **kwargs: MockDraftLLM())

    state = draft_answer(state)

    # The user prompt passed to the draft LLM must contain the original query
    user_prompts = [msg[1] for msg in captured_messages if msg[0] == "user"]
    assert len(user_prompts) == 1
    assert original_q in user_prompts[0]


def test_cleaning_skipped_when_no_company_found(monkeypatch):
    """
    Verifies that when no company is found (filters=None or company=None),
    query cleaning is completely skipped and the raw query is passed to search backends.
    """
    captured_dense_queries = []
    captured_sparse_queries = []

    def mock_qdrant(query, filters, limit):
        captured_dense_queries.append(query)
        return [{"id": "d1", "text": "dense text"}]

    def mock_bm25(query, filters, limit):
        captured_sparse_queries.append(query)
        return [{"id": "s1", "text": "sparse text"}]

    import backend.agent.retrieval
    monkeypatch.setattr(backend.agent.retrieval, "search_qdrant", mock_qdrant)
    monkeypatch.setattr(backend.agent.retrieval, "search_bm25", mock_bm25)

    raw_query = "According to general filings, what is the standard operating margin in 2024?"

    # Case A: filters=None
    retrieve(raw_query, top_k=5, filters=None)
    assert captured_dense_queries[-1] == raw_query
    assert captured_sparse_queries[-1] == raw_query

    # Case B: filters={"fiscal_year": "2024"} but company is None
    retrieve(raw_query, top_k=5, filters={"fiscal_year": "2024"})
    assert captured_dense_queries[-1] == raw_query
    assert captured_sparse_queries[-1] == raw_query


def test_per_year_merge_returns_chunks_from_each_named_year(monkeypatch):
    """
    Verifies that for multi-year queries with a company filter,
    retrieve() queries each named fiscal year separately and merges chunks
    alternatingly from every year.
    """
    captured_filters = []

    def mock_qdrant(query, filters, limit):
        captured_filters.append(filters)
        yr = (filters or {}).get("fiscal_year", "unknown")
        return [
            {"id": f"chunk_{yr}_dense_{i}", "text": f"Dense {yr} {i}", "fiscal_year": yr, "company": "Apple"}
            for i in range(limit)
        ]

    def mock_bm25(query, filters, limit):
        yr = (filters or {}).get("fiscal_year", "unknown")
        return [
            {"id": f"chunk_{yr}_sparse_{i}", "text": f"Sparse {yr} {i}", "fiscal_year": yr, "company": "Apple"}
            for i in range(limit)
        ]

    import backend.agent.retrieval
    monkeypatch.setattr(backend.agent.retrieval, "search_qdrant", mock_qdrant)
    monkeypatch.setattr(backend.agent.retrieval, "search_bm25", mock_bm25)

    q = "Compare Apple's net sales between fiscal year 2023 and fiscal year 2024"
    results = retrieve(q, top_k=8, filters={"company": "Apple"})

    # Must query both 2023 and 2024
    queried_years = [f.get("fiscal_year") for f in captured_filters if f]
    assert "2023" in queried_years
    assert "2024" in queried_years

    # Must return 8 chunks total
    assert len(results) == 8

    # Chunks must alternate between 2023 and 2024
    years_in_results = [r.get("fiscal_year") for r in results]
    assert years_in_results == ["2023", "2024", "2023", "2024", "2023", "2024", "2023", "2024"]


def test_baseline_config_uses_raw_query(monkeypatch):
    """
    Verifies that the baseline config in run_ragas_eval.py always uses the raw query
    with vector-only search (filters=None, no cleaning).
    """
    captured_query = None
    captured_filters = "SENTINEL"

    def mock_search_qdrant(query, filters, limit):
        nonlocal captured_query, captured_filters
        captured_query = query
        captured_filters = filters
        return [{"id": "chunk_1", "text": "Sample baseline filing context."}]

    class MockBaselineLLM:
        def invoke(self, messages, **kwargs):
            mock_resp = MagicMock()
            mock_resp.content = "Baseline draft response."
            return mock_resp

    import backend.eval.run_ragas_eval
    monkeypatch.setattr(backend.eval.run_ragas_eval, "search_qdrant", mock_search_qdrant)
    monkeypatch.setattr(backend.eval.run_ragas_eval, "get_llm_with_fallback", lambda **kwargs: MockBaselineLLM())

    raw_query = "According to Apple's fiscal year 2024 Form 10-K, what were total net sales?"
    ans, ctxs, c_ids, b_info = run_baseline_pipeline(query=raw_query, gen_model="mock-gemini")

    assert captured_query == raw_query  # Uncleaned raw query
    assert captured_filters is None     # No metadata filters
    assert len(c_ids) == 1
    assert "chunk_1" in c_ids
