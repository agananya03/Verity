"""
Unit tests for Verity LangGraph self-correcting routing and execution traces.

Tests:
- Grounded on first pass (0 retries, final_grounded=True)
- Grounded after 1 retry (1 retry, critique_feedback populated, final_grounded=True)
- Never grounded (2 retries reached, finalizer strips ungrounded claims, final_grounded=False)

All LLM calls and retrieval operations are mocked with monkeypatch;
no real API keys or network calls are required.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from backend.agent.graph import verity_graph
from backend.agent.nodes.classifier import ClassificationOutput
from backend.agent.nodes.critic import CriticVerdictOutput
from backend.agent.state import AgentState


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLMWrapper:
    """Mock FallbackLLMWrapper returning scripted responses based on structured_schema or prompt."""

    def __init__(self, invoke_fn):
        self.invoke_fn = invoke_fn
        self.schema = None

    def with_structured_output(self, schema: Any, **kwargs: Any):
        wrapper = MockLLMWrapper(self.invoke_fn)
        wrapper.schema = schema
        return wrapper

    def invoke(self, messages: Any, **kwargs: Any):
        return self.invoke_fn(messages, self.schema)


def test_grounded_first_pass(monkeypatch):
    """
    Scenario 1: The draft is grounded on the first pass.
    Critic verdict is grounded=True -> routes immediately to finalizer with 0 retries.
    """
    # 1. Mock retriever
    mock_chunks = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "text": "Apple net sales in fiscal 2024 were $391,035 million.",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 7",
            "source_path": "apple_2024.html",
        }
    ]
    monkeypatch.setattr("backend.agent.nodes.retriever.retrieve", lambda *args, **kwargs: mock_chunks)

    # 2. Mock LLM calls
    def mock_invoke(messages, schema):
        if schema == ClassificationOutput:
            return ClassificationOutput(
                classification="simple_lookup",
                extracted_filters={"company": "Apple", "fiscal_year": "2024"},
            )
        elif schema == CriticVerdictOutput:
            return CriticVerdictOutput(grounded=True, unsupported_claims=[])
        else:
            # Draft node
            return MockLLMResponse("Apple net sales were $391,035 million in fiscal year 2024 [1].")

    monkeypatch.setattr(
        "backend.agent.nodes.classifier.get_llm_with_fallback",
        lambda *args, **kwargs: MockLLMWrapper(mock_invoke),
    )
    monkeypatch.setattr(
        "backend.agent.nodes.draft.get_llm_with_fallback",
        lambda *args, **kwargs: MockLLMWrapper(mock_invoke),
    )
    monkeypatch.setattr(
        "backend.agent.nodes.critic.get_llm_with_fallback",
        lambda *args, **kwargs: MockLLMWrapper(mock_invoke),
    )

    initial_state: AgentState = {
        "query": "What was Apple's net sales in 2024?",
        "original_query": "What was Apple's net sales in 2024?",
        "rewritten_query": "What was Apple's net sales in 2024?",
        "conversation_history": [],
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result = verity_graph.invoke(initial_state)

    assert result["retry_count"] == 0
    assert result["final_grounded"] is True
    assert result["unsupported_claims_final"] == []
    assert "$391,035 million" in result["final_answer"]
    assert len(result["citations"]) > 0


def test_grounded_after_one_retry(monkeypatch):
    """
    Scenario 2: The draft is ungrounded on pass 1, but corrected on pass 2.
    Critic verdict: pass 1 grounded=False -> revise -> draft -> pass 2 grounded=True -> finalizer.
    Trace: retry_count=1, critique_feedback populated, final_grounded=True.
    """
    mock_chunks = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "text": "Microsoft Azure revenue grew 29% in fiscal 2024.",
            "company": "Microsoft",
            "fiscal_year": "2024",
            "item_section": "Item 7",
            "source_path": "msft_2024.html",
        }
    ]
    monkeypatch.setattr("backend.agent.nodes.retriever.retrieve", lambda *args, **kwargs: mock_chunks)

    critic_calls = 0

    def mock_invoke(messages, schema):
        nonlocal critic_calls
        if schema == ClassificationOutput:
            return ClassificationOutput(
                classification="simple_lookup",
                extracted_filters={"company": "Microsoft", "fiscal_year": "2024"},
            )
        elif schema == CriticVerdictOutput:
            critic_calls += 1
            if critic_calls == 1:
                return CriticVerdictOutput(
                    grounded=False,
                    unsupported_claims=["Azure growth was 50%"],
                )
            else:
                return CriticVerdictOutput(grounded=True, unsupported_claims=[])
        else:
            # Draft node
            if critic_calls == 0:
                return MockLLMResponse("Microsoft Azure grew 29% [1]. Azure growth was 50% [1].")
            else:
                return MockLLMResponse("Microsoft Azure grew 29% [1].")

    monkeypatch.setattr("backend.agent.nodes.classifier.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))
    monkeypatch.setattr("backend.agent.nodes.draft.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))

    initial_state: AgentState = {
        "query": "What was Azure's revenue growth in 2024?",
        "original_query": "What was Azure's revenue growth in 2024?",
        "rewritten_query": "What was Azure's revenue growth in 2024?",
        "conversation_history": [],
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result = verity_graph.invoke(initial_state)

    assert result["retry_count"] == 1
    assert result["final_grounded"] is True
    assert result["unsupported_claims_final"] == []
    assert "Azure grew 29%" in result["final_answer"]
    assert "Azure growth was 50%" not in result["final_answer"]


def test_never_grounded_strips_sentences(monkeypatch):
    """
    Scenario 3: The draft remains ungrounded across all retry attempts.
    Critic verdict: pass 1 False -> revise (retry 1) -> pass 2 False -> revise (retry 2) -> pass 3 False.
    MAX_RETRIES=2 reached -> routes to finalizer.
    Finalizer strips sentences containing the ungrounded claims.
    """
    mock_chunks = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "text": "NVIDIA compute and networking revenue was $115B in fiscal 2025.",
            "company": "NVIDIA",
            "fiscal_year": "2025",
            "item_section": "Item 7",
            "source_path": "nvda_2025.html",
        }
    ]
    monkeypatch.setattr("backend.agent.nodes.retriever.retrieve", lambda *args, **kwargs: mock_chunks)

    def mock_invoke(messages, schema):
        if schema == ClassificationOutput:
            return ClassificationOutput(
                classification="simple_lookup",
                extracted_filters={"company": "NVIDIA", "fiscal_year": "2025"},
            )
        elif schema == CriticVerdictOutput:
            return CriticVerdictOutput(
                grounded=False,
                unsupported_claims=["NVIDIA plans to release electric cars"],
            )
        else:
            return MockLLMResponse(
                "NVIDIA compute revenue reached $115B in 2025 [1]. NVIDIA plans to release electric cars [1]."
            )

    monkeypatch.setattr("backend.agent.nodes.classifier.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))
    monkeypatch.setattr("backend.agent.nodes.draft.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: MockLLMWrapper(mock_invoke))

    initial_state: AgentState = {
        "query": "What was NVIDIA revenue in 2025?",
        "original_query": "What was NVIDIA revenue in 2025?",
        "rewritten_query": "What was NVIDIA revenue in 2025?",
        "conversation_history": [],
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result = verity_graph.invoke(initial_state)

    assert result["retry_count"] == 2
    assert result["final_grounded"] is False
    assert "NVIDIA plans to release electric cars" in result["unsupported_claims_final"]
    # Finalizer must have stripped the sentence about electric cars
    assert "electric cars" not in result["final_answer"]
    # The grounded sentence is retained
    assert "NVIDIA compute revenue reached $115B" in result["final_answer"]
