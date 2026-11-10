"""
Unit tests for the Critic Node in backend/agent/nodes/critic.py.

Tests:
1. Critic prompt content and constraints:
   - Enforces checking only factual assertions (numbers, named facts, causal/comparative statements).
   - Ignores framing phrases, headings, bullet labels, and citation markers.
   - Mandates verbatim sentences/clauses for unsupported_claims, and empty list when supported.
2. Structured output parsing with Pydantic CriticVerdictOutput.
3. Mocked LLM execution and verdict handling:
   - Grounded draft evaluation (empty unsupported_claims list).
   - Ungrounded draft evaluation (verbatim unsupported claims extracted).
   - Consistency enforcement between grounded flag and unsupported_claims.
   - Preservation of state['first_critic_verdict'] across retry passes.
"""

from typing import Any
import pytest
from pydantic import ValidationError

from backend.agent.nodes.critic import (
    CRITIC_PROMPT,
    CriticVerdictOutput,
    critique,
)
from backend.agent.state import AgentState


class MockStructuredLLM:
    """Mock LLM returning a scripted CriticVerdictOutput."""

    def __init__(self, return_value: CriticVerdictOutput):
        self.return_value = return_value
        self.last_messages = None

    def with_structured_output(self, schema: Any, **kwargs: Any):
        return self

    def invoke(self, messages: Any, **kwargs: Any):
        self.last_messages = messages
        return self.return_value


def test_critic_prompt_content():
    """Verifies that CRITIC_PROMPT contains the required tightening instructions."""
    # Factual assertions check
    assert "factual assertions" in CRITIC_PROMPT.lower()
    assert "numbers" in CRITIC_PROMPT.lower()
    assert "causal or comparative" in CRITIC_PROMPT.lower()

    # Ignored elements
    assert "introductory or framing phrases" in CRITIC_PROMPT.lower()
    assert "based on the filings" in CRITIC_PROMPT.lower()
    assert "headings" in CRITIC_PROMPT.lower()
    assert "bullet labels" in CRITIC_PROMPT.lower()
    assert "citation markers" in CRITIC_PROMPT.lower()

    # Verbatim and empty list constraints
    assert "verbatim sentences or clauses" in CRITIC_PROMPT.lower()
    assert "empty list" in CRITIC_PROMPT.lower()


def test_critic_verdict_output_pydantic_parsing():
    """Tests Pydantic parsing and validation of CriticVerdictOutput."""
    # Valid grounded output
    v1 = CriticVerdictOutput(grounded=True, unsupported_claims=[])
    assert v1.grounded is True
    assert v1.unsupported_claims == []

    # Valid ungrounded output with verbatim claims
    v2 = CriticVerdictOutput(
        grounded=False,
        unsupported_claims=["Apple acquired Tesla for $100B in 2024."]
    )
    assert v2.grounded is False
    assert v2.unsupported_claims == ["Apple acquired Tesla for $100B in 2024."]

    # Default unsupported_claims should be empty list
    v3 = CriticVerdictOutput(grounded=True)
    assert v3.unsupported_claims == []


def test_critique_grounded_answer(monkeypatch):
    """
    Tests critique() node when the draft answer is fully supported.
    Framing phrases and citations should be present in draft, and critic should return
    grounded=True with empty unsupported_claims list.
    """
    mock_verdict = CriticVerdictOutput(grounded=True, unsupported_claims=[])
    mock_llm = MockStructuredLLM(mock_verdict)
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: mock_llm)

    state: AgentState = {
        "query": "What was Apple's revenue in 2024?",
        "original_query": "What was Apple's revenue in 2024?",
        "rewritten_query": "What was Apple's revenue in 2024?",
        "conversation_history": [],
        "classification": "simple_lookup",
        "extracted_filters": {"company": "Apple", "fiscal_year": "2024"},
        "retrieved_chunks": [
            {
                "id": "c1",
                "chunk_id": "c1",
                "text": "Total net sales were $391,035 million in fiscal year 2024.",
                "company": "Apple",
                "fiscal_year": "2024",
                "item_section": "Item 7",
            }
        ],
        "draft_answer": "Based on the filings, Apple's total net sales were $391,035 million in fiscal year 2024 [1].",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result_state = critique(state)

    # Assert critic_verdict
    assert result_state["critic_verdict"]["grounded"] is True
    assert result_state["critic_verdict"]["unsupported_claims"] == []
    assert result_state["first_critic_verdict"]["grounded"] is True
    assert result_state["first_critic_verdict"]["unsupported_claims"] == []

    # Verify prompt messages passed to LLM
    system_msg = mock_llm.last_messages[0][1]
    user_msg = mock_llm.last_messages[1][1]
    assert "CHECK ONLY SUBSTANTIVE FACTUAL ASSERTIONS" in system_msg
    assert "Based on the filings" in user_msg


def test_critique_ungrounded_answer_verbatim_claim(monkeypatch):
    """
    Tests critique() node when the draft contains an ungrounded factual assertion.
    Critic should return grounded=False and verbatim sentence in unsupported_claims.
    """
    unsupported_sentence = "Apple's cloud revenue reached $150 billion in fiscal year 2024."
    mock_verdict = CriticVerdictOutput(
        grounded=False,
        unsupported_claims=[unsupported_sentence]
    )
    mock_llm = MockStructuredLLM(mock_verdict)
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: mock_llm)

    state: AgentState = {
        "query": "What was Apple's cloud revenue in 2024?",
        "original_query": "What was Apple's cloud revenue in 2024?",
        "rewritten_query": "What was Apple's cloud revenue in 2024?",
        "conversation_history": [],
        "classification": "simple_lookup",
        "extracted_filters": {"company": "Apple", "fiscal_year": "2024"},
        "retrieved_chunks": [
            {
                "id": "c1",
                "chunk_id": "c1",
                "text": "Services net sales were $85,200 million in fiscal year 2024.",
                "company": "Apple",
                "fiscal_year": "2024",
                "item_section": "Item 7",
            }
        ],
        "draft_answer": (
            "According to the SEC filings:\n"
            "- **Services Overview:** Services net sales were $85,200 million [1].\n"
            f"- **Cloud:** {unsupported_sentence}"
        ),
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result_state = critique(state)

    assert result_state["critic_verdict"]["grounded"] is False
    assert result_state["critic_verdict"]["unsupported_claims"] == [unsupported_sentence]
    assert result_state["first_critic_verdict"]["grounded"] is False
    assert result_state["first_critic_verdict"]["unsupported_claims"] == [unsupported_sentence]


def test_critique_consistency_enforcement(monkeypatch):
    """
    Tests that if the LLM produces conflicting output (e.g. grounded=True but
    unsupported_claims has entries), critique() enforces grounded=False.
    """
    mock_verdict = CriticVerdictOutput(
        grounded=True,
        unsupported_claims=["Apple projected 50% growth in 2025."]
    )
    mock_llm = MockStructuredLLM(mock_verdict)
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: mock_llm)

    state: AgentState = {
        "query": "Apple growth 2025?",
        "original_query": "Apple growth 2025?",
        "rewritten_query": "Apple growth 2025?",
        "conversation_history": [],
        "classification": "simple_lookup",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "Apple projected 50% growth in 2025.",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    result_state = critique(state)
    assert result_state["critic_verdict"]["grounded"] is False
    assert result_state["critic_verdict"]["unsupported_claims"] == ["Apple projected 50% growth in 2025."]


def test_first_critic_verdict_preservation(monkeypatch):
    """
    Tests that state['first_critic_verdict'] is populated on pass 1 and
    preserved on subsequent critique passes.
    """
    mock_verdict_pass1 = CriticVerdictOutput(
        grounded=False,
        unsupported_claims=["Ungrounded claim 1."]
    )
    mock_llm = MockStructuredLLM(mock_verdict_pass1)
    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", lambda *args, **kwargs: mock_llm)

    state: AgentState = {
        "query": "Q",
        "original_query": "Q",
        "rewritten_query": "Q",
        "conversation_history": [],
        "classification": "simple_lookup",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "Draft 1",
        "citations": [],
        "critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_answer": "",
        "final_grounded": False,
        "unsupported_claims_final": [],
        "is_decline": False,
    }

    # Pass 1
    s1 = critique(state)
    assert s1["first_critic_verdict"]["grounded"] is False
    assert s1["first_critic_verdict"]["unsupported_claims"] == ["Ungrounded claim 1."]

    # Pass 2: model now returns grounded=True
    mock_llm.return_value = CriticVerdictOutput(grounded=True, unsupported_claims=[])
    s1["draft_answer"] = "Draft 2 revised"
    s1["retry_count"] = 1
    s2 = critique(s1)

    # Current verdict is updated to grounded=True
    assert s2["critic_verdict"]["grounded"] is True
    assert s2["critic_verdict"]["unsupported_claims"] == []
    # But first_critic_verdict remains the initial pass 1 verdict
    assert s2["first_critic_verdict"]["grounded"] is False
    assert s2["first_critic_verdict"]["unsupported_claims"] == ["Ungrounded claim 1."]


def test_model_env_vars_and_defaults(monkeypatch):
    """
    Asserts:
    1. Setting GEN_MODEL changes the model the draft node uses (fallback MODEL_DRAFT).
    2. Setting CRITIC_MODEL changes the model the critic node uses (fallback MODEL_CRITIC).
    3. Code defaults are draft: gemini-3.5-flash-lite, critic: gemini-3.1-flash-lite.
    """
    from unittest.mock import MagicMock
    from backend.agent.nodes.critic import get_critic_model
    from backend.agent.nodes.draft import draft_answer, get_draft_model

    # 1. Defaults when unset
    monkeypatch.delenv("GEN_MODEL", raising=False)
    monkeypatch.delenv("MODEL_DRAFT", raising=False)
    monkeypatch.delenv("CRITIC_MODEL", raising=False)
    monkeypatch.delenv("MODEL_CRITIC", raising=False)

    assert get_draft_model() == "gemini-3.5-flash-lite"
    assert get_critic_model() == "gemini-3.1-flash-lite"

    # 2. Setting GEN_MODEL overrides default and changes model passed to get_llm_with_fallback
    monkeypatch.setenv("GEN_MODEL", "custom-gen-model-v1")
    assert get_draft_model() == "custom-gen-model-v1"

    captured_draft_models = []
    class MockDraftLLM:
        def invoke(self, messages):
            return MagicMock(content="Draft answer [1]")

    def mock_get_llm_draft(primary_model, *args, **kwargs):
        captured_draft_models.append(primary_model)
        return MockDraftLLM()

    monkeypatch.setattr("backend.agent.nodes.draft.get_llm_with_fallback", mock_get_llm_draft)

    state_draft: AgentState = {
        "query": "Test query",
        "retrieved_chunks": [{"id": "1", "text": "chunk text"}],
    }
    draft_answer(state_draft)
    assert captured_draft_models == ["custom-gen-model-v1"]

    # 3. Setting CRITIC_MODEL overrides default and changes model passed to get_llm_with_fallback
    monkeypatch.setenv("CRITIC_MODEL", "custom-critic-model-v2")
    assert get_critic_model() == "custom-critic-model-v2"

    captured_critic_models = []
    class MockCriticLLM:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return CriticVerdictOutput(grounded=True, unsupported_claims=[])

    def mock_get_llm_critic(primary_model, *args, **kwargs):
        captured_critic_models.append(primary_model)
        return MockCriticLLM()

    monkeypatch.setattr("backend.agent.nodes.critic.get_llm_with_fallback", mock_get_llm_critic)

    state_critic: AgentState = {
        "draft_answer": "Draft answer",
        "retrieved_chunks": [{"id": "1", "text": "chunk text"}],
    }
    critique(state_critic)
    assert captured_critic_models == ["custom-critic-model-v2"]

    # 4. Fallback when primary env var is unset
    monkeypatch.delenv("GEN_MODEL", raising=False)
    monkeypatch.setenv("MODEL_DRAFT", "fallback-draft-model")
    assert get_draft_model() == "fallback-draft-model"

    monkeypatch.delenv("CRITIC_MODEL", raising=False)
    monkeypatch.setenv("MODEL_CRITIC", "fallback-critic-model")
    assert get_critic_model() == "fallback-critic-model"

