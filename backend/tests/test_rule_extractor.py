"""
Unit tests for rule-based metadata extractor and classifier bypass (no network).

Tests:
1. Extraction of canonical company names from tickers and aliases.
2. Extraction of fiscal years from varied formatting ("fiscal year 2025", "FY2025", "FY 2024", "FY'23", "2024").
3. Ambiguity detection on multi-company and multi-year queries.
4. Correct fallback when nothing is found.
5. Inferred query classification ('simple_lookup', 'comparison', 'numeric').
6. LLM call bypass when extraction succeeds unambiguously vs invocation when ambiguous.
"""

import pytest
from unittest.mock import MagicMock

from backend.agent.nodes.classifier import (
    ClassificationOutput,
    classify,
    extract_metadata_rules,
)
from backend.agent.state import AgentState


def test_rule_extractor_canonical_companies():
    # Apple
    c, y, intent, ambig = extract_metadata_rules("What was Apple's total revenue in fiscal 2024?")
    assert c == "Apple"
    assert y == "2024"
    assert not ambig

    c, _, _, _ = extract_metadata_rules("According to AAPL 2024 10-K filing...")
    assert c == "Apple"

    # Microsoft
    c, y, _, _ = extract_metadata_rules("What was Microsoft's cloud revenue in FY2025?")
    assert c == "Microsoft"
    assert y == "2025"

    c, _, _, _ = extract_metadata_rules("Explain MSFT operating margins in 2023")
    assert c == "Microsoft"

    # Alphabet / Google
    c, _, _, _ = extract_metadata_rules("What did Google report for advertising revenue in 2024?")
    assert c == "Alphabet"

    c, _, _, _ = extract_metadata_rules("Review Alphabet 10-K for FY 2023")
    assert c == "Alphabet"

    c, _, _, _ = extract_metadata_rules("According to GOOGL disclosures in 2024...")
    assert c == "Alphabet"

    # Amazon
    c, _, _, _ = extract_metadata_rules("What was Amazon's net income in fiscal year 2024?")
    assert c == "Amazon"

    c, _, _, _ = extract_metadata_rules("What were AMZN AWS segment revenues in 2024?")
    assert c == "Amazon"

    # NVIDIA
    c, y, _, _ = extract_metadata_rules("What was NVIDIA's Data Center revenue in FY2024?")
    assert c == "NVIDIA"
    assert y == "2024"

    c, _, _, _ = extract_metadata_rules("Check NVDA automotive growth in 2024")
    assert c == "NVIDIA"


def test_rule_extractor_fiscal_year_patterns():
    # "fiscal year 2025"
    _, y, _, _ = extract_metadata_rules("Apple fiscal year 2025 revenue")
    assert y == "2025"

    # "FY2025"
    _, y, _, _ = extract_metadata_rules("Microsoft FY2025 capital expenditures")
    assert y == "2025"

    # "FY 2024"
    _, y, _, _ = extract_metadata_rules("NVIDIA FY 2024 data center revenue")
    assert y == "2024"

    # "FY'24"
    _, y, _, _ = extract_metadata_rules("Amazon FY'24 operating cash flows")
    assert y == "2024"

    # Standalone "2024"
    _, y, _, _ = extract_metadata_rules("What were Apple net sales in 2024?")
    assert y == "2024"


def test_rule_extractor_ambiguity_detection():
    # Multiple companies -> ambiguous
    c, y, intent, ambig = extract_metadata_rules("Compare Apple and Microsoft revenue growth in 2024")
    assert ambig is True
    assert intent == "comparison"

    # Multiple years -> ambiguous
    c, y, intent, ambig = extract_metadata_rules("How did Amazon revenue change from 2023 to 2024?")
    assert ambig is True
    assert intent == "comparison"


def test_rule_extractor_finds_nothing():
    # Neither company nor fiscal year
    c, y, intent, ambig = extract_metadata_rules("What are the key risk factors in cloud computing?")
    assert c is None
    assert y is None
    assert not ambig


def test_rule_extractor_intent_inference():
    # Comparison
    _, _, intent, _ = extract_metadata_rules("Compare Google and Microsoft cloud market share")
    assert intent == "comparison"

    # Numeric
    _, _, intent, _ = extract_metadata_rules("What was Apple's gross margin percentage in fiscal year 2024?")
    assert intent == "numeric"

    # Simple lookup
    _, _, intent, _ = extract_metadata_rules("What subscription-based services are listed in Apple's 10-K for 2024?")
    assert intent == "simple_lookup"


def test_classify_node_bypasses_llm_when_unambiguous(monkeypatch):
    """
    Ensures that when rule-based extraction succeeds, get_llm_with_fallback is NOT called.
    """
    def fail_if_llm_called(*args, **kwargs):
        pytest.fail("LLM was invoked but should have been bypassed by the rule-based extractor!")

    monkeypatch.setattr("backend.agent.nodes.classifier.get_llm_with_fallback", fail_if_llm_called)

    state: AgentState = {
        "query": "According to Apple's fiscal year 2025 Form 10-K, what subscription services are listed?",
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "first_critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_grounded": False,
        "unsupported_claims_final": [],
        "final_answer": "",
    }

    result_state = classify(state)

    assert result_state["classification"] == "simple_lookup"
    assert result_state["extracted_filters"]["company"] == "Apple"
    assert result_state["extracted_filters"]["fiscal_year"] == "2025"


def test_classify_node_calls_llm_when_ambiguous_or_nothing_found(monkeypatch):
    """
    Ensures that when rule-based extraction is ambiguous, the structured LLM is called as fallback.
    """
    llm_called = []

    class MockStructuredLLM:
        def invoke(self, messages):
            llm_called.append(True)
            return ClassificationOutput(
                classification="comparison",
                company=None,
                fiscal_year=None,
            )

    class MockLLMWrapper:
        def with_structured_output(self, schema, **kwargs):
            return MockStructuredLLM()

    monkeypatch.setattr(
        "backend.agent.nodes.classifier.get_llm_with_fallback",
        lambda *args, **kwargs: MockLLMWrapper(),
    )

    state: AgentState = {
        "query": "Compare Apple and Microsoft AI strategies in 2023 and 2024",
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "first_critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_grounded": False,
        "unsupported_claims_final": [],
        "final_answer": "",
    }

    result_state = classify(state)

    assert len(llm_called) == 1
    assert result_state["classification"] == "comparison"


def test_classifier_bypass_specific_scenarios(monkeypatch):
    """
    Verifies the specific bypass behavior:
    1. "Tesla FY2023 revenue" -> out-of-corpus company: not bypassed (LLM called).
    2. "Apple FY2025 ..." -> company found: bypassed with company "Apple" and fiscal_year "2025" (str).
    3. "Apple and Microsoft in 2024" -> ambiguous: not bypassed (LLM called).
    """
    llm_calls = []

    class MockStructuredLLM:
        def invoke(self, messages):
            llm_calls.append(messages)
            return ClassificationOutput(
                classification="simple_lookup",
                company=None,
                fiscal_year="2023",
            )

    class MockLLMWrapper:
        def with_structured_output(self, schema, **kwargs):
            return MockStructuredLLM()

    monkeypatch.setattr(
        "backend.agent.nodes.classifier.get_llm_with_fallback",
        lambda *args, **kwargs: MockLLMWrapper(),
    )

    # 1. Tesla FY2023 revenue: out-of-corpus company, year only -> NOT bypassed
    state1: AgentState = {
        "query": "Tesla FY2023 revenue",
        "classification": "",
        "extracted_filters": {},
    }
    classify(state1)
    assert len(llm_calls) == 1

    # 2. Apple FY2025 ... -> company found -> BYPASSED
    state2: AgentState = {
        "query": "Apple FY2025 total net sales and operating income",
        "classification": "",
        "extracted_filters": {},
    }
    res2 = classify(state2)
    assert len(llm_calls) == 1  # No new LLM call
    assert res2["extracted_filters"]["company"] == "Apple"
    assert res2["extracted_filters"]["fiscal_year"] == "2025"
    assert isinstance(res2["extracted_filters"]["fiscal_year"], str)

    # 3. Apple and Microsoft in 2024 -> ambiguous -> NOT bypassed (LLM called)
    state3: AgentState = {
        "query": "Apple and Microsoft in 2024",
        "classification": "",
        "extracted_filters": {},
    }
    classify(state3)
    assert len(llm_calls) == 2

