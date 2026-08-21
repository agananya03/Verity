"""
Classifier Node for Verity's Agentic LangGraph Workflow.

Role:
- Analyzes the incoming financial query submitted by the user.
- Classifies query intent into one of: 'simple_lookup', 'comparison', or 'numeric'.
- Extracts metadata entities (such as company name and fiscal year) into state['extracted_filters'].
- Uses Gemini via ChatGoogleGenerativeAI with Pydantic structured output.
"""

import os
from pathlib import Path
from typing import Dict, Literal, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from backend.agent.llm_client import get_llm_with_fallback
from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# System & Classification Prompt
CLASSIFIER_PROMPT = """You are an expert financial query analyzer for SEC 10-K filing retrieval.

Your task is to:
1. Classify the user's financial query into exactly one of three categories:
   - "simple_lookup": Query asks for a single factual piece of information, metric, or disclosure for a specific company/year (e.g., "What was Apple's R&D expense in 2024?").
   - "comparison": Query asks to compare financial performance, metrics, or risks across multiple companies, years, or segments (e.g., "Compare Microsoft and Apple revenue growth from 2023 to 2024").
   - "numeric": Query requires mathematical calculation, balance sheet reconciliation, ratio computation, or tabular data analysis.

2. Extract explicit metadata filters:
   - "company": Extract the primary target company name mentioned (e.g., "Apple", "Microsoft", "Amazon") if a single company is targeted, otherwise null.
   - "fiscal_year": Extract the explicit 4-digit fiscal year mentioned (e.g., "2024", "2023"), otherwise null.

Analyze the query carefully and output structured JSON matching the requested schema.
"""


class ClassificationOutput(BaseModel):
    """Pydantic model for structured query classification output."""
    classification: Literal["simple_lookup", "comparison", "numeric"] = Field(
        description="Category of the query: 'simple_lookup', 'comparison', or 'numeric'."
    )
    company: Optional[str] = Field(
        default=None,
        description="Extracted target company name (e.g., 'Apple', 'Microsoft'), or None if unmentioned or multiple."
    )
    fiscal_year: Optional[str] = Field(
        default=None,
        description="Extracted explicit 4-digit fiscal year (e.g., '2024', '2023'), or None if unmentioned."
    )


# Model Configuration
# Classifier node only performs structured classification (routing + filter extraction),
# so the lighter model is sufficient and preserves free-tier quota for the draft node.
MODEL_CLASSIFIER = os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite")


def classify(state: AgentState) -> AgentState:
    """
    Classifies query intent and extracts metadata filters using Gemini structured output.
    Updates state['classification'] and state['extracted_filters'].
    """
    query = state.get("query", "")
    api_key = os.getenv("GOOGLE_API_KEY")

    llm = get_llm_with_fallback(
        primary_model=MODEL_CLASSIFIER,
        fallback_models=[],
        google_api_key=api_key or None,
        temperature=0.0,
        request_timeout=30.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(ClassificationOutput)

    messages = [
        ("system", CLASSIFIER_PROMPT),
        ("user", f"Query: {query}"),
    ]

    result: ClassificationOutput = structured_llm.invoke(messages)

    extracted_filters: Dict[str, str] = state.get("extracted_filters") or {}
    if result.company:
        extracted_filters["company"] = result.company
    if result.fiscal_year:
        extracted_filters["fiscal_year"] = str(result.fiscal_year)

    state["classification"] = result.classification
    state["extracted_filters"] = extracted_filters

    return state
