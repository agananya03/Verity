"""
Classifier Node for Verity's Agentic LangGraph Workflow.

Role:
- Analyzes incoming financial queries submitted by the user.
- Extracts metadata entities (such as company name and fiscal year) into state['extracted_filters'].
- Classifies query intent into 'simple_lookup', 'comparison', or 'numeric'.
- Uses a fast, deterministic rule-based extractor first to bypass LLM calls when unambiguous,
  calling Gemini structured output only when nothing is found or when the query is ambiguous.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from backend.agent.llm_client import get_llm_with_fallback
from backend.agent.state import AgentState

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# System & Classification Prompt for LLM fallback
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
MODEL_CLASSIFIER = os.getenv(
    "CLASSIFIER_MODEL", os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite")
)

# Canonical company dictionary with aliases & tickers
COMPANY_ALIASES: Dict[str, List[str]] = {
    "Apple": ["apple", "aapl"],
    "Microsoft": ["microsoft", "msft"],
    "Alphabet": ["alphabet", "google", "googl", "goog"],
    "Amazon": ["amazon", "amzn", "aws"],
    "NVIDIA": ["nvidia", "nvda"],
}

# Regex pattern for fiscal year variants
# Matches: "fiscal year 2025", "fiscal 2025", "FY2025", "FY 2025", "FY'25", "FY25", or standalone "2024"
_FY_PREFIX_REGEX = re.compile(
    r"\b(?:fiscal\s+year|fiscal\s+yr|fiscal|fy)\s*['’]?\s*(20\d{2}|\d{2})\b",
    re.IGNORECASE,
)
_STANDALONE_YEAR_REGEX = re.compile(r"\b(20[1-3]\d)\b")
_COMPARISON_REGEX = re.compile(r"\b(compare|comparison|versus|vs\.?|difference|differ|growth\s+from)\b", re.IGNORECASE)
_NUMERIC_REGEX = re.compile(
    r"\b(percentage|percent|%|ratio|margin|calculate|reconcil|sum|total\s+shares|dilution)\b",
    re.IGNORECASE,
)


def extract_metadata_rules(query: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Extracts company name and fiscal year from query using deterministic rules.

    Returns:
        (company, fiscal_year, inferred_classification, is_ambiguous)
        where:
          - company is the canonical company name if exactly one is detected.
          - fiscal_year is the 4-digit string if exactly one year is detected.
          - inferred_classification is 'comparison', 'numeric', or 'simple_lookup'.
          - is_ambiguous is True if multiple companies or multiple fiscal years are found.
    """
    if not query or not query.strip():
        return None, None, None, True

    q_lower = query.lower()

    # 1. Company Detection
    detected_companies: Set[str] = set()
    for canonical, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, q_lower):
                detected_companies.add(canonical)
                break

    # 2. Fiscal Year Detection
    detected_years: Set[str] = set()
    for match in _FY_PREFIX_REGEX.finditer(query):
        raw_year = match.group(1)
        if len(raw_year) == 2:
            year_val = f"20{raw_year}"
        else:
            year_val = raw_year
        detected_years.add(year_val)

    # If no prefix match, check standalone 4-digit years in valid range (2015-2035)
    if not detected_years:
        for match in _STANDALONE_YEAR_REGEX.finditer(query):
            detected_years.add(match.group(1))

    # 3. Ambiguity Check
    is_ambiguous = len(detected_companies) > 1 or len(detected_years) > 1

    company = next(iter(detected_companies)) if len(detected_companies) == 1 else None
    fiscal_year = next(iter(detected_years)) if len(detected_years) == 1 else None

    # 4. Infer classification
    if _COMPARISON_REGEX.search(query) or len(detected_companies) > 1 or len(detected_years) > 1:
        inferred_classification = "comparison"
    elif _NUMERIC_REGEX.search(query):
        inferred_classification = "numeric"
    else:
        inferred_classification = "simple_lookup"

    return company, fiscal_year, inferred_classification, is_ambiguous


def classify(state: AgentState) -> AgentState:
    """
    Classifies query intent and extracts metadata filters.
    Bypasses LLM call if the rule-based extractor finds an unambiguous match.
    Calls Gemini structured output only when nothing is found or when the query is ambiguous.
    """
    query = state.get("query", "")

    # Step 1: Attempt fast rule-based extraction
    company, fiscal_year, inferred_intent, is_ambiguous = extract_metadata_rules(query)

    # If unambiguous and found a company, bypass LLM
    # (Bypassing requires a company so out-of-corpus entities like Tesla never receive year-only filters)
    if not is_ambiguous and company is not None:
        extracted_filters: Dict[str, str] = state.get("extracted_filters") or {}
        extracted_filters["company"] = company
        if fiscal_year:
            extracted_filters["fiscal_year"] = str(fiscal_year)

        state["classification"] = inferred_intent or "simple_lookup"
        state["extracted_filters"] = extracted_filters
        logger.info(
            f"Rule-based classifier match: company={company}, fiscal_year={fiscal_year}, "
            f"classification={state['classification']} (bypassed LLM call)"
        )
        return state

    # Step 2: Fall back to Gemini structured LLM when nothing found or ambiguous
    logger.info(
        f"Rule-based classifier inconclusive (found_nothing={company is None and fiscal_year is None}, "
        f"is_ambiguous={is_ambiguous}). Invoking LLM classifier..."
    )
    api_key = os.getenv("GOOGLE_API_KEY")

    classifier_model = os.getenv(
        "CLASSIFIER_MODEL", os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite")
    )
    llm = get_llm_with_fallback(
        primary_model=classifier_model,
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

    extracted_filters = state.get("extracted_filters") or {}
    if result.company:
        extracted_filters["company"] = result.company
    if result.fiscal_year:
        extracted_filters["fiscal_year"] = str(result.fiscal_year)

    state["classification"] = result.classification
    state["extracted_filters"] = extracted_filters

    return state
