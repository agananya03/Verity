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
   - "fiscal_year": Extract the single explicit 4-digit fiscal year mentioned (e.g., "2024", "2023"), or null if unmentioned or multiple.
   - "fiscal_years": Extract all explicit 4-digit fiscal years mentioned as a list (e.g., ["2023", "2024"]), or empty if none.

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
        description="Extracted single explicit 4-digit fiscal year (e.g., '2024', '2023'), or None if unmentioned or multiple."
    )
    fiscal_years: List[str] = Field(
        default_factory=list,
        description="List of all explicit 4-digit fiscal years mentioned (e.g., ['2023', '2024']), or empty if unmentioned."
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

    # Also check standalone 4-digit years in valid range (2015-2035)
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


def extract_company_and_year(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts canonical company and single fiscal year from query using deterministic rules.
    Returns (company, fiscal_year). For multi-year or ambiguous queries, fiscal_year is None.
    """
    company, fiscal_year, _, is_ambiguous = extract_metadata_rules(query)
    if is_ambiguous:
        return company, None
    return company, fiscal_year


def extract_all_fiscal_years(query: str) -> List[str]:
    """
    Extracts all fiscal years mentioned in the query in order of appearance.
    Returns a deduplicated list of 4-digit year strings, e.g. ['2023', '2024'].
    """
    if not query or not query.strip():
        return []

    matches: List[Tuple[int, str]] = []

    for m in _FY_PREFIX_REGEX.finditer(query):
        raw_year = m.group(1)
        year_val = f"20{raw_year}" if len(raw_year) == 2 else raw_year
        matches.append((m.start(), year_val))

    for m in _STANDALONE_YEAR_REGEX.finditer(query):
        matches.append((m.start(), m.group(1)))

    matches.sort(key=lambda x: x[0])

    years: List[str] = []
    seen: Set[str] = set()
    for _, yr in matches:
        if yr not in seen:
            seen.add(yr)
            years.append(yr)

    return years


def extract_company_and_years(query: str) -> Tuple[Optional[str], List[str]]:
    """
    Extracts canonical company and list of all fiscal years mentioned in the query.
    Returns (company, fiscal_years). For multi-year queries, fiscal_years contains
    all detected years in order of appearance.
    """
    company, _ = extract_company_and_year(query)
    years = extract_all_fiscal_years(query)
    return company, years


def strip_query_metadata(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> str:
    """clean_v1: Alias for strip_query_metadata_v1 for backward compatibility."""
    return strip_query_metadata_v1(query, company=company, fiscal_year=fiscal_year)


def strip_query_metadata_v1(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> str:
    """
    clean_v1:
    Strips preamble boilerplate and metadata terms from a query that are already
    enforced by retrieval filters, including company name.
    """
    if not query or not query.strip():
        return ""

    q = query.strip()

    # 1. Preamble removal
    if fiscal_year and "/" not in str(fiscal_year):
        q = re.sub(
            r"^\s*(?:According\s+to|Based\s+on|In|Per)\s+(?:the\s+)?(?:(?:Item\s+[0-9A-Z]+|[A-Za-z]+)\s+section\s+of\s+)?(?:[A-Za-z0-9'’\s-]*?)(?:Form\s+)?10-K(?:\s+filings?)?\s*,\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
    else:
        q = re.sub(
            r"^\s*Based\s+on\s+(?:[A-Za-z0-9'’\s-]*?)(?:Form\s+)?10-K(?:\s+filings?)?\s*,\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )

    # 2. Section references
    q = re.sub(r"\bthe\s+Item\s+[0-9A-Z]+\s+section\s+of\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bthe\s+[A-Za-z]+\s+section\s+of\b", "", q, flags=re.IGNORECASE)

    # 3. Form 10-K references
    q = re.sub(r"\b(?:Form\s+)?10-K(?:\s+filings?)?\b", "", q, flags=re.IGNORECASE)

    # 4. Fiscal year removal: ONLY when single fiscal_year filter is active
    if fiscal_year and "/" not in str(fiscal_year):
        fy_str = str(fiscal_year)
        q = re.sub(r"\bduring\s+fiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b,?", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bin\s+fiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b,?", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfy\s*'?" + re.escape(fy_str) + r"\b", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\b" + re.escape(fy_str) + r"\b", "", q)

    # 5. Company removal: when company filter is active
    if company:
        comp_pattern = r"\b" + re.escape(company) + r"(?:'s|’s)?\b"
        q = re.sub(comp_pattern, "", q, flags=re.IGNORECASE)

    # 6. Clean punctuation & whitespace artifacts
    q = re.sub(r"\s*,\s*", ", ", q)
    q = re.sub(r"(?:,\s*)+", ", ", q)
    q = re.sub(r"^\s*,\s*", "", q)
    q = re.sub(r"\s+", " ", q).strip()

    if q and q[0].islower():
        q = q[0].upper() + q[1:]

    return q


def strip_query_metadata_v2(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> str:
    """
    clean_v2:
    Removes ONLY:
    1. The leading preamble ("According to / Based on / In / Per ... Form 10-K,")
    2. "Form 10-K" and "10-K filing(s)"
    3. "fiscal year <yyyy>" / "in fiscal <yyyy>" / "during fiscal <yyyy>" phrases
       when a single-year filter is active (preserves years for multi-year queries)
    4. "the Item N section of" or "the <X> section of" clauses
    
    MUST NOT remove the company name from the body of the question (so
    product names like "Apple Vision Pro", "Apple Watch", "Apple Card" survive).
    """
    if not query or not query.strip():
        return ""

    q = query.strip()

    # 1. Leading preamble removal ("According to / Based on / In / Per ... Form 10-K,")
    q = re.sub(
        r"^\s*(?:According\s+to|Based\s+on|In|Per)\s+(?:the\s+)?(?:(?:Item\s+[0-9A-Z]+|[A-Za-z]+)\s+section\s+of\s+)?(?:[A-Za-z0-9'’\s-]*?)(?:Form\s+)?10-K(?:\s+filings?)?\s*,\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )

    # 2. Section references: "the Item N section of", "the <X> section of"
    q = re.sub(r"\bthe\s+Item\s+[0-9A-Z]+\s+section\s+of\b\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bthe\s+[A-Za-z]+\s+section\s+of\b\s*", "", q, flags=re.IGNORECASE)

    # 3. Form 10-K references
    q = re.sub(r"\b(?:Form\s+)?10-K(?:\s+filings?)?\b", "", q, flags=re.IGNORECASE)

    # 4. Fiscal year removal: ONLY when single fiscal_year filter is active
    if fiscal_year and "/" not in str(fiscal_year):
        fy_str = str(fiscal_year)
        q = re.sub(r"\bduring\s+fiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b,?", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bin\s+fiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b,?", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfiscal\s+(?:year\s+)?'?" + re.escape(fy_str) + r"\b", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfy\s*'?" + re.escape(fy_str) + r"\b", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\b" + re.escape(fy_str) + r"\b", "", q)

    # NOTE: In clean_v2, the company name is NEVER removed from the body!

    # 5. Clean punctuation & whitespace artifacts
    q = re.sub(r"\s*,\s*", ", ", q)
    q = re.sub(r"(?:,\s*)+", ", ", q)
    q = re.sub(r"^\s*,\s*", "", q)
    q = re.sub(r"\s+", " ", q).strip()

    if q and q[0].islower():
        q = q[0].upper() + q[1:]

    return q


def strip_query_metadata_v3(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> str:
    """
    clean_v3:
    Removes ONLY:
    1. Leading preamble ("According to / Based on / In / Per ... Form 10-K,")
    2. "Form 10-K" and "10-K" references
    3. Disambiguation clauses like "per {Company}'s fiscal {year} annual report",
       "in {Company}'s FY{year} 10-K", "for fiscal {year}"
    4. Phrases like "fiscal year 2024", "FY2024", "fiscal 2024"

    MUST NOT remove:
    - Company name (e.g. "Alphabet", "Amazon", "Apple", "Microsoft", "NVIDIA")
    - Calendar dates with years (e.g. "June 30, 2024", "December 31, 2023", "January 29, 2025")
    """
    if not query or not query.strip():
        return ""

    q = query.strip()

    # 1. Leading preamble removal ("According to / Based on / In / Per ... Form 10-K,")
    q = re.sub(
        r"^\s*(?:According\s+to|Based\s+on|In|Per)\s+(?:the\s+)?(?:(?:Item\s+[0-9A-Z]+|[A-Za-z]+)\s+section\s+of\s+)?(?:[A-Za-z0-9'’\s-]*?)(?:Form\s+)?10-K(?:\s+filings?)?\s*,\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )

    # 2. Form 10-K references
    q = re.sub(r"\b(?:Form\s+)?10-K(?:\s+filings?)?\b", "", q, flags=re.IGNORECASE)

    # 3. Disambiguation phrases added during review
    q = re.sub(r",?\s*per\s+[A-Za-z0-9'’\s-]+?'s\s+fiscal\s+\d{4}\s+annual\s+report\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*in\s+[A-Za-z0-9'’\s-]+?'s\s+FY\s*'?\d{2,4}(?:\s+10-K)?\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*in\s+[A-Za-z0-9'’\s-]+?'s\s+fiscal\s+\d{4}\s+filing\b", "", q, flags=re.IGNORECASE)

    # 4. Phrases like "fiscal year 2024", "fiscal 2024", "FY2024", "FY 2024"
    # Note: does NOT match bare year like "2024" in "June 30, 2024"
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*fiscal\s+year\s+\d{4}\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*fiscal\s+\d{4}\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*FY\s*'?\d{2,4}\b", "", q, flags=re.IGNORECASE)

    # Whitespace and punctuation cleanup
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\s+([,?.])", r"\1", q)
    q = re.sub(r",\s*\?", "?", q)
    q = re.sub(r",\s*$", "", q)
    return q


def classify(state: AgentState) -> AgentState:
    """
    Classifies query intent and extracts metadata filters.
    Bypasses LLM call if the rule-based extractor finds an unambiguous match.
    Calls Gemini structured output only when nothing is found or when the query is ambiguous.
    """
    query = state.get("query", "")
    all_years = extract_all_fiscal_years(query)

    # Step 1: Attempt fast rule-based extraction
    company, fiscal_year, inferred_intent, is_ambiguous = extract_metadata_rules(query)

    # Case 1A: Unambiguous single-company & single-year (or single company zero-year)
    if not is_ambiguous and company is not None:
        extracted_filters: Dict[str, Any] = state.get("extracted_filters") or {}
        extracted_filters["company"] = company
        if fiscal_year:
            extracted_filters["fiscal_year"] = str(fiscal_year)
            extracted_filters["fiscal_years"] = [str(fiscal_year)]

        state["classification"] = inferred_intent or "simple_lookup"
        state["extracted_filters"] = extracted_filters
        logger.info(
            f"Rule-based classifier match: company={company}, fiscal_year={fiscal_year}, "
            f"classification={state['classification']} (bypassed LLM call)"
        )
        return state

    # Case 1B: Single company with multiple years detected (multi-year query)
    # Extracts company and all years without requiring an expensive LLM call
    if company is not None and len(all_years) >= 2:
        extracted_filters = state.get("extracted_filters") or {}
        extracted_filters["company"] = company
        extracted_filters["fiscal_years"] = all_years
        state["classification"] = inferred_intent or "comparison"
        state["extracted_filters"] = extracted_filters
        logger.info(
            f"Rule-based multi-year classifier match: company={company}, fiscal_years={all_years}, "
            f"classification={state['classification']} (bypassed LLM call)"
        )
        return state

    # Step 2: Fall back to Gemini structured LLM when nothing found or ambiguous across companies
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

    # Use LLM returned years (or fallback to extract_all_fiscal_years on query)
    years = result.fiscal_years or []
    if not years and result.fiscal_year:
        years = [result.fiscal_year]
    if not years:
        years = all_years

    if len(years) >= 2:
        extracted_filters["fiscal_years"] = years
        extracted_filters.pop("fiscal_year", None)
    elif len(years) == 1:
        extracted_filters["fiscal_year"] = str(years[0])
        extracted_filters["fiscal_years"] = [str(years[0])]

    state["classification"] = result.classification
    state["extracted_filters"] = extracted_filters

    return state
