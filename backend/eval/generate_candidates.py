"""
Held-Out Benchmark Candidate Generation and Validation Script.

Builds a held-out benchmark candidate pool kept strictly separate from the dev set.
1. Samples ~90 chunks from backend/data/processed/chunks.json, stratified across:
   - All 5 companies (Apple, Microsoft, Alphabet, Amazon, NVIDIA)
   - All fiscal years present
   - Items 1, 1A, 7, 8 (including table-heavy Item 7/8 chunks)
   - Strictly excludes any chunks in backend/eval/golden_set.json (dev set).
2. Generates ~12 multi-year comparison question pairs (same company, two fiscal years).
3. Generates ~12 plausible unanswerable questions (out-of-corpus companies, years, or non-10-K facts; 0 LLM calls).
4. Uses BENCH_MODEL (gemini-3.5-flash-lite) through QuotaLedger (default max 110 calls, honoring ledger stops).
5. Validates candidates:
   - supporting_quote must be an exact substring of chunk text
   - Rejects questions with 6-gram overlap with chunk text
   - Rejects duplicate questions
   - Cross-filing uniqueness: compares supporting_quote against chunks of the same company in other fiscal years.
     If it repeats verbatim in another year without the question explicitly naming its year, flags AMBIGUOUS_REPEATED_ACROSS_YEARS
     and sets default review status to reject.
6. Outputs:
   - Dry run writes strictly to candidates_dryrun.json and review_dryrun.csv.
   - Real run writes to candidates.json and review.csv.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Project root path setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

from backend.eval.quota_ledger import QuotaLedger, global_quota_ledger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DEFAULT_DEV_SET_PATH = BASE_DIR / "eval" / "golden_set.json"
DEFAULT_HELDOUT_SET_PATH = BASE_DIR / "eval" / "golden_heldout.json"

DEFAULT_OUTPUT_JSON = BASE_DIR / "eval" / "results" / "candidates.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "eval" / "results" / "review.csv"
DEFAULT_DRYRUN_JSON = BASE_DIR / "eval" / "results" / "candidates_dryrun.json"
DEFAULT_DRYRUN_CSV = BASE_DIR / "eval" / "results" / "review_dryrun.csv"

DEFAULT_ROUND2_OUTPUT_JSON = BASE_DIR / "eval" / "golden_round2_candidates.json"
DEFAULT_ROUND2_OUTPUT_CSV = BASE_DIR / "eval" / "golden_round2_review.csv"
DEFAULT_ROUND2_DRYRUN_JSON = BASE_DIR / "eval" / "golden_round2_candidates_dryrun.json"
DEFAULT_ROUND2_DRYRUN_CSV = BASE_DIR / "eval" / "golden_round2_review_dryrun.csv"

# Unanswerable question templates (out-of-corpus companies, years, or non-10-K facts; cost 0 LLM calls)
UNANSWERABLE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "company": "Tesla",
        "fiscal_year": "2022",
        "category": "unanswerable",
        "question": "What was Tesla's total automotive regulatory credit revenue in fiscal year 2022?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Uber",
        "fiscal_year": "2024",
        "category": "unanswerable",
        "question": "What was Uber's Mobility gross bookings figure for full fiscal year 2024?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Meta",
        "fiscal_year": "2021",
        "category": "unanswerable",
        "question": "How many daily active users (DAUs) did Meta report across its Family of Apps in fiscal year 2021?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Netflix",
        "fiscal_year": "2023",
        "category": "unanswerable",
        "question": "What was Netflix's average monthly revenue per paying membership in the UCAN region in fiscal 2023?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Spotify",
        "fiscal_year": "2024",
        "category": "unanswerable",
        "question": "What was Spotify's ad-supported monthly active user count at the end of fiscal year 2024?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Airbnb",
        "fiscal_year": "2023",
        "category": "unanswerable",
        "question": "What was Airbnb's total Nights and Seats Booked volume reported in fiscal year 2023?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Salesforce",
        "fiscal_year": "2023",
        "category": "unanswerable",
        "question": "What was Salesforce's total subscription and support revenue for Service Cloud in fiscal year 2023?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Intel",
        "fiscal_year": "2022",
        "category": "unanswerable",
        "question": "What was Intel Foundry Services (IFS) operating income in fiscal year 2022?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Oracle",
        "fiscal_year": "2023",
        "category": "unanswerable",
        "question": "What was Oracle Cloud Infrastructure (OCI) consumption revenue growth reported in fiscal year 2023?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Cisco",
        "fiscal_year": "2024",
        "category": "unanswerable",
        "question": "What was Cisco's total Secure, Agile Networks revenue reported in fiscal year 2024?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Apple",
        "fiscal_year": "2020",
        "category": "unanswerable",
        "question": "What was Apple's total research and development expense in fiscal year 2020?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "NVIDIA",
        "fiscal_year": "2021",
        "category": "unanswerable",
        "question": "What was NVIDIA's professional visualization revenue in fiscal year 2021?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Amazon",
        "fiscal_year": "2021",
        "category": "unanswerable",
        "question": "What were Amazon's net sales from physical stores reported in fiscal year 2021?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Amazon",
        "fiscal_year": "2024",
        "category": "unanswerable",
        "question": "What was the intraday high price of Amazon common stock on October 14, 2024?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
    {
        "company": "Microsoft",
        "fiscal_year": "2025",
        "category": "unanswerable",
        "question": "What is the secret internal hardware architecture codename for Microsoft's 2028 cloud AI chips?",
        "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
    },
]


# =============================================================================
# Validation Logic
# =============================================================================
def normalize_text_for_matching(text: str) -> str:
    """Normalizes spaces and typographical quotes for robust substring alignment."""
    t = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip()


def validate_supporting_quote(quote: str, chunk_text: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates that supporting_quote is an exact substring of chunk_text.
    Rejects any quote containing '...' or '…' unless that exact text appears in the chunk.
    If exact match fails but normalized whitespace matches, extracts the exact verbatim slice.
    Returns (is_valid, reason, verbatim_matched_quote).
    """
    if not quote or len(quote.strip()) < 10:
        return False, "REJECT_QUOTE_EMPTY_OR_TOO_SHORT", None

    # Ellipsis check: reject if quote contains '...' or '…' not present in chunk_text
    if "..." in quote and "..." not in chunk_text:
        return False, "REJECT_QUOTE_CONTAINS_ELLIPSIS", None
    if "…" in quote and "…" not in chunk_text:
        return False, "REJECT_QUOTE_CONTAINS_ELLIPSIS", None

    # 1. Direct exact substring match
    if quote in chunk_text:
        return True, "VALID", quote

    # 2. Normalized whitespace / quote matching
    norm_q = normalize_text_for_matching(quote)
    norm_chunk = normalize_text_for_matching(chunk_text)

    if norm_q in norm_chunk:
        words = norm_q.split()
        if len(words) >= 3:
            first_words = re.escape(words[0]) + r"\s+" + re.escape(words[1])
            last_words = re.escape(words[-2]) + r"\s+" + re.escape(words[-1])
            pattern = re.compile(first_words + r".*?" + last_words, re.DOTALL)
            m = pattern.search(chunk_text)
            if m:
                return True, "VALID_NORMALIZED", m.group(0)
        return True, "VALID_NORMALIZED", quote

    return False, "REJECT_QUOTE_NOT_IN_CHUNK", None


def extract_word_tokens(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: List[str], n: int = 6) -> Set[Tuple[str, ...]]:
    """Builds set of contiguous n-gram tuples from tokens."""
    if len(tokens) < n:
        return set()
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def validate_6gram_overlap(question: str, chunk_text: str, n: int = 6) -> Tuple[bool, str]:
    """
    Rejects questions that share a 6-gram or longer contiguous token sequence with chunk text.
    """
    q_tokens = extract_word_tokens(question)
    c_tokens = extract_word_tokens(chunk_text)

    q_ngrams = get_ngrams(q_tokens, n=n)
    if not q_ngrams:
        return True, "VALID"

    c_ngrams = get_ngrams(c_tokens, n=n)
    overlap = q_ngrams.intersection(c_ngrams)
    if overlap:
        example = " ".join(next(iter(overlap)))
        return False, f"REJECT_6GRAM_OVERLAP: '{example}'"

    return True, "VALID"


def validate_no_preamble(question: str) -> Tuple[bool, str]:
    """Rejects questions starting with 'According to'."""
    q_clean = question.strip().lower()
    if q_clean.startswith("according to"):
        return False, "REJECT_PREAMBLE_ACCORDING_TO"
    return True, "VALID"


def validate_duplicate(
    question: str,
    existing_questions: List[str],
    threshold: float = 0.85,
) -> Tuple[bool, str]:
    """Rejects questions that duplicate or nearly duplicate an existing question."""
    q_tokens = set(extract_word_tokens(question))
    if not q_tokens:
        return False, "REJECT_EMPTY_QUESTION"

    q_norm = " ".join(extract_word_tokens(question))
    for exist_q in existing_questions:
        e_norm = " ".join(extract_word_tokens(exist_q))
        if q_norm == e_norm:
            return False, f"REJECT_DUPLICATE_EXACT: matches '{exist_q[:50]}...'"
        e_tokens = set(extract_word_tokens(exist_q))
        if not e_tokens:
            continue
        jaccard = len(q_tokens & e_tokens) / len(q_tokens | e_tokens)
        if jaccard >= threshold:
            return False, f"REJECT_DUPLICATE_SIMILAR (jaccard={jaccard:.2f}): matches '{exist_q[:50]}...'"

    return True, "VALID"


COMMON_CONTEXT_WORDS: Set[str] = {
    # Grammar, determiners, pronouns, prepositions
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "they", "their", "them", "we", "our", "us", "which", "what",
    "who", "whom", "whose", "where", "when", "why", "how", "all", "each", "every",
    "both", "neither", "either", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "first", "second", "third", "other", "into",
    "than", "then", "more", "most", "less", "least", "some", "any", "no", "not",
    "only", "same", "such", "also", "well", "even", "about", "after", "before",
    "between", "under", "over", "above", "below", "up", "down", "out", "off",
    "through", "during", "including", "included", "excludes", "excluded",
    # Financial & filing terminology
    "company", "corporation", "inc", "corp", "co", "firm", "group", "enterprise",
    "fiscal", "year", "years", "annual", "quarter", "quarterly", "period", "periods",
    "ended", "ending", "date", "december", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "form",
    "10-k", "10k", "sec", "filing", "filings", "report", "reported", "reporting",
    "statement", "statements", "consolidated", "item", "items", "section", "sections",
    "total", "net", "sales", "revenue", "revenues", "income", "incomes", "expense", "expenses",
    "cost", "costs", "dollar", "dollars", "share", "shares", "stock", "common",
    "million", "millions", "billion", "billions", "thousand", "thousands",
    "percent", "percentage", "rate", "rates", "basis", "points", "amount", "amounts",
    "cash", "flow", "flows", "asset", "assets", "liability", "liabilities", "equity",
    "operating", "operations", "operation", "service", "services", "product", "products",
    "segment", "segments", "business", "businesses", "capital", "expenditure", "expenditures",
    "capex", "depreciation", "amortization", "tax", "taxes", "effective", "interest",
    "debt", "gain", "gains", "loss", "losses", "profit", "margins", "margin", "gross",
    "research", "development", "administrative", "general", "selling", "marketing",
    "marketable", "securities", "repurchase", "repurchases", "dividend", "dividends",
    "diluted", "basic", "per", "earning", "earnings", "carrying", "value", "values", "fair",
    "note", "notes", "table", "information", "accounting", "policy", "policies",
    "united", "states", "america", "international", "global", "geographic", "region",
    "headcount", "employees", "workforce", "personnel", "officer", "officers", "executive",
    "approximately", "estimated", "estimates", "estimate", "increase", "decrease", "higher", "lower", "compared",
    "prior", "current", "subsequent", "growth", "payable", "receivable", "inventory",
    "goodwill", "intangible", "lease", "leases", "commitments", "contingencies",
    # Additional common English & finance words (avoid false-positive leakage)
    "awards", "award", "plans", "plan", "unit", "units", "options", "option",
    "grant", "grants", "granted", "vesting", "compensation", "benefit", "benefits",
    "provisions", "provision", "allowance", "allowances", "contract", "contracts",
    "agreement", "agreements", "settlement", "settlements", "restructuring",
    "severance", "claims", "litigation", "regulatory", "regulations", "requirements",
    "requirement", "factors", "factor", "conditions", "condition", "management",
    "customer", "customers", "supplier", "suppliers", "vendor", "vendors", "distribution",
}


def validate_answer_leakage(
    question: str,
    ground_truth: str,
    fiscal_year: str,
    company: str = "",
    item_section: str = "",
) -> Tuple[bool, str]:
    """
    Rejects a question if distinctive tokens from the ground_truth_answer
    (numbers with 3+ digits, years, and capitalized or rare words not already
    in the company/section context) appear in the question, except the fiscal
    year of the target filing.
    Ignores common English and finance words and sentence-initial capitalized words.
    Flag: REJECT_ANSWER_LEAKAGE.
    """
    if not question or not ground_truth:
        return True, "VALID"

    # Extract target filing years to exempt
    exempt_years = set(re.findall(r"\b202[0-9]\b", str(fiscal_year)))

    # 1. Numbers with 3+ digits and 4-digit years in ground_truth
    gt_num_tokens = set(re.findall(r"\b\d{3,}\b", ground_truth))
    comma_matches = re.findall(r"\b\d{1,3}(?:,\d{3})+\b", ground_truth)
    for cm in comma_matches:
        gt_num_tokens.add(cm)
        gt_num_tokens.add(cm.replace(",", ""))

    for num in sorted(gt_num_tokens):
        # Exclude target filing fiscal year
        if num in exempt_years:
            continue
        # Check if number appears in question
        num_pattern = r"(?<!\d)" + re.escape(num) + r"(?!\d)"
        if re.search(num_pattern, question):
            return False, f"REJECT_ANSWER_LEAKAGE: '{num}'"

    # 2. Capitalized or rare distinctive words in ground_truth
    context_words = set(COMMON_CONTEXT_WORDS)
    if company:
        for part in re.findall(r"\b\w+\b", company.lower()):
            context_words.add(part)
        comp_lower = company.lower()
        if "apple" in comp_lower:
            context_words.update(["apple", "iphone", "ipad", "mac", "watch"])
        elif "microsoft" in comp_lower:
            context_words.update(["microsoft", "azure", "windows", "xbox"])
        elif "alphabet" in comp_lower:
            context_words.update(["alphabet", "google", "youtube", "android"])
        elif "amazon" in comp_lower:
            context_words.update(["amazon", "aws", "prime"])
        elif "nvidia" in comp_lower:
            context_words.update(["nvidia", "geforce", "cuda"])

    if item_section:
        for part in re.findall(r"\b\w+\b", item_section.lower()):
            context_words.add(part)

    # Scan ground_truth by sentence to ignore sentence-initial capitalization
    sentences = re.split(r"(?<=[.!?])\s+", ground_truth.strip())
    for sent in sentences:
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", sent)
        for idx, token in enumerate(words):
            # Ignore sentence-initial capitalized words in multi-word sentences
            if idx == 0 and len(words) > 1:
                continue

            clean_tok = token.strip("-_")
            if len(clean_tok) < 4:
                continue
            lower_tok = clean_tok.lower()
            if lower_tok in context_words:
                continue

            if clean_tok[0].isupper():
                token_pattern = r"\b" + re.escape(clean_tok) + r"\b"
                if re.search(token_pattern, question, re.IGNORECASE):
                    return False, f"REJECT_ANSWER_LEAKAGE: '{clean_tok}'"

    return True, "VALID"


def validate_quote_unique_across_filings(
    supporting_quote: str,
    question: str,
    company: str,
    fiscal_year: str,
    all_chunks: List[Dict[str, Any]],
) -> Tuple[bool, str, bool]:
    """
    Normalizes whitespace and compares the supporting_quote against ALL chunks of the same company
    in OTHER fiscal years.
    If it appears verbatim there, the item is ambiguous unless the question explicitly names its own fiscal year;
    flags it AMBIGUOUS_REPEATED_ACROSS_YEARS and sets default review status to reject (should_reject = True).
    Returns (is_valid, flag_message, should_reject).
    """
    if not supporting_quote or len(supporting_quote.strip()) < 10:
        return True, "VALID", False

    norm_quote = normalize_text_for_matching(supporting_quote)
    target_comp = company.lower()
    target_years = set(re.findall(r"\b202[0-9]\b", str(fiscal_year)))

    # Chunks of same company in other fiscal years
    other_year_chunks = [
        c for c in all_chunks
        if c.get("company", "").lower() == target_comp
        and (not target_years or str(c.get("fiscal_year", "")).strip() not in target_years)
    ]

    matched_other_years: Set[str] = set()
    for c in other_year_chunks:
        if norm_quote in normalize_text_for_matching(c.get("text", "")):
            matched_other_years.add(str(c.get("fiscal_year", "")))

    if matched_other_years:
        q_lower = question.lower()
        if target_years:
            names_own_year = all(y in q_lower for y in target_years)
        else:
            names_own_year = str(fiscal_year).lower() in q_lower

        other_yrs_str = ", ".join(sorted(matched_other_years))
        if not names_own_year:
            flag_msg = f"AMBIGUOUS_REPEATED_ACROSS_YEARS (quote also in FY{other_yrs_str}; question omits year)"
            return False, flag_msg, True  # Set default review status to reject
        else:
            flag_msg = f"REPEATED_ACROSS_YEARS_DISAMBIGUATED (quote in FY{other_yrs_str}; question names target year)"
            return True, flag_msg, False

    return True, "VALID", False


def validate_cross_filing_name_ambiguity(
    question: str,
    company: str,
    fiscal_year: str,
) -> List[str]:
    """
    Secondary check: flags questions omitting company name or fiscal year context.
    """
    q_lower = question.lower()
    has_company = (
        company.lower() in q_lower
        or (company == "Apple" and any(k in q_lower for k in ["iphone", "ipad", "mac", "vision pro", "apple"]))
        or (company == "Microsoft" and any(k in q_lower for k in ["azure", "windows", "xbox", "microsoft"]))
        or (company == "Alphabet" and any(k in q_lower for k in ["google", "youtube", "android", "alphabet"]))
        or (company == "Amazon" and any(k in q_lower for k in ["aws", "prime", "amazon"]))
        or (company == "NVIDIA" and any(k in q_lower for k in ["geforce", "cuda", "hopper", "blackwell", "nvidia"]))
    )

    years = re.findall(r"\b202[0-9]\b", q_lower)
    has_year = len(years) > 0

    flags: List[str] = []
    if not has_company:
        flags.append("FLAG_NO_COMPANY_NAME")
    if not has_year and fiscal_year != "None":
        flags.append("FLAG_NO_FISCAL_YEAR")

    return flags


# =============================================================================
# Filing Year and Ground Truth Support Validators
# =============================================================================
COMMON_GT_STOPWORDS: Set[str] = {
    "that", "this", "with", "from", "have", "were", "which", "their", "about",
    "between", "fiscal", "year", "company", "report", "filing", "stated", "noted",
    "discussed", "compared", "during", "these", "under", "other", "also", "into",
    "over", "such", "than", "each", "both", "more", "most", "some", "only", "same",
    "been", "being", "having", "first", "second", "third", "fourth", "annual",
}


def extract_question_years(question: str) -> Set[str]:
    """
    Extracts all fiscal years (4-digit 1990-2040, or FY-form) named in the question.
    Handles FY2024, FY24, fiscal 2024, 2024, etc.
    """
    years = set()
    # FY forms: FY2024, FY24, FY 2024
    for m in re.finditer(r"\bFY\s*(\d{2,4})\b", question, re.IGNORECASE):
        y = m.group(1)
        if len(y) == 2:
            years.add("20" + y)
        elif len(y) == 4:
            years.add(y)

    # 4-digit calendar years 1990-2040
    for m in re.finditer(r"\b(19\d\d|20[0-4]\d)\b", question):
        years.add(m.group(1))

    return years


def validate_question_year_matches_filing(
    question: str,
    fiscal_year: str,
    is_multi_year: bool = False,
    is_unanswerable: bool = False,
) -> Tuple[bool, str]:
    """
    Validates that any fiscal year named in the question matches the filing's fiscal year:
    - For single-chunk items: if question names a year differing from chunk's fiscal_year, reject.
    - For multi-year items: if question names a year that is not one of the two filings' years, reject.
    - Unanswerable inquiries (no gold chunk) are exempt.
    Returns (is_valid, message).
    """
    if is_unanswerable:
        return True, "VALID"

    q_years = extract_question_years(question)
    if not q_years:
        return True, "VALID"

    if is_multi_year or "/" in fiscal_year:
        valid_years = {str(y).strip() for y in fiscal_year.split("/") if y.strip()}
        mismatched = q_years - valid_years
        if mismatched:
            sorted_m = ", ".join(sorted(mismatched))
            sorted_v = "/".join(sorted(valid_years))
            return False, f"REJECT_QUESTION_YEAR_MISMATCH (question names {sorted_m}, but filings are {sorted_v})"
    else:
        valid_year = str(fiscal_year).strip()
        mismatched = q_years - {valid_year}
        if mismatched:
            sorted_m = ", ".join(sorted(mismatched))
            return False, f"REJECT_QUESTION_YEAR_MISMATCH (question names {sorted_m}, but filing is {valid_year})"

    return True, "VALID"


def validate_multi_year_adjacent_years(fiscal_year_str: str) -> Tuple[bool, str]:
    """
    Requires adjacent fiscal years for multi-year items (e.g. 2023/2024, 2024/2025, 2023/2024/2025).
    Rejects non-adjacent years (e.g. 2023/2025).
    """
    years = [int(y.strip()) for y in fiscal_year_str.split("/") if y.strip().isdigit()]
    if len(years) not in (2, 3):
        return False, "REJECT_MULTI_YEAR_NON_ADJACENT_YEARS (requires two or three fiscal years)"
    s_years = sorted(years)
    for i in range(len(s_years) - 1):
        if s_years[i + 1] - s_years[i] != 1:
            return False, f"REJECT_MULTI_YEAR_NON_ADJACENT_YEARS ({s_years[i]} and {s_years[i+1]} are not adjacent)"
    return True, "VALID"


def validate_ground_truth_supported_by_quotes(
    ground_truth: str,
    supporting_quote_str: str,
) -> Tuple[bool, str]:
    """
    Flags for human attention if ground truth in a multi-year comparison item
    is not clearly supported by all quotes:
    - Requires at least 2 distinct supporting quotes.
    - Each quote must have substantive content (at least 6 words, not just titles/signatures).
    - Ground truth must share at least 2 substantive content terms/figures with each quote.
    - Neither quote should end with dangling incomplete punctuation or prepositions.
    """
    quotes = [q.strip() for q in supporting_quote_str.split(" || ") if q.strip()]
    if len(quotes) < 2:
        return False, "FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES (missing distinct quotes)"

    dangling = (",", ";", ":", "•", "-", " of", " in", " for", " with", " regarding", " to")
    w_gt = set(re.findall(r"\b[a-z0-9]{3,}\b", ground_truth.lower())) - COMMON_GT_STOPWORDS

    for idx, q in enumerate(quotes, start=1):
        if len(q.split()) < 6 or len(q) < 30:
            return False, f"FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES (Quote {idx} is too brief, <6 words or <30 chars)"
        if q.rstrip().endswith(dangling):
            return False, f"FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES (Quote {idx} ends with dangling punctuation or incomplete phrase)"
        w_q = set(re.findall(r"\b[a-z0-9]{3,}\b", q.lower())) - COMMON_GT_STOPWORDS
        ov = w_gt & w_q
        if len(ov) < 2:
            return False, f"FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES (Quote {idx} shares only {len(ov)} content terms with GT)"

    return True, "VALID"

    return True, "VALID"


def run_all_validators(
    candidate: Dict[str, Any],
    chunk_text: str,
    existing_questions: List[str],
    all_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[str], bool]:
    """
    Runs the full validation suite on a generated question candidate.
    Returns (is_acceptable, flags_list, should_reject).
    """
    flags: List[str] = []
    is_acceptable = True
    should_reject = False

    # Check preamble
    ok_pre, msg_pre = validate_no_preamble(candidate["question"])
    if not ok_pre:
        flags.append(msg_pre)
        is_acceptable = False

    # Check 6-gram overlap
    ok_6g, msg_6g = validate_6gram_overlap(candidate["question"], chunk_text)
    if not ok_6g:
        flags.append(msg_6g)
        is_acceptable = False

    # Check supporting quote substring
    quote = candidate.get("supporting_quote", "")
    ok_q, msg_q, verbatim = validate_supporting_quote(quote, chunk_text)
    if not ok_q:
        flags.append(msg_q)
        is_acceptable = False
    elif verbatim:
        candidate["supporting_quote"] = verbatim

    # Check duplicate
    ok_dup, msg_dup = validate_duplicate(candidate["question"], existing_questions)
    if not ok_dup:
        flags.append(msg_dup)
        is_acceptable = False

    # Check answer leakage
    ok_leak, msg_leak = validate_answer_leakage(
        question=candidate["question"],
        ground_truth=candidate.get("ground_truth", ""),
        fiscal_year=str(candidate.get("fiscal_year", "")),
        company=candidate.get("company", ""),
        item_section=candidate.get("section", ""),
    )
    if not ok_leak:
        flags.append(msg_leak)
        is_acceptable = False

    # Check quote uniqueness across other fiscal years of the same company
    if all_chunks:
        ok_rep, msg_rep, rej_flag = validate_quote_unique_across_filings(
            supporting_quote=candidate.get("supporting_quote", ""),
            question=candidate["question"],
            company=candidate.get("company", ""),
            fiscal_year=str(candidate.get("fiscal_year", "")),
            all_chunks=all_chunks,
        )
        if not ok_rep:
            flags.append(msg_rep)
            if rej_flag:
                should_reject = True
                is_acceptable = False
        elif msg_rep != "VALID":
            flags.append(msg_rep)

    # Secondary name-based check
    sec_flags = validate_cross_filing_name_ambiguity(
        candidate["question"],
        candidate.get("company", ""),
        str(candidate.get("fiscal_year", "")),
    )
    flags.extend(sec_flags)

    # Check question fiscal year matches filing year
    is_my = len(candidate.get("source_chunk_ids", [])) > 1 or candidate.get("category") == "multi_hop"
    is_un = (
        candidate.get("category") == "unanswerable"
        or str(candidate.get("id", "")).startswith("heldout_un_")
        or str(candidate.get("id", "")).startswith("round2_un_")
    )
    ok_yr, msg_yr = validate_question_year_matches_filing(
        question=candidate["question"],
        fiscal_year=str(candidate.get("fiscal_year", "")),
        is_multi_year=is_my,
        is_unanswerable=is_un,
    )
    if not ok_yr:
        flags.append(msg_yr)
        is_acceptable = False
        should_reject = True

    # Any failed validator defaults to should_reject = True
    if not is_acceptable or should_reject:
        should_reject = True
        is_acceptable = False

    if not flags:
        flags.append("PASS")

    return is_acceptable, flags, should_reject


# =============================================================================
# Stratified Sampling & Exclusion
# =============================================================================
def is_table_chunk(chunk: Dict[str, Any]) -> bool:
    """Identifies table-heavy chunks by markdown table syntax or high numeric token ratio."""
    text = chunk.get("text", "")
    if "|" in text:
        return True
    digit_count = sum(c.isdigit() for c in text)
    return len(text) > 0 and (digit_count / len(text)) > 0.08


def get_excluded_chunk_ids(
    chunks: List[Dict[str, Any]],
    exclude_golden_paths: List[Path],
    exclude_adjacent: bool = True,
) -> Set[str]:
    """
    Excludes any chunk whose id is a source_chunk_id in any provided golden set file,
    plus chunks within +/-1 position of them in the same filing/section, so round 2
    is strictly disjoint from earlier benchmark sets.
    """
    # Group chunks by (company, fiscal_year, item_section) preserving file sequence order
    group_chunks: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    chunk_to_pos: Dict[str, Tuple[Tuple[str, str, str], int]] = {}
    for c in chunks:
        key = (c.get("company", ""), str(c.get("fiscal_year", "")), c.get("item_section", ""))
        pos = len(group_chunks[key])
        group_chunks[key].append(c)
        cid = c.get("chunk_id")
        if cid:
            chunk_to_pos[cid] = (key, pos)

    gold_chunk_ids: Set[str] = set()
    for gp in exclude_golden_paths:
        p = Path(gp)
        if not p.exists():
            p_alt = BASE_DIR / "eval" / p.name
            if p_alt.exists():
                p = p_alt
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for it in items:
                        gold_chunk_ids.update(it.get("source_chunk_ids", []))
                        if it.get("chunk_id"):
                            gold_chunk_ids.add(it["chunk_id"])
            except Exception as e:
                logger.warning(f"Could not load exclusion golden file {p}: {e}")

    excluded_ids: Set[str] = set(gold_chunk_ids)
    if exclude_adjacent:
        for cid in gold_chunk_ids:
            if cid in chunk_to_pos:
                key, pos = chunk_to_pos[cid]
                c_list = group_chunks[key]
                if pos - 1 >= 0:
                    excluded_ids.add(c_list[pos - 1]["chunk_id"])
                if pos + 1 < len(c_list):
                    excluded_ids.add(c_list[pos + 1]["chunk_id"])

    logger.info(
        f"Exclusion filter: {len(gold_chunk_ids)} direct golden chunk IDs -> {len(excluded_ids)} total excluded IDs (with +/- 1 adjacent positions)."
    )
    return excluded_ids


def sample_heldout_chunks(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    target_total: int = 130,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Performs stratified sampling of single-chunk targets:
    - 5 companies: Apple, Microsoft, Alphabet, Amazon, NVIDIA
    - All fiscal years present for each company (3 years per company)
    - Stratifies across Items 1, 1A, 7, 8 (including table-heavy chunks)
    - If target_total <= 90: returns 90 chunks (6 per fiscal year = 18 per company)
    - If target_total > 90 (e.g. 130 for Round 2): returns 130 chunks (26 per company: 9, 9, 8 across 3 FYs)
    - Strictly excludes any chunk IDs in dev_chunk_ids.
    """
    rng = random.Random(seed)
    heldout_pool = [c for c in chunks if c.get("chunk_id") not in dev_chunk_ids]

    companies = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
    sampled: List[Dict[str, Any]] = []

    pool: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for c in heldout_pool:
        comp = c.get("company")
        yr = str(c.get("fiscal_year"))
        sec = c.get("item_section")
        if comp in companies and sec in ["Item 1", "Item 1A", "Item 7", "Item 8"]:
            pool[comp][yr][sec].append(c)

    # Round 1 logic: 90 total (6 per year)
    if target_total <= 90:
        for comp in sorted(companies):
            comp_years = sorted(pool[comp].keys())
            for yr in comp_years:
                # Item 1 (1 chunk)
                sec1_pool = list(pool[comp][yr]["Item 1"])
                rng.shuffle(sec1_pool)
                if sec1_pool:
                    sampled.append(sec1_pool[0])

                # Item 1A (1 chunk)
                sec1a_pool = list(pool[comp][yr]["Item 1A"])
                rng.shuffle(sec1a_pool)
                if sec1a_pool:
                    sampled.append(sec1a_pool[0])

                # Item 7 (2 chunks: 1 table if possible, 1 narrative)
                sec7_pool = list(pool[comp][yr]["Item 7"])
                rng.shuffle(sec7_pool)
                t7 = [c for c in sec7_pool if is_table_chunk(c)]
                n7 = [c for c in sec7_pool if not is_table_chunk(c)]
                chosen_7 = []
                if t7:
                    chosen_7.append(t7[0])
                if n7:
                    chosen_7.append(n7[0])
                while len(chosen_7) < 2 and sec7_pool:
                    cand = sec7_pool.pop(0)
                    if cand not in chosen_7:
                        chosen_7.append(cand)
                sampled.extend(chosen_7[:2])

                # Item 8 (2 chunks: 1 table if possible, 1 narrative)
                sec8_pool = list(pool[comp][yr]["Item 8"])
                rng.shuffle(sec8_pool)
                t8 = [c for c in sec8_pool if is_table_chunk(c)]
                n8 = [c for c in sec8_pool if not is_table_chunk(c)]
                chosen_8 = []
                if t8:
                    chosen_8.append(t8[0])
                if n8:
                    chosen_8.append(n8[0])
                while len(chosen_8) < 2 and sec8_pool:
                    cand = sec8_pool.pop(0)
                    if cand not in chosen_8:
                        chosen_8.append(cand)
                sampled.extend(chosen_8[:2])
        return sampled

    # Round 2 logic: 130 total (26 per company, [9, 9, 8] across 3 years)
    for comp in sorted(companies):
        years = sorted(pool[comp].keys())
        target_per_year = [9, 9, 8] if len(years) == 3 else [target_total // (len(companies) * len(years))] * len(years)
        for yr_idx, yr in enumerate(years):
            t_total = target_per_year[yr_idx] if yr_idx < len(target_per_year) else 8
            yr_sampled: List[Dict[str, Any]] = []
            sec_targets = {
                "Item 1": 1,
                "Item 1A": 2,
                "Item 7": 3 if t_total == 9 else 2,
                "Item 8": 3,
            }
            for sec in ["Item 1", "Item 1A", "Item 7", "Item 8"]:
                avail = list(pool[comp][yr][sec])
                rng.shuffle(avail)
                tgt = sec_targets.get(sec, 2)
                if sec in ["Item 7", "Item 8"]:
                    t_chunks = [c for c in avail if is_table_chunk(c)]
                    n_chunks = [c for c in avail if not is_table_chunk(c)]
                    chosen: List[Dict[str, Any]] = []
                    if t_chunks:
                        chosen.append(t_chunks.pop(0))
                    if n_chunks:
                        chosen.append(n_chunks.pop(0))
                    rem = t_chunks + n_chunks
                    rng.shuffle(rem)
                    while len(chosen) < tgt and rem:
                        chosen.append(rem.pop(0))
                    yr_sampled.extend(chosen[:tgt])
                else:
                    yr_sampled.extend(avail[:tgt])

            # Backfill if any section ran short of target
            if len(yr_sampled) < t_total:
                existing_ids = {c["chunk_id"] for c in yr_sampled}
                all_yr = [
                    c for s in ["Item 7", "Item 8", "Item 1A", "Item 1"]
                    for c in pool[comp][yr][s]
                    if c["chunk_id"] not in existing_ids
                ]
                rng.shuffle(all_yr)
                yr_sampled.extend(all_yr[:(t_total - len(yr_sampled))])

            sampled.extend(yr_sampled[:t_total])

    return sampled[:target_total]


def sample_multi_year_groups(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    target_groups: int = 25,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """
    Samples ~25 multi-year chunk groups (2-year pairs and 3-year triplets) from the same company
    across adjacent fiscal years:
    - 5 companies * 5 items each = 25 multi-year candidate targets
    - For each company with years [y0, y1, y2]:
      - 2 pairs from (y0, y1) in Item 7 and Item 8
      - 2 pairs from (y1, y2) in Item 7 and Item 8
      - 1 triplet from (y0, y1, y2) in Item 7 or Item 8
    - All pairs and triplets strictly require adjacent fiscal years.
    """
    rng = random.Random(seed + 100)
    pool = [c for c in chunks if c.get("chunk_id") not in dev_chunk_ids]

    companies = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
    comp_sec_yr: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for c in pool:
        comp = c.get("company")
        yr = str(c.get("fiscal_year"))
        sec = c.get("item_section")
        if comp in companies and sec in ["Item 7", "Item 8", "Item 1"]:
            comp_sec_yr[comp][sec][yr].append(c)

    all_groups: List[List[Dict[str, Any]]] = []
    groups_per_comp = max(1, target_groups // len(companies))

    for comp in sorted(companies):
        comp_groups: List[List[Dict[str, Any]]] = []
        years = sorted({str(c.get("fiscal_year")) for c in pool if c.get("company") == comp})

        if len(years) >= 2:
            # 1. Pairs from (y0, y1)
            for sec in ["Item 7", "Item 8"]:
                l0 = comp_sec_yr[comp][sec].get(years[0], [])
                l1 = comp_sec_yr[comp][sec].get(years[1], [])
                if l0 and l1:
                    comp_groups.append([rng.choice(l0), rng.choice(l1)])

        if len(years) >= 3:
            # 2. Pairs from (y1, y2)
            for sec in ["Item 7", "Item 8"]:
                l1 = comp_sec_yr[comp][sec].get(years[1], [])
                l2 = comp_sec_yr[comp][sec].get(years[2], [])
                if l1 and l2:
                    comp_groups.append([rng.choice(l1), rng.choice(l2)])

            # 3. Triplet from (y0, y1, y2)
            sec = "Item 7" if comp_sec_yr[comp]["Item 7"].get(years[0]) else "Item 8"
            l0 = comp_sec_yr[comp][sec].get(years[0], [])
            l1 = comp_sec_yr[comp][sec].get(years[1], [])
            l2 = comp_sec_yr[comp][sec].get(years[2], [])
            if l0 and l1 and l2:
                comp_groups.append([rng.choice(l0), rng.choice(l1), rng.choice(l2)])

        # Pad or trim to groups_per_comp
        all_groups.extend(comp_groups[:groups_per_comp])

    return all_groups[:target_groups]


def sample_multi_year_pairs(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    target_pairs: int = 12,
    seed: int = 42,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Backward-compatible sampling of 2-chunk pairs from adjacent fiscal years.
    """
    groups = sample_multi_year_groups(chunks, dev_chunk_ids, target_groups=target_pairs, seed=seed)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for g in groups:
        if len(g) >= 2:
            pairs.append((g[0], g[1]))
    return pairs[:target_pairs]


# =============================================================================
# Question Styles & Prompt Templates
# =============================================================================
QUESTION_STYLES: List[str] = [
    "natural prose",
    "short direct question",
    "keyword-style phrase",
    "paraphrase (no copied terms from the chunk)",
    '"what does the filing say about" style',
    "comparison-of-two-figures style where the chunk allows",
]

SINGLE_CHUNK_PROMPT_TEMPLATE = """You are an expert financial analyst building an evaluation benchmark for SEC 10-K RAG systems.

Given the following 10-K text chunk from {company} (Fiscal Year: {fiscal_year}, Section: {item_section}):

---
{text_chunk}
---

Your task is to formulate a high-quality benchmark question-answer item.
Output strict JSON with these 4 keys:
{{
  "question": "string",
  "ground_truth_answer": "string",
  "supporting_quote": "string",
  "category": "simple_lookup" | "numeric_precision" | "multi_hop-like single-chunk" | "other"
}}

Strict Requirements:
1. The question must be answerable from this chunk ALONE.
2. The question must NOT start with "According to" or "Based on".
3. The question must NOT copy more than 5 consecutive words from the chunk (paraphrase or use natural financial terminology).
4. REQUIRED QUESTION STYLE: You MUST formulate the question using this exact style: "{question_style}".
   - "natural prose": Full, grammatically complete sentence with natural financial prose.
   - "short direct question": Concise, direct question with minimal filler words.
   - "keyword-style phrase": A concise keyword/search inquiry phrase (e.g. "Apple FY2023 R&D expenditure total").
   - "paraphrase (no copied terms from the chunk)": Completely paraphrased using synonyms and no copied terms from the chunk.
   - "\"what does the filing say about\" style": Phrased as "What does the filing state regarding [topic]?" or "What does the filing say about [topic]?"
   - "comparison-of-two-figures style where the chunk allows": Formulate a question comparing two figures/metrics or periods present in this chunk (or concise lookup if only one figure exists).
5. For table chunks or financial metrics, ask for specific numbers with exact units (e.g. '$ in millions', 'percentage', 'shares').
6. supporting_quote MUST be an EXACT, VERBATIM substring copied directly from the text chunk above that proves the answer. Do NOT use ellipsis ('...' or '…') in the quote unless that exact ellipsis is present in the chunk.
7. Do NOT leak distinctive parts of the answer (e.g. specific target figures, future years, or secret entities) inside the question itself.
"""


MULTI_YEAR_PROMPT_TEMPLATE = """You are an expert financial analyst building a multi-hop evaluation benchmark for SEC 10-K RAG systems.

Given two 10-K text chunks from {company} across two different fiscal years:

--- CHUNK 1 (Fiscal Year: {year1}, Section: {section1}) ---
{text1}

--- CHUNK 2 (Fiscal Year: {year2}, Section: {section2}) ---
{text2}
---

Formulate a multi-year comparison question that REQUIRES information from BOTH chunks to answer.
Output strict JSON with these 4 keys:
{{
  "question": "string",
  "ground_truth_answer": "string",
  "supporting_quotes": ["verbatim quote from chunk 1", "verbatim quote from chunk 2"],
  "category": "multi_hop"
}}

Strict Requirements:
1. The question must require synthesizing facts or numbers from BOTH fiscal years (e.g. comparing trends, growth rates, changes in accounting policies, or shift in risk factors).
2. The question must NOT start with "According to" or "Based on".
3. The question must NOT copy more than 5 consecutive words from either chunk.
4. Each supporting quote MUST be an EXACT, VERBATIM substring copied directly from its respective chunk that proves that year's part of the answer. Do NOT use ellipsis ('...' or '…') in quotes.
5. Do NOT leak distinctive parts of the answer inside the question itself.
"""


def format_multi_year_prompt(company: str, chunk_list: List[Dict[str, Any]]) -> str:
    """Formats prompt template for multi-year comparison across 2 or 3 chunks."""
    n = len(chunk_list)
    word_n = "two" if n == 2 else ("three" if n == 3 else str(n))
    chunks_text = []
    for idx, c in enumerate(chunk_list, start=1):
        chunks_text.append(
            f"--- CHUNK {idx} (Fiscal Year: {c.get('fiscal_year', '')}, Section: {c.get('item_section', '')}) ---\n"
            f"{c.get('text', '')[:2000]}"
        )
    joined_chunks = "\n\n".join(chunks_text)
    quotes_spec = ", ".join(f'"verbatim quote from chunk {i}"' for i in range(1, n + 1))
    return f"""You are an expert financial analyst building a multi-hop evaluation benchmark for SEC 10-K RAG systems.

Given {word_n} 10-K text chunks from {company} across {word_n} different fiscal years:

{joined_chunks}
---

Formulate a multi-year comparison question that REQUIRES information from ALL {word_n} chunks to answer.
Output strict JSON with these 4 keys:
{{
  "question": "string",
  "ground_truth_answer": "string",
  "supporting_quotes": [{quotes_spec}],
  "category": "multi_hop"
}}

Strict Requirements:
1. The question must require synthesizing facts or numbers from ALL {word_n} fiscal years.
2. The question must NOT start with "According to" or "Based on".
3. The question must NOT copy more than 5 consecutive words from any chunk.
4. Each supporting quote MUST be an EXACT, VERBATIM substring copied directly from its respective chunk.
"""


def _extract_content_text(resp: Any) -> str:
    """Safely extracts text content from LangChain response even when content is a list of parts."""
    content = resp.content if hasattr(resp, "content") else resp
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(getattr(p, "text"))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def generate_candidate_from_chunk_llm(
    chunk: Dict[str, Any],
    bench_model: str,
    ledger: QuotaLedger,
    question_style: str = "natural prose",
) -> Optional[Dict[str, Any]]:
    """
    Invokes BENCH_MODEL through QuotaLedger to generate a candidate QA item.
    Checks quota budget before calling and records usage.
    """
    ledger.check_budget_or_stop(bench_model)

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=bench_model,
        google_api_key=api_key or None,
        temperature=0.2,
        request_timeout=45.0,
        max_retries=1,
    )

    prompt = SINGLE_CHUNK_PROMPT_TEMPLATE.format(
        company=chunk.get("company", ""),
        fiscal_year=chunk.get("fiscal_year", ""),
        item_section=chunk.get("item_section", ""),
        text_chunk=chunk.get("text", "")[:3500],
        question_style=question_style,
    )

    try:
        resp = llm.invoke([("user", prompt)])
        ledger.record_call(bench_model, count=1)
        content = _extract_content_text(resp)

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        return {
            "question": data.get("question", "").strip(),
            "ground_truth": data.get("ground_truth_answer", "").strip(),
            "supporting_quote": data.get("supporting_quote", "").strip(),
            "category": data.get("category", "simple_lookup"),
        }
    except Exception as e:
        logger.warning(f"LLM call failed for chunk {chunk.get('chunk_id')}: {e}")
        return None


def generate_multi_year_candidate_llm(
    *args,
    bench_model: str,
    ledger: QuotaLedger,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Invokes BENCH_MODEL through QuotaLedger to generate a multi-year comparison item.
    Accepts either a list of chunks, or (c1, c2) positional arguments.
    """
    ledger.check_budget_or_stop(bench_model)

    if len(args) == 1 and isinstance(args[0], list):
        chunk_list = args[0]
    elif len(args) >= 2:
        chunk_list = list(args[:2])
    else:
        raise ValueError("generate_multi_year_candidate_llm requires a list of chunks or (c1, c2)")

    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=bench_model,
        google_api_key=api_key or None,
        temperature=0.2,
        request_timeout=45.0,
        max_retries=1,
    )

    comp = chunk_list[0].get("company", "")
    prompt = format_multi_year_prompt(company=comp, chunk_list=chunk_list)

    try:
        resp = llm.invoke([("user", prompt)])
        ledger.record_call(bench_model, count=1)
        content = _extract_content_text(resp)

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        quotes = data.get("supporting_quotes", [])
        quote_str = " || ".join(quotes) if isinstance(quotes, list) else str(quotes)
        return {
            "question": data.get("question", "").strip(),
            "ground_truth": data.get("ground_truth_answer", "").strip(),
            "supporting_quote": quote_str,
            "category": "multi_hop",
        }
    except Exception as e:
        logger.warning(f"LLM multi-year call failed: {e}")
        return None


# =============================================================================
# CSV and JSON Writers
# =============================================================================
def extract_chunk_excerpt(chunk_text: str, quote: str, window: int = 125) -> str:
    """Extracts approximately 250 characters around the quote from chunk_text."""
    if not chunk_text or not quote:
        return ""
    idx = chunk_text.find(quote)
    if idx == -1:
        norm_chunk = " ".join(chunk_text.split())
        norm_quote = " ".join(quote.split())
        idx = norm_chunk.find(norm_quote)
        if idx != -1:
            chunk_text = norm_chunk
            quote = norm_quote
        else:
            return chunk_text[:250] + (" [...]" if len(chunk_text) > 250 else "")

    start = max(0, idx - window)
    end = min(len(chunk_text), idx + len(quote) + window)
    prefix = "[...] " if start > 0 else ""
    suffix = " [...]" if end < len(chunk_text) else ""
    return prefix + chunk_text[start:end].strip() + suffix


def write_candidates_json(candidates: List[Dict[str, Any]], path: Path) -> None:
    """Writes candidates list to JSON, asserting candidate ID uniqueness."""
    ids = [c["id"] for c in candidates if "id" in c]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Candidate IDs must be unique! Found duplicates: {duplicates}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(candidates)} candidates to JSON: {path}")


def write_review_csv(
    candidates: List[Dict[str, Any]],
    path: Path,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Writes candidates to review CSV with columns:
    id,company,fiscal_year,section,category,question_style,question,ground_truth,supporting_quote,chunk_text_excerpt,final_status,reject_reason,rewrite_applied,original_question,validator_flags,accept,edit,reject
    Asserts uniqueness of candidate IDs.
    """
    ids = [c["id"] for c in candidates if "id" in c]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Candidate IDs must be unique! Found duplicates: {duplicates}")

    if chunks is None and DEFAULT_CHUNKS_PATH.exists():
        try:
            with open(DEFAULT_CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception:
            chunks = []
    chunk_lookup = {c["chunk_id"]: c for c in (chunks or []) if c.get("chunk_id")}

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "company",
        "fiscal_year",
        "section",
        "category",
        "question_style",
        "question",
        "ground_truth",
        "supporting_quote",
        "chunk_text_excerpt",
        "final_status",
        "reject_reason",
        "rewrite_applied",
        "original_question",
        "validator_flags",
        "accept",
        "edit",
        "reject",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            is_rejected = bool(c.get("default_reject", False))
            flags = c.get("validator_flags", ["PASS"])

            # Reject reason: extract all failing flags
            reject_reasons = [
                f for f in flags
                if any(f.startswith(p) for p in ["REJECT_", "AMBIGUOUS_", "CHUNK1_", "CHUNK2_", "FLAG_"])
            ]
            reject_reason_str = "; ".join(reject_reasons) if (is_rejected and reject_reasons) else ""

            # Excerpt extraction
            excerpt_str = ""
            sq = c.get("supporting_quote", "")
            sc_ids = c.get("source_chunk_ids", [])
            cid = c.get("chunk_id") or (sc_ids[0] if sc_ids else None)

            if len(sc_ids) > 1:
                quotes = sq.split(" || ")
                excerpts = []
                for i, s_id in enumerate(sc_ids):
                    c_obj = chunk_lookup.get(s_id, {})
                    c_text = c_obj.get("text", "")
                    q_text = quotes[i] if i < len(quotes) else ""
                    ex = extract_chunk_excerpt(c_text, q_text)
                    c_yr = c_obj.get("fiscal_year", f"Chunk {i+1}")
                    excerpts.append(f"[{c_yr}]: {ex}")
                excerpt_str = " || ".join(excerpts)
            elif cid:
                c_obj = chunk_lookup.get(cid, {})
                c_text = c_obj.get("text", "")
                excerpt_str = extract_chunk_excerpt(c_text, sq)

            writer.writerow({
                "id": c.get("id", ""),
                "company": c.get("company", ""),
                "fiscal_year": c.get("fiscal_year", ""),
                "section": c.get("section", ""),
                "category": c.get("category", ""),
                "question_style": c.get("question_style", ""),
                "question": c.get("question", ""),
                "ground_truth": c.get("ground_truth", ""),
                "supporting_quote": sq,
                "chunk_text_excerpt": excerpt_str,
                "final_status": "REJECT" if is_rejected else "ACCEPT",
                "reject_reason": reject_reason_str,
                "rewrite_applied": c.get("rewrite_applied", ""),
                "original_question": c.get("original_question", ""),
                "validator_flags": "; ".join(flags),
                "accept": "" if is_rejected else "x",
                "edit": "",
                "reject": "x" if is_rejected else "",
            })
    logger.info(f"Saved {len(candidates)} candidates to review CSV: {path}")


# =============================================================================
# Main Pipeline Runner
# =============================================================================
def run_generation_pipeline(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    exclude_golden_paths: Optional[List[Path]] = None,
    dev_set_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    bench_model: str = "gemini-3.5-flash-lite",
    max_calls: int = 110,
    round_num: int = 2,
    target_single: Optional[int] = None,
    target_multi: Optional[int] = None,
    target_unanswerable: Optional[int] = None,
    limit: Optional[int] = None,
    delay: float = 5.0,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates candidate generation, stratification, validation, and writing for benchmark rounds.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    # Resolve target counts and ID prefixes by round
    if round_num == 2:
        target_s = target_single or 130
        target_m = target_multi or 25
        target_u = target_unanswerable or 15
        id_prefix = "round2"
        default_exclude = [DEFAULT_DEV_SET_PATH, DEFAULT_HELDOUT_SET_PATH]
        def_dry_json = DEFAULT_ROUND2_DRYRUN_JSON
        def_dry_csv = DEFAULT_ROUND2_DRYRUN_CSV
        def_real_json = DEFAULT_ROUND2_OUTPUT_JSON
        def_real_csv = DEFAULT_ROUND2_OUTPUT_CSV
    else:
        target_s = target_single or 90
        target_m = target_multi or 12
        target_u = target_unanswerable or 12
        id_prefix = "heldout"
        default_exclude = [DEFAULT_DEV_SET_PATH]
        def_dry_json = DEFAULT_DRYRUN_JSON
        def_dry_csv = DEFAULT_DRYRUN_CSV
        def_real_json = DEFAULT_OUTPUT_JSON
        def_real_csv = DEFAULT_OUTPUT_CSV

    # Build exclusion set with +/- 1 adjacent positions in same filing/section
    eff_exclude_paths: List[Path] = []
    if exclude_golden_paths:
        eff_exclude_paths = [Path(p) for p in exclude_golden_paths]
    elif dev_set_path:
        eff_exclude_paths = [Path(dev_set_path)]
    else:
        eff_exclude_paths = default_exclude

    excluded_chunk_ids = get_excluded_chunk_ids(chunks, eff_exclude_paths, exclude_adjacent=True)

    # 1. Stratified sampling of single chunks
    sampled_single_chunks = sample_heldout_chunks(chunks, excluded_chunk_ids, target_total=target_s, seed=seed)

    # If --limit N is specified, generate only the first N planned single-chunk items
    if limit is not None and limit > 0:
        sampled_single_chunks = sampled_single_chunks[:limit]
        sampled_multi_groups: List[List[Dict[str, Any]]] = []
        logger.info(f"--limit {limit} active: generating only the first {limit} single-chunk items (multi-year LLM calls omitted).")
    else:
        # 2. Sample multi-year comparison groups (pairs and triplets across adjacent fiscal years)
        sampled_multi_groups = sample_multi_year_groups(chunks, excluded_chunk_ids, target_groups=target_m, seed=seed)

    # 3. Unanswerable items (100% fixed zero-cost templates, always included)
    unanswerable_items = list(UNANSWERABLE_TEMPLATES[:target_u])

    planned_single_calls = len(sampled_single_chunks)
    planned_multi_calls = len(sampled_multi_groups)
    total_planned_calls = planned_single_calls + planned_multi_calls

    ledger = QuotaLedger(max_calls=max_calls)
    used_today = ledger.get_used_today(bench_model)
    daily_limit = ledger.get_limit(bench_model)
    rem_budget = ledger.get_remaining_budget(bench_model)

    if dry_run:
        target_json = output_json_path or def_dry_json
        target_csv = output_csv_path or def_dry_csv
    else:
        target_json = output_json_path or def_real_json
        target_csv = output_csv_path or def_real_csv

    print("\n" + "=" * 80)
    print(f"BENCHMARK ROUND {round_num} CANDIDATE GENERATION PLAN")
    print("=" * 80)
    print(f"Corpus Chunks:                 {len(chunks)}")
    print(f"Excluded Golden Sets:          {', '.join(p.name for p in eff_exclude_paths)}")
    print(f"Excluded Chunks (+/-1 adj):    {len(excluded_chunk_ids)} (strictly disjoint from earlier benchmark sets)")
    print(f"Sampled Single Chunks:         {planned_single_calls} (stratified across 5 companies, all FYs, Items 1/1A/7/8)")
    print(f"Sampled Multi-Year Groups:     {planned_multi_calls} (across 5 companies, 2-year pairs and 3-year triplets)")
    print(f"Fixed Unanswerable Items:      {len(unanswerable_items)} (Fixed templates; cost EXACTLY 0 LLM calls)")
    print(f"Total Benchmark Target Items:  {planned_single_calls + planned_multi_calls + len(unanswerable_items)}")
    print("-" * 80)
    print("PLANNED LLM CALL COUNT & QUOTA LEDGER USAGE:")
    print(f"  Target Model:                {bench_model}")
    print(f"  Max Calls Cap:               {max_calls}")
    print(f"  Planned Single-Chunk Calls:  {planned_single_calls}")
    print(f"  Planned Multi-Year Calls:    {planned_multi_calls}")
    print(f"  TOTAL PLANNED LLM CALLS:     {total_planned_calls}")
    print(f"  Unanswerable Items Call Cost:0 calls (100% fixed zero-cost templates)")
    print(f"  Used Today (Pacific):        {used_today} requests")
    print(f"  Effective Daily Limit:       {daily_limit} requests (0.8 safety factor applied to 500)")
    print(f"  Remaining Budget Today:      {rem_budget} requests")
    print(f"  Budget Sufficient:           {'YES' if rem_budget >= total_planned_calls else 'NO (quota ledger will stop gracefully; resumable)'}")
    print("=" * 80)

    # Breakdown by company & section
    comp_sec_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    table_chunk_count = 0
    for c in sampled_single_chunks:
        comp_sec_count[c["company"]][c["item_section"]] += 1
        if is_table_chunk(c):
            table_chunk_count += 1

    print("\nStratification Breakdown (Single Chunks):")
    for comp in sorted(comp_sec_count.keys()):
        sections_str = ", ".join(f"{s}: {cnt}" for s, cnt in sorted(comp_sec_count[comp].items()))
        total_comp = sum(comp_sec_count[comp].values())
        print(f"  - {comp:<10} ({total_comp} chunks) -> {sections_str}")
    print(f"  Table-heavy Item 7/8 chunks included: {table_chunk_count}/{planned_single_calls}")

    if dry_run:
        print("\n[DRY-RUN MODE] Zero LLM calls executed. Preparing dry-run template outputs...")
        print(f"Note: The {len(unanswerable_items)} unanswerable items are fixed templates that cost zero LLM calls.")
        dry_candidates: List[Dict[str, Any]] = []

        # Single chunk templates
        for idx, chunk in enumerate(sampled_single_chunks, start=1):
            cid = chunk.get("chunk_id", "")
            comp = chunk.get("company", "")
            yr = str(chunk.get("fiscal_year", ""))
            sec = chunk.get("item_section", "")
            style = QUESTION_STYLES[(idx - 1) % len(QUESTION_STYLES)]
            item_id = f"{id_prefix}_{idx:03d}" if round_num == 2 else f"{id_prefix}_{idx:02d}"
            dry_candidates.append({
                "id": item_id,
                "company": comp,
                "fiscal_year": yr,
                "section": sec,
                "category": "numeric_precision" if is_table_chunk(chunk) else "simple_lookup",
                "question_style": style,
                "question": f"[DRY-RUN TEMPLATE: {style}] Question for {comp} FY{yr} ({sec}) chunk {cid[:8]}",
                "ground_truth": "[DRY-RUN TEMPLATE] Ground truth answer derived from chunk text.",
                "supporting_quote": chunk.get("text", "")[:120].strip(),
                "validator_flags": ["PASS"],
                "default_reject": False,
                "chunk_id": cid,
                "source_chunk_ids": [cid],
            })

        # Multi-year templates
        for idx, group in enumerate(sampled_multi_groups, start=1):
            comp = group[0].get("company", "")
            yrs_str = "/".join(str(c.get("fiscal_year", "")) for c in group)
            fy_joined = " and ".join("FY" + str(c.get("fiscal_year", "")) for c in group)
            secs_str = "/".join(c.get("item_section", "") for c in group)
            dry_candidates.append({
                "id": f"{id_prefix}_mh_{idx:02d}",
                "company": comp,
                "fiscal_year": yrs_str,
                "section": secs_str,
                "category": "multi_hop",
                "question_style": "comparison-of-two-figures style where the chunk allows",
                "question": f"[DRY-RUN TEMPLATE] Compare {comp} {fy_joined} metric",
                "ground_truth": "[DRY-RUN TEMPLATE] Multi-year synthesized answer.",
                "supporting_quote": " || ".join(c.get("text", "")[:60].strip() for c in group),
                "validator_flags": ["PASS"],
                "default_reject": False,
                "source_chunk_ids": [c.get("chunk_id") for c in group],
            })

        # Unanswerable templates
        for idx, u in enumerate(unanswerable_items, start=1):
            dry_candidates.append({
                "id": f"{id_prefix}_un_{idx:02d}",
                "company": u["company"],
                "fiscal_year": u["fiscal_year"],
                "section": "None",
                "category": "unanswerable",
                "question_style": "unanswerable inquiry",
                "question": u["question"],
                "ground_truth": u["ground_truth"],
                "supporting_quote": "",
                "validator_flags": ["PASS"],
                "default_reject": False,
                "source_chunk_ids": [],
            })

        write_candidates_json(dry_candidates, target_json)
        write_review_csv(dry_candidates, target_csv, chunks=chunks)

        print("\nDry-run completed successfully.")
        print(f"Generated {len(dry_candidates)} candidate templates.")
        print(f"  Dry-run JSON output: {target_json}")
        print(f"  Dry-run CSV output:  {target_csv}")
        return {
            "status": "dry_run_complete",
            "planned_calls": total_planned_calls,
            "remaining_budget": rem_budget,
            "total_candidates": len(dry_candidates),
            "output_json": str(target_json),
            "output_csv": str(target_csv),
        }

    # Full LLM Generation Mode (with resumability support)
    candidates: List[Dict[str, Any]] = []
    completed_chunk_ids: Set[str] = set()
    completed_multi_keys: Set[Tuple[str, ...]] = set()

    # Load existing candidates if resuming a partial previous run
    if target_json.exists():
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    candidates.append(item)
                    if item.get("chunk_id"):
                        completed_chunk_ids.add(item["chunk_id"])
                    for sc in item.get("source_chunk_ids", []):
                        completed_chunk_ids.add(sc)
                    if len(item.get("source_chunk_ids", [])) > 1:
                        completed_multi_keys.add(tuple(sorted(item["source_chunk_ids"])))
            logger.info(
                f"Resuming generation: loaded {len(candidates)} previously generated items from {target_json}. "
                f"Skipping already processed chunks."
            )
        except Exception as e:
            logger.warning(f"Could not load existing {target_json} for resume: {e}")

    existing_questions: List[str] = [c.get("question", "") for c in candidates if c.get("question")]

    quota_exhausted = False
    new_calls_made = 0

    # 1. Single chunk generation
    for idx, chunk in enumerate(sampled_single_chunks, start=1):
        cid = chunk.get("chunk_id", "")
        if cid in completed_chunk_ids:
            logger.info(f"[{idx}/{planned_single_calls}] Chunk {cid[:8]} already generated; skipping.")
            continue

        if ledger.get_remaining_budget(bench_model) <= 0:
            logger.warning(f"Quota ledger daily limit ({daily_limit}) reached today! Stopping gracefully. Resumable next day.")
            quota_exhausted = True
            break

        if new_calls_made > 0 and delay > 0:
            logger.info(f"Waiting {delay}s between calls to stay under rate limit (15 RPM)...")
            time.sleep(delay)

        item_id = f"{id_prefix}_{idx:03d}" if round_num == 2 else f"{id_prefix}_{idx:02d}"
        comp = chunk.get("company", "")
        yr = str(chunk.get("fiscal_year", ""))
        sec = chunk.get("item_section", "")
        style = QUESTION_STYLES[(idx - 1) % len(QUESTION_STYLES)]

        logger.info(f"[{idx}/{planned_single_calls}] Calling {bench_model} ({style}) for {comp} FY{yr} {sec} ({cid[:8]})...")
        try:
            res = generate_candidate_from_chunk_llm(
                chunk, bench_model=bench_model, ledger=ledger, question_style=style
            )
            new_calls_made += 1
        except Exception as e:
            logger.warning(f"LLM call interrupted/failed: {e}")
            quota_exhausted = True
            break

        if not res:
            logger.warning(f"Failed to generate item for {item_id}")
            continue

        cand_record = {
            "id": item_id,
            "company": comp,
            "fiscal_year": yr,
            "section": sec,
            "category": res["category"],
            "question_style": style,
            "question": res["question"],
            "ground_truth": res["ground_truth"],
            "supporting_quote": res["supporting_quote"],
            "chunk_id": cid,
            "source_chunk_ids": [cid],
        }

        # Run validators
        is_ok, flags, default_reject = run_all_validators(
            candidate=cand_record,
            chunk_text=chunk.get("text", ""),
            existing_questions=existing_questions,
            all_chunks=chunks,
        )
        cand_record["validator_flags"] = flags
        cand_record["default_reject"] = default_reject
        candidates.append(cand_record)
        completed_chunk_ids.add(cid)
        existing_questions.append(res["question"])

        # Detailed item console print
        review_status = "REJECT" if default_reject else "ACCEPT"
        print(f"\n" + "-" * 78)
        print(f"ITEM [{idx}/{planned_single_calls}]: {item_id} | {comp} FY{yr} ({sec})")
        print(f"Chunk ID:         {cid}")
        print(f"Question Style:   {style}")
        print(f"Question:         {res['question']}")
        print(f"Ground Truth:     {res['ground_truth']}")
        print(f"Supporting Quote: {res['supporting_quote']}")
        print(f"Validator Status: {'PASSED' if is_ok else 'FAILED'}")
        print(f"Validator Flags:  {'; '.join(flags)}")
        print(f"Review Status:    {review_status}")
        print(f"Quota Ledger:     Used today: {ledger.get_used_today(bench_model)}/{ledger.get_limit(bench_model)} (Remaining: {ledger.get_remaining_budget(bench_model)})")
        print("-" * 78)

        # Write intermediate progress every 5 items
        if len(candidates) % 5 == 0:
            write_candidates_json(candidates, target_json)
            write_review_csv(candidates, target_csv, chunks=chunks)

    # 2. Multi-year generation (only if quota remains and not limited to single chunks)
    if not quota_exhausted and sampled_multi_groups:
        for idx, group in enumerate(sampled_multi_groups, start=1):
            group_key = tuple(sorted(c.get("chunk_id", "") for c in group))
            if group_key in completed_multi_keys:
                logger.info(f"[{idx}/{planned_multi_calls}] Multi-year group already generated; skipping.")
                continue

            if ledger.get_remaining_budget(bench_model) <= 0:
                logger.warning(f"Quota ledger daily limit reached today! Stopping gracefully. Resumable next day.")
                quota_exhausted = True
                break

            if new_calls_made > 0 and delay > 0:
                time.sleep(delay)

            item_id = f"{id_prefix}_mh_{idx:02d}"
            comp = group[0].get("company", "")
            yrs_str = "/".join(str(c.get("fiscal_year", "")) for c in group)
            secs_str = "/".join(c.get("item_section", "") for c in group)

            logger.info(f"[{idx}/{planned_multi_calls}] Generating multi-year comparison for {comp} FY{yrs_str}...")
            try:
                res = generate_multi_year_candidate_llm(group, bench_model=bench_model, ledger=ledger)
                new_calls_made += 1
            except Exception as e:
                logger.warning(f"LLM multi-year call interrupted: {e}")
                quota_exhausted = True
                break

            if not res:
                continue

            cand_record = {
                "id": item_id,
                "company": comp,
                "fiscal_year": yrs_str,
                "section": secs_str,
                "category": "multi_hop",
                "question_style": "comparison-of-two-figures style where the chunk allows",
                "question": res["question"],
                "ground_truth": res["ground_truth"],
                "supporting_quote": res["supporting_quote"],
                "source_chunk_ids": [c.get("chunk_id") for c in group],
                "validator_flags": ["PASS"],
                "default_reject": False,
            }
            candidates.append(cand_record)
            completed_multi_keys.add(group_key)
            existing_questions.append(res["question"])

    # 3. Add unanswerable questions (cost 0 LLM calls; only add if not already present)
    if unanswerable_items:
        existing_un_qs = {c.get("question") for c in candidates if c.get("category") == "unanswerable"}
        un_idx = 1
        for u in unanswerable_items:
            if u["question"] not in existing_un_qs:
                candidates.append({
                    "id": f"{id_prefix}_un_{un_idx:02d}",
                    "company": u["company"],
                    "fiscal_year": u["fiscal_year"],
                    "section": "None",
                    "category": "unanswerable",
                    "question_style": "unanswerable inquiry",
                    "question": u["question"],
                    "ground_truth": u["ground_truth"],
                    "supporting_quote": "",
                    "validator_flags": ["PASS"],
                    "default_reject": False,
                    "source_chunk_ids": [],
                })
            un_idx += 1

    write_candidates_json(candidates, target_json)
    write_review_csv(candidates, target_csv, chunks=chunks)

    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION PIPELINE STATUS")
    print("=" * 80)
    print(f"Total Saved Candidates:     {len(candidates)}")
    print(f"Output JSON:                {target_json}")
    print(f"Output Review CSV:          {target_csv}")
    print(f"New Calls Made This Run:    {new_calls_made}")
    print(f"Model:                      {bench_model}")
    print(f"Ledger Used Today:          {ledger.get_used_today(bench_model)} / {ledger.get_limit(bench_model)}")
    print(f"Remaining Budget Today:     {ledger.get_remaining_budget(bench_model)}")
    if quota_exhausted:
        print("NOTICE: Quota limit reached. Run the script again tomorrow to resume without re-asking existing items.")
    print("=" * 80 + "\n")

    return {
        "status": "generation_complete" if not quota_exhausted else "quota_paused",
        "total_generated": len(candidates),
        "output_json": str(target_json),
        "output_csv": str(target_csv),
        "calls_made": new_calls_made,
        "remaining_budget": ledger.get_remaining_budget(bench_model),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate candidates for evaluation benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned sampling, quota ledger usage, and generate dry-run templates with zero LLM calls.")
    parser.add_argument("--round", type=int, default=2, help="Benchmark round number (1 or 2; default: 2)")
    parser.add_argument("--exclude-golden", nargs="+", default=None, help="Paths to golden set JSON files to exclude with +/- 1 adjacent positions")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.json")
    parser.add_argument("--dev-set-path", type=str, default=None, help="Legacy path to dev golden_set.json (deprecated in favor of --exclude-golden)")
    parser.add_argument("--output-json", type=str, default=None, help="Output candidates JSON path")
    parser.add_argument("--output-csv", type=str, default=None, help="Output review CSV path")
    parser.add_argument("--model", type=str, default=os.getenv("BENCH_MODEL", "gemini-3.5-flash-lite"), help="BENCH_MODEL name")
    parser.add_argument("--max-calls", type=int, default=110, help="Maximum LLM calls cap (default: 110)")
    parser.add_argument("--target-single", type=int, default=None, help="Target number of single-chunk items (default: 130 for Round 2, 90 for Round 1)")
    parser.add_argument("--target-multi", type=int, default=None, help="Target number of multi-year items (default: 25 for Round 2, 12 for Round 1)")
    parser.add_argument("--target-unanswerable", type=int, default=None, help="Target number of unanswerable items (default: 15 for Round 2, 12 for Round 1)")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N planned single-chunk items")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between LLM calls (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    exclude_paths = [Path(p) for p in args.exclude_golden] if args.exclude_golden else None

    run_generation_pipeline(
        chunks_path=Path(args.chunks_path),
        exclude_golden_paths=exclude_paths,
        dev_set_path=Path(args.dev_set_path) if args.dev_set_path else None,
        output_json_path=Path(args.output_json) if args.output_json else None,
        output_csv_path=Path(args.output_csv) if args.output_csv else None,
        bench_model=args.model,
        max_calls=args.max_calls,
        round_num=args.round,
        target_single=args.target_single,
        target_multi=args.target_multi,
        target_unanswerable=args.target_unanswerable,
        limit=args.limit,
        delay=args.delay,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
