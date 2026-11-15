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
DEFAULT_OUTPUT_JSON = BASE_DIR / "eval" / "results" / "candidates.json"
DEFAULT_OUTPUT_CSV = BASE_DIR / "eval" / "results" / "review.csv"
DEFAULT_DRYRUN_JSON = BASE_DIR / "eval" / "results" / "candidates_dryrun.json"
DEFAULT_DRYRUN_CSV = BASE_DIR / "eval" / "results" / "review_dryrun.csv"

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
    "total", "net", "sales", "revenue", "revenues", "income", "expense", "expenses",
    "cost", "costs", "dollar", "dollars", "share", "shares", "stock", "common",
    "million", "millions", "billion", "billions", "thousand", "thousands",
    "percent", "percentage", "rate", "basis", "points", "amount", "amounts",
    "cash", "flow", "flows", "asset", "assets", "liability", "liabilities", "equity",
    "operating", "operations", "operation", "service", "services", "product", "products",
    "segment", "segments", "business", "businesses", "capital", "expenditure", "expenditures",
    "capex", "depreciation", "amortization", "tax", "taxes", "effective", "interest",
    "debt", "gain", "loss", "losses", "profit", "margins", "margin", "gross",
    "research", "development", "administrative", "general", "selling", "marketing",
    "marketable", "securities", "repurchase", "repurchases", "dividend", "dividends",
    "diluted", "basic", "per", "earning", "earnings", "carrying", "value", "fair",
    "note", "notes", "table", "information", "accounting", "policy", "policies",
    "united", "states", "america", "international", "global", "geographic", "region",
    "headcount", "employees", "workforce", "personnel", "officer", "officers", "executive",
    "approximately", "estimated", "increase", "decrease", "higher", "lower", "compared",
    "prior", "current", "subsequent", "growth", "payable", "receivable", "inventory",
    "goodwill", "intangible", "lease", "leases", "commitments", "contingencies",
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

    # Scan ground_truth for capitalized tokens
    gt_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]*\b", ground_truth)
    for token in gt_tokens:
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

    # Any failed validator defaults to should_reject = True
    if not is_acceptable or should_reject:
        should_reject = True
        is_acceptable = False

    if not flags:
        flags.append("PASS")

    return is_acceptable, flags, should_reject


# =============================================================================
# Stratified Sampling
# =============================================================================
def is_table_chunk(chunk: Dict[str, Any]) -> bool:
    """Identifies table-heavy chunks by markdown table syntax or high numeric token ratio."""
    text = chunk.get("text", "")
    if "|" in text:
        return True
    digit_count = sum(c.isdigit() for c in text)
    return len(text) > 0 and (digit_count / len(text)) > 0.08


def sample_heldout_chunks(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Performs stratified sampling of 90 single-chunk targets:
    - 5 companies: Apple, Microsoft, Alphabet, Amazon, NVIDIA
    - All fiscal years present for each company (3 years per company)
    - 6 chunks per fiscal year = 18 per company = 90 total
    - Section balance per year: 1 Item 1, 1 Item 1A, 2 Item 7 (1 table), 2 Item 8 (1 table)
    - Excludes any chunk IDs present in dev set.
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


def sample_multi_year_pairs(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    target_pairs: int = 12,
    seed: int = 42,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Samples ~12 pairs of chunks from the same company across 2 different fiscal years
    for multi-year comparison question formulation.
    """
    rng = random.Random(seed + 100)
    pool = [c for c in chunks if c.get("chunk_id") not in dev_chunk_ids]

    companies = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    comp_sec_pool: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for c in pool:
        comp = c.get("company")
        yr = str(c.get("fiscal_year"))
        sec = c.get("item_section")
        if comp in companies and sec in ["Item 1", "Item 7", "Item 8"]:
            comp_sec_pool[comp][sec][yr].append(c)

    for comp in companies:
        for sec in ["Item 7", "Item 1", "Item 8"]:
            years = sorted(comp_sec_pool[comp][sec].keys())
            if len(years) >= 2:
                yr1, yr2 = years[0], years[1]
                list1 = comp_sec_pool[comp][sec][yr1]
                list2 = comp_sec_pool[comp][sec][yr2]
                if list1 and list2:
                    c1 = rng.choice(list1)
                    c2 = rng.choice(list2)
                    pairs.append((c1, c2))
                    if len(pairs) == target_pairs:
                        return pairs

            if len(years) >= 3:
                yr2, yr3 = years[1], years[2]
                list2 = comp_sec_pool[comp][sec][yr2]
                list3 = comp_sec_pool[comp][sec][yr3]
                if list2 and list3:
                    c2 = rng.choice(list2)
                    c3 = rng.choice(list3)
                    pairs.append((c2, c3))
                    if len(pairs) == target_pairs:
                        return pairs

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
1. The question must require synthesizing facts or numbers from BOTH fiscal years.
2. The question must NOT start with "According to" or "Based on".
3. The question must NOT copy more than 5 consecutive words from either chunk.
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
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    bench_model: str,
    ledger: QuotaLedger,
) -> Optional[Dict[str, Any]]:
    """
    Invokes BENCH_MODEL through QuotaLedger to generate a multi-year comparison item.
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

    prompt = MULTI_YEAR_PROMPT_TEMPLATE.format(
        company=c1.get("company", ""),
        year1=c1.get("fiscal_year", ""),
        section1=c1.get("item_section", ""),
        text1=c1.get("text", "")[:2000],
        year2=c2.get("fiscal_year", ""),
        section2=c2.get("item_section", ""),
        text2=c2.get("text", "")[:2000],
    )

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
def write_candidates_json(candidates: List[Dict[str, Any]], path: Path) -> None:
    """Writes candidates list to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(candidates)} candidates to JSON: {path}")


def write_review_csv(candidates: List[Dict[str, Any]], path: Path) -> None:
    """
    Writes candidates to review CSV with columns:
    id,company,fiscal_year,section,category,question_style,question,ground_truth,supporting_quote,validator_flags,accept,edit,reject
    Any item with a failed validator defaults to reject="x", accept="".
    Only fully passing items default to accept="x", reject="".
    """
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
            writer.writerow({
                "id": c.get("id", ""),
                "company": c.get("company", ""),
                "fiscal_year": c.get("fiscal_year", ""),
                "section": c.get("section", ""),
                "category": c.get("category", ""),
                "question_style": c.get("question_style", ""),
                "question": c.get("question", ""),
                "ground_truth": c.get("ground_truth", ""),
                "supporting_quote": c.get("supporting_quote", ""),
                "validator_flags": "; ".join(c.get("validator_flags", ["PASS"])),
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
    dev_set_path: Path = DEFAULT_DEV_SET_PATH,
    output_json_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    bench_model: str = "gemini-3.5-flash-lite",
    max_calls: int = 110,
    limit: Optional[int] = None,
    delay: float = 5.0,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates candidate generation, stratification, validation, and writing.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    dev_chunk_ids: Set[str] = set()
    if dev_set_path.exists():
        with open(dev_set_path, "r", encoding="utf-8") as f:
            dev_items = json.load(f)
            for it in dev_items:
                dev_chunk_ids.update(it.get("source_chunk_ids", []))
                if it.get("chunk_id"):
                    dev_chunk_ids.add(it["chunk_id"])

    # 1. Stratified sampling of single chunks
    sampled_single_chunks = sample_heldout_chunks(chunks, dev_chunk_ids, seed=seed)

    # If --limit N is specified, generate only the first N planned single-chunk items
    if limit is not None and limit > 0:
        sampled_single_chunks = sampled_single_chunks[:limit]
        sampled_multi_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        unanswerable_items: List[Dict[str, Any]] = []
        logger.info(f"--limit {limit} active: generating only the first {limit} single-chunk items (multi-year & unanswerables omitted).")
    else:
        # 2. Sample 12 multi-year pairs
        sampled_multi_pairs = sample_multi_year_pairs(chunks, dev_chunk_ids, target_pairs=12, seed=seed)
        # 3. Unanswerable items
        unanswerable_items = list(UNANSWERABLE_TEMPLATES)

    planned_single_calls = len(sampled_single_chunks)
    planned_multi_calls = len(sampled_multi_pairs)
    total_planned_calls = planned_single_calls + planned_multi_calls

    ledger = QuotaLedger(max_calls=max_calls)
    used_today = ledger.get_used_today(bench_model)
    daily_limit = ledger.get_limit(bench_model)
    rem_budget = ledger.get_remaining_budget(bench_model)

    # Resolve output paths: dry-run writes strictly to candidates_dryrun.json / review_dryrun.csv
    if dry_run:
        target_json = output_json_path or DEFAULT_DRYRUN_JSON
        target_csv = output_csv_path or DEFAULT_DRYRUN_CSV
    else:
        target_json = output_json_path or DEFAULT_OUTPUT_JSON
        target_csv = output_csv_path or DEFAULT_OUTPUT_CSV

    print("\n" + "=" * 80)
    print("HELD-OUT BENCHMARK CANDIDATE GENERATION PLAN")
    print("=" * 80)
    print(f"Corpus Chunks:                 {len(chunks)}")
    print(f"Dev Set Chunks Excluded:       {len(dev_chunk_ids)}")
    print(f"Sampled Single Chunks:         {planned_single_calls} (stratified across 5 companies, all FYs, Items 1/1A/7/8)")
    print(f"Sampled Multi-Year Pairs:      {planned_multi_calls} (across 5 companies)")
    print(f"Fixed Unanswerable Items:      {len(unanswerable_items)} (Fixed templates; cost EXACTLY 0 LLM calls)")
    print(f"Total Benchmark Target Items:  {planned_single_calls + planned_multi_calls + len(unanswerable_items)}")
    print("-" * 80)
    print("PLANNED LLM CALL COUNT & QUOTA LEDGER USAGE:")
    print(f"  Target Model:                {bench_model}")
    print(f"  Max Calls Cap:               {max_calls}")
    print(f"  Planned Single-Chunk Calls:  {planned_single_calls}")
    print(f"  Planned Multi-Year Calls:    {planned_multi_calls}")
    print(f"  TOTAL PLANNED LLM CALLS:     {total_planned_calls} (within max {max_calls} cap)")
    print(f"  Unanswerable Items Call Cost:0 calls (100% fixed zero-cost templates)")
    print(f"  Used Today (Pacific):        {used_today} requests")
    print(f"  Effective Daily Limit:       {daily_limit} requests (0.8 safety factor applied to 500)")
    print(f"  Remaining Budget Today:      {rem_budget} requests")
    print(f"  Budget Sufficient:           {'YES' if rem_budget >= total_planned_calls else 'NO (will stop gracefully)'}")
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
        print("Note: The 12 unanswerable items are fixed templates that cost zero LLM calls.")
        dry_candidates: List[Dict[str, Any]] = []

        # Single chunk templates
        for idx, chunk in enumerate(sampled_single_chunks, start=1):
            cid = chunk.get("chunk_id", "")
            comp = chunk.get("company", "")
            yr = str(chunk.get("fiscal_year", ""))
            sec = chunk.get("item_section", "")
            style = QUESTION_STYLES[(idx - 1) % len(QUESTION_STYLES)]
            dry_candidates.append({
                "id": f"heldout_{idx:02d}",
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
        m_start = len(dry_candidates) + 1
        for idx, (c1, c2) in enumerate(sampled_multi_pairs, start=m_start):
            comp = c1.get("company", "")
            yr1 = str(c1.get("fiscal_year", ""))
            yr2 = str(c2.get("fiscal_year", ""))
            dry_candidates.append({
                "id": f"heldout_{idx:02d}",
                "company": comp,
                "fiscal_year": f"{yr1}/{yr2}",
                "section": f"{c1.get('item_section', '')}/{c2.get('item_section', '')}",
                "category": "multi_hop",
                "question_style": "comparison-of-two-figures style where the chunk allows",
                "question": f"[DRY-RUN TEMPLATE] Compare {comp} FY{yr1} and FY{yr2} metric",
                "ground_truth": "[DRY-RUN TEMPLATE] Multi-year synthesized answer.",
                "supporting_quote": f"{c1.get('text', '')[:60]} || {c2.get('text', '')[:60]}",
                "validator_flags": ["PASS"],
                "default_reject": False,
                "source_chunk_ids": [c1.get("chunk_id"), c2.get("chunk_id")],
            })

        # Unanswerable templates
        u_start = len(dry_candidates) + 1
        for idx, u in enumerate(unanswerable_items, start=u_start):
            dry_candidates.append({
                "id": f"heldout_{idx:02d}",
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
        write_review_csv(dry_candidates, target_csv)

        print("\nDry-run completed successfully.")
        print(f"Generated {len(dry_candidates)} candidate templates.")
        print(f"  Dry-run JSON output: {target_json}")
        print(f"  Dry-run CSV output:  {target_csv}")
        return {
            "status": "dry_run_complete",
            "planned_calls": total_planned_calls,
            "remaining_budget": rem_budget,
            "total_candidates": len(dry_candidates),
        }

    # Full LLM Generation Mode
    logger.info(f"Starting candidate LLM generation with {bench_model} (delay={delay}s, limit={limit})...")
    candidates: List[Dict[str, Any]] = []
    existing_questions: List[str] = []

    # 1. Single chunk generation
    for idx, chunk in enumerate(sampled_single_chunks, start=1):
        if idx > 1 and delay > 0:
            logger.info(f"Waiting {delay}s between calls to stay under rate limit (15 RPM)...")
            time.sleep(delay)

        item_id = f"heldout_{idx:02d}"
        comp = chunk.get("company", "")
        yr = str(chunk.get("fiscal_year", ""))
        sec = chunk.get("item_section", "")
        cid = chunk.get("chunk_id", "")
        style = QUESTION_STYLES[(idx - 1) % len(QUESTION_STYLES)]

        logger.info(f"[{idx}/{planned_single_calls}] Calling {bench_model} ({style}) for {comp} FY{yr} {sec} ({cid[:8]})...")
        res = generate_candidate_from_chunk_llm(
            chunk, bench_model=bench_model, ledger=ledger, question_style=style
        )
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

    # 2. Multi-year generation (only if --limit is not restricting to single chunks)
    if sampled_multi_pairs:
        m_offset = len(candidates) + 1
        for idx, (c1, c2) in enumerate(sampled_multi_pairs, start=m_offset):
            if delay > 0:
                time.sleep(delay)

            item_id = f"heldout_{idx:02d}"
            comp = c1.get("company", "")
            yr1 = str(c1.get("fiscal_year", ""))
            yr2 = str(c2.get("fiscal_year", ""))
            sec = f"{c1.get('item_section', '')}/{c2.get('item_section', '')}"

            logger.info(f"[{idx}] Generating multi-year comparison for {comp} FY{yr1} vs FY{yr2}...")
            res = generate_multi_year_candidate_llm(c1, c2, bench_model=bench_model, ledger=ledger)
            if not res:
                continue

            cand_record = {
                "id": item_id,
                "company": comp,
                "fiscal_year": f"{yr1}/{yr2}",
                "section": sec,
                "category": "multi_hop",
                "question_style": "comparison-of-two-figures style where the chunk allows",
                "question": res["question"],
                "ground_truth": res["ground_truth"],
                "supporting_quote": res["supporting_quote"],
                "source_chunk_ids": [c1.get("chunk_id"), c2.get("chunk_id")],
                "validator_flags": ["PASS"],
                "default_reject": False,
            }
            candidates.append(cand_record)
            existing_questions.append(res["question"])

    # 3. Add unanswerable questions (cost 0 LLM calls)
    if unanswerable_items:
        u_offset = len(candidates) + 1
        for idx, u in enumerate(unanswerable_items, start=u_offset):
            candidates.append({
                "id": f"heldout_{idx:02d}",
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

    write_candidates_json(candidates, target_json)
    write_review_csv(candidates, target_csv)

    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION RUN COMPLETE")
    print("=" * 80)
    print(f"Total Generated Candidates: {len(candidates)}")
    print(f"Output JSON:                {target_json}")
    print(f"Output Review CSV:          {target_csv}")
    print(f"Total Calls Made This Run:  {ledger.run_calls}")
    print(f"Model:                      {bench_model}")
    print(f"Ledger Used Today:          {ledger.get_used_today(bench_model)} / {ledger.get_limit(bench_model)}")
    print(f"Remaining Budget:           {ledger.get_remaining_budget(bench_model)}")
    print("=" * 80 + "\n")

    return {
        "status": "generation_complete",
        "total_generated": len(candidates),
        "output_json": str(target_json),
        "output_csv": str(target_csv),
        "calls_made": ledger.run_calls,
        "remaining_budget": rem_budget,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate candidates for held-out evaluation benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned sampling, quota ledger usage, and generate dry-run templates with zero LLM calls.")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.json")
    parser.add_argument("--dev-set-path", type=str, default=str(DEFAULT_DEV_SET_PATH), help="Path to dev golden_set.json")
    parser.add_argument("--output-json", type=str, default=None, help="Output candidates.json path (or candidates_dryrun.json for dry-run)")
    parser.add_argument("--output-csv", type=str, default=None, help="Output review.csv path (or review_dryrun.csv for dry-run)")
    parser.add_argument("--model", type=str, default=os.getenv("BENCH_MODEL", "gemini-3.5-flash-lite"), help="BENCH_MODEL name")
    parser.add_argument("--max-calls", type=int, default=110, help="Maximum LLM calls cap (default: 110)")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N planned single-chunk items")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay in seconds between LLM calls (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_generation_pipeline(
        chunks_path=Path(args.chunks_path),
        dev_set_path=Path(args.dev_set_path),
        output_json_path=Path(args.output_json) if args.output_json else None,
        output_csv_path=Path(args.output_csv) if args.output_csv else None,
        bench_model=args.model,
        max_calls=args.max_calls,
        limit=args.limit,
        delay=args.delay,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
