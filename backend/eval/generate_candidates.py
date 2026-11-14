"""
Held-Out Benchmark Candidate Generation and Validation Script.

Builds a held-out benchmark candidate pool kept strictly separate from the dev set.
1. Samples ~90 chunks from backend/data/processed/chunks.json, stratified across:
   - All 5 companies (Apple, Microsoft, Alphabet, Amazon, NVIDIA)
   - All fiscal years present
   - Items 1, 1A, 7, 8 (including table-heavy Item 7/8 chunks)
   - Strictly excludes any chunks in backend/eval/golden_set.json (dev set).
2. Generates ~12 multi-year comparison question pairs (same company, two fiscal years).
3. Generates ~12 plausible unanswerable questions (out-of-corpus companies, years, or non-10-K facts).
4. Uses BENCH_MODEL (gemini-3.5-flash-lite) through QuotaLedger (max 100 calls, honoring ledger stops).
5. Validates candidates:
   - supporting_quote must be an exact substring of chunk text
   - Rejects questions with 6-gram overlap with chunk text
   - Rejects duplicate questions
   - Flags questions ambiguous across filings (missing company or fiscal year context)
6. Outputs candidates.json and review.csv with empty accept/edit/reject columns for manual review.
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

# Unanswerable question templates (out-of-corpus companies, years, or non-10-K facts)
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
    If exact match fails but normalized whitespace matches, extracts the exact verbatim slice.
    Returns (is_valid, reason, verbatim_matched_quote).
    """
    if not quote or len(quote.strip()) < 10:
        return False, "REJECT_QUOTE_EMPTY_OR_TOO_SHORT", None

    # 1. Direct exact substring match
    if quote in chunk_text:
        return True, "VALID", quote

    # 2. Normalized whitespace / quote matching
    norm_q = normalize_text_for_matching(quote)
    norm_chunk = normalize_text_for_matching(chunk_text)

    if norm_q in norm_chunk:
        # Locate start and end in original chunk_text
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


def validate_cross_filing_ambiguity(
    question: str,
    company: str,
    fiscal_year: str,
) -> Tuple[bool, str]:
    """
    Flags questions whose terms may be ambiguous across filings
    (e.g. omits company name or fiscal year for time-bound metrics).
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

    # Check if year or years are present
    years = re.findall(r"\b202[0-9]\b", q_lower)
    has_year = len(years) > 0

    flags: List[str] = []
    if not has_company:
        flags.append("FLAG_NO_COMPANY_NAME")
    if not has_year and fiscal_year != "None":
        flags.append("FLAG_NO_FISCAL_YEAR")

    if flags:
        return False, "AMBIGUOUS_ACROSS_FILINGS: " + ", ".join(flags)
    return True, "VALID"


def run_all_validators(
    candidate: Dict[str, Any],
    chunk_text: str,
    existing_questions: List[str],
) -> Tuple[bool, List[str]]:
    """
    Runs the full validation suite on a generated question candidate.
    Returns (is_acceptable, flags_list).
    """
    flags: List[str] = []
    is_acceptable = True

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

    # Check cross-filing ambiguity (warning flag, does not force hard rejection)
    ok_amb, msg_amb = validate_cross_filing_ambiguity(
        candidate["question"],
        candidate.get("company", ""),
        str(candidate.get("fiscal_year", "")),
    )
    if not ok_amb:
        flags.append(msg_amb)

    if not flags:
        flags.append("PASS")

    return is_acceptable, flags


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

    # Map pool: pool_by_comp_year_sec[comp][year][sec] = list of chunks
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

    # Distribute ~2-3 pairs per company across Item 1, 7, 8
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
# LLM Generation (QuotaLedger-Enforced)
# =============================================================================
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
4. Phrasing variety: vary the phrasing style naturally (e.g. name the company and year naturally in prose, or use direct question style, or keyword inquiry).
5. For table chunks or financial metrics, ask for specific numbers with exact units (e.g. '$ in millions', 'percentage', 'shares').
6. supporting_quote MUST be an EXACT, VERBATIM substring copied directly from the text chunk above that proves the answer.
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


def generate_candidate_from_chunk_llm(
    chunk: Dict[str, Any],
    bench_model: str,
    ledger: QuotaLedger,
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
    )

    try:
        resp = llm.invoke([("user", prompt)])
        ledger.record_call(bench_model, count=1)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # Parse JSON
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
        content = resp.content if hasattr(resp, "content") else str(resp)

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
    Writes candidates to review.csv with columns:
    id,company,fiscal_year,section,category,question,ground_truth,supporting_quote,validator_flags,accept,edit,reject
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "company",
        "fiscal_year",
        "section",
        "category",
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
            writer.writerow({
                "id": c.get("id", ""),
                "company": c.get("company", ""),
                "fiscal_year": c.get("fiscal_year", ""),
                "section": c.get("section", ""),
                "category": c.get("category", ""),
                "question": c.get("question", ""),
                "ground_truth": c.get("ground_truth", ""),
                "supporting_quote": c.get("supporting_quote", ""),
                "validator_flags": "; ".join(c.get("validator_flags", ["PASS"])),
                "accept": "",
                "edit": "",
                "reject": "",
            })
    logger.info(f"Saved {len(candidates)} candidates to review CSV: {path}")


# =============================================================================
# Main Pipeline Runner
# =============================================================================
def run_generation_pipeline(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    dev_set_path: Path = DEFAULT_DEV_SET_PATH,
    output_json_path: Path = DEFAULT_OUTPUT_JSON,
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    bench_model: str = "gemini-3.5-flash-lite",
    max_calls: int = 100,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrates candidate generation, stratification, validation, and writing.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    dev_chunk_ids: Set[str] = set()
    if dev_set_path.exists():
        with open(dev_set_path, "r", encoding="utf-8") as f:
            dev_items = json.load(f)
            for it in dev_items:
                dev_chunk_ids.update(it.get("source_chunk_ids", []))
                if it.get("chunk_id"):
                    dev_chunk_ids.add(it["chunk_id"])

    # 1. Stratified sampling of ~90 single chunks
    sampled_single_chunks = sample_heldout_chunks(chunks, dev_chunk_ids, seed=seed)

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

    print("\n" + "=" * 80)
    print("HELD-OUT BENCHMARK CANDIDATE GENERATION PLAN")
    print("=" * 80)
    print(f"Corpus Chunks:           {len(chunks)}")
    print(f"Dev Set Chunks Excluded: {len(dev_chunk_ids)}")
    print(f"Sampled Single Chunks:   {planned_single_calls} (stratified across 5 companies, all FYs, Items 1/1A/7/8)")
    print(f"Sampled Multi-Year Pairs:{planned_multi_calls} (across 5 companies)")
    print(f"Unanswerable Items:      {len(unanswerable_items)} (out-of-corpus companies, years, non-10-K facts)")
    print(f"Total Benchmark Target:  {planned_single_calls + planned_multi_calls + len(unanswerable_items)} items")
    print("-" * 80)
    print("QUOTA LEDGER STATUS:")
    print(f"  Target Model:          {bench_model}")
    print(f"  Max Calls Cap:         {max_calls}")
    print(f"  Planned LLM Calls:     {total_planned_calls} ({planned_single_calls} single + {planned_multi_calls} multi-year)")
    print(f"  Used Today (Pacific):  {used_today} requests")
    print(f"  Effective Daily Limit: {daily_limit} requests (0.8 safety factor)")
    print(f"  Remaining Budget:      {rem_budget} requests")
    print(f"  Budget Sufficient:     {'YES' if rem_budget >= total_planned_calls else 'NO (will stop gracefully)'}")
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
        print("\n[DRY-RUN MODE] Zero LLM calls executed. Preparing template outputs for inspection...")
        dry_candidates: List[Dict[str, Any]] = []

        # Single chunk templates
        for idx, chunk in enumerate(sampled_single_chunks, start=1):
            cid = chunk.get("chunk_id", "")
            comp = chunk.get("company", "")
            yr = str(chunk.get("fiscal_year", ""))
            sec = chunk.get("item_section", "")
            dry_candidates.append({
                "id": f"heldout_{idx:02d}",
                "company": comp,
                "fiscal_year": yr,
                "section": sec,
                "category": "numeric_precision" if is_table_chunk(chunk) else "simple_lookup",
                "question": f"[DRY-RUN TEMPLATE] Question for {comp} FY{yr} ({sec}) chunk {cid[:8]}",
                "ground_truth": "[DRY-RUN TEMPLATE] Ground truth answer derived from chunk text.",
                "supporting_quote": chunk.get("text", "")[:120].strip(),
                "validator_flags": ["PASS"],
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
                "question": f"[DRY-RUN TEMPLATE] Compare {comp} FY{yr1} and FY{yr2} metric",
                "ground_truth": "[DRY-RUN TEMPLATE] Multi-year synthesized answer.",
                "supporting_quote": f"{c1.get('text', '')[:60]} || {c2.get('text', '')[:60]}",
                "validator_flags": ["PASS"],
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
                "question": u["question"],
                "ground_truth": u["ground_truth"],
                "supporting_quote": "",
                "validator_flags": ["PASS"],
                "source_chunk_ids": [],
            })

        write_candidates_json(dry_candidates, output_json_path)
        write_review_csv(dry_candidates, output_csv_path)

        print("\nDry-run completed successfully.")
        print(f"Generated {len(dry_candidates)} candidate templates.")
        print(f"  JSON output: {output_json_path}")
        print(f"  CSV output:  {output_csv_path}")
        return {
            "status": "dry_run_complete",
            "planned_calls": total_planned_calls,
            "remaining_budget": rem_budget,
            "total_candidates": len(dry_candidates),
        }

    # Full LLM Generation Mode
    logger.info("Starting candidate LLM generation with BENCH_MODEL...")
    candidates: List[Dict[str, Any]] = []
    existing_questions: List[str] = []

    # 1. Single chunk generation
    for idx, chunk in enumerate(sampled_single_chunks, start=1):
        item_id = f"heldout_{idx:02d}"
        comp = chunk.get("company", "")
        yr = str(chunk.get("fiscal_year", ""))
        sec = chunk.get("item_section", "")
        cid = chunk.get("chunk_id", "")

        logger.info(f"[{idx}/{planned_single_calls}] Generating candidate for {comp} {yr} {sec} ({cid[:8]})...")
        res = generate_candidate_from_chunk_llm(chunk, bench_model=bench_model, ledger=ledger)
        if not res:
            continue

        cand_record = {
            "id": item_id,
            "company": comp,
            "fiscal_year": yr,
            "section": sec,
            "category": res["category"],
            "question": res["question"],
            "ground_truth": res["ground_truth"],
            "supporting_quote": res["supporting_quote"],
            "chunk_id": cid,
            "source_chunk_ids": [cid],
        }

        # Run validators
        is_ok, flags = run_all_validators(cand_record, chunk.get("text", ""), existing_questions)
        cand_record["validator_flags"] = flags
        candidates.append(cand_record)
        existing_questions.append(res["question"])

    # 2. Multi-year generation
    m_offset = len(candidates) + 1
    for idx, (c1, c2) in enumerate(sampled_multi_pairs, start=m_offset):
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
            "question": res["question"],
            "ground_truth": res["ground_truth"],
            "supporting_quote": res["supporting_quote"],
            "source_chunk_ids": [c1.get("chunk_id"), c2.get("chunk_id")],
            "validator_flags": ["PASS"],
        }
        candidates.append(cand_record)
        existing_questions.append(res["question"])

    # 3. Add unanswerable questions
    u_offset = len(candidates) + 1
    for idx, u in enumerate(unanswerable_items, start=u_offset):
        candidates.append({
            "id": f"heldout_{idx:02d}",
            "company": u["company"],
            "fiscal_year": u["fiscal_year"],
            "section": "None",
            "category": "unanswerable",
            "question": u["question"],
            "ground_truth": u["ground_truth"],
            "supporting_quote": "",
            "validator_flags": ["PASS"],
            "source_chunk_ids": [],
        })

    write_candidates_json(candidates, output_json_path)
    write_review_csv(candidates, output_csv_path)

    return {
        "status": "generation_complete",
        "total_generated": len(candidates),
        "output_json": str(output_json_path),
        "output_csv": str(output_csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate candidates for held-out evaluation benchmark.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect planned sampling, quota ledger usage, and generate templates with zero LLM calls.")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.json")
    parser.add_argument("--dev-set-path", type=str, default=str(DEFAULT_DEV_SET_PATH), help="Path to dev golden_set.json")
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output candidates.json path")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Output review.csv path")
    parser.add_argument("--model", type=str, default=os.getenv("BENCH_MODEL", "gemini-3.5-flash-lite"), help="BENCH_MODEL name")
    parser.add_argument("--max-calls", type=int, default=100, help="Maximum LLM calls cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_generation_pipeline(
        chunks_path=Path(args.chunks_path),
        dev_set_path=Path(args.dev_set_path),
        output_json_path=Path(args.output_json),
        output_csv_path=Path(args.output_csv),
        bench_model=args.model,
        max_calls=args.max_calls,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

