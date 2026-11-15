"""
Post-processing and multi-year regeneration for Verity benchmark candidates.

Rules:
1. Missing company: For every candidate whose question lacks the company name,
   prepend "At {Company}, " and lowercase the first letter of original question.
   Keep original_question in metadata and log every rewrite. Keyword-style items
   that already name the company are untouched.
2. Repeated across years: For items flagged AMBIGUOUS_REPEATED_ACROSS_YEARS,
   append a disambiguating phrase to the question, rotating through:
   - "per {Company}'s fiscal {year} annual report"
   - "in {Company}'s FY{year} 10-K"
   - "for fiscal {year}"
   Re-run every validator on rewritten question. If it passes, set accept and record
   rewrite. If it fails another validator, leave rejected with reason.
3. Comparison style: If question_style is "comparison-of-two-figures style where the chunk allows"
   and ground truth contains fewer than two distinct numeric figures (or question asks
   to compare something the supporting quote does not contain), reject with reason
   REJECT_COMPARISON_NOT_SUPPORTED.
4. Multi-year: Discard existing 12 multi-year candidates and regenerate 14 items
   covering all 5 companies (at least 2 per company, Apple at most 2, distinct pairs)
   through QuotaLedger (BENCH_MODEL, --delay 5, at most 20 calls).
   Each must pass verbatim-quote validators for BOTH chunks.
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

from backend.eval.generate_candidates import (
    MULTI_YEAR_PROMPT_TEMPLATE,
    _extract_content_text,
    extract_question_years,
    run_all_validators,
    validate_6gram_overlap,
    validate_answer_leakage,
    validate_ground_truth_supported_by_quotes,
    validate_multi_year_adjacent_years,
    validate_no_preamble,
    validate_question_year_matches_filing,
    validate_supporting_quote,
    write_candidates_json,
    write_review_csv,
)
from backend.eval.quota_ledger import QuotaLedger

DEFAULT_BENCH_MODEL = "gemini-3.5-flash-lite"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CANDIDATES_PATH = BASE_DIR / "eval" / "results" / "candidates.json"
DEFAULT_REVIEW_PATH = BASE_DIR / "eval" / "results" / "review.csv"
DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DEFAULT_DEV_PATH = BASE_DIR / "eval" / "golden_set.json"

DISAMBIGUATION_PHRASES: List[str] = [
    "per {company}'s fiscal {year} annual report",
    "in {company}'s FY{year} 10-K",
    "for fiscal {year}",
]


def count_distinct_numeric_figures(text: str) -> int:
    """
    Extracts distinct numeric figures from text, ignoring standalone 4-digit years.
    Handles integers, floats, currency amounts, and percentages.
    """
    raw_nums = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text)
    cleaned = set()
    for n in raw_nums:
        val = n.replace(",", "")
        # Ignore standalone 4-digit calendar years 1990-2040
        if re.fullmatch(r"(?:19|20)\d\d", val):
            continue
        cleaned.add(val)
    return len(cleaned)


def fix_repeated_across_years(
    candidates: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Disambiguates items flagged AMBIGUOUS_REPEATED_ACROSS_YEARS by appending
    a rotating phrase, re-validating the item, and accepting if all validators pass.
    """
    chunk_lookup = {c.get("chunk_id"): c for c in all_chunks if c.get("chunk_id")}
    rewritten_count = 0
    now_passed_count = 0

    existing_questions = [c["question"] for c in candidates if "question" in c]

    for idx, c in enumerate(candidates):
        flags = c.get("validator_flags", [])
        if not any("AMBIGUOUS_REPEATED_ACROSS_YEARS" in f for f in flags):
            continue

        comp = c.get("company", "")
        yr = str(c.get("fiscal_year", ""))
        phrase_template = DISAMBIGUATION_PHRASES[rewritten_count % len(DISAMBIGUATION_PHRASES)]
        phrase = phrase_template.format(company=comp, year=yr)

        orig_q = c.get("question", "").strip()
        if "original_question" not in c:
            c["original_question"] = orig_q

        # Append phrase cleanly before terminal question mark if present
        if orig_q.endswith("?"):
            q_base = orig_q[:-1].rstrip()
            new_q = f"{q_base}, {phrase}?"
        else:
            new_q = f"{orig_q}, {phrase}"

        c["question"] = new_q
        rewritten_count += 1
        logger.info(f"[{c.get('id')}] Disambiguating across years: '{orig_q}' -> '{new_q}'")

        # Find chunk text
        cid = c.get("chunk_id") or (c.get("source_chunk_ids", [None])[0])
        chunk_obj = chunk_lookup.get(cid, {})
        chunk_text = chunk_obj.get("text", "")

        # Re-run all validators
        other_questions = [q for q in existing_questions if q != orig_q]
        is_ok, new_flags, should_reject = run_all_validators(
            candidate=c,
            chunk_text=chunk_text,
            existing_questions=other_questions,
            all_chunks=all_chunks,
        )

        c["validator_flags"] = new_flags
        if is_ok and not should_reject:
            c["default_reject"] = False
            c["rewrite_applied"] = f"Disambiguated repeated quote: appended '{phrase}'"
            now_passed_count += 1
            logger.info(f"[{c.get('id')}] Now PASSED after year disambiguation.")
        else:
            c["default_reject"] = True
            c["rewrite_applied"] = f"Disambiguated repeated quote: appended '{phrase}' (remains rejected: {'; '.join(new_flags)})"
            logger.warning(f"[{c.get('id')}] Remains rejected after year disambiguation: {new_flags}")

    return candidates, rewritten_count, now_passed_count


def fix_missing_company(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    For every candidate whose question lacks the company name,
    prepend "At {Company}, " and lowercase the first letter of the original question.
    Keyword-style items that already name the company are untouched.
    """
    rewritten_count = 0
    for c in candidates:
        comp = c.get("company", "")
        q = c.get("question", "").strip()
        if not comp or not q:
            continue

        # Check if question contains the company name
        if comp.lower() in q.lower():
            continue

        if "original_question" not in c:
            c["original_question"] = q

        # Prepend "At {Company}, " and lowercase first letter of original question
        first_char = q[0].lower()
        rest = q[1:]
        new_q = f"At {comp}, {first_char}{rest}"
        c["question"] = new_q

        existing_rewrite = c.get("rewrite_applied")
        rewrite_note = f"Prepended 'At {comp}, '"
        c["rewrite_applied"] = f"{existing_rewrite}; {rewrite_note}" if existing_rewrite else rewrite_note

        rewritten_count += 1
        logger.info(f"[{c.get('id')}] Added missing company: '{q}' -> '{new_q}'")

    return candidates, rewritten_count


def fix_comparison_style(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    For items with question_style == "comparison-of-two-figures style where the chunk allows",
    rejects if ground truth contains fewer than two distinct numeric figures.
    """
    rejected_count = 0
    for c in candidates:
        style = c.get("question_style", "")
        if "comparison-of-two-figures" not in style:
            continue

        # Ignore multi-year synthesized items (handled by multi-year validation)
        if len(c.get("source_chunk_ids", [])) > 1 or c.get("category") == "multi_hop":
            continue

        gt = c.get("ground_truth", "")
        distinct_figures = count_distinct_numeric_figures(gt)
        if distinct_figures < 2:
            c["default_reject"] = True
            flags = [f for f in c.get("validator_flags", []) if f != "PASS"]
            if not any("REJECT_COMPARISON_NOT_SUPPORTED" in f for f in flags):
                flags.append(f"REJECT_COMPARISON_NOT_SUPPORTED (found {distinct_figures} distinct figures in GT)")
            c["validator_flags"] = flags
            rejected_count += 1
            logger.info(f"[{c.get('id')}] Rejected comparison style: only {distinct_figures} distinct figures in GT: '{gt[:50]}...'")

    return candidates, rejected_count


def sample_multi_year_candidate_pairs(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    targets: Dict[str, int],
    seed: int = 42,
) -> Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """
    Samples distinct chunk pairs for each company across different fiscal years.
    Returns a pool of pairs for each company.
    """
    rng = random.Random(seed + 77)
    pool = [c for c in chunks if c.get("chunk_id") not in dev_chunk_ids and len(c.get("text", "")) > 300]

    by_comp_sec_yr: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for c in pool:
        comp = c.get("company")
        yr = str(c.get("fiscal_year"))
        sec = c.get("item_section")
        if comp in targets and sec in ["Item 7", "Item 8", "Item 1", "Item 1A"]:
            by_comp_sec_yr[comp][sec][yr].append(c)

    pairs_by_company: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)

    for comp, target_n in targets.items():
        comp_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen_pair_ids: Set[Tuple[str, str]] = set()

        # Prioritize matching sections across distinct fiscal years
        sections = ["Item 7", "Item 8", "Item 1", "Item 1A"]
        for sec in sections:
            years = sorted(by_comp_sec_yr[comp][sec].keys())
            if len(years) < 2:
                continue

            # Generate adjacent combinations of years only
            year_combos = []
            for i in range(len(years) - 1):
                try:
                    if abs(int(years[i]) - int(years[i + 1])) == 1:
                        year_combos.append((years[i], years[i + 1]))
                except ValueError:
                    pass
            for y1, y2 in year_combos:
                if y1 == y2:
                    continue
                list1 = by_comp_sec_yr[comp][sec][y1]
                list2 = by_comp_sec_yr[comp][sec][y2]
                if not list1 or not list2:
                    continue

                shuffled1 = list(list1)
                shuffled2 = list(list2)
                rng.shuffle(shuffled1)
                rng.shuffle(shuffled2)

                for c1 in shuffled1[:3]:
                    for c2 in shuffled2[:3]:
                        p_key = (c1["chunk_id"], c2["chunk_id"])
                        if p_key not in seen_pair_ids:
                            seen_pair_ids.add(p_key)
                            comp_pairs.append((c1, c2))

        rng.shuffle(comp_pairs)
        pairs_by_company[comp] = comp_pairs

    return pairs_by_company


def generate_single_multi_year_candidate(
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    bench_model: str,
    ledger: QuotaLedger,
) -> Optional[Dict[str, Any]]:
    """
    Invokes BENCH_MODEL through QuotaLedger to formulate a multi-year comparison item.
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
        if isinstance(quotes, str):
            quotes = [q.strip() for q in quotes.split("||")]
        return {
            "question": data.get("question", "").strip(),
            "ground_truth": data.get("ground_truth_answer", "").strip(),
            "supporting_quotes": quotes,
            "category": "multi_hop",
        }
    except Exception as e:
        logger.warning(f"Multi-year LLM generation error: {e}")
        return None


def validate_multi_year_item(
    item: Dict[str, Any],
    c1: Dict[str, Any],
    c2: Dict[str, Any],
    existing_questions: List[str],
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Validates a multi-year comparison item against both chunks:
    - Verbatim quote substring in both chunks (no ellipses).
    - No preamble.
    - No 6-gram overlap with either chunk.
    - No answer leakage.
    - Company and years named in question.
    Returns (is_valid, flags, combined_verbatim_quote).
    """
    flags: List[str] = []
    is_valid = True

    quotes = item.get("supporting_quotes", [])
    if len(quotes) < 2:
        return False, ["REJECT_MULTI_YEAR_NEEDS_TWO_QUOTES"], None

    q1, q2 = str(quotes[0]), str(quotes[1])

    # Validate quote 1 on chunk 1
    ok1, r1, v1 = validate_supporting_quote(q1, c1.get("text", ""))
    if not ok1:
        flags.append(f"CHUNK1_QUOTE_FAIL: {r1}")
        is_valid = False

    # Validate quote 2 on chunk 2
    ok2, r2, v2 = validate_supporting_quote(q2, c2.get("text", ""))
    if not ok2:
        flags.append(f"CHUNK2_QUOTE_FAIL: {r2}")
        is_valid = False

    # Check preamble
    ok_pre, msg_pre = validate_no_preamble(item["question"])
    if not ok_pre:
        flags.append(msg_pre)
        is_valid = False

    # Check 6-gram overlap with both chunks
    ok_6g1, msg_6g1 = validate_6gram_overlap(item["question"], c1.get("text", ""))
    if not ok_6g1:
        flags.append(f"CHUNK1_{msg_6g1}")
        is_valid = False
    ok_6g2, msg_6g2 = validate_6gram_overlap(item["question"], c2.get("text", ""))
    if not ok_6g2:
        flags.append(f"CHUNK2_{msg_6g2}")
        is_valid = False

    # Check answer leakage
    ok_leak, msg_leak = validate_answer_leakage(
        question=item["question"],
        ground_truth=item["ground_truth"],
        fiscal_year=f"{c1.get('fiscal_year')}/{c2.get('fiscal_year')}",
        company=c1.get("company", ""),
    )
    if not ok_leak:
        flags.append(msg_leak)
        is_valid = False

    # Check adjacent fiscal years
    yr1_str = str(c1.get("fiscal_year", ""))
    yr2_str = str(c2.get("fiscal_year", ""))
    ok_adj, msg_adj = validate_multi_year_adjacent_years(f"{yr1_str}/{yr2_str}")
    if not ok_adj:
        flags.append(msg_adj)
        is_valid = False

    # Check question fiscal years match filing years
    ok_yr, msg_yr = validate_question_year_matches_filing(
        question=item["question"],
        fiscal_year=f"{yr1_str}/{yr2_str}",
        is_multi_year=True,
    )
    if not ok_yr:
        flags.append(msg_yr)
        is_valid = False

    combined_quote = f"{v1 or q1} || {v2 or q2}"

    # Check ground truth supported by both quotes (flag for human attention)
    ok_supp, msg_supp = validate_ground_truth_supported_by_quotes(
        ground_truth=item.get("ground_truth", ""),
        supporting_quote_str=combined_quote,
    )
    if not ok_supp:
        flags.append(msg_supp)
        # Note: flags for human review

    if not flags:
        flags.append("PASS")

    return is_valid, flags, combined_quote


def regenerate_14_multi_year_candidates(
    chunks: List[Dict[str, Any]],
    dev_chunk_ids: Set[str],
    bench_model: str = DEFAULT_BENCH_MODEL,
    ledger: Optional[QuotaLedger] = None,
    delay: float = 5.0,
    max_calls: int = 20,
    existing_questions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Regenerates 14 multi-year candidates across all 5 companies:
    Alphabet: 3, Amazon: 3, Apple: 2, Microsoft: 3, NVIDIA: 3.
    Capped at max_calls (<= 20) with --delay 5 between calls.
    """
    if ledger is None:
        ledger = QuotaLedger(max_calls=max_calls)

    target_counts = {
        "Alphabet": 3,
        "Amazon": 3,
        "Apple": 2,
        "Microsoft": 3,
        "NVIDIA": 3,
    }

    pairs_by_company = sample_multi_year_candidate_pairs(chunks, dev_chunk_ids, target_counts)
    multi_year_results: List[Dict[str, Any]] = []
    questions_seen = list(existing_questions or [])
    calls_made = 0

    print("\n" + "=" * 80)
    print("REGENERATING 14 MULTI-YEAR CANDIDATES ACROSS 5 COMPANIES")
    print(f"Target Distribution: Alphabet=3, Amazon=3, Apple=2, Microsoft=3, NVIDIA=3 (Total=14)")
    print(f"Max Calls Cap: {max_calls} | Delay: {delay}s | Model: {bench_model}")
    print("=" * 80)

    for comp in ["Alphabet", "Amazon", "Apple", "Microsoft", "NVIDIA"]:
        target_n = target_counts[comp]
        comp_pairs = pairs_by_company.get(comp, [])
        accepted_comp = 0
        pair_idx = 0

        logger.info(f"Generating {target_n} multi-year items for {comp} (available candidate pairs: {len(comp_pairs)})...")

        while accepted_comp < target_n and pair_idx < len(comp_pairs):
            if calls_made >= max_calls:
                logger.warning(f"Hit max calls limit ({max_calls}). Stopping multi-year regeneration.")
                break

            if calls_made > 0 and delay > 0:
                logger.info(f"Waiting {delay}s between calls...")
                time.sleep(delay)

            c1, c2 = comp_pairs[pair_idx]
            pair_idx += 1
            yr1, yr2 = str(c1.get("fiscal_year")), str(c2.get("fiscal_year"))
            cid1, cid2 = c1.get("chunk_id", ""), c2.get("chunk_id", "")

            calls_made += 1
            logger.info(f"[{calls_made}/{max_calls}] Calling {bench_model} for {comp} FY{yr1} vs FY{yr2}...")

            res = generate_single_multi_year_candidate(c1, c2, bench_model=bench_model, ledger=ledger)
            if not res:
                logger.warning(f"Failed to generate candidate for {comp} FY{yr1} vs FY{yr2}.")
                continue

            # Ensure company name is present
            q_text = res["question"]
            if comp.lower() not in q_text.lower():
                first_char = q_text[0].lower()
                q_text = f"At {comp}, {first_char}{q_text[1:]}"
                res["question"] = q_text

            is_valid, flags, combined_quote = validate_multi_year_item(
                res, c1, c2, existing_questions=questions_seen
            )

            if is_valid:
                accepted_comp += 1
                cand_record = {
                    "company": comp,
                    "fiscal_year": f"{yr1}/{yr2}",
                    "section": f"{c1.get('item_section')}/{c2.get('item_section')}",
                    "category": "multi_hop",
                    "question_style": "comparison-of-two-figures style where the chunk allows",
                    "question": res["question"],
                    "ground_truth": res["ground_truth"],
                    "supporting_quote": combined_quote,
                    "source_chunk_ids": [cid1, cid2],
                    "validator_flags": ["PASS"],
                    "default_reject": False,
                }
                multi_year_results.append(cand_record)
                questions_seen.append(res["question"])
                logger.info(f"PASSED multi-year item [{accepted_comp}/{target_n}] for {comp}: '{res['question'][:60]}...'")
            else:
                logger.warning(f"Multi-year candidate failed validation ({flags}). Retrying with next pair.")

        if calls_made >= max_calls:
            break

    logger.info(f"Multi-year regeneration complete: {len(multi_year_results)} items generated in {calls_made} calls.")
    return multi_year_results


def run_fix_pipeline(
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    dev_path: Path = DEFAULT_DEV_PATH,
    bench_model: str = DEFAULT_BENCH_MODEL,
    delay: float = 5.0,
    max_calls: int = 20,
    seed: int = 42,
    skip_multi_year: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline to post-process candidates, regenerate multi-year items, and write results.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates: List[Dict[str, Any]] = json.load(f)

    dev_chunk_ids: Set[str] = set()
    if dev_path.exists():
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_items = json.load(f)
            for it in dev_items:
                dev_chunk_ids.update(it.get("source_chunk_ids", []))

    logger.info(f"Loaded {len(candidates)} existing candidates. Starting post-processing...")

    # Separate existing items into single-chunk, old multi-year, and unanswerables
    single_chunks: List[Dict[str, Any]] = []
    unanswerables: List[Dict[str, Any]] = []
    old_multi_years: List[Dict[str, Any]] = []

    for c in candidates:
        cat = c.get("category")
        sc_len = len(c.get("source_chunk_ids", []))
        if cat == "unanswerable" or sc_len == 0:
            unanswerables.append(c)
        elif cat == "multi_hop" or sc_len > 1:
            old_multi_years.append(c)
        else:
            single_chunks.append(c)

    logger.info(f"Split candidates: {len(single_chunks)} single-chunk, {len(old_multi_years)} old multi-year (to discard), {len(unanswerables)} unanswerables.")

    # 1. Rule 2: Repeated across years disambiguation (run first so phrase with company name avoids redundant prefix)
    single_chunks, ambig_rewrites, ambig_now_passed = fix_repeated_across_years(single_chunks, chunks)

    # 2. Rule 1: Missing company name prepend
    single_chunks, comp_rewrites = fix_missing_company(single_chunks)

    # 3. Rule 3: Comparison style validation
    single_chunks, comp_style_rejections = fix_comparison_style(single_chunks)

    # 3b. Rule 2: Question year matches filing year validation (single chunks)
    year_mismatches_single = 0
    for c in single_chunks:
        ok_yr, msg_yr = validate_question_year_matches_filing(
            question=c["question"],
            fiscal_year=str(c.get("fiscal_year", "")),
            is_multi_year=False,
            is_unanswerable=False,
        )
        if not ok_yr:
            c["default_reject"] = True
            flags = [f for f in c.get("validator_flags", []) if f != "PASS"]
            if not any(f.startswith("REJECT_QUESTION_YEAR_MISMATCH") for f in flags):
                flags.append(msg_yr)
            c["validator_flags"] = flags
            year_mismatches_single += 1

    # 4. Rule 5: Multi-year regeneration
    if skip_multi_year:
        logger.info("Skipping multi-year regeneration (--skip-multi-year active). Keeping existing multi-year items.")
        new_multi_years = old_multi_years
    else:
        existing_q_pool = [c["question"] for c in single_chunks]
        ledger = QuotaLedger(max_calls=max_calls)
        new_multi_years = regenerate_14_multi_year_candidates(
            chunks=chunks,
            dev_chunk_ids=dev_chunk_ids,
            bench_model=bench_model,
            ledger=ledger,
            delay=delay,
            max_calls=max_calls,
            existing_questions=existing_q_pool,
        )

    # Re-validate multi-year candidates with the new rules (adjacent years, question year match, GT supported)
    non_adjacent_my = 0
    year_mismatches_my = 0
    gt_unsupported_my = 0
    for c in new_multi_years:
        fy_str = str(c.get("fiscal_year", ""))
        # 1. Adjacent years check
        ok_adj, msg_adj = validate_multi_year_adjacent_years(fy_str)
        if not ok_adj:
            c["default_reject"] = True
            flags = [f for f in c.get("validator_flags", []) if f != "PASS"]
            if not any(f.startswith("REJECT_MULTI_YEAR_NON_ADJACENT_YEARS") for f in flags):
                flags.append(msg_adj)
            c["validator_flags"] = flags
            non_adjacent_my += 1

        # 2. Question year matches filings check
        ok_yr, msg_yr = validate_question_year_matches_filing(
            question=c["question"],
            fiscal_year=fy_str,
            is_multi_year=True,
            is_unanswerable=False,
        )
        if not ok_yr:
            c["default_reject"] = True
            flags = [f for f in c.get("validator_flags", []) if f != "PASS"]
            if not any(f.startswith("REJECT_QUESTION_YEAR_MISMATCH") for f in flags):
                flags.append(msg_yr)
            c["validator_flags"] = flags
            year_mismatches_my += 1

        # 3. Ground truth supported by both quotes check (flag for human attention)
        ok_supp, msg_supp = validate_ground_truth_supported_by_quotes(
            ground_truth=c.get("ground_truth", ""),
            supporting_quote_str=c.get("supporting_quote", ""),
        )
        if not ok_supp:
            flags = [f for f in c.get("validator_flags", []) if f != "PASS"]
            if not any(f.startswith("FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES") for f in flags):
                flags.append(msg_supp)
            c["validator_flags"] = flags
            c["default_reject"] = True
            gt_unsupported_my += 1

    # Cleanly renumber IDs with distinct prefixes
    for idx, c in enumerate(single_chunks, start=1):
        c["id"] = f"heldout_{idx:02d}"
    for idx, c in enumerate(new_multi_years, start=1):
        c["id"] = f"heldout_mh_{idx:02d}"
    for idx, c in enumerate(unanswerables, start=1):
        c["id"] = f"heldout_un_{idx:02d}"

    # Combine all candidates: single-chunks + new multi-years + unanswerables
    all_final = single_chunks + new_multi_years + unanswerables

    # Rule 1: Ensure EVERY candidate question names its company
    all_final, comp_rewrites = fix_missing_company(all_final)

    # Write candidates.json and review.csv (asserting unique IDs and extracting excerpts)
    write_candidates_json(all_final, candidates_path)
    write_review_csv(all_final, review_path, chunks=chunks)

    stats = {
        "total_candidates": len(all_final),
        "single_chunk_count": len(single_chunks),
        "multi_year_count": len(new_multi_years),
        "unanswerable_count": len(unanswerables),
        "company_rewrites": comp_rewrites,
        "ambiguous_year_rewrites": ambig_rewrites,
        "ambiguous_year_now_passed": ambig_now_passed,
        "comparison_style_rejections": comp_style_rejections,
        "year_mismatches_single": year_mismatches_single,
        "non_adjacent_multi_year": non_adjacent_my,
        "year_mismatches_multi_year": year_mismatches_my,
        "gt_unsupported_multi_year": gt_unsupported_my,
    }
    logger.info(f"Pipeline complete! Stats: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Fix and post-process candidate benchmark items.")
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH, help="Path to candidates.json")
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH, help="Path to review.csv")
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH, help="Path to chunks.json")
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH, help="Path to golden_set.json")
    parser.add_argument("--bench-model", type=str, default=DEFAULT_BENCH_MODEL, help="Model for multi-year calls")
    parser.add_argument("--delay", type=float, default=5.0, help="Seconds between LLM calls")
    parser.add_argument("--max-calls", type=int, default=20, help="Maximum calls for multi-year regeneration")
    parser.add_argument("--skip-multi-year", action="store_true", help="Skip LLM multi-year regeneration")
    args = parser.parse_args()

    run_fix_pipeline(
        candidates_path=args.candidates_path,
        review_path=args.review_path,
        chunks_path=args.chunks_path,
        dev_path=args.dev_path,
        bench_model=args.bench_model,
        delay=args.delay,
        max_calls=args.max_calls,
        skip_multi_year=args.skip_multi_year,
    )


if __name__ == "__main__":
    main()
