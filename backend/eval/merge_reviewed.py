"""
Merge Reviewed CSV into Final Golden Held-Out Benchmark.

Reads the manually reviewed review.csv:
1. Reads CSV as UTF-8. Uses accept=x rows only.
2. If the edit column is non-empty, it replaces the question text. Stores the
   original question as original_question in the item metadata.
3. Re-runs validators on every edited question (leakage, 6-gram overlap,
   year-matches-filing, company named, no preamble). Lists any failures in a printed
   report and still writes them, flagged "needs_attention": true.
4. Resolves source_chunk_ids strictly by matching supporting_quote against the
   CURRENT chunks.json (restricted to the matching company and fiscal year).
   FAILS LOUDLY if any quote cannot be resolved.
5. Adds per-item metadata: reviewed_by "claude-llm-review", answer_quality_eligible
   true except for multi-year items (heldout_mh_*), which are false.
6. Outputs backend/eval/golden_heldout.json in the standard benchmark schema,
   printing counts by category, company, section and eligibility.
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.eval.generate_candidates import (
    validate_6gram_overlap,
    validate_answer_leakage,
    validate_no_preamble,
    validate_question_year_matches_filing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REVIEW_CSV = BASE_DIR / "eval" / "results" / "review.csv"
DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "eval" / "golden_heldout.json"


def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace and typography for substring location."""
    t = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip()


def find_matching_chunk_for_quote(
    quote: str,
    company: str,
    fiscal_year_str: str,
    chunks: List[Dict[str, Any]],
    preferred_section: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds the exact chunk in chunks.json containing quote as a substring,
    restricted to the given company and fiscal year(s).
    """
    if not quote or len(quote.strip()) < 8:
        return None

    # Parse target years (handles single year '2024' or multi-year '2023/2024')
    target_years = set(re.findall(r"\b202[0-9]\b", fiscal_year_str))

    # Candidate chunk pool matching company and year
    pool = [
        c for c in chunks
        if c.get("company", "").lower() == company.lower()
        and (not target_years or str(c.get("fiscal_year", "")) in target_years)
    ]

    # 1. Exact verbatim match
    exact_matches = [c for c in pool if quote in c.get("text", "")]
    if len(exact_matches) == 1:
        return exact_matches[0]
    elif len(exact_matches) > 1:
        if preferred_section:
            sec_matches = [c for c in exact_matches if c.get("item_section") == preferred_section]
            if sec_matches:
                return sec_matches[0]
        return exact_matches[0]

    # 2. Normalized whitespace match
    norm_q = normalize_whitespace(quote)
    norm_matches = [c for c in pool if norm_q in normalize_whitespace(c.get("text", ""))]
    if len(norm_matches) == 1:
        return norm_matches[0]
    elif len(norm_matches) > 1:
        if preferred_section:
            sec_matches = [c for c in norm_matches if c.get("item_section") == preferred_section]
            if sec_matches:
                return sec_matches[0]
        return norm_matches[0]

    # 3. Fallback across all chunks for that company
    company_pool = [c for c in chunks if c.get("company", "").lower() == company.lower()]
    broad_matches = [
        c for c in company_pool
        if quote in c.get("text", "") or norm_q in normalize_whitespace(c.get("text", ""))
    ]
    if broad_matches:
        return broad_matches[0]

    return None


def is_marked_truthy(val: Optional[str]) -> bool:
    """Checks if a review column marker is checked/truthy."""
    if not val:
        return False
    return val.strip().lower() in ["x", "1", "yes", "true", "accept", "accepted", "y", "ok"]


def is_marked_reject(val: Optional[str]) -> bool:
    """Checks if reject column is checked."""
    if not val:
        return False
    return val.strip().lower() in ["x", "1", "yes", "true", "reject", "rejected", "y", "no"]


def merge_review_csv(
    review_csv_path: Path = DEFAULT_REVIEW_CSV,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    output_json_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Parses review.csv as UTF-8, uses accept=x rows only, applies edits,
    re-validates edited questions, resolves source_chunk_ids, attaches metadata,
    and writes golden_heldout.json.
    """
    target_output_path = output_path or output_json_path or DEFAULT_OUTPUT_JSON
    if not review_csv_path.exists():
        raise FileNotFoundError(f"Review CSV file not found: {review_csv_path}")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Processed chunks corpus file not found: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    corpus_chunk_ids: Set[str] = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
    logger.info(f"Loaded {len(chunks)} chunks from corpus ({len(corpus_chunk_ids)} unique chunk_ids).")

    with open(review_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    logger.info(f"Read {len(raw_rows)} rows from {review_csv_path}.")

    heldout_items: List[Dict[str, Any]] = []
    unresolved_errors: List[str] = []
    edited_validation_reports: List[Dict[str, Any]] = []

    for row in raw_rows:
        row_id = row.get("id", "").strip()
        comp = row.get("company", "").strip()
        fy = row.get("fiscal_year", "").strip()
        sec = row.get("section", "").strip()
        cat = row.get("category", "simple_lookup").strip()
        orig_q = row.get("question", "").strip()
        gt = row.get("ground_truth", "").strip()
        quote_field = row.get("supporting_quote", "").strip()
        style = row.get("question_style", "").strip()

        # Requirement 1: Use accept=x rows only
        if not is_marked_truthy(row.get("accept")):
            logger.info(f"Skipping non-accepted item: {row_id}")
            continue

        if is_marked_reject(row.get("reject")):
            logger.info(f"Skipping rejected item: {row_id}")
            continue

        # Requirement 1: If edit column is non-empty, it replaces question text
        edit_val = (row.get("edit") or "").strip()
        was_edited = bool(edit_val)
        if was_edited:
            if edit_val.startswith("{") and edit_val.endswith("}"):
                try:
                    edit_dict = json.loads(edit_val)
                    q = edit_dict.get("question", edit_val)
                except Exception:
                    q = edit_val
            else:
                q = edit_val
        else:
            q = orig_q

        is_unanswerable = (cat == "unanswerable" or row_id.startswith("heldout_un_"))
        is_multi_year = row_id.startswith("heldout_mh_") or cat == "multi_hop" or "/" in fy

        # Requirement 3: Resolve source_chunk_ids for answerable items
        resolved_chunk_ids: List[str] = []
        resolved_contexts: List[str] = []
        resolved_quotes: List[str] = []
        matched_chunks: List[Dict[str, Any]] = []

        if is_unanswerable:
            expected_decline = True
        else:
            expected_decline = False
            individual_quotes = [
                sq.strip()
                for sq in re.split(r"\s*\|\|\s*|\n---\n", quote_field)
                if sq.strip()
            ]

            if not individual_quotes:
                err = f"Item '{row_id}' ({comp} FY{fy}) has no supporting_quote specified!"
                unresolved_errors.append(err)
                continue

            for quote in individual_quotes:
                matching_chunk = find_matching_chunk_for_quote(
                    quote=quote,
                    company=comp,
                    fiscal_year_str=fy,
                    chunks=chunks,
                    preferred_section=sec,
                )

                if not matching_chunk:
                    err = (
                        f"Item '{row_id}' ({comp} FY{fy}, {sec}): Supporting quote could NOT be found in chunks.json!\n"
                        f"  Quote: \"{quote[:120]}...\""
                    )
                    unresolved_errors.append(err)
                    continue

                cid = matching_chunk.get("chunk_id")
                if not cid:
                    unresolved_errors.append(f"Matching chunk for item '{row_id}' is missing 'chunk_id'!")
                    continue

                matched_chunks.append(matching_chunk)
                if cid not in resolved_chunk_ids:
                    resolved_chunk_ids.append(cid)
                    resolved_contexts.append(matching_chunk.get("text", ""))
                    resolved_quotes.append(quote)

        # Requirement 2: Re-run validators on every edited question
        edit_failures: List[str] = []
        if was_edited:
            # Preamble check
            ok_pre, msg_pre = validate_no_preamble(q)
            if not ok_pre:
                edit_failures.append(msg_pre)

            # Company name check
            if comp and comp.lower() not in q.lower():
                edit_failures.append(f"MISSING_COMPANY_NAME: '{comp}' not named in question")

            # Year-matches-filing check
            ok_yr, msg_yr = validate_question_year_matches_filing(
                question=q,
                fiscal_year=fy,
                is_multi_year=is_multi_year,
                is_unanswerable=is_unanswerable,
            )
            if not ok_yr:
                edit_failures.append(msg_yr)

            # Answer leakage check
            ok_leak, msg_leak = validate_answer_leakage(
                question=q,
                ground_truth=gt,
                fiscal_year=fy,
                company=comp,
                item_section=sec,
            )
            if not ok_leak:
                edit_failures.append(msg_leak)

            # 6-gram overlap check against matched chunks
            for mc in matched_chunks:
                ok_6g, msg_6g = validate_6gram_overlap(q, mc.get("text", ""))
                if not ok_6g:
                    edit_failures.append(msg_6g)

            edited_validation_reports.append({
                "id": row_id,
                "company": comp,
                "fiscal_year": fy,
                "original_question": orig_q,
                "edited_question": q,
                "passed": len(edit_failures) == 0,
                "failures": edit_failures,
            })

        # Requirement 4: Per-item metadata
        meta: Dict[str, Any] = {
            "reviewed_by": "claude-llm-review",
            "answer_quality_eligible": False if row_id.startswith("heldout_mh_") else True,
            "section": sec,
            "question_style": style,
        }
        if was_edited or row.get("original_question"):
            meta["original_question"] = orig_q if was_edited else row.get("original_question")
        if was_edited:
            meta["needs_attention"] = bool(edit_failures)
            if edit_failures:
                meta["validator_failures"] = edit_failures
            else:
                meta["needs_attention"] = False

        heldout_items.append({
            "id": row_id,
            "category": cat,
            "company": comp,
            "fiscal_year": fy,
            "section": sec,
            "expected_decline": expected_decline,
            "question": q,
            "ground_truth": gt,
            "contexts": resolved_contexts,
            "chunk_id": resolved_chunk_ids[0] if resolved_chunk_ids else None,
            "source_chunk_ids": resolved_chunk_ids,
            "source_quotes": resolved_quotes,
            "metadata": meta,
        })

    # Requirement 3: Fail loudly if any quote cannot be resolved
    if unresolved_errors:
        err_msg = (
            f"\n{'=' * 80}\n"
            f"CRITICAL ERROR: Failed to resolve {len(unresolved_errors)} supporting quote(s) against chunks.json!\n"
            + "\n".join(f"  - {e}" for e in unresolved_errors[:10])
        )
        if len(unresolved_errors) > 10:
            err_msg += f"\n  ... and {len(unresolved_errors) - 10} more unresolved quotes."
        err_msg += f"\n{'=' * 80}\n"
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Validate that every source_chunk_id exists in corpus
    all_source_ids = [cid for it in heldout_items for cid in it.get("source_chunk_ids", [])]
    missing_ids = [cid for cid in all_source_ids if cid not in corpus_chunk_ids]
    if missing_ids:
        raise ValueError(
            f"Integrity check failed: {len(missing_ids)} source_chunk_ids do not exist in chunks.json: {missing_ids[:5]}"
        )

    # Requirement 5: Write backend/eval/golden_heldout.json
    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_output_path, "w", encoding="utf-8") as f:
        json.dump(heldout_items, f, indent=2, ensure_ascii=False)

    # Print Validation Report for Edited Questions
    print("\n" + "=" * 80)
    print("EDITED QUESTIONS VALIDATION REPORT")
    print("=" * 80)
    print(f"Total Edited Questions Re-validated: {len(edited_validation_reports)}")
    failed_edits = [r for r in edited_validation_reports if not r["passed"]]
    passed_edits = [r for r in edited_validation_reports if r["passed"]]
    print(f"  - Passed: {len(passed_edits)}")
    print(f"  - Needs Attention: {len(failed_edits)}")
    if failed_edits:
        print("\n  FAILURES LIST (flagged needs_attention: true):")
        for f_rep in failed_edits:
            print(f"    * [{f_rep['id']}] {f_rep['company']} FY{f_rep['fiscal_year']}:")
            print(f"      Edited Q: {f_rep['edited_question']}")
            print(f"      Failures: {'; '.join(f_rep['failures'])}")
    else:
        print("  All edited questions passed all validators without errors.")
    print("=" * 80 + "\n")

    # Print Breakdown Counts
    cat_counts = Counter(it["category"] for it in heldout_items)
    comp_counts = Counter(it["company"] for it in heldout_items)
    sec_counts = Counter(it.get("section", "None") for it in heldout_items)
    elig_counts = Counter(bool(it.get("metadata", {}).get("answer_quality_eligible", False)) for it in heldout_items)

    print("=" * 80)
    print("GOLDEN HELDOUT BENCHMARK DATASET SUMMARY")
    print("=" * 80)
    print(f"Output File:     {target_output_path}")
    print(f"Total Questions: {len(heldout_items)}")

    print("\nBreakdown by Category:")
    for c, cnt in sorted(cat_counts.items()):
        print(f"  - {c:<25}: {cnt:>3}")
    print("\nBreakdown by Company:")
    for comp, cnt in sorted(comp_counts.items()):
        print(f"  - {comp:<25}: {cnt:>3}")
    print("\nBreakdown by Section:")
    for s, cnt in sorted(sec_counts.items()):
        print(f"  - {s:<25}: {cnt:>3}")
    print("\nBreakdown by Answer Quality Eligibility:")
    print(f"  - Eligible (True)        : {elig_counts[True]:>3}")
    print(f"  - Ineligible (False)     : {elig_counts[False]:>3}")
    print("=" * 80 + "\n")

    return heldout_items


def main():
    parser = argparse.ArgumentParser(description="Merge reviewed CSV into golden_heldout.json.")
    parser.add_argument("--review-csv", type=str, default=str(DEFAULT_REVIEW_CSV), help="Path to review.csv")
    parser.add_argument("--chunks-path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.json")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output golden_heldout.json path")
    args = parser.parse_args()

    merge_review_csv(
        review_csv_path=Path(args.review_csv),
        chunks_path=Path(args.chunks_path),
        output_json_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()
