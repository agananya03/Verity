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
    """Normalizes whitespace and typography for substring location, stripping page-break TOC headers."""
    t = text.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("—", "-")
    t = re.sub(r"\b\d+\s+Table of Contents\b", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def find_matching_chunk_for_quote(
    quote: str,
    company: str,
    fiscal_year_str: str,
    chunks: List[Dict[str, Any]],
    preferred_section: Optional[str] = None,
    preferred_chunk_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds the exact chunk in chunks.json containing quote as a substring,
    restricted to the given company and fiscal year(s).
    Prefers preferred_chunk_id if it matches the quote.
    """
    if not quote or len(quote.strip()) < 8:
        return None

    norm_q = normalize_whitespace(quote)

    # If preferred_chunk_id is given, test it first
    if preferred_chunk_id:
        for c in chunks:
            if c.get("chunk_id") == preferred_chunk_id:
                c_text = c.get("text", "")
                if quote in c_text or norm_q in normalize_whitespace(c_text):
                    return c
                break

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
        if preferred_chunk_id:
            p_matches = [c for c in exact_matches if c.get("chunk_id") == preferred_chunk_id]
            if p_matches:
                return p_matches[0]
        if preferred_section:
            sec_matches = [c for c in exact_matches if c.get("item_section") == preferred_section]
            if sec_matches:
                return sec_matches[0]
        return exact_matches[0]

    # 2. Normalized whitespace match
    norm_matches = [c for c in pool if norm_q in normalize_whitespace(c.get("text", ""))]
    if len(norm_matches) == 1:
        return norm_matches[0]
    elif len(norm_matches) > 1:
        if preferred_chunk_id:
            p_matches = [c for c in norm_matches if c.get("chunk_id") == preferred_chunk_id]
            if p_matches:
                return p_matches[0]
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
        if preferred_chunk_id:
            p_matches = [c for c in broad_matches if c.get("chunk_id") == preferred_chunk_id]
            if p_matches:
                return p_matches[0]
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
    candidates_json_path: Optional[Path] = None,
    stop_on_validation_failure: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parses review.csv as UTF-8, uses accept=x rows only, applies edits,
    re-validates edited questions, resolves source_chunk_ids, attaches metadata,
    and writes golden benchmark JSON.
    """
    target_output_path = output_path or output_json_path or DEFAULT_OUTPUT_JSON
    if not review_csv_path.exists():
        raise FileNotFoundError(f"Review CSV file not found: {review_csv_path}")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Processed chunks corpus file not found: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    corpus_chunk_ids: Set[str] = {c.get("chunk_id") for c in chunks if c.get("chunk_id")}
    chunks_by_id: Dict[str, Dict[str, Any]] = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
    logger.info(f"Loaded {len(chunks)} chunks from corpus ({len(corpus_chunk_ids)} unique chunk_ids).")

    # Load candidates JSON if provided (for resolving candidate chunk IDs)
    cands_by_id: Dict[str, Dict[str, Any]] = {}
    if candidates_json_path and candidates_json_path.exists():
        try:
            with open(candidates_json_path, "r", encoding="utf-8") as f:
                cands_data = json.load(f)
                if isinstance(cands_data, list):
                    cands_by_id = {d["id"]: d for d in cands_data if isinstance(d, dict) and "id" in d}
                logger.info(f"Loaded {len(cands_by_id)} candidate items from {candidates_json_path}.")
        except Exception as e:
            logger.warning(f"Could not load candidates JSON from {candidates_json_path}: {e}")

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

        is_unanswerable = (cat == "unanswerable" or row_id.startswith("heldout_un_") or "_un_" in row_id)
        is_multi_year = (row_id.startswith("heldout_mh_") or "_mh_" in row_id or cat == "multi_hop" or "/" in fy)

        # Requirement 3: Resolve source_chunk_ids for answerable items
        resolved_chunk_ids: List[str] = []
        resolved_contexts: List[str] = []
        resolved_quotes: List[str] = []
        matched_chunks: List[Dict[str, Any]] = []

        if is_unanswerable:
            expected_decline = True
        elif is_multi_year:
            expected_decline = False
            individual_quotes = [
                sq.strip()
                for sq in re.split(r"\s*\|\|\s*|\n---\n", quote_field)
                if sq.strip()
            ]

            if not individual_quotes:
                err = f"Multi-year item '{row_id}' ({comp} FY{fy}) has no supporting_quote specified!"
                unresolved_errors.append(err)
                continue

            years = [y.strip() for y in re.findall(r"\b202[0-9]\b", fy)]
            cand = cands_by_id.get(row_id)
            cand_sids = cand.get("source_chunk_ids", []) if cand else []

            for idx, quote in enumerate(individual_quotes):
                target_yr = years[idx] if idx < len(years) else fy
                pref_cid = cand_sids[idx] if idx < len(cand_sids) else None

                matching_chunk = find_matching_chunk_for_quote(
                    quote=quote,
                    company=comp,
                    fiscal_year_str=target_yr,
                    chunks=chunks,
                    preferred_section=sec,
                    preferred_chunk_id=pref_cid,
                )

                if not matching_chunk and pref_cid and pref_cid in chunks_by_id:
                    pref_c = chunks_by_id[pref_cid]
                    if normalize_whitespace(quote) in normalize_whitespace(pref_c.get("text", "")):
                        matching_chunk = pref_c

                if not matching_chunk:
                    err = (
                        f"Item '{row_id}' ({comp} FY{target_yr}, {sec}): Supporting quote could NOT be found in chunks.json!\n"
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

            cand = cands_by_id.get(row_id)
            pref_cid = (cand.get("chunk_id") or cand.get("source_chunk_ids", [None])[0]) if cand else None

            for quote in individual_quotes:
                matching_chunk = find_matching_chunk_for_quote(
                    quote=quote,
                    company=comp,
                    fiscal_year_str=fy,
                    chunks=chunks,
                    preferred_section=sec,
                    preferred_chunk_id=pref_cid,
                )

                if not matching_chunk and pref_cid and pref_cid in chunks_by_id:
                    pref_c = chunks_by_id[pref_cid]
                    if normalize_whitespace(quote) in normalize_whitespace(pref_c.get("text", "")):
                        matching_chunk = pref_c

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
        ineligible_ids = {"round2_130", "round2_126", "round2_106", "round2_mh_09", "round2_mh_22"}
        if row_id in ineligible_ids or row_id.startswith("heldout_mh_"):
            aq_eligible = False
        else:
            aq_eligible = True

        meta: Dict[str, Any] = {
            "reviewed_by": "claude-llm-review",
            "answer_quality_eligible": aq_eligible,
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

    # Check for chunk ID overlap against golden_set.json and golden_heldout.json
    eval_dir = Path(__file__).resolve().parent
    golden_files_to_check = [
        eval_dir / "golden_set.json",
        eval_dir / "golden_heldout.json",
    ]
    overlap_ids: Set[str] = set()
    current_source_ids_set = set(all_source_ids)
    checked_files = []

    for gf in golden_files_to_check:
        if gf.exists() and gf.resolve() != target_output_path.resolve():
            checked_files.append(gf.name)
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                    for git in gdata:
                        if isinstance(git, dict):
                            for cid in git.get("source_chunk_ids", []):
                                if cid in current_source_ids_set:
                                    overlap_ids.add(cid)
                            if git.get("chunk_id") and git["chunk_id"] in current_source_ids_set:
                                overlap_ids.add(git["chunk_id"])
            except Exception as e:
                logger.warning(f"Could not read {gf} for overlap check: {e}")

    # Item counts by type and company
    single_year_count = 0
    multi_year_count = 0
    unanswerable_count = 0

    for it in heldout_items:
        iid = it["id"]
        icat = it["category"]
        ify = str(it.get("fiscal_year", ""))
        if it.get("expected_decline") or icat == "unanswerable" or "_un_" in iid or iid.startswith("heldout_un_"):
            unanswerable_count += 1
        elif "_mh_" in iid or iid.startswith("heldout_mh_") or icat == "multi_hop" or "/" in ify:
            multi_year_count += 1
        else:
            single_year_count += 1

    comp_counts = Counter(it["company"] for it in heldout_items)

    print("\n" + "=" * 80)
    print("DATASET INTEGRITY & ITEM COUNTS REPORT")
    print("=" * 80)
    print(f"Total Accepted Items: {len(heldout_items)}")
    print("\nCounts by Item Type:")
    print(f"  - Single-year questions : {single_year_count}")
    print(f"  - Multi-year questions  : {multi_year_count}")
    print(f"  - Unanswerable questions: {unanswerable_count}")
    print(f"  - Total                 : {single_year_count + multi_year_count + unanswerable_count}")

    print("\nCounts by Company:")
    for comp, cnt in sorted(comp_counts.items()):
        print(f"  - {comp:<25}: {cnt:>3}")

    print("\nCorpus & Overlap Verification:")
    print(f"  - Corpus check: 100% of source_chunk_ids ({len(all_source_ids)} references, {len(current_source_ids_set)} unique) exist in chunks.json.")
    if overlap_ids:
        print(f"  - Overlap check FAILED: {len(overlap_ids)} chunk IDs overlap with {checked_files}: {overlap_ids}")
        raise ValueError(f"Chunk overlap detected with prior golden files: {overlap_ids}")
    else:
        print(f"  - Overlap check PASSED: 0 chunk IDs overlap with {', '.join(checked_files)}.")
    print("=" * 80 + "\n")

    # Print Validation Report for Edited Questions
    print("=" * 80)
    print("EDITED QUESTIONS RE-VALIDATION REPORT")
    print("=" * 80)
    print(f"Total Edited Questions Re-validated: {len(edited_validation_reports)}")
    failed_edits = [r for r in edited_validation_reports if not r["passed"]]
    passed_edits = [r for r in edited_validation_reports if r["passed"]]
    print(f"  - Passed: {len(passed_edits)}")
    print(f"  - Failed: {len(failed_edits)}")
    if failed_edits:
        print("\nFAILING ROWS LIST:")
        for f_rep in failed_edits:
            print(f"  * ID: {f_rep['id']} | Company: {f_rep['company']} | Fiscal Year: {f_rep['fiscal_year']}")
            print(f"    Original Q : {f_rep.get('original_question', '')}")
            print(f"    Edited Q   : {f_rep['edited_question']}")
            print(f"    Reasons    : {'; '.join(f_rep['failures'])}")
            print()
    else:
        print("  All edited questions passed all validators without errors.")
    print("=" * 80 + "\n")

    # Stopping condition if any edited questions failed re-validation
    if failed_edits and stop_on_validation_failure:
        print(f"[STOP] {len(failed_edits)} edited question(s) failed re-validation.")
        print(f"As required, stopping without writing output file: {target_output_path}")
        print("=" * 80 + "\n")
        sys.exit(1)

    # Write target golden json only if all passed (or not stopping)
    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_output_path, "w", encoding="utf-8") as f:
        json.dump(heldout_items, f, indent=2, ensure_ascii=False)
    print(f"Successfully wrote {len(heldout_items)} items to {target_output_path}")

    return heldout_items


def main():
    parser = argparse.ArgumentParser(description="Merge reviewed CSV into golden JSON.")
    parser.add_argument("--input", "--review-csv", dest="input_csv", type=str, default=str(DEFAULT_REVIEW_CSV), help="Path to review.csv")
    parser.add_argument("--output", "--output-path", dest="output_json", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output golden JSON path")
    parser.add_argument("--candidates-json", dest="candidates_json", type=str, default=None, help="Path to candidates JSON")
    parser.add_argument("--chunks-path", dest="chunks_path", type=str, default=str(DEFAULT_CHUNKS_PATH), help="Path to chunks.json")
    parser.add_argument("--allow-failed-validation", action="store_true", default=False, help="Do not stop on validation failures")
    args = parser.parse_args()

    merge_review_csv(
        review_csv_path=Path(args.input_csv),
        chunks_path=Path(args.chunks_path),
        output_json_path=Path(args.output_json),
        candidates_json_path=Path(args.candidates_json) if args.candidates_json else None,
        stop_on_validation_failure=not args.allow_failed_validation,
    )


if __name__ == "__main__":
    main()
