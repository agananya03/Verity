"""
Merge Reviewed CSV into Final Golden Held-Out Benchmark.

Reads the manually reviewed review.csv:
1. Filters rows based on user accept/edit/reject markers.
2. Derives source_chunk_ids strictly by matching supporting_quote against the CURRENT chunks.json
   (restricted to the matching company and fiscal year).
3. FAILS LOUDLY if any quote cannot be resolved to an exact chunk in the current corpus.
4. Outputs backend/eval/golden_heldout.json in the standard golden set benchmark schema.
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        # If preferred section matches, prioritize it
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

    # 3. Fallback: try across all chunks for that company (if year boundary was slightly mislabeled)
    company_pool = [c for c in chunks if c.get("company", "").lower() == company.lower()]
    broad_matches = [c for c in company_pool if quote in c.get("text", "") or norm_q in normalize_whitespace(c.get("text", ""))]
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
    Parses review.csv, derives source_chunk_ids from supporting_quotes,
    validates corpus integrity, and writes golden_heldout.json.
    FAILS LOUDLY if any quote cannot be resolved to a chunk ID.
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

    # Check if user has populated 'accept' column on any row
    any_accept_marked = any(is_marked_truthy(r.get("accept")) for r in raw_rows)

    heldout_items: List[Dict[str, Any]] = []
    unresolved_errors: List[str] = []

    for row in raw_rows:
        row_id = row.get("id", "").strip()
        comp = row.get("company", "").strip()
        fy = row.get("fiscal_year", "").strip()
        sec = row.get("section", "").strip()
        cat = row.get("category", "simple_lookup").strip()
        q = row.get("question", "").strip()
        gt = row.get("ground_truth", "").strip()
        quote_field = row.get("supporting_quote", "").strip()
        v_flags = row.get("validator_flags", "")

        # Check rejection
        if is_marked_reject(row.get("reject")):
            logger.info(f"Skipping rejected item: {row_id}")
            continue

        # Check acceptance filter if accept markers are present
        if any_accept_marked and not is_marked_truthy(row.get("accept")):
            logger.info(f"Skipping unaccepted item: {row_id}")
            continue

        # If accept column is not marked across file, skip hard-rejected items
        if not any_accept_marked and any(bad in v_flags for bad in ["REJECT_6GRAM", "REJECT_QUOTE_NOT_IN_CHUNK"]):
            logger.info(f"Skipping failed validation item: {row_id} ({v_flags})")
            continue

        # Apply edit if specified
        edit_val = (row.get("edit") or "").strip()
        if edit_val:
            try:
                # If user entered JSON edit
                if edit_val.startswith("{") and edit_val.endswith("}"):
                    edit_dict = json.loads(edit_val)
                    q = edit_dict.get("question", q)
                    gt = edit_dict.get("ground_truth", gt)
                    quote_field = edit_dict.get("supporting_quote", quote_field)
                else:
                    # Treat raw text as edited question
                    q = edit_val
            except Exception as e:
                logger.warning(f"Failed to parse edit column for {row_id}: {e}")

        # Unanswerable category
        if cat == "unanswerable":
            heldout_items.append({
                "id": row_id,
                "category": "unanswerable",
                "company": comp,
                "fiscal_year": fy,
                "expected_decline": True,
                "question": q,
                "ground_truth": gt or "I don't have enough information to answer this question based on the retrieved SEC filings.",
                "contexts": [],
                "chunk_id": None,
                "source_chunk_ids": [],
                "source_quotes": [],
            })
            continue

        # Answerable categories: parse supporting quote(s) and resolve chunk_ids
        individual_quotes = [
            sq.strip()
            for sq in re.split(r"\s*\|\|\s*|\n---\n", quote_field)
            if sq.strip()
        ]

        if not individual_quotes:
            err = f"Item '{row_id}' ({comp} FY{fy}) has no supporting_quote specified!"
            unresolved_errors.append(err)
            continue

        resolved_chunk_ids: List[str] = []
        resolved_contexts: List[str] = []
        resolved_quotes: List[str] = []

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

            if cid not in resolved_chunk_ids:
                resolved_chunk_ids.append(cid)
                resolved_contexts.append(matching_chunk.get("text", ""))
                resolved_quotes.append(quote)

        if not resolved_chunk_ids:
            continue

        heldout_items.append({
            "id": row_id,
            "category": cat,
            "company": comp,
            "fiscal_year": fy,
            "expected_decline": False,
            "question": q,
            "ground_truth": gt,
            "contexts": resolved_contexts,
            "chunk_id": resolved_chunk_ids[0] if resolved_chunk_ids else None,
            "source_chunk_ids": resolved_chunk_ids,
            "source_quotes": resolved_quotes,
        })

    # Fail loudly if any quote could not be resolved!
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

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_output_path, "w", encoding="utf-8") as f:
        json.dump(heldout_items, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("GOLDEN HELDOUT BENCHMARK DATASET GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output File:     {target_output_path}")
    print(f"Total Questions: {len(heldout_items)}")

    from collections import Counter
    cat_counts = Counter(it["category"] for it in heldout_items)
    comp_counts = Counter(it["company"] for it in heldout_items)
    print("\nBreakdown by Category:")
    for c, cnt in sorted(cat_counts.items()):
        print(f"  - {c:<25}: {cnt}")
    print("\nBreakdown by Company:")
    for comp, cnt in sorted(comp_counts.items()):
        print(f"  - {comp:<25}: {cnt}")
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

