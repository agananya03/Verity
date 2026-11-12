"""
Inspect Retrieval Misses for Verity Evaluation.

Reads:
- backend/eval/results/improved_answers.json
- backend/eval/golden_set.json
- backend/data/processed/chunks.json (or backend/artifacts/bm25_inputs.json)

Uses:
- ZERO LLM calls
- ZERO quota

For each item prints:
- id
- question
- gold source_chunk_ids
- top-8 retrieved chunk ids with rank, item_section, fiscal_year, and 100-char preview
- whether hit@8 is True or False
- whether the item's source_quotes text appears verbatim in any retrieved chunk text
  (this separates a retrieval failure from a bad remap)
- for misses, prints the text of the gold chunk, trimmed to 400 chars around the quote,
  and its fiscal_year and company.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANSWERS_PATH = BASE_DIR / "eval" / "results" / "improved_answers.json"
DEFAULT_GOLDEN_PATH = BASE_DIR / "eval" / "golden_set.json"
DEFAULT_CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
DEFAULT_BM25_INPUTS_PATH = BASE_DIR / "artifacts" / "bm25_inputs.json"


def load_corpus_lookup(
    chunks_path: Optional[Path] = None,
    bm25_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Loads all chunks into a dictionary indexed by chunk_id."""
    c_path = chunks_path or DEFAULT_CHUNKS_PATH
    b_path = bm25_path or DEFAULT_BM25_INPUTS_PATH

    target_path = None
    if c_path.exists():
        target_path = c_path
    elif b_path.exists():
        target_path = b_path
    else:
        raise FileNotFoundError(
            f"No corpus chunk file found at {c_path} or {b_path}."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup: Dict[str, Dict[str, Any]] = {}
    for item in data:
        cid = item.get("chunk_id") or item.get("id")
        if cid:
            lookup[str(cid)] = item
    return lookup


def trim_around_quote(text: str, quote: str, max_chars: int = 400) -> str:
    """Trims text to ~max_chars centered around the quote or its key words."""
    if not text:
        return ""
    if not quote:
        return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")

    idx = text.find(quote)
    if idx == -1:
        # Try finding by first few words
        words = [w for w in quote.split() if len(w) > 3][:5]
        sub = " ".join(words)
        if sub:
            idx = text.find(sub)

    if idx == -1:
        # Case-insensitive partial search
        clean_q = re.sub(r"[^\w\s]", "", quote).lower()
        words = clean_q.split()[:4]
        if words:
            sub = " ".join(words)
            clean_t = re.sub(r"[^\w\s]", " ", text).lower()
            idx = clean_t.find(sub)

    if idx == -1:
        idx = 0

    quote_len = len(quote)
    if len(text) <= max_chars:
        return text.strip()

    half_remain = max(0, (max_chars - min(quote_len, 200)) // 2)
    start = max(0, idx - half_remain)
    end = min(len(text), start + max_chars)
    if end - start < max_chars and start > 0:
        start = max(0, end - max_chars)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", " ").strip() + suffix


def inspect_retrieval_misses(
    answers_path: Path = DEFAULT_ANSWERS_PATH,
    golden_set_path: Path = DEFAULT_GOLDEN_PATH,
    chunks_path: Optional[Path] = None,
) -> None:
    """Inspects top-8 retrieved chunks vs gold chunks for every evaluated item."""
    if not answers_path.exists():
        print(f"Error: answers file not found at: {answers_path}")
        return

    if not golden_set_path.exists():
        print(f"Error: golden set file not found at: {golden_set_path}")
        return

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_list = json.load(f)

    golden_lookup = {item["id"]: item for item in golden_list}
    chunk_lookup = load_corpus_lookup(chunks_path=chunks_path)

    items: List[Dict[str, Any]] = answers_data.get("items", [])
    if not items:
        print(f"No items found in {answers_path}.")
        return

    total_items = len(items)
    hit_count = 0
    miss_count = 0
    quote_in_retrieved_count = 0

    print("=" * 100)
    print(f"VERITY RETRIEVAL MISS INSPECTOR")
    print(f"Answers File:   {answers_path}")
    print(f"Golden Set:     {golden_set_path}")
    print(f"Total Evaluated Questions: {total_items}")
    print("=" * 100)

    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"item_{idx:02d}")
        question = item.get("question", "")
        gold_item = golden_lookup.get(item_id, item)

        # Get gold source chunk IDs and quotes
        gold_cids: List[str] = gold_item.get("source_chunk_ids") or (
            [gold_item["chunk_id"]] if gold_item.get("chunk_id") else []
        )
        source_quotes: List[str] = gold_item.get("source_quotes", [])

        # Get top-8 retrieved chunk IDs
        retrieved_cids: List[str] = (item.get("retrieved_chunk_ids") or [])[:8]

        # Determine hit@8
        is_hit_8 = bool(gold_cids and any(cid in retrieved_cids for cid in gold_cids))
        if is_hit_8:
            hit_count += 1
        else:
            miss_count += 1

        # Check if source_quotes appear verbatim in any retrieved chunk text
        retrieved_chunks = [chunk_lookup.get(cid, {}) for cid in retrieved_cids]
        quote_matches: List[Tuple[str, int, str]] = []  # (quote, rank, cid)

        for q in source_quotes:
            q_clean = q.strip().lower()
            if not q_clean:
                continue
            for r_idx, (r_cid, r_chunk) in enumerate(zip(retrieved_cids, retrieved_chunks), start=1):
                r_text = (r_chunk.get("text") or "").lower()
                if q_clean in r_text or (len(q_clean) > 40 and q_clean[:40] in r_text):
                    quote_matches.append((q, r_idx, r_cid))
                    break

        has_quote_in_retrieved = len(quote_matches) > 0
        if has_quote_in_retrieved:
            quote_in_retrieved_count += 1

        # Print item header
        print(f"\n[{idx}/{total_items}] ITEM: {item_id}")
        print(f"Question:             {question}")
        print(f"Gold source_chunk_ids: {gold_cids if gold_cids else '[] (Unanswerable/None)'}")
        print(f"hit@8:                {is_hit_8}")

        # Explain quote presence in retrieved chunks
        if has_quote_in_retrieved:
            match_details = ", ".join(f"Rank #{r} (`{cid}`)" for _, r, cid in quote_matches)
            print(f"source_quotes in top-8: True ({match_details})")
            if not is_hit_8:
                print("  ↳ NOTE: Text quote was retrieved under an overlapping/duplicate chunk id!")
        else:
            print("source_quotes in top-8: False (True Retrieval Failure: key evidence missing from top-8)")

        # Print top-8 retrieved chunks table
        print("\n  Top-8 Retrieved Chunks:")
        print(f"  {'Rank':<5} | {'Chunk ID':<38} | {'Section':<10} | {'FY':<6} | {'Preview'}")
        print(f"  {'-'*5}-+-{'-'*38}-+-{'-'*10}-+-{'-'*6}-+-{'-'*40}")

        for rank, r_cid in enumerate(retrieved_cids, start=1):
            r_chunk = chunk_lookup.get(r_cid, {})
            sec = r_chunk.get("item_section", "Unknown")
            fy = str(r_chunk.get("fiscal_year", "Unknown"))
            preview = (r_chunk.get("text", "")[:100]).replace("\n", " ").strip()
            is_gold_marker = " [GOLD]" if r_cid in gold_cids else ""
            print(f"  #{rank:<4} | {r_cid + is_gold_marker:<38} | {sec:<10} | {fy:<6} | \"{preview}...\"")

        # For misses, print the gold chunk text trimmed to 400 chars around the quote
        if not is_hit_8 and gold_cids:
            print("\n  >>> MISS DETAILS (Gold Chunk Text):")
            for g_cid in gold_cids:
                g_chunk = chunk_lookup.get(g_cid)
                if not g_chunk:
                    print(f"    Gold chunk {g_cid} NOT FOUND in corpus lookup.")
                    continue
                g_comp = g_chunk.get("company", "Unknown")
                g_fy = g_chunk.get("fiscal_year", "Unknown")
                g_sec = g_chunk.get("item_section", "Unknown")
                g_text = g_chunk.get("text", "")

                # Pick corresponding quote if available
                matching_quote = source_quotes[0] if source_quotes else ""
                trimmed = trim_around_quote(g_text, matching_quote, max_chars=400)

                print(f"    Gold Chunk ID: `{g_cid}`")
                print(f"    Company:       {g_comp}")
                print(f"    Fiscal Year:   {g_fy}")
                print(f"    Section:       {g_sec}")
                print(f"    Evidence Text (trimmed ~400 chars around quote):")
                print(f"      \"{trimmed}\"")

        print("-" * 100)

    # Print summary
    hit_rate = (hit_count / total_items * 100.0) if total_items > 0 else 0.0
    print("\n" + "=" * 100)
    print("RETRIEVAL INSPECTION SUMMARY")
    print("=" * 100)
    print(f"Total Evaluated:              {total_items}")
    print(f"Hits (hit@8 == True):         {hit_count} ({hit_rate:.1f}%)")
    print(f"Misses (hit@8 == False):       {miss_count}")
    print(f"Quote Present in Top-8:       {quote_in_retrieved_count} / {total_items}")
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieval misses for Verity RAG evaluation.")
    parser.add_argument(
        "--answers",
        type=Path,
        default=DEFAULT_ANSWERS_PATH,
        help="Path to answers file (defaults to backend/eval/results/improved_answers.json)",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_PATH,
        help="Path to golden set JSON (defaults to backend/eval/golden_set.json)",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to chunks JSON (defaults to backend/data/processed/chunks.json)",
    )
    args = parser.parse_args()

    inspect_retrieval_misses(
        answers_path=args.answers,
        golden_set_path=args.golden_set,
        chunks_path=args.chunks,
    )


if __name__ == "__main__":
    main()

