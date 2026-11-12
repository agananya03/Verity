"""
Diagnostic Tool for Retrieval Pipeline Performance.

Analyzes:
1. Corpus token lengths using BAAI/bge-small-en-v1.5 tokenizer:
   - min, median, p90, max, and % > 512 tokens.
   - Breakdown by company and item_section.
2. Search diagnostics for golden_01:
   - Gold chunk token length & token offset of the quote.
   - Dense-only rank & top-5 IDs.
   - BM25-only rank & top-5 IDs.
   - RRF fused rank.
3. Search diagnostics for golden_07 and golden_29:
   - Repeats step 2 for each.
4. Top-8 frequency of chunk 97e04e65-82f0-5217-bb0d-57f54d63ccf8 across all 19 answerable golden questions:
   - Dense, BM25, and fused top-8 presence and ranks.

Zero LLM calls and zero Gemini quota used.
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer

# Suppress HF Hub unauthenticated warnings and telemetry
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging to show info cleanly
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
BM25_INPUTS_PATH = BASE_DIR / "artifacts" / "bm25_inputs.json"
GOLDEN_SET_PATH = BASE_DIR / "eval" / "golden_set.json"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

TARGET_OVERVIEW_CHUNK_ID = "97e04e65-82f0-5217-bb0d-57f54d63ccf8"
TARGET_GOLD_CHUNK_ID = "dfa8abbf-4dda-57f5-b67f-6ada36cf6dae"


def load_chunks() -> List[Dict[str, Any]]:
    """Loads chunks from chunks.json or bm25_inputs.json fallback."""
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    elif BM25_INPUTS_PATH.exists():
        with open(BM25_INPUTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Could not find chunks at {CHUNKS_PATH} or {BM25_INPUTS_PATH}")


def load_golden_set() -> List[Dict[str, Any]]:
    """Loads golden evaluation set."""
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_stats(lengths: List[int]) -> Dict[str, Any]:
    """Computes min, median, p90, max, and percentage > 512."""
    if not lengths:
        return {"min": 0, "median": 0, "p90": 0, "max": 0, "pct_gt_512": 0.0, "count": 0}
    arr = np.array(lengths)
    return {
        "count": len(lengths),
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(np.max(arr)),
        "pct_gt_512": float(np.mean(arr > 512) * 100.0),
    }


# =============================================================================
# Step 1: Token Lengths Analysis
# =============================================================================
def run_step_1_token_lengths(chunks: List[Dict[str, Any]], tokenizer: Any) -> Tuple[List[int], Dict[str, Any]]:
    print("=" * 80)
    print("STEP 1: TOKEN LENGTHS ANALYSIS (BAAI/bge-small-en-v1.5)")
    print("=" * 80)
    print(f"Total chunks loaded: {len(chunks):,}")
    print("Tokenizing corpus chunks...")

    all_lengths: List[int] = []
    by_company: Dict[str, List[int]] = defaultdict(list)
    by_section: Dict[str, List[int]] = defaultdict(list)
    by_comp_sec: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for c in chunks:
        text = c.get("text", "")
        # Encode with special tokens as the transformer model receives them
        token_count = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        all_lengths.append(token_count)

        comp = c.get("company", "Unknown")
        sec = c.get("item_section", "Unknown")
        by_company[comp].append(token_count)
        by_section[sec].append(token_count)
        by_comp_sec[(comp, sec)].append(token_count)

    overall_stats = calculate_stats(all_lengths)
    print("\n--- Overall Corpus Token Lengths ---")
    print(f"  Total Chunks:      {overall_stats['count']:,}")
    print(f"  Min tokens:        {overall_stats['min']}")
    print(f"  Median tokens:     {overall_stats['median']:.1f}")
    print(f"  P90 tokens:        {overall_stats['p90']:.1f}")
    print(f"  Max tokens:        {overall_stats['max']}")
    print(f"  Chunks > 512 tok:  {overall_stats['pct_gt_512']:.2f}% ({sum(1 for l in all_lengths if l > 512):,} chunks)")

    print("\n--- Breakdown by Company ---")
    print(f"{'Company':<15} | {'Count':<7} | {'Min':<5} | {'Median':<8} | {'P90':<8} | {'Max':<5} | {'% > 512':<8}")
    print("-" * 70)
    for comp in sorted(by_company.keys()):
        s = calculate_stats(by_company[comp])
        print(f"{comp:<15} | {s['count']:<7} | {s['min']:<5} | {s['median']:<8.1f} | {s['p90']:<8.1f} | {s['max']:<5} | {s['pct_gt_512']:<7.2f}%")

    print("\n--- Breakdown by Item Section ---")
    print(f"{'Section':<15} | {'Count':<7} | {'Min':<5} | {'Median':<8} | {'P90':<8} | {'Max':<5} | {'% > 512':<8}")
    print("-" * 70)
    for sec in sorted(by_section.keys()):
        s = calculate_stats(by_section[sec])
        print(f"{sec:<15} | {s['count']:<7} | {s['min']:<5} | {s['median']:<8.1f} | {s['p90']:<8.1f} | {s['max']:<5} | {s['pct_gt_512']:<7.2f}%")

    return all_lengths, overall_stats


# =============================================================================
# Helper for Gold Chunk & Quote Token Offset
# =============================================================================
def find_quote_token_offset(chunk_text: str, quote: str, tokenizer: Any) -> Tuple[int, int, int]:
    """
    Returns (char_offset, token_offset, total_tokens) for quote within chunk_text.
    Token offset is the 0-indexed position in input_ids where the quote begins.
    """
    enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    total_tokens = len(input_ids)

    char_offset = chunk_text.find(quote)
    if char_offset == -1:
        # Fallback partial matching if needed
        words = [w for w in quote.split() if len(w) > 3][:4]
        sub = " ".join(words)
        char_offset = chunk_text.find(sub)

    token_offset = -1
    if char_offset != -1:
        for idx, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= char_offset < e or s == char_offset:
                token_offset = idx
                break

    return char_offset, token_offset, total_tokens


# =============================================================================
# Steps 2 & 3: Gold Chunk Diagnostics for golden_01, golden_07, golden_29
# =============================================================================
def run_steps_2_and_3_gold_chunk_search(
    golden_set: List[Dict[str, Any]],
    chunks_lookup: Dict[str, Dict[str, Any]],
    tokenizer: Any,
) -> None:
    from backend.agent.retrieval import search_qdrant, search_bm25, reciprocal_rank_fusion

    target_ids = ["golden_01", "golden_07", "golden_29"]
    item_map = {item["id"]: item for item in golden_set if item["id"] in target_ids}

    print("\n" + "=" * 80)
    print("STEPS 2 & 3: GOLD CHUNK SEARCH DIAGNOSTICS (golden_01, golden_07, golden_29)")
    print(f"Target Gold Chunk: {TARGET_GOLD_CHUNK_ID}")
    print("=" * 80)

    gold_chunk = chunks_lookup.get(TARGET_GOLD_CHUNK_ID)
    if not gold_chunk:
        print(f"ERROR: Target gold chunk {TARGET_GOLD_CHUNK_ID} not found in corpus lookup!")
        return

    gold_text = gold_chunk.get("text", "")
    filters = {"company": "Apple", "fiscal_year": "2025"}

    for item_id in target_ids:
        item = item_map.get(item_id)
        if not item:
            print(f"Warning: {item_id} not found in golden set.")
            continue

        q = item["question"]
        quotes = item.get("source_quotes", [])
        quote = quotes[0] if quotes else item.get("ground_truth", "")

        char_off, tok_off, total_tok = find_quote_token_offset(gold_text, quote, tokenizer)

        print(f"\n--------------------------------------------------------------------------------")
        print(f"Item ID:   {item_id}")
        print(f"Question:  {q}")
        print(f"Filter:    {filters}")
        print(f"Quote:     \"{quote[:120]}...\"")
        print(f"Gold Chunk Stats:")
        print(f"  - Total Token Length:    {total_tok} tokens")
        print(f"  - Quote Char Offset:     {char_off}")
        print(f"  - Quote Token Offset:    {tok_off} (out of {total_tok} tokens)")
        if tok_off >= 512:
            print(f"  ⚠️ ALERT: Quote token offset ({tok_off}) EXCEEDS model limit 512! Truncated by transformer embedding.")
        else:
            print(f"  ✓ Quote token offset ({tok_off}) is within 512 limit.")

        # Run dense search (limit=50)
        dense_results = search_qdrant(query=q, filters=filters, limit=50)
        dense_ids = [d["id"] for d in dense_results]
        dense_rank = (dense_ids.index(TARGET_GOLD_CHUNK_ID) + 1) if TARGET_GOLD_CHUNK_ID in dense_ids else None
        dense_rank_str = f"Rank {dense_rank}" if dense_rank is not None else "not in top 50"

        # Run BM25 search (limit=50)
        bm25_results = search_bm25(query=q, filters=filters, limit=50)
        bm25_ids = [b["id"] for b in bm25_results]
        bm25_rank = (bm25_ids.index(TARGET_GOLD_CHUNK_ID) + 1) if TARGET_GOLD_CHUNK_ID in bm25_ids else None
        bm25_rank_str = f"Rank {bm25_rank}" if bm25_rank is not None else "not in top 50"

        # Run RRF fusion (limit=50)
        fused_results = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_k=50)
        fused_ids = [f["id"] for f in fused_results]
        fused_rank = (fused_ids.index(TARGET_GOLD_CHUNK_ID) + 1) if TARGET_GOLD_CHUNK_ID in fused_ids else None
        fused_rank_str = f"Rank {fused_rank}" if fused_rank is not None else "not in top 50"

        print(f"\nRetrieval Performance for {item_id}:")
        print(f"  Dense (Qdrant) Rank:  {dense_rank_str}")
        print(f"    Top-5 Dense IDs:")
        for rk, cid in enumerate(dense_ids[:5], start=1):
            marker = "  <-- GOLD CHUNK" if cid == TARGET_GOLD_CHUNK_ID else ""
            if cid == TARGET_OVERVIEW_CHUNK_ID:
                marker = "  <-- OVERVIEW CHUNK (97e04e65...)"
            print(f"      {rk}. {cid}{marker}")

        print(f"  BM25 (Sparse) Rank:   {bm25_rank_str}")
        print(f"    Top-5 BM25 IDs:")
        for rk, cid in enumerate(bm25_ids[:5], start=1):
            marker = "  <-- GOLD CHUNK" if cid == TARGET_GOLD_CHUNK_ID else ""
            if cid == TARGET_OVERVIEW_CHUNK_ID:
                marker = "  <-- OVERVIEW CHUNK (97e04e65...)"
            print(f"      {rk}. {cid}{marker}")

        print(f"  RRF Fused Rank:       {fused_rank_str}")
        print(f"    Top-5 Fused IDs:")
        for rk, cid in enumerate(fused_ids[:5], start=1):
            marker = "  <-- GOLD CHUNK" if cid == TARGET_GOLD_CHUNK_ID else ""
            if cid == TARGET_OVERVIEW_CHUNK_ID:
                marker = "  <-- OVERVIEW CHUNK (97e04e65...)"
            print(f"      {rk}. {cid}{marker}")


# =============================================================================
# Step 4: Chunk 97e04e65-82f0-5217-bb0d-57f54d63ccf8 Frequency in Top-8
# =============================================================================
def run_step_4_target_chunk_frequency(
    golden_set: List[Dict[str, Any]],
) -> None:
    from backend.agent.retrieval import search_qdrant, search_bm25, reciprocal_rank_fusion
    from backend.agent.nodes.classifier import extract_metadata_rules

    print("\n" + "=" * 80)
    print("STEP 4: FREQUENCY OF CHUNK 97e04e65-82f0-5217-bb0d-57f54d63ccf8 IN TOP-8")
    print(f"Target Chunk: {TARGET_OVERVIEW_CHUNK_ID} (Apple FY2025 Item 7 Overview)")
    print("=" * 80)

    answerable = [item for item in golden_set if item.get("category") != "unanswerable"]
    print(f"Total answerable golden items: {len(answerable)}")

    dense_top8_hits = 0
    bm25_top8_hits = 0
    fused_top8_hits = 0

    results_table = []

    for item in answerable:
        g_id = item["id"]
        q = item["question"]
        gold_comp = item.get("company", "Apple")
        gold_fy = item.get("fiscal_year", "")

        # Extract filters identically to agentic pipeline
        comp, fy, _, is_ambig = extract_metadata_rules(q)
        filters = {}
        if comp:
            filters["company"] = comp
        if fy and not is_ambig:
            filters["fiscal_year"] = str(fy)
        elif "/" not in str(gold_fy) and gold_fy:
            # Fallback to item metadata if rule parser had no year but single year
            filters["fiscal_year"] = str(gold_fy)

        # Run dense (top 8)
        dense_hits = search_qdrant(query=q, filters=filters, limit=8)
        dense_ids = [d["id"] for d in dense_hits]
        in_dense = TARGET_OVERVIEW_CHUNK_ID in dense_ids
        dense_rank = (dense_ids.index(TARGET_OVERVIEW_CHUNK_ID) + 1) if in_dense else "-"
        if in_dense:
            dense_top8_hits += 1

        # Run BM25 (top 8)
        bm25_hits = search_bm25(query=q, filters=filters, limit=8)
        bm25_ids = [b["id"] for b in bm25_hits]
        in_bm25 = TARGET_OVERVIEW_CHUNK_ID in bm25_ids
        bm25_rank = (bm25_ids.index(TARGET_OVERVIEW_CHUNK_ID) + 1) if in_bm25 else "-"
        if in_bm25:
            bm25_top8_hits += 1

        # Run Fused (top 8)
        # Fuse top 20 dense + sparse to get top 8
        dense_for_fusion = search_qdrant(query=q, filters=filters, limit=20)
        bm25_for_fusion = search_bm25(query=q, filters=filters, limit=20)
        fused_hits = reciprocal_rank_fusion(dense_for_fusion, bm25_for_fusion, k=60, top_k=8)
        fused_ids = [f["id"] for f in fused_hits]
        in_fused = TARGET_OVERVIEW_CHUNK_ID in fused_ids
        fused_rank = (fused_ids.index(TARGET_OVERVIEW_CHUNK_ID) + 1) if in_fused else "-"
        if in_fused:
            fused_top8_hits += 1

        results_table.append({
            "id": g_id,
            "fy": filters.get("fiscal_year", "None (multi-year)"),
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
            "fused_rank": fused_rank,
            "question": q[:50] + "...",
        })

    print(f"\n{'ID':<11} | {'FY Filter':<18} | {'Dense Rank':<11} | {'BM25 Rank':<10} | {'Fused Rank':<11} | {'Question Preview'}")
    print("-" * 95)
    for row in results_table:
        print(f"{row['id']:<11} | {row['fy']:<18} | {str(row['dense_rank']):<11} | {str(row['bm25_rank']):<10} | {str(row['fused_rank']):<11} | {row['question']}")

    print("\n--- Summary: Top-8 Appearances of 97e04e65-82f0-5217-bb0d-57f54d63ccf8 ---")
    print(f"  Dense (Qdrant) Top-8:   {dense_top8_hits:>2} / {len(answerable)} ({dense_top8_hits / len(answerable) * 100:.1f}%)")
    print(f"  BM25 (Sparse)  Top-8:   {bm25_top8_hits:>2} / {len(answerable)} ({bm25_top8_hits / len(answerable) * 100:.1f}%)")
    print(f"  Fused (RRF)    Top-8:   {fused_top8_hits:>2} / {len(answerable)} ({fused_top8_hits / len(answerable) * 100:.1f}%)")


# =============================================================================
# Main Entrypoint
# =============================================================================
def main():
    print("Loading tokenizer:", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print("Loading corpus chunks...")
    chunks = load_chunks()
    chunks_lookup = {str(c.get("chunk_id") or c.get("id")): c for c in chunks}

    print("Loading golden set...")
    golden_set = load_golden_set()

    # Step 1
    run_step_1_token_lengths(chunks, tokenizer)

    # Steps 2 & 3
    run_steps_2_and_3_gold_chunk_search(golden_set, chunks_lookup, tokenizer)

    # Step 4
    run_step_4_target_chunk_frequency(golden_set)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE. NO RETRIEVAL CODE HAS BEEN MODIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    main()

