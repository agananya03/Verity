#!/usr/bin/env python3
"""
Production Parity Verification Script
======================================
Runs the real production retrieve() function on single-year answerable items
from golden_set.json (dev set, n=14) and golden_heldout.json (round-1 heldout, n=62).

Verifies exact parity with the pre-registered fusion sweep results:
  - RETRIEVAL_STRATEGY="clean_v3_rrf10" (new default):
      Dev Hit@8     : 14/14 (100.0%)
      Heldout Hit@8 : 44/62 ( 71.0%)
      Combined Hit@8: 58/76 ( 76.3%)
  - RETRIEVAL_STRATEGY="legacy":
      Dev Hit@8     : 14/14 (100.0%)
      Heldout Hit@8 : 35/62 ( 56.5%)
      Combined Hit@8: 49/76 ( 64.5%)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DEV_PATH = REPO_ROOT / "backend" / "eval" / "golden_set.json"
GOLDEN_HELDOUT_PATH = REPO_ROOT / "backend" / "eval" / "golden_heldout.json"


def evaluate_items(
    items: List[Dict[str, Any]],
    top_k: int = 8,
    use_explicit_filters: bool = False,
    pool_limit: Optional[int] = None,
) -> Tuple[int, int, List[bool]]:
    """Runs production retrieve() for each item and checks Hit@top_k."""
    from backend.agent.retrieval import retrieve
    from backend.eval.retrieval_ablation import derive_query_filters

    hits = []
    for it in items:
        q = it["question"]
        gold_ids = set(it["source_chunk_ids"])
        filters = derive_query_filters(q) if use_explicit_filters else None
        req_k = pool_limit if pool_limit is not None else top_k
        retrieved_chunks = retrieve(query=q, top_k=req_k, filters=filters)
        retrieved_ids = [str(c.get("id") or c.get("chunk_id")) for c in retrieved_chunks[:top_k]]
        is_hit = any(cid in gold_ids for cid in retrieved_ids)
        hits.append(is_hit)
    return sum(hits), len(hits), hits


def run_verification() -> None:
    from backend.agent.retrieval import is_qdrant_connected

    print("=" * 80)
    print("VERIFYING PRODUCTION RETRIEVAL PARITY")
    print("=" * 80)
    print(f"Qdrant Live Connection: {is_qdrant_connected()}")

    with open(GOLDEN_DEV_PATH, "r", encoding="utf-8") as f:
        dev_raw = json.load(f)
    with open(GOLDEN_HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout_raw = json.load(f)

    dev_items = [
        x for x in dev_raw
        if x.get("source_chunk_ids") and "/" not in str(x.get("fiscal_year", ""))
    ]
    heldout_items = [
        x for x in heldout_raw
        if x.get("source_chunk_ids") and "/" not in str(x.get("fiscal_year", ""))
    ]
    combined_items = dev_items + heldout_items

    print(f"Dataset items: Dev={len(dev_items)}, Heldout={len(heldout_items)}, Combined={len(combined_items)}\n")

    # -------------------------------------------------------------------------
    # 1. Test New Production Default: clean_v3_rrf10 (standalone query, auto-extraction)
    # -------------------------------------------------------------------------
    os.environ["RETRIEVAL_STRATEGY"] = "clean_v3_rrf10"
    print("--- 1. Testing Default RETRIEVAL_STRATEGY='clean_v3_rrf10' (auto-extracting filters) ---")

    dev_k, dev_n, _ = evaluate_items(dev_items, top_k=8, use_explicit_filters=False)
    held_k, held_n, _ = evaluate_items(heldout_items, top_k=8, use_explicit_filters=False)
    comb_k, comb_n = dev_k + held_k, dev_n + held_n

    print(f"Dev Set Hit@8         : {dev_k}/{dev_n} ({dev_k/dev_n:5.1%}) [Expected: 14/14 (100.0%)]")
    print(f"Round-1 Heldout Hit@8 : {held_k}/{held_n} ({held_k/held_n:5.1%}) [Expected: 44/62 (71.0%)]")
    print(f"Combined Hit@8        : {comb_k}/{comb_n} ({comb_k/comb_n:5.1%}) [Expected: 58/76 (76.3%)]")

    assert dev_k == 14, f"Dev hit count mismatch: expected 14, got {dev_k}"
    assert held_k == 44, f"Heldout hit count mismatch: expected 44, got {held_k}"
    assert comb_k == 58, f"Combined hit count mismatch: expected 58, got {comb_k}"
    print(">>> PASS: clean_v3_rrf10 matches fusion sweep results with 100% parity!\n")

    # -------------------------------------------------------------------------
    # 2. Test Legacy Mode: legacy (production retriever node with extracted filters)
    # -------------------------------------------------------------------------
    os.environ["RETRIEVAL_STRATEGY"] = "legacy"
    print("--- 2. Testing RETRIEVAL_STRATEGY='legacy' (ablation top-30 candidate pool) ---")

    leg_dev_k, leg_dev_n, _ = evaluate_items(dev_items, top_k=8, use_explicit_filters=True, pool_limit=30)
    leg_held_k, leg_held_n, _ = evaluate_items(heldout_items, top_k=8, use_explicit_filters=True, pool_limit=30)
    leg_comb_k, leg_comb_n = leg_dev_k + leg_held_k, leg_dev_n + leg_held_n

    print(f"Dev Set Hit@8         : {leg_dev_k}/{leg_dev_n} ({leg_dev_k/leg_dev_n:5.1%}) [Expected: 14/14 (100.0%)]")
    print(f"Round-1 Heldout Hit@8 : {leg_held_k}/{leg_held_n} ({leg_held_k/leg_held_n:5.1%}) [Expected: 35/62 (56.5%)]")
    print(f"Combined Hit@8        : {leg_comb_k}/{leg_comb_n} ({leg_comb_k/leg_comb_n:5.1%}) [Expected: 49/76 (64.5%)]")

    assert leg_dev_k == 14, f"Legacy Dev hit count mismatch: expected 14, got {leg_dev_k}"
    assert leg_held_k == 35, f"Legacy Heldout hit count mismatch: expected 35, got {leg_held_k}"
    assert leg_comb_k == 49, f"Legacy Combined hit count mismatch: expected 49, got {leg_comb_k}"
    print(">>> PASS: legacy strategy reproduces exact baseline ablation numbers with 100% parity!\n")

    print("=" * 80)
    print("ALL PRODUCTION PARITY CHECKS PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
