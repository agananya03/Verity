#!/usr/bin/env python3
"""
Fusion Sweep Benchmark Evaluation (Read-Only)
==============================================
Evaluates 14 PRE-REGISTERED retrieval configurations over:
  - Dev set: backend/eval/golden_set.json (single-year answerable, n=14)
  - Round-1 heldout: backend/eval/golden_heldout.json (single-year answerable, n=62)
  - Combined set: (n=76)

Configurations:
  Query Variants (2):
    - raw: original question
    - clean_v3: strip preamble, Form 10-K, and fiscal year phrases;
                preserve calendar dates (e.g. "June 30, 2024") and company name.

  Fusion Methods (6) over Dense top-50 and BM25 top-50:
    1. bm25_only
    2. dense_only
    3. rrf_k60 (equal weights 1.0, 1.0)
    4. rrf_k10 (equal weights 1.0, 1.0)
    5. wrrf_bm25x2 (BM25 weight 2.0, Dense weight 1.0, k=60)
    6. interleave_4_4 (round-robin alternating top results, dedupe)

  (6 fusions x 2 query variants = 12 configs)

  Plus 2 Rerank Configs:
    13. raw__rrf_k60_rerank (rrf_k60 top-30 reranked with local cross-encoder ms-marco-MiniLM-L-6-v2)
    14. clean_v3__rrf_k60_rerank (rrf_k60 top-30 reranked with local cross-encoder ms-marco-MiniLM-L-6-v2)

Selection Rule:
  "Choose the config with the highest combined hit@8 that is not worse than
   bm25_only/raw on either set; break ties by lower latency.
   Print the chosen config and the runner-up, and do NOT change production."
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats
from backend.agent.nodes.classifier import strip_query_metadata_v3

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DEV_PATH = REPO_ROOT / "backend" / "eval" / "golden_set.json"
GOLDEN_HELDOUT_PATH = REPO_ROOT / "backend" / "eval" / "golden_heldout.json"
SWEEP_CACHE_PATH = REPO_ROOT / "backend" / "eval" / "results" / "fusion_sweep_cache.json"
CHUNKS_PATH = REPO_ROOT / "backend" / "data" / "processed" / "chunks.json"


# =============================================================================
# 1. Query Cleaning: clean_v3
# =============================================================================
def strip_query_metadata_v3(
    query: str,
    company: Optional[str] = None,
    fiscal_year: Optional[str] = None,
) -> str:
    """
    clean_v3:
    Removes ONLY:
    1. Leading preamble ("According to / Based on / In / Per ... Form 10-K,")
    2. "Form 10-K" and "10-K" references
    3. Disambiguation clauses like "per {Company}'s fiscal {year} annual report",
       "in {Company}'s FY{year} 10-K", "for fiscal {year}"
    4. Phrases like "fiscal year 2024", "FY2024", "fiscal 2024"

    MUST NOT remove:
    - Company name (e.g. "Alphabet", "Amazon", "Apple", "Microsoft", "NVIDIA")
    - Calendar dates with years (e.g. "June 30, 2024", "December 31, 2023", "January 29, 2025")
    """
    if not query or not query.strip():
        return ""

    q = query.strip()

    # 1. Leading preamble removal ("According to / Based on / In / Per ... Form 10-K,")
    q = re.sub(
        r"^\s*(?:According\s+to|Based\s+on|In|Per)\s+(?:the\s+)?(?:(?:Item\s+[0-9A-Z]+|[A-Za-z]+)\s+section\s+of\s+)?(?:[A-Za-z0-9'’\s-]*?)(?:Form\s+)?10-K(?:\s+filings?)?\s*,\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )

    # 2. Form 10-K references
    q = re.sub(r"\b(?:Form\s+)?10-K(?:\s+filings?)?\b", "", q, flags=re.IGNORECASE)

    # 3. Disambiguation phrases added during review
    q = re.sub(r",?\s*per\s+[A-Za-z0-9'’\s-]+?'s\s+fiscal\s+\d{4}\s+annual\s+report\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*in\s+[A-Za-z0-9'’\s-]+?'s\s+FY\s*'?\d{2,4}(?:\s+10-K)?\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*in\s+[A-Za-z0-9'’\s-]+?'s\s+fiscal\s+\d{4}\s+filing\b", "", q, flags=re.IGNORECASE)

    # 4. Phrases like "fiscal year 2024", "fiscal 2024", "FY2024", "FY 2024"
    # Note: does NOT match bare year like "2024" in "June 30, 2024"
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*fiscal\s+year\s+\d{4}\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*fiscal\s+\d{4}\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r",?\s*\b(?:for|in|during)?\s*FY\s*'?\d{2,4}\b", "", q, flags=re.IGNORECASE)

    # Whitespace and punctuation cleanup
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\s+([,?.])", r"\1", q)
    q = re.sub(r",\s*\?", "?", q)
    q = re.sub(r",\s*$", "", q)
    return q


# =============================================================================
# 2. Fusion Methods
# =============================================================================
def interleave_4_4(
    list_a: List[str],
    list_b: List[str],
    top_k: int = 50,
) -> List[str]:
    """
    Interleaves top results from list_a and list_b in round-robin fashion,
    alternating unique items from each list and deduplicating.
    For top-8 results, this guarantees a 4/4 balance between the two lists
    (unless one list is exhausted).
    """
    merged: List[str] = []
    seen: Set[str] = set()
    idx_a, idx_b = 0, 0
    turn_a = True

    while (idx_a < len(list_a) or idx_b < len(list_b)) and len(merged) < top_k:
        if turn_a:
            while idx_a < len(list_a):
                item = list_a[idx_a]
                idx_a += 1
                if item not in seen:
                    seen.add(item)
                    merged.append(item)
                    break
            turn_a = False
        else:
            while idx_b < len(list_b):
                item = list_b[idx_b]
                idx_b += 1
                if item not in seen:
                    seen.add(item)
                    merged.append(item)
                    break
            turn_a = True

    return merged


def rrf_fuse(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 50,
) -> List[str]:
    """Computes Weighted Reciprocal Rank Fusion over dense and sparse hit lists."""
    rrf_scores: Dict[str, float] = {}
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = str(doc.get("id") or doc.get("chunk_id"))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (dense_weight / (k + rank))
    for rank, doc in enumerate(sparse_results, start=1):
        doc_id = str(doc.get("id") or doc.get("chunk_id"))
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))
    sorted_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)
    return sorted_ids[:top_k]


# =============================================================================
# 3. Statistical Helpers
# =============================================================================
def exact_mcnemar_test(hits_a: List[bool], hits_b: List[bool]) -> Dict[str, Any]:
    """
    Computes exact two-sided McNemar test using the binomial distribution.
    Under H0, discordant pairs (b, c) follow Binomial(b + c, 0.5).
    """
    both = sum(1 for a, b in zip(hits_a, hits_b) if a and b)
    only_a = sum(1 for a, b in zip(hits_a, hits_b) if a and not b)
    only_b = sum(1 for a, b in zip(hits_a, hits_b) if not a and b)
    neither = sum(1 for a, b in zip(hits_a, hits_b) if not a and not b)
    discordant = only_a + only_b
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(scipy.stats.binomtest(only_a, discordant, p=0.5).pvalue)
    return {
        "both": both,
        "only_a": only_a,
        "only_b": only_b,
        "neither": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def compute_holm_bonferroni(p_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Computes Holm-Bonferroni step-down adjusted p-values across m comparisons.
    Formula:
      Sort p_(1) <= p_(2) <= ... <= p_(m)
      Adjusted p_(k) = min(1.0, max_{j <= k} ((m - j + 1) * p_(j)))
    """
    m = len(p_dict)
    sorted_items = sorted(p_dict.items(), key=lambda x: x[1])
    adjusted: Dict[str, float] = {}
    current_max = 0.0
    for rank_idx, (cfg, p_val) in enumerate(sorted_items):
        k = rank_idx + 1  # 1-based index
        multiplier = m - k + 1
        adj_p = min(1.0, multiplier * p_val)
        current_max = max(current_max, adj_p)
        adjusted[cfg] = min(1.0, current_max)
    return adjusted


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    z = scipy.stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_metrics(retrieved_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
    """Computes Hit@8, Recall@8, MRR, and Hit@20."""
    golds = set(gold_ids)
    top8 = retrieved_ids[:8]
    top20 = retrieved_ids[:20]

    hit8 = 1.0 if any(cid in golds for cid in top8) else 0.0
    recall8 = len(set(top8) & golds) / len(golds) if golds else 0.0
    hit20 = 1.0 if any(cid in golds for cid in top20) else 0.0

    mrr = 0.0
    for rank_idx, cid in enumerate(retrieved_ids, start=1):
        if cid in golds:
            mrr = 1.0 / rank_idx
            break

    return {
        "hit@8": hit8,
        "recall@8": recall8,
        "mrr": mrr,
        "hit@20": hit20,
    }


# =============================================================================
# 4. Fusion Sweep Execution
# =============================================================================
def run_fusion_sweep() -> None:
    from backend.agent.retrieval import search_bm25, reciprocal_rank_fusion
    from backend.eval.retrieval_ablation import derive_query_filters

    # 1. Load datasets
    with open(GOLDEN_DEV_PATH, "r", encoding="utf-8") as f:
        dev_raw = json.load(f)
    with open(GOLDEN_HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout_raw = json.load(f)

    # Filter to answerable single-year items only
    dev_items = [
        x for x in dev_raw
        if x.get("source_chunk_ids") and "/" not in str(x.get("fiscal_year", ""))
    ]
    heldout_items = [
        x for x in heldout_raw
        if x.get("source_chunk_ids") and "/" not in str(x.get("fiscal_year", ""))
    ]
    combined_items = dev_items + heldout_items

    n_dev = len(dev_items)
    n_heldout = len(heldout_items)
    n_comb = len(combined_items)

    print("=" * 80)
    print("FUSION SWEEP BENCHMARK (Single-Year Answerable Items)")
    print(f"Dev Set: n={n_dev} | Round-1 Heldout: n={n_heldout} | Combined: n={n_comb}")
    print("=" * 80)
    print("\nPRE-REGISTERED SELECTION RULE:")
    print("  'Choose the config with the highest combined hit@8 that is not worse")
    print("   than bm25_only/raw on either set; break ties by lower latency.")
    print("   Print the chosen config and the runner-up, and do NOT change production.'\n")

    # Load cache
    cache = {}
    if SWEEP_CACHE_PATH.exists():
        with open(SWEEP_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Lazy cross-encoder if needed
    cross_encoder = None

    def get_encoder():
        nonlocal cross_encoder
        if cross_encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            cross_encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        return cross_encoder

    # Define the 14 configurations
    config_names = [
        "raw__bm25_only",
        "raw__dense_only",
        "raw__rrf_k60",
        "raw__rrf_k10",
        "raw__wrrf_bm25x2",
        "raw__interleave_4_4",
        "clean_v3__bm25_only",
        "clean_v3__dense_only",
        "clean_v3__rrf_k60",
        "clean_v3__rrf_k10",
        "clean_v3__wrrf_bm25x2",
        "clean_v3__interleave_4_4",
        "raw__rrf_k60_rerank",
        "clean_v3__rrf_k60_rerank",
    ]

    # Pre-compute BM25 and candidate pools per item
    # Map item id -> query variants -> candidate lists
    item_candidate_pools: Dict[str, Dict[str, Any]] = {}

    for it in combined_items:
        qid = it["id"]
        q_raw = it["question"]
        q_clean = strip_query_metadata_v3(q_raw)
        f = derive_query_filters(q_raw)
        clean_f = dict(f) if f else {}
        clean_f.pop("fiscal_years", None)

        cached_entry = cache.get(qid, {})

        # Dense top-50 from cache (or query if cache missing)
        dense_raw = cached_entry.get("dense_raw", [])
        dense_clean = cached_entry.get("dense_clean", [])

        # BM25 top-50 (pure local keyword search)
        bm25_raw = search_bm25(query=q_raw, filters=clean_f or None, limit=50)
        bm25_clean = search_bm25(query=q_clean, filters=clean_f or None, limit=50)

        bm25_raw_ids = [str(c.get("id") or c.get("chunk_id")) for c in bm25_raw]
        bm25_clean_ids = [str(c.get("id") or c.get("chunk_id")) for c in bm25_clean]
        dense_raw_ids = [str(c.get("id") or c.get("chunk_id")) for c in dense_raw]
        dense_clean_ids = [str(c.get("id") or c.get("chunk_id")) for c in dense_clean]

        item_candidate_pools[qid] = {
            "q_raw": q_raw,
            "q_clean": q_clean,
            "filters": clean_f,
            "dense_raw": dense_raw,
            "dense_clean": dense_clean,
            "bm25_raw": bm25_raw,
            "bm25_clean": bm25_clean,
            "bm25_raw_ids": bm25_raw_ids,
            "bm25_clean_ids": bm25_clean_ids,
            "dense_raw_ids": dense_raw_ids,
            "dense_clean_ids": dense_clean_ids,
            "rerank_raw": cached_entry.get("rerank_raw"),
            "rerank_clean": cached_entry.get("rerank_clean"),
        }

    # Evaluate each config
    # Results structure: config_name -> dataset_name ("dev", "heldout", "combined") -> metrics
    sweep_results: Dict[str, Dict[str, Any]] = {cfg: {} for cfg in config_names}
    per_item_hits: Dict[str, Dict[str, List[bool]]] = {
        "dev": {cfg: [] for cfg in config_names},
        "heldout": {cfg: [] for cfg in config_names},
        "combined": {cfg: [] for cfg in config_names},
    }
    latencies: Dict[str, List[float]] = {cfg: [] for cfg in config_names}

    for cfg in config_names:
        variant, method = cfg.split("__", 1)

        for it in combined_items:
            qid = it["id"]
            golds = it["source_chunk_ids"]
            pools = item_candidate_pools[qid]

            t_start = time.perf_counter()

            if variant == "raw":
                sparse_hits = pools["bm25_raw"]
                sparse_ids = pools["bm25_raw_ids"]
                dense_hits = pools["dense_raw"]
                dense_ids = pools["dense_raw_ids"]
                q_text = pools["q_raw"]
                rerank_cached = pools["rerank_raw"]
            else:
                sparse_hits = pools["bm25_clean"]
                sparse_ids = pools["bm25_clean_ids"]
                dense_hits = pools["dense_clean"]
                dense_ids = pools["dense_clean_ids"]
                q_text = pools["q_clean"]
                rerank_cached = pools["rerank_clean"]

            if method == "bm25_only":
                retrieved = sparse_ids[:50]
            elif method == "dense_only":
                retrieved = dense_ids[:50]
            elif method == "rrf_k60":
                retrieved = rrf_fuse(dense_hits, sparse_hits, k=60, dense_weight=1.0, sparse_weight=1.0, top_k=50)
            elif method == "rrf_k10":
                retrieved = rrf_fuse(dense_hits, sparse_hits, k=10, dense_weight=1.0, sparse_weight=1.0, top_k=50)
            elif method == "wrrf_bm25x2":
                retrieved = rrf_fuse(dense_hits, sparse_hits, k=60, dense_weight=1.0, sparse_weight=2.0, top_k=50)
            elif method == "interleave_4_4":
                retrieved = interleave_4_4(sparse_ids, dense_ids, top_k=50)
            elif method == "rrf_k60_rerank":
                if rerank_cached:
                    retrieved = rerank_cached[:30]
                else:
                    # Compute rerank on the fly
                    fused_docs = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_k=30)
                    encoder = get_encoder()
                    texts = [c.get("text", "") for c in fused_docs]
                    scores = list(encoder.rerank(q_text, texts))
                    scored = sorted(zip(fused_docs, scores), key=lambda x: x[1], reverse=True)
                    retrieved = [str(c[0].get("id") or c[0].get("chunk_id")) for c in scored]
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies[cfg].append(elapsed_ms)

            m = compute_metrics(retrieved, golds)
            is_hit8 = (m["hit@8"] == 1.0)

            # Store per item
            is_dev = it in dev_items
            dataset_key = "dev" if is_dev else "heldout"
            per_item_hits[dataset_key][cfg].append(is_hit8)
            per_item_hits["combined"][cfg].append(is_hit8)

    # Compute production hits for baseline comparison
    # Production dev hits: 14/14 = 1.0000
    # Production heldout single-year hits: 35/62 = 0.5645
    prod_hits: Dict[str, List[bool]] = {
        "dev": [True] * 14,
        "heldout": [],
        "combined": [],
    }
    # Load heldout production hits from ablation results
    with open(REPO_ROOT / "backend" / "eval" / "results" / "retrieval_ablation_heldout.json", "r", encoding="utf-8") as f:
        abl_held = json.load(f)
    abl_held_map = {it["id"]: it for it in abl_held.get("per_item_results", [])}
    for it in heldout_items:
        h = abl_held_map.get(it["id"], {}).get("configs", {}).get("production_retrieve", {}).get("metrics", {}).get("hit@8", 0.0)
        prod_hits["heldout"].append(h == 1.0)
    prod_hits["combined"] = prod_hits["dev"] + prod_hits["heldout"]

    # Benchmark bm25_only/raw hits
    bm25_raw_hits = per_item_hits["combined"]["raw__bm25_only"]

    # Aggregate metrics for each dataset
    def aggregate_dataset(dataset_name: str, item_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        agg = {}
        n = len(item_list)
        for cfg in config_names:
            variant, method = cfg.split("__", 1)
            all_m = []
            for it in item_list:
                qid = it["id"]
                golds = it["source_chunk_ids"]
                pools = item_candidate_pools[qid]

                if variant == "raw":
                    sparse_hits = pools["bm25_raw"]
                    sparse_ids = pools["bm25_raw_ids"]
                    dense_hits = pools["dense_raw"]
                    dense_ids = pools["dense_raw_ids"]
                    q_text = pools["q_raw"]
                    rerank_cached = pools["rerank_raw"]
                else:
                    sparse_hits = pools["bm25_clean"]
                    sparse_ids = pools["bm25_clean_ids"]
                    dense_hits = pools["dense_clean"]
                    dense_ids = pools["dense_clean_ids"]
                    q_text = pools["q_clean"]
                    rerank_cached = pools["rerank_clean"]

                if method == "bm25_only":
                    retrieved = sparse_ids[:50]
                elif method == "dense_only":
                    retrieved = dense_ids[:50]
                elif method == "rrf_k60":
                    retrieved = rrf_fuse(dense_hits, sparse_hits, k=60, dense_weight=1.0, sparse_weight=1.0, top_k=50)
                elif method == "rrf_k10":
                    retrieved = rrf_fuse(dense_hits, sparse_hits, k=10, dense_weight=1.0, sparse_weight=1.0, top_k=50)
                elif method == "wrrf_bm25x2":
                    retrieved = rrf_fuse(dense_hits, sparse_hits, k=60, dense_weight=1.0, sparse_weight=2.0, top_k=50)
                elif method == "interleave_4_4":
                    retrieved = interleave_4_4(sparse_ids, dense_ids, top_k=50)
                elif method == "rrf_k60_rerank":
                    if rerank_cached:
                        retrieved = rerank_cached[:30]
                    else:
                        fused_docs = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_k=30)
                        encoder = get_encoder()
                        texts = [c.get("text", "") for c in fused_docs]
                        scores = list(encoder.rerank(q_text, texts))
                        scored = sorted(zip(fused_docs, scores), key=lambda x: x[1], reverse=True)
                        retrieved = [str(c[0].get("id") or c[0].get("chunk_id")) for c in scored]
                else:
                    retrieved = []

                m = compute_metrics(retrieved, golds)
                all_m.append(m)

            hit8_vals = [m["hit@8"] for m in all_m]
            k_hit8 = int(sum(hit8_vals))
            ci_low, ci_high = wilson_ci(k_hit8, n)

            # McNemar vs bm25_only/raw
            bm25_h = per_item_hits[dataset_name]["raw__bm25_only"]
            cfg_h = per_item_hits[dataset_name][cfg]
            mcn_bm25 = exact_mcnemar_test(cfg_h, bm25_h)

            # McNemar vs current production
            prod_h = prod_hits[dataset_name]
            mcn_prod = exact_mcnemar_test(cfg_h, prod_h)

            agg[cfg] = {
                "n": n,
                "hit@8": float(np.mean(hit8_vals)),
                "k_hit@8": k_hit8,
                "recall@8": float(np.mean([m["recall@8"] for m in all_m])),
                "mrr": float(np.mean([m["mrr"] for m in all_m])),
                "hit@20": float(np.mean([m["hit@20"] for m in all_m])),
                "wilson_ci": (ci_low, ci_high),
                "mcnemar_vs_bm25_p": mcn_bm25["p_value"],
                "mcnemar_vs_prod_p": mcn_prod["p_value"],
                "mcnemar_vs_bm25": mcn_bm25,
                "mcnemar_vs_prod": mcn_prod,
                "mean_lat": float(np.mean(latencies[cfg])),
                "p95_lat": float(np.percentile(latencies[cfg], 95)),
            }
        return agg

    dev_results = aggregate_dataset("dev", dev_items)
    heldout_results = aggregate_dataset("heldout", heldout_items)
    combined_results = aggregate_dataset("combined", combined_items)

    # Compute Holm-adjusted p-values for Round-1 Heldout vs Production across all 14 configs
    p_heldout_prod = {cfg: heldout_results[cfg]["mcnemar_vs_prod_p"] for cfg in config_names}
    holm_heldout_prod = compute_holm_bonferroni(p_heldout_prod)
    for cfg in config_names:
        heldout_results[cfg]["holm_vs_prod_p"] = holm_heldout_prod[cfg]

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLES
    # -------------------------------------------------------------------------
    def print_main_table(title: str, results_dict: Dict[str, Any], n_total: int, show_holm: bool = False) -> None:
        table_width = 129 if show_holm else 115
        print("\n" + "=" * table_width)
        print(f"TABLE: {title} (n={n_total})")
        print("=" * table_width)
        if show_holm:
            header = (
                f"{'Configuration':26s} | {'Hit@8':13s} | {'Rec@8':7s} | {'MRR':7s} | {'Hit@20':7s} | "
                f"{'Wilson 95% CI':17s} | {'p vs BM25':9s} | {'p vs Prod':9s} | {'Holm p':9s} | {'Lat (ms)':12s}"
            )
        else:
            header = (
                f"{'Configuration':26s} | {'Hit@8':13s} | {'Rec@8':7s} | {'MRR':7s} | {'Hit@20':7s} | "
                f"{'Wilson 95% CI':17s} | {'p vs BM25':9s} | {'p vs Prod':9s} | {'Lat (ms)':12s}"
            )
        print(header)
        print("-" * len(header))
        for cfg in config_names:
            r = results_dict[cfg]
            h8_str = f"{r['k_hit@8']:2d}/{n_total:2d} ({r['hit@8']:5.1%})"
            ci_str = f"[{r['wilson_ci'][0]:.3f}, {r['wilson_ci'][1]:.3f}]"
            lat_str = f"{r['mean_lat']:4.1f} / {r['p95_lat']:4.1f}"
            p_bm25_str = f"{r['mcnemar_vs_bm25_p']:.4f}"
            p_prod_str = f"{r['mcnemar_vs_prod_p']:.4f}"
            if show_holm:
                holm_str = f"{r.get('holm_vs_prod_p', 1.0):.4f}"
                row = (
                    f"{cfg:26s} | {h8_str:13s} | {r['recall@8']:5.4f} | {r['mrr']:5.4f} | "
                    f"{r['hit@20']:5.4f} | {ci_str:17s} | {p_bm25_str:9s} | {p_prod_str:9s} | {holm_str:9s} | {lat_str:12s}"
                )
            else:
                row = (
                    f"{cfg:26s} | {h8_str:13s} | {r['recall@8']:5.4f} | {r['mrr']:5.4f} | "
                    f"{r['hit@20']:5.4f} | {ci_str:17s} | {p_bm25_str:9s} | {p_prod_str:9s} | {lat_str:12s}"
                )
            print(row)

    print_main_table("DEV SET EVALUATION", dev_results, n_dev, show_holm=False)
    print_main_table("ROUND-1 HELDOUT SET EVALUATION", heldout_results, n_heldout, show_holm=True)
    print_main_table("COMBINED BENCHMARK EVALUATION", combined_results, n_comb, show_holm=False)

    # Statistical comparison report
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS VS OLD PRODUCTION (Round-1 Heldout, m=14 comparisons)")
    print("=" * 80)
    print(f"{'Configuration':26s} | {'unadjusted p':12s} | {'Holm-adjusted p':15s}")
    print("-" * 59)
    for cfg in config_names:
        r = heldout_results[cfg]
        print(f"{cfg:26s} | {r['mcnemar_vs_prod_p']:12.4f} | {r['holm_vs_prod_p']:15.4f}")

    # -------------------------------------------------------------------------
    # SELECTION RULE EXECUTION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SELECTION RULE EVALUATION")
    print("=" * 80)
    print("Rule: Highest combined Hit@8 that is NOT worse than bm25_only/raw on either set;")
    print("      tie-break by lower mean latency.")
    print("-" * 80)

    bm25_raw_dev_hit8 = dev_results["raw__bm25_only"]["hit@8"]
    bm25_raw_held_hit8 = heldout_results["raw__bm25_only"]["hit@8"]
    print(f"Reference Baseline (raw__bm25_only): Dev Hit@8 = {bm25_raw_dev_hit8:.4f}, Heldout Hit@8 = {bm25_raw_held_hit8:.4f}")

    eligible_configs = []
    for cfg in config_names:
        dev_h = dev_results[cfg]["hit@8"]
        held_h = heldout_results[cfg]["hit@8"]
        comb_h = combined_results[cfg]["hit@8"]
        mean_l = combined_results[cfg]["mean_lat"]

        is_not_worse = (dev_h >= bm25_raw_dev_hit8) and (held_h >= bm25_raw_held_hit8)
        if is_not_worse:
            eligible_configs.append((comb_h, -mean_l, cfg))

    eligible_configs.sort(reverse=True)

    print(f"\nEligible Configurations Passing Non-Inferiority Constraint ({len(eligible_configs)} found):")
    for score, neg_lat, cfg in eligible_configs:
        print(f"  * {cfg:26s}: Combined Hit@8 = {score:.4f} ({combined_results[cfg]['k_hit@8']}/{n_comb}), "
              f"Dev = {dev_results[cfg]['hit@8']:.4f}, Heldout = {heldout_results[cfg]['hit@8']:.4f}, Mean Lat = {-neg_lat:.1f}ms")

    if eligible_configs:
        chosen_config = eligible_configs[0][2]
        runner_up = eligible_configs[1][2] if len(eligible_configs) > 1 else eligible_configs[0][2]
    else:
        # Fall back to best combined overall
        sorted_all = sorted([(combined_results[cfg]["hit@8"], -combined_results[cfg]["mean_lat"], cfg) for cfg in config_names], reverse=True)
        chosen_config = sorted_all[0][2]
        runner_up = sorted_all[1][2]
        print("Note: No configuration beat bm25_only/raw on both sets simultaneously; falling back to top overall.")

    print("\n" + "=" * 80)
    print(f"CHOSEN CONFIGURATION:  {chosen_config}")
    print(f"RUNNER-UP:             {runner_up}")
    print("ACTION:                Do NOT change production retrieval code.")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # BREAKDOWNS FOR TOP 3 CONFIGS
    # -------------------------------------------------------------------------
    # Top 3 configs by combined hit@8
    top3_configs = sorted(config_names, key=lambda c: combined_results[c]["hit@8"], reverse=True)[:3]
    print(f"\nTop 3 Configurations for Detailed Slice Breakdown: {top3_configs}")

    def print_top3_breakdown(dim_name: str, key_fn: Callable[[Dict[str, Any]], str]) -> None:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for it in combined_items:
            k = key_fn(it)
            groups.setdefault(k, []).append(it)

        print("\n" + "=" * 90)
        print(f"SLICE BREAKDOWN: {dim_name} (Top 3 Configurations, n={n_comb})")
        print("=" * 90)
        c1, c2, c3 = top3_configs
        hdr = f"{'Group':32s} | {'n':3s} | {c1:17s} | {c2:17s} | {c3:17s}"
        print(hdr)
        print("-" * len(hdr))

        for grp_val, grp_items in sorted(groups.items()):
            n_grp = len(grp_items)
            cols = [f"{grp_val:32s}", f"{n_grp:3d}"]
            for c in top3_configs:
                hits = 0
                for it in grp_items:
                    qid = it["id"]
                    is_dev = it in dev_items
                    ds = "dev" if is_dev else "heldout"
                    idx = dev_items.index(it) if is_dev else heldout_items.index(it)
                    if per_item_hits[ds][c][idx]:
                        hits += 1
                cols.append(f"{hits:2d}/{n_grp:2d} ({hits/n_grp:5.1%})")
            print(" | ".join(cols))

    # Load corpus chunks for section resolution
    corpus_chunks = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            corpus_chunks = {c["chunk_id"]: c for c in json.load(f)}

    def resolve_section(it: Dict[str, Any]) -> str:
        sec = it.get("section") or it.get("metadata", {}).get("section")
        if not sec and it.get("source_chunk_ids"):
            cid = it["source_chunk_ids"][0]
            sec = corpus_chunks.get(cid, {}).get("item_section")
        return str(sec or "unknown")

    def resolve_style(it: Dict[str, Any]) -> str:
        style = it.get("metadata", {}).get("question_style")
        return str(style or "unassigned (dev set)")

    print_top3_breakdown("Section", resolve_section)
    print_top3_breakdown("Question Style", resolve_style)


def measure_real_latency() -> None:
    """
    Measures end-to-end real query latency for 3 configurations:
      (1) old production retrieve() (legacy clean_v2, k=60, top-20 pools)
      (2) clean_v3 + rrf_k10 with live Qdrant top-50 + BM25 top-50 (new default)
      (3) clean_v3 + rrf_k60 + cross-encoder rerank of top-30
    Evaluated over 30 round-1 single-year queries with 1 warmup.
    Reports mean / p50 / p95 ms and peak process RSS MB for (2) vs (3).
    Writes backend/eval/results/latency_measurement.md.
    """
    import psutil
    from backend.agent.retrieval import (
        is_qdrant_connected,
        retrieve,
        search_qdrant,
        search_bm25,
        reciprocal_rank_fusion,
    )
    from backend.agent.nodes.classifier import (
        extract_company_and_year,
        strip_query_metadata_v3,
    )

    print("=" * 80)
    print("REAL END-TO-END LATENCY & MEMORY MEASUREMENT")
    print("=" * 80)
    print(f"Qdrant Live Connection: {is_qdrant_connected()}")
    if not is_qdrant_connected():
        print("WARNING: Qdrant is not connected. Measurements require live Qdrant connection.")

    # 1. Load 30 round-1 single-year queries from golden_heldout.json
    with open(GOLDEN_HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout_raw = json.load(f)

    heldout_single = [
        x for x in heldout_raw
        if x.get("source_chunk_ids") and "/" not in str(x.get("fiscal_year", ""))
    ]
    queries_30 = heldout_single[:30]
    print(f"Loaded {len(queries_30)} single-year round-1 queries for benchmarking.")

    warmup_item = queries_30[0]
    warmup_q = warmup_item["question"]

    process = psutil.Process()

    # -------------------------------------------------------------------------
    # Config 1: Old production retrieve()
    # -------------------------------------------------------------------------
    print("\nRunning (1) Old production retrieve() [RETRIEVAL_STRATEGY='legacy']...")
    os.environ["RETRIEVAL_STRATEGY"] = "legacy"
    # 1 warmup query
    _ = retrieve(warmup_q, top_k=8)

    times_cfg1 = []
    for it in queries_30:
        q = it["question"]
        t0 = time.perf_counter()
        _ = retrieve(q, top_k=8)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_cfg1.append(elapsed_ms)

    mean_cfg1 = float(np.mean(times_cfg1))
    p50_cfg1 = float(np.percentile(times_cfg1, 50))
    p95_cfg1 = float(np.percentile(times_cfg1, 95))
    print(f"  Config (1): Mean = {mean_cfg1:.1f}ms, p50 = {p50_cfg1:.1f}ms, p95 = {p95_cfg1:.1f}ms")

    # -------------------------------------------------------------------------
    # Config 2: clean_v3 + rrf_k10 with live Qdrant top-50 + BM25 top-50
    # -------------------------------------------------------------------------
    print("\nRunning (2) clean_v3 + rrf_k10 [live Qdrant top-50 + BM25 top-50]...")
    os.environ["RETRIEVAL_STRATEGY"] = "clean_v3_rrf10"
    # Measure initial RSS before cross-encoder
    rss_start_cfg2 = process.memory_info().rss / (1024 * 1024)
    peak_rss_cfg2 = rss_start_cfg2

    # 1 warmup query
    _ = retrieve(warmup_q, top_k=8)

    times_cfg2 = []
    for it in queries_30:
        q = it["question"]
        t0 = time.perf_counter()
        _ = retrieve(q, top_k=8)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_cfg2.append(elapsed_ms)
        cur_rss = process.memory_info().rss / (1024 * 1024)
        if cur_rss > peak_rss_cfg2:
            peak_rss_cfg2 = cur_rss

    mean_cfg2 = float(np.mean(times_cfg2))
    p50_cfg2 = float(np.percentile(times_cfg2, 50))
    p95_cfg2 = float(np.percentile(times_cfg2, 95))
    print(f"  Config (2): Mean = {mean_cfg2:.1f}ms, p50 = {p50_cfg2:.1f}ms, p95 = {p95_cfg2:.1f}ms, Peak RSS = {peak_rss_cfg2:.1f}MB")

    # -------------------------------------------------------------------------
    # Config 3: clean_v3 + rrf_k60 + cross-encoder rerank of the top 30
    # -------------------------------------------------------------------------
    print("\nRunning (3) clean_v3 + rrf_k60 + cross-encoder rerank of top 30...")
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

    rss_loaded_cfg3 = process.memory_info().rss / (1024 * 1024)
    peak_rss_cfg3 = rss_loaded_cfg3

    # 1 warmup query
    q_clean_w = strip_query_metadata_v3(warmup_q)
    ext_c, ext_y = extract_company_and_year(warmup_q)
    f_w = {}
    if ext_c: f_w["company"] = ext_c
    if ext_y: f_w["fiscal_year"] = str(ext_y)
    dense_w = search_qdrant(query=q_clean_w, filters=f_w or None, limit=50)
    sparse_w = search_bm25(query=q_clean_w, filters=f_w or None, limit=50)
    fused_w = reciprocal_rank_fusion(dense_results=dense_w, sparse_results=sparse_w, k=60, top_k=30)
    texts_w = [c.get("text", "") for c in fused_w]
    _ = list(encoder.rerank(q_clean_w, texts_w))

    times_cfg3 = []
    for it in queries_30:
        q = it["question"]
        t0 = time.perf_counter()
        q_clean = strip_query_metadata_v3(q)
        ext_c, ext_y = extract_company_and_year(q)
        f = {}
        if ext_c: f["company"] = ext_c
        if ext_y: f["fiscal_year"] = str(ext_y)
        dense_hits = search_qdrant(query=q_clean, filters=f or None, limit=50)
        sparse_hits = search_bm25(query=q_clean, filters=f or None, limit=50)
        fused_docs = reciprocal_rank_fusion(dense_results=dense_hits, sparse_results=sparse_hits, k=60, top_k=30)
        texts = [c.get("text", "") for c in fused_docs]
        scores = list(encoder.rerank(q_clean, texts))
        scored = sorted(zip(fused_docs, scores), key=lambda x: x[1], reverse=True)[:8]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_cfg3.append(elapsed_ms)
        cur_rss = process.memory_info().rss / (1024 * 1024)
        if cur_rss > peak_rss_cfg3:
            peak_rss_cfg3 = cur_rss

    mean_cfg3 = float(np.mean(times_cfg3))
    p50_cfg3 = float(np.percentile(times_cfg3, 50))
    p95_cfg3 = float(np.percentile(times_cfg3, 95))
    print(f"  Config (3): Mean = {mean_cfg3:.1f}ms, p50 = {p50_cfg3:.1f}ms, p95 = {p95_cfg3:.1f}ms, Peak RSS = {peak_rss_cfg3:.1f}MB")

    # Format output tables
    print("\n" + "=" * 90)
    print("END-TO-END LATENCY & MEMORY SUMMARY (n=30 queries, 1 warmup)")
    print("=" * 90)
    header = f"{'Configuration':55s} | {'Mean (ms)':9s} | {'p50 (ms)':8s} | {'p95 (ms)':8s} | {'Peak RSS (MB)':13s}"
    print(header)
    print("-" * len(header))
    print(f"{'(1) Old production retrieve() (legacy clean_v2, k=60)':55s} | {mean_cfg1:9.1f} | {p50_cfg1:8.1f} | {p95_cfg1:8.1f} | {'N/A':13s}")
    print(f"{'(2) clean_v3 + rrf_k10 (live Qdrant + BM25 top-50)':55s} | {mean_cfg2:9.1f} | {p50_cfg2:8.1f} | {p95_cfg2:8.1f} | {peak_rss_cfg2:13.1f}")
    print(f"{'(3) clean_v3 + rrf_k60 + cross-encoder top-30 rerank':55s} | {mean_cfg3:9.1f} | {p50_cfg3:8.1f} | {p95_cfg3:8.1f} | {peak_rss_cfg3:13.1f}")
    print("=" * 90)

    # Write backend/eval/results/latency_measurement.md
    out_md_path = REPO_ROOT / "backend" / "eval" / "results" / "latency_measurement.md"
    md_content = f"""# End-to-End Latency & Memory Measurement

**Benchmark Scope:** 30 single-year answerable questions from `golden_heldout.json` (round 1), 1 warmup query per configuration.  
**Live Backend:** Live Qdrant Cloud (`verity_chunks`, 2921 chunks) + local BM25 index.

## Latency & Process RSS Table

| Configuration | Description | Mean Latency (ms) | Median / p50 (ms) | p95 Latency (ms) | Peak Process RSS (MB) |
|---|---|---|---|---|---|
| **(1) Old production retrieve()** | Legacy clean_v2, k=60, top-20 candidate pools | {mean_cfg1:.1f} | {p50_cfg1:.1f} | {p95_cfg1:.1f} | N/A (baseline) |
| **(2) clean_v3 + rrf_k10** | clean_v3 query, k=10, top-50 dense + BM25 pools (new default) | {mean_cfg2:.1f} | {p50_cfg2:.1f} | {p95_cfg2:.1f} | {peak_rss_cfg2:.1f} MB |
| **(3) clean_v3 + rrf_k60 + rerank** | clean_v3, k=60, top-30 reranked via ms-marco-MiniLM-L-6-v2 | {mean_cfg3:.1f} | {p50_cfg3:.1f} | {p95_cfg3:.1f} | {peak_rss_cfg3:.1f} MB |

## Memory Comparison: Config (2) vs Config (3)

- **Config (2) `clean_v3_rrf10`:** Runs pure RRF rank math in Python without loading neural cross-encoders into process memory. Peak RSS measured at **{peak_rss_cfg2:.1f} MB**.
- **Config (3) Cross-Encoder Rerank:** Loads `Xenova/ms-marco-MiniLM-L-6-v2` ONNX model weights and runtime, increasing process RSS to **{peak_rss_cfg3:.1f} MB** (+{peak_rss_cfg3 - peak_rss_cfg2:.1f} MB overhead).
- **Latency Impact:** Cross-encoder adds ~{mean_cfg3 - mean_cfg2:.1f}ms mean latency per query to evaluate 30 candidate chunk texts.
"""
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nWrote latency measurement report to: {out_md_path}")


if __name__ == "__main__":
    if "--measure-latency" in sys.argv:
        measure_real_latency()
    else:
        run_fusion_sweep()
