#!/usr/bin/env python3
"""
Held-Out Benchmark Retrieval Diagnostic Script (Read-Only)
==========================================================
Analyzes the 64 answerable questions in backend/eval/golden_heldout.json.
Evaluates:
  1. Filter extraction accuracy (company, fiscal year, both) and lists mismatches.
  2. Hit@8 for bm25_filtered, dense_filtered, rrf_raw_filtered, production retrieve()
     broken down by category, section, question_style, company, fiscal_year,
     whether question names a year, and whether extractor produced a year filter.
  3. Pool quality (Hit@20, Hit@50, and rank distributions 1-8, 9-20, 21-50, >50).
  4. Paired comparisons (production vs bm25, rrf_raw vs bm25) with exact McNemar
     p-values and Wilson 95% confidence intervals.
  5. Near-duplicate noise analysis (token Jaccard >= 0.7 within same company/year).
  6. 12 stratified miss samples with gold chunk excerpts, per-config ranks, and top-3 retrieved.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import scipy.stats

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_HELDOUT_PATH = REPO_ROOT / "backend" / "eval" / "golden_heldout.json"
ABLATION_RESULTS_PATH = REPO_ROOT / "backend" / "eval" / "results" / "retrieval_ablation_heldout.json"
POOL_CACHE_PATH = REPO_ROOT / "backend" / "eval" / "results" / "diagnose_pool_cache.json"
CHUNKS_PATH = REPO_ROOT / "backend" / "data" / "processed" / "chunks.json"


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = scipy.stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, center - margin), min(1.0, center + margin)


def get_token_set(text: str) -> Set[str]:
    """Extracts lowercased word tokens from text."""
    return set(re.findall(r"\w+", text.lower()))


def compute_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def run_diagnostics() -> None:
    # -------------------------------------------------------------------------
    # 0. Load Datasets
    # -------------------------------------------------------------------------
    with open(GOLDEN_HELDOUT_PATH, "r", encoding="utf-8") as f:
        golden_items = json.load(f)

    # Filter to answerable items only (64 items)
    answerable_items = [it for it in golden_items if it.get("source_chunk_ids")]
    golden_map = {it["id"]: it for it in answerable_items}
    n_items = len(answerable_items)

    with open(ABLATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)
    ablation_item_map = {it["id"]: it for it in ablation_data.get("per_item_results", [])}

    # Load corpus chunks
    corpus_chunks: Dict[str, Dict[str, Any]] = {}
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
            corpus_chunks = {c["chunk_id"]: c for c in raw_chunks}

    # Load pool cache if present
    pool_cache: Dict[str, Dict[str, List[str]]] = {}
    if POOL_CACHE_PATH.exists():
        with open(POOL_CACHE_PATH, "r", encoding="utf-8") as f:
            pool_cache = json.load(f)

    # Import rule extractors
    from backend.agent.nodes.classifier import extract_all_fiscal_years, extract_company_and_year
    from backend.eval.retrieval_ablation import derive_query_filters

    configs = ["bm25_filtered", "dense_filtered", "rrf_raw_filtered", "production_retrieve"]

    print("=" * 80)
    print(f"HELD-OUT BENCHMARK RETRIEVAL DIAGNOSTICS (n={n_items} answerable items)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Filter Extraction Accuracy
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. FILTER EXTRACTION ACCURACY")
    print("=" * 80)

    comp_correct = 0
    year_correct = 0
    both_correct = 0
    mismatches = []

    for it in answerable_items:
        qid = it["id"]
        q = it["question"]
        gold_comp = it.get("company")
        gold_fy = str(it.get("fiscal_year"))

        ext_comp, ext_fy = extract_company_and_year(q)
        all_ext_fys = extract_all_fiscal_years(q)

        comp_ok = (ext_comp == gold_comp)
        if "/" in gold_fy:
            gold_years = set(gold_fy.split("/"))
            year_ok = (set(all_ext_fys) == gold_years)
        else:
            year_ok = (ext_fy == gold_fy)

        if comp_ok:
            comp_correct += 1
        if year_ok:
            year_correct += 1
        if comp_ok and year_ok:
            both_correct += 1
        else:
            mismatches.append({
                "id": qid,
                "question": q,
                "gold_comp": gold_comp,
                "gold_fy": gold_fy,
                "ext_comp": ext_comp,
                "ext_fy": ext_fy,
                "all_ext_fys": all_ext_fys,
                "comp_ok": comp_ok,
                "year_ok": year_ok,
            })

    print(f"Total Evaluated Questions:  {n_items}")
    print(f"Company Extraction Accuracy: {comp_correct}/{n_items} ({comp_correct / n_items:.1%})")
    print(f"Year Extraction Accuracy:    {year_correct}/{n_items} ({year_correct / n_items:.1%})")
    print(f"Both Accurate:              {both_correct}/{n_items} ({both_correct / n_items:.1%})")
    print(f"\nTotal Extraction Mismatches: {len(mismatches)}")
    print("-" * 80)

    for m in mismatches:
        print(f"Item ID: {m['id']}")
        print(f"  Question: {m['question']}")
        print(f"  Gold:     Company={m['gold_comp']}, Fiscal Year={m['gold_fy']}")
        print(f"  Extract:  Company={m['ext_comp']}, Single Year={m['ext_fy']}, All Years={m['all_ext_fys']}")
        print(f"  Status:   Company Match={m['comp_ok']}, Year Match={m['year_ok']}")
        print("-" * 80)

    # -------------------------------------------------------------------------
    # 2. Hit@8 Breakdown by Slices
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. HIT@8 BREAKDOWN BY SLICE & ATTRIBUTE")
    print("=" * 80)

    # Helper to retrieve Hit@8 boolean for each config
    item_hits: Dict[str, Dict[str, bool]] = {}
    for it in answerable_items:
        qid = it["id"]
        abl_it = ablation_item_map.get(qid, {})
        cfg_res = abl_it.get("configs", {})
        item_hits[qid] = {}
        for cfg in configs:
            h8 = cfg_res.get(cfg, {}).get("metrics", {}).get("hit@8", 0.0)
            item_hits[qid][cfg] = (h8 == 1.0)

    def print_breakdown(dimension_name: str, group_fn: Any) -> None:
        groups: Dict[str, List[str]] = {}
        for it in answerable_items:
            key = str(group_fn(it))
            groups.setdefault(key, []).append(it["id"])

        print(f"\n--- Dimension: {dimension_name} ---")
        header = f"{'Group':32s} | {'n':3s} | {'bm25_filtered':15s} | {'dense_filtered':15s} | {'rrf_raw_filtered':16s} | {'production_retrieve':19s}"
        print(header)
        print("-" * len(header))

        for grp_val, qids in sorted(groups.items()):
            n_grp = len(qids)
            row = [f"{grp_val:32s}", f"{n_grp:3d}"]
            for cfg in configs:
                n_hit = sum(1 for qid in qids if item_hits[qid][cfg])
                pct = (n_hit / n_grp) * 100.0 if n_grp > 0 else 0.0
                row.append(f"{n_hit:2d}/{n_grp:2d} ({pct:5.1f}%)")
            print(" | ".join(row))

    print_breakdown("Category", lambda it: it.get("category", "unknown"))
    print_breakdown("Section", lambda it: it.get("section") or it.get("metadata", {}).get("section", "unknown"))
    print_breakdown("Question Style", lambda it: it.get("metadata", {}).get("question_style", "unknown"))
    print_breakdown("Company", lambda it: it.get("company", "unknown"))
    print_breakdown("Fiscal Year", lambda it: it.get("fiscal_year", "unknown"))
    print_breakdown("Question Names a Year", lambda it: len(extract_all_fiscal_years(it["question"])) > 0)
    print_breakdown("Extractor Produced Year Filter", lambda it: derive_query_filters(it["question"]).get("fiscal_year") is not None)

    # -------------------------------------------------------------------------
    # 3. Pool Quality & Rank Distributions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. POOL QUALITY & RANK DISTRIBUTION")
    print("=" * 80)

    # Compute ranks up to 50
    config_ranks: Dict[str, List[Optional[int]]] = {cfg: [] for cfg in configs}
    for it in answerable_items:
        qid = it["id"]
        golds = set(it["source_chunk_ids"])
        for cfg in configs:
            cand_ids = pool_cache.get(qid, {}).get(cfg, [])
            if not cand_ids:
                # Fall back to ablation top 30
                cand_ids = ablation_item_map.get(qid, {}).get("configs", {}).get(cfg, {}).get("retrieved_chunk_ids", [])
            found_rank: Optional[int] = None
            for idx, cid in enumerate(cand_ids, start=1):
                if cid in golds:
                    found_rank = idx
                    break
            config_ranks[cfg].append(found_rank)

    print(f"{'Configuration':22s} | {'Hit@8':15s} | {'Hit@20':15s} | {'Hit@50':15s}")
    print("-" * 75)
    for cfg in configs:
        ranks = config_ranks[cfg]
        h8 = sum(1 for r in ranks if r is not None and 1 <= r <= 8)
        h20 = sum(1 for r in ranks if r is not None and 1 <= r <= 20)
        h50 = sum(1 for r in ranks if r is not None and 1 <= r <= 50)
        print(f"{cfg:22s} | {h8:2d}/{n_items} ({h8/n_items:5.1%}) | {h20:2d}/{n_items} ({h20/n_items:5.1%}) | {h50:2d}/{n_items} ({h50/n_items:5.1%})")

    print("\nRank Distributions for BM25-Only and Dense-Only:")
    print(f"{'Configuration':20s} | {'Rank 1-8':13s} | {'Rank 9-20':13s} | {'Rank 21-50':13s} | {'Rank >50':13s}")
    print("-" * 80)
    for cfg in ["bm25_filtered", "dense_filtered"]:
        ranks = config_ranks[cfg]
        r1_8 = sum(1 for r in ranks if r is not None and 1 <= r <= 8)
        r9_20 = sum(1 for r in ranks if r is not None and 9 <= r <= 20)
        r21_50 = sum(1 for r in ranks if r is not None and 21 <= r <= 50)
        r_gt50 = sum(1 for r in ranks if r is None or r > 50)
        print(f"{cfg:20s} | {r1_8:2d} ({r1_8/n_items:5.1%})  | {r9_20:2d} ({r9_20/n_items:5.1%})  | {r21_50:2d} ({r21_50/n_items:5.1%})  | {r_gt50:2d} ({r_gt50/n_items:5.1%})")

    # -------------------------------------------------------------------------
    # 4. Paired Comparisons & Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. PAIRED COMPARISONS & WILSON 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    print("\nWilson 95% Confidence Intervals for Hit@8:")
    print(f"{'Configuration':22s} | {'Hit@8':8s} | {'Accuracy':8s} | {'95% Confidence Interval':25s}")
    print("-" * 72)
    for cfg in configs:
        k_hit = sum(1 for it in answerable_items if item_hits[it["id"]][cfg])
        low, high = wilson_ci(k_hit, n_items)
        print(f"{cfg:22s} | {k_hit:2d}/{n_items:2d}  | {k_hit/n_items:6.1%}   | [{low:.4f}, {high:.4f}] ({low:5.1%} - {high:5.1%})")

    def run_paired_mcnemar(name_a: str, name_b: str) -> None:
        both = 0
        only_a = 0
        only_b = 0
        neither = 0
        for it in answerable_items:
            qid = it["id"]
            hit_a = item_hits[qid][name_a]
            hit_b = item_hits[qid][name_b]
            if hit_a and hit_b:
                both += 1
            elif hit_a and not hit_b:
                only_a += 1
            elif not hit_a and hit_b:
                only_b += 1
            else:
                neither += 1

        disc = only_a + only_b
        p_val = scipy.stats.binomtest(only_a, disc, p=0.5).pvalue if disc > 0 else 1.0

        print(f"\nPaired Comparison: {name_a} (A) vs {name_b} (B)")
        print(f"  Both Hit (A=1, B=1):       {both:2d} ({both/n_items:5.1%})")
        print(f"  Only A Hit (A=1, B=0):     {only_a:2d} ({only_a/n_items:5.1%})")
        print(f"  Only B Hit (A=0, B=1):     {only_b:2d} ({only_b/n_items:5.1%})")
        print(f"  Neither Hit (A=0, B=0):    {neither:2d} ({neither/n_items:5.1%})")
        print(f"  Discordant Pairs (b + c):  {disc}")
        print(f"  Exact McNemar Test p-value: {p_val:.6f} {'(Significant at p<0.10)' if p_val < 0.10 else '(Not statistically significant)'}")

    run_paired_mcnemar("production_retrieve", "bm25_filtered")
    run_paired_mcnemar("rrf_raw_filtered", "bm25_filtered")

    # -------------------------------------------------------------------------
    # 5. Near-Duplicate Noise
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. NEAR-DUPLICATE NOISE ANALYSIS")
    print("=" * 80)

    # Group corpus chunks by (company, fiscal_year)
    corpus_by_comp_year: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in corpus_chunks.values():
        key = (str(c.get("company")), str(c.get("fiscal_year")))
        corpus_by_comp_year.setdefault(key, []).append(c)

    gold_duplicate_info: Dict[str, List[Dict[str, Any]]] = {}
    for it in answerable_items:
        qid = it["id"]
        gold_cids = it["source_chunk_ids"]
        dups_found = []
        for gid in gold_cids:
            gchunk = corpus_chunks.get(gid)
            if not gchunk:
                continue
            comp = str(gchunk.get("company"))
            fy = str(gchunk.get("fiscal_year"))
            gtoks = get_token_set(gchunk.get("text", ""))
            candidates = corpus_by_comp_year.get((comp, fy), [])
            for cand in candidates:
                cid = cand.get("chunk_id")
                if cid == gid:
                    continue
                ctoks = get_token_set(cand.get("text", ""))
                jacc = compute_token_jaccard(gtoks, ctoks)
                if jacc >= 0.7:
                    dups_found.append({
                        "gold_chunk_id": gid,
                        "duplicate_chunk_id": cid,
                        "jaccard": jacc,
                        "dup_section": cand.get("item_section"),
                    })
        if dups_found:
            gold_duplicate_info[qid] = dups_found

    n_with_dup = len(gold_duplicate_info)
    print(f"Total Gold Items: {n_items}")
    print(f"Gold Chunks with Near-Duplicate (Token Jaccard >= 0.7 within same company & year): {n_with_dup}/{n_items} ({n_with_dup/n_items:.1%})")

    # Check how many misses are near-duplicate cases
    prod_misses = [it["id"] for it in answerable_items if not item_hits[it["id"]]["production_retrieve"]]
    bm25_misses = [it["id"] for it in answerable_items if not item_hits[it["id"]]["bm25_filtered"]]
    prod_miss_dups = [qid for qid in prod_misses if qid in gold_duplicate_info]
    bm25_miss_dups = [qid for qid in bm25_misses if qid in gold_duplicate_info]

    print(f"\nProduction Retrieve Misses ({len(prod_misses)} total):")
    print(f"  Misses with Near-Duplicate Gold: {len(prod_miss_dups)} / {len(prod_misses)} ({len(prod_miss_dups)/len(prod_misses):.1%})")
    print(f"BM25 Filtered Misses ({len(bm25_misses)} total):")
    print(f"  Misses with Near-Duplicate Gold: {len(bm25_miss_dups)} / {len(bm25_misses)} ({len(bm25_miss_dups)/len(bm25_misses):.1%})")

    for qid, dups in gold_duplicate_info.items():
        hit_p = "HIT" if item_hits[qid]["production_retrieve"] else "MISS"
        hit_b = "HIT" if item_hits[qid]["bm25_filtered"] else "MISS"
        for d in dups:
            print(f"  * {qid} [Prod: {hit_p}, BM25: {hit_b}]: Gold {d['gold_chunk_id']} has duplicate {d['duplicate_chunk_id']} "
                  f"(Jaccard={d['jaccard']:.3f}, Section={d['dup_section']})")

    # -------------------------------------------------------------------------
    # 6. Miss Samples (Stratified across categories and companies)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. STRATIFIED MISS SAMPLES (12 Items)")
    print("=" * 80)

    stratified_sample_ids = [
        "heldout_04",    # Alphabet  | numeric_precision
        "heldout_14",    # Alphabet  | simple_lookup
        "heldout_23",    # Amazon    | numeric_precision
        "heldout_25",    # Amazon    | simple_lookup
        "heldout_32",    # Amazon    | simple_lookup
        "heldout_37",    # Apple     | numeric_precision
        "heldout_39",    # Apple     | simple_lookup
        "heldout_45",    # Apple     | other
        "heldout_58",    # Microsoft | numeric_precision
        "heldout_mh_10", # Microsoft | multi_hop
        "heldout_75",    # NVIDIA    | numeric_precision
        "heldout_79",    # NVIDIA    | simple_lookup
    ]

    for idx, sid in enumerate(stratified_sample_ids, start=1):
        g_it = golden_map[sid]
        abl_it = ablation_item_map[sid]
        golds = set(g_it["source_chunk_ids"])
        gold_id = g_it["source_chunk_ids"][0]
        gold_chunk = corpus_chunks.get(gold_id, {})

        # Query actually used by production
        query_used = abl_it.get("configs", {}).get("production_retrieve", {}).get("query_used", g_it["question"])

        # First 200 chars of gold chunk
        gold_excerpt = gold_chunk.get("text", "")[:200].replace("\n", " ").strip()

        # Ranks across configs
        ranks_desc = []
        for cfg in configs:
            cids = pool_cache.get(sid, {}).get(cfg, [])
            if not cids:
                cids = abl_it.get("configs", {}).get(cfg, {}).get("retrieved_chunk_ids", [])
            r: Optional[int] = None
            for r_idx, c in enumerate(cids, start=1):
                if c in golds:
                    r = r_idx
                    break
            ranks_desc.append(f"{cfg}: {r if r is not None else '>50'}")

        # Top 3 retrieved chunks for production_retrieve
        top3_ids = abl_it.get("configs", {}).get("production_retrieve", {}).get("retrieved_chunk_ids", [])[:3]

        print(f"\n[Sample {idx}/12] Item ID: {sid}")
        print(f"Company:  {g_it.get('company')} | Fiscal Year: {g_it.get('fiscal_year')} | Category: {g_it.get('category')} | Section: {g_it.get('section')}")
        print(f"Question: {g_it.get('question')}")
        print(f"Query Actually Used: \"{query_used}\"")
        print(f"Gold Chunk Excerpt (ID: {gold_id}, Section: {gold_chunk.get('item_section', 'N/A')}):")
        print(f"  \"{gold_excerpt}...\"")
        print(f"Ranks Across Configurations: {', '.join(ranks_desc)}")
        print("Top-3 Retrieved Chunks (production retrieve()):")
        for rank_pos, cid in enumerate(top3_ids, start=1):
            cand_c = corpus_chunks.get(cid, {})
            cand_text = cand_c.get("text", "")[:150].replace("\n", " ").strip()
            print(f"  #{rank_pos} [Company: {cand_c.get('company')}, FY: {cand_c.get('fiscal_year')}, Section: {cand_c.get('item_section')}]: \"{cand_text}...\"")
        print("-" * 80)


if __name__ == "__main__":
    run_diagnostics()
