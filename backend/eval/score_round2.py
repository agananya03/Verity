#!/usr/bin/env python3
"""
Pre-Registered Round 2 Scoring Script
=====================================
Scores the round 2 benchmark (backend/eval/golden_round2.json) strictly following
backend/eval/round2_preregistration.md.

Pre-Registered Configurations (scored exactly once):
  - Config A: RETRIEVAL_STRATEGY=legacy (old production via real retrieve())
  - Config B: RETRIEVAL_STRATEGY=clean_v3_rrf10 (new production via real retrieve())
  - Config C: raw query, BM25-only, extractor's company/year filters, top-50 pool.
              For multi-year items, filter by company and any extracted years, top-50 pool.

Outputs:
  - backend/eval/results/round2_scoring.json
  - backend/eval/results/round2_scoring.md
  - backend/eval/results/round2_per_item.csv

Safety Guards:
  - Zero Gemini API calls.
  - Production retrieval code, golden files, and frozen result files are NOT modified.
  - Write-once guard: Refuses to run if round2_scoring.json exists unless --force is passed.
  - Stamped with CRLF-normalized SHA256 of golden_round2.json, git commit hash, and timestamp.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.stats

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_ROUND2_PATH = REPO_ROOT / "backend" / "eval" / "golden_round2.json"
RESULTS_DIR = REPO_ROOT / "backend" / "eval" / "results"
OUTPUT_JSON_PATH = RESULTS_DIR / "round2_scoring.json"
OUTPUT_MD_PATH = RESULTS_DIR / "round2_scoring.md"
OUTPUT_CSV_PATH = RESULTS_DIR / "round2_per_item.csv"


# =============================================================================
# 1. Stamping & Write-Once Guard
# =============================================================================
def compute_crlf_normalized_sha256(file_path: Path) -> str:
    """Computes SHA-256 of file with normalized newlines (CRLF -> LF)."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    normalized = content.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash, or 'unknown' if not in git."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def check_write_once_guard(results_path: Path, force: bool = False) -> None:
    """Enforces write-once pre-registration rule unless force is specified."""
    if results_path.exists():
        if not force:
            msg = (
                f"REFUSING TO RUN: {results_path} already exists!\n"
                "Pre-registration rule: Round 2 must only be scored ONCE.\n"
                "Execution halted to protect evaluation integrity. Use --force to override."
            )
            print("=" * 80, file=sys.stderr)
            print(msg, file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            raise FileExistsError(msg)
        else:
            print("!" * 80, file=sys.stderr)
            print("WARNING: --force specified. Overwriting existing round2_scoring.json.", file=sys.stderr)
            print("THIS BREAKS THE PRE-REGISTRATION PROTOCOL.", file=sys.stderr)
            print("!" * 80, file=sys.stderr)


# =============================================================================
# 2. Statistical Helpers & Metrics
# =============================================================================
def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    z = float(scipy.stats.norm.ppf((1.0 + confidence) / 2.0))
    p = float(k) / float(n)
    denominator = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denominator
    margin = (z / denominator) * math.sqrt(p * (1.0 - p) / float(n) + (z**2) / (4.0 * (float(n) ** 2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_test(hits_test: List[bool], hits_base: List[bool]) -> Dict[str, Any]:
    """
    Computes exact two-sided McNemar test using the binomial distribution.
    Under H0, discordant pairs (b, c) follow Binomial(b + c, 0.5).
      b = hits_test True, hits_base False
      c = hits_test False, hits_base True
    """
    both = sum(1 for t, b in zip(hits_test, hits_base) if t and b)
    only_test = sum(1 for t, b in zip(hits_test, hits_base) if t and not b)
    only_base = sum(1 for t, b in zip(hits_test, hits_base) if not t and b)
    neither = sum(1 for t, b in zip(hits_test, hits_base) if not t and not b)
    discordant = only_test + only_base
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(scipy.stats.binomtest(only_test, discordant, p=0.5).pvalue)
    return {
        "both": both,
        "only_test": only_test,
        "only_base": only_base,
        "neither": neither,
        "discordant": discordant,
        "p_value": p_value,
    }


def find_first_gold_rank(retrieved_ids: List[str], gold_ids: Set[str]) -> Optional[int]:
    """Returns 1-based rank of the first gold chunk in retrieved_ids, or None."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return rank
    return None


def compute_item_metrics(retrieved_ids: List[str], gold_chunk_ids: List[str]) -> Dict[str, Any]:
    """
    Computes deterministic retrieval metrics for a single item:
      - hit@8: 1.0 if any gold chunk in top 8, else 0.0
      - recall@8: fraction of gold chunks captured in top 8
      - mrr: reciprocal rank of the first gold chunk in top 8 (MRR@8), else 0.0
      - hit@20: 1.0 if any gold chunk in top 20, else 0.0
      - first_gold_rank: 1-based index in the full retrieved candidate list, or None
    """
    golds = set(str(cid).strip() for cid in gold_chunk_ids if str(cid).strip())
    top8 = [str(cid).strip() for cid in retrieved_ids[:8] if str(cid).strip()]
    top20 = [str(cid).strip() for cid in retrieved_ids[:20] if str(cid).strip()]

    hit8 = 1.0 if any(cid in golds for cid in top8) else 0.0
    recall8 = len(set(top8) & golds) / max(1, len(golds)) if golds else 0.0
    hit20 = 1.0 if any(cid in golds for cid in top20) else 0.0

    matching_ranks_8 = [rank for rank, cid in enumerate(top8, start=1) if cid in golds]
    mrr8 = 1.0 / matching_ranks_8[0] if matching_ranks_8 else 0.0

    first_rank_all = find_first_gold_rank(retrieved_ids, golds)

    return {
        "hit@8": hit8,
        "recall@8": recall8,
        "mrr": mrr8,
        "hit@20": hit20,
        "first_gold_rank": first_rank_all,
    }


# =============================================================================
# 3. Retrieval Configurations Execution
# =============================================================================
def extract_query_filters(query: str) -> Dict[str, Any]:
    """Derives metadata filters from query using deterministic classifier rules."""
    from backend.agent.nodes.classifier import extract_all_fiscal_years, extract_company_and_year

    company, fiscal_year = extract_company_and_year(query)
    all_years = extract_all_fiscal_years(query)

    filters: Dict[str, Any] = {}
    if company:
        filters["company"] = company
    if len(all_years) >= 2:
        filters["fiscal_years"] = all_years
    elif fiscal_year:
        filters["fiscal_year"] = str(fiscal_year)
    return filters


def run_config_a(query: str, filters: Dict[str, Any]) -> List[str]:
    """
    Config A: RETRIEVAL_STRATEGY=legacy (old production retrieve()).
    Queries with clean_v2, k=60, top-20 pools.
    """
    from backend.agent.retrieval import retrieve

    os.environ["RETRIEVAL_STRATEGY"] = "legacy"
    chunks = retrieve(query=query, top_k=20, filters=filters)
    return [str(c.get("id") or c.get("chunk_id")) for c in chunks if (c.get("id") or c.get("chunk_id"))]


def run_config_b(query: str, filters: Dict[str, Any]) -> List[str]:
    """
    Config B: RETRIEVAL_STRATEGY=clean_v3_rrf10 (new production retrieve()).
    Queries with clean_v3, k=10, top-50 dense + top-50 BM25 candidate pools.
    """
    from backend.agent.retrieval import retrieve

    os.environ["RETRIEVAL_STRATEGY"] = "clean_v3_rrf10"
    chunks = retrieve(query=query, top_k=50, filters=filters)
    return [str(c.get("id") or c.get("chunk_id")) for c in chunks if (c.get("id") or c.get("chunk_id"))]


def run_config_c(query: str, filters: Dict[str, Any], limit: int = 50) -> List[str]:
    """
    Config C: raw query, BM25-only, extractor's company/year filters, top-50 pool.
    For multi-year items, filter by company and any of the extracted years, top-50 pool.
    """
    from backend.agent.retrieval import bm25_chunks, bm25_index, tokenize_bm25

    if not bm25_index or not bm25_chunks:
        return []

    tokenized = tokenize_bm25(query)
    if not tokenized:
        return []

    scores = bm25_index.get_scores(tokenized)

    company = filters.get("company")
    fiscal_years = filters.get("fiscal_years")
    fiscal_year = filters.get("fiscal_year")
    years_set = set(str(y) for y in fiscal_years) if fiscal_years else ({str(fiscal_year)} if fiscal_year else set())

    scored = []
    for idx, (chunk, score) in enumerate(zip(bm25_chunks, scores)):
        if company and chunk.get("company") != company:
            continue
        if years_set and str(chunk.get("fiscal_year")) not in years_set:
            continue
        cid = str(chunk.get("chunk_id") or chunk.get("id") or idx)
        scored.append((cid, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:limit]]


# =============================================================================
# 4. Main Scoring Routine
# =============================================================================
def score_round2(force: bool = False) -> Dict[str, Any]:
    """Executes the pre-registered Round-2 evaluation and generates outputs."""
    check_write_once_guard(OUTPUT_JSON_PATH, force=force)

    # 1. Output Stamping
    sha256_hash = compute_crlf_normalized_sha256(GOLDEN_ROUND2_PATH)
    commit_hash = get_git_commit_hash()
    timestamp_iso = datetime.now().isoformat()
    timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("VERITY RETRIEVAL BENCHMARK: ROUND 2 PRE-REGISTERED EVALUATION")
    print("=" * 80)
    print(f"Timestamp:          {timestamp_readable}")
    print(f"Git Commit:         {commit_hash}")
    print(f"golden_round2.json: SHA-256 (CRLF-normalized) = {sha256_hash}")
    print("=" * 80)

    # 2. Load and Filter Items
    with open(GOLDEN_ROUND2_PATH, "r", encoding="utf-8") as f:
        all_raw_items = json.load(f)

    # Exclude the 14 unanswerable items from retrieval scoring
    answerable_items = [
        it
        for it in all_raw_items
        if it.get("source_chunk_ids")
        and not it.get("expected_decline")
        and it.get("category") != "unanswerable"
    ]
    unanswerable_items = [it for it in all_raw_items if it not in answerable_items]

    single_year_items = [it for it in answerable_items if "/" not in str(it.get("fiscal_year", ""))]
    multi_year_items = [it for it in answerable_items if "/" in str(it.get("fiscal_year", ""))]

    n_ans = len(answerable_items)
    n_sy = len(single_year_items)
    n_my = len(multi_year_items)

    print(f"Total Corpus Items Loaded:      {len(all_raw_items)}")
    print(f"Answerable Items to Score:       {n_ans} (62 single-year, 6 multi-year)")
    print(f"Unanswerable Items Excluded:     {len(unanswerable_items)}")
    print("-" * 80)

    # 3. Descriptive Extra: Extractor Accuracy
    from backend.agent.nodes.classifier import extract_all_fiscal_years, extract_company_and_year

    comp_correct = 0
    year_correct = 0
    both_correct = 0
    extractor_mismatches = []

    for it in answerable_items:
        q = it["question"]
        gold_comp = it.get("company")
        gold_fy = str(it.get("fiscal_year"))

        ext_comp, ext_fy = extract_company_and_year(q)
        all_ext_fys = extract_all_fiscal_years(q)

        comp_ok = ext_comp == gold_comp
        if "/" in gold_fy:
            gold_years = set(gold_fy.split("/"))
            year_ok = set(all_ext_fys) == gold_years
        else:
            year_ok = ext_fy == gold_fy

        if comp_ok:
            comp_correct += 1
        if year_ok:
            year_correct += 1
        if comp_ok and year_ok:
            both_correct += 1
        else:
            extractor_mismatches.append({
                "id": it["id"],
                "question": q,
                "gold_comp": gold_comp,
                "gold_fy": gold_fy,
                "ext_comp": ext_comp,
                "ext_fy": ext_fy,
                "all_ext_fys": all_ext_fys,
                "comp_ok": comp_ok,
                "year_ok": year_ok,
            })

    extractor_summary = {
        "n_items": n_ans,
        "company_accuracy": comp_correct / n_ans if n_ans else 0.0,
        "company_correct": comp_correct,
        "year_accuracy": year_correct / n_ans if n_ans else 0.0,
        "year_correct": year_correct,
        "both_accuracy": both_correct / n_ans if n_ans else 0.0,
        "both_correct": both_correct,
        "mismatches": extractor_mismatches,
    }

    # 4. Run Retrieval for Configs A, B, C
    configs = ["A", "B", "C"]
    per_item_results: List[Dict[str, Any]] = []

    print("\nExecuting Retrieval across all 68 items for Configs A, B, C...")
    t0_eval = time.perf_counter()

    for idx, it in enumerate(answerable_items, start=1):
        qid = it["id"]
        q = it["question"]
        gold_cids = it.get("source_chunk_ids", [])
        filters = extract_query_filters(q)

        ret_a = run_config_a(q, filters)
        ret_b = run_config_b(q, filters)
        ret_c = run_config_c(q, filters)

        m_a = compute_item_metrics(ret_a, gold_cids)
        m_b = compute_item_metrics(ret_b, gold_cids)
        m_c = compute_item_metrics(ret_c, gold_cids)

        q_style = it.get("question_style") or it.get("metadata", {}).get("question_style") or "unknown"
        is_multi_year = "/" in str(it.get("fiscal_year", ""))

        per_item_results.append({
            "id": qid,
            "question": q,
            "company": it.get("company"),
            "fiscal_year": str(it.get("fiscal_year")),
            "section": it.get("section"),
            "question_style": q_style,
            "is_multi_year": is_multi_year,
            "gold_chunk_ids": gold_cids,
            "retrieved_A": ret_a,
            "retrieved_B": ret_b,
            "retrieved_C": ret_c,
            "metrics": {
                "A": m_a,
                "B": m_b,
                "C": m_c,
            },
        })

    eval_duration = time.perf_counter() - t0_eval
    print(f"Retrieval complete in {eval_duration:.1f}s.")

    # 5. Aggregate Metrics for Each Sub-Benchmark
    def aggregate_slice(item_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(item_list)
        if n == 0:
            return {}
        summary: Dict[str, Any] = {}
        for cfg in configs:
            hit8_list = [it["metrics"][cfg]["hit@8"] for it in item_list]
            recall8_list = [it["metrics"][cfg]["recall@8"] for it in item_list]
            mrr_list = [it["metrics"][cfg]["mrr"] for it in item_list]
            hit20_list = [it["metrics"][cfg]["hit@20"] for it in item_list]

            k_hit8 = int(sum(hit8_list))
            k_hit20 = int(sum(hit20_list))
            ci_hit8_low, ci_hit8_high = wilson_ci(k_hit8, n)
            ci_hit20_low, ci_hit20_high = wilson_ci(k_hit20, n)

            summary[cfg] = {
                "n": n,
                "k_hit@8": k_hit8,
                "hit@8": float(np.mean(hit8_list)),
                "hit@8_ci": (ci_hit8_low, ci_hit8_high),
                "recall@8": float(np.mean(recall8_list)),
                "mrr": float(np.mean(mrr_list)),
                "k_hit@20": k_hit20,
                "hit@20": float(np.mean(hit20_list)),
                "hit@20_ci": (ci_hit20_low, ci_hit20_high),
            }

        # McNemar tests on hit@8
        b_hits = [it["metrics"]["B"]["hit@8"] == 1.0 for it in item_list]
        a_hits = [it["metrics"]["A"]["hit@8"] == 1.0 for it in item_list]
        c_hits = [it["metrics"]["C"]["hit@8"] == 1.0 for it in item_list]

        summary["mcnemar_B_vs_A"] = exact_mcnemar_test(b_hits, a_hits)
        summary["mcnemar_B_vs_C"] = exact_mcnemar_test(b_hits, c_hits)
        return summary

    sy_results = [it for it in per_item_results if not it["is_multi_year"]]
    my_results = [it for it in per_item_results if it["is_multi_year"]]

    agg_all = aggregate_slice(per_item_results)
    agg_sy = aggregate_slice(sy_results)
    agg_my = aggregate_slice(my_results)

    # 6. Single-Year Hit@8 Breakdowns (by Company, Section, Question Style)
    def compute_dimension_breakdown(key_name: str) -> Dict[str, Dict[str, Any]]:
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for it in sy_results:
            val = str(it.get(key_name) or "unknown")
            categories.setdefault(val, []).append(it)

        breakdown: Dict[str, Dict[str, Any]] = {}
        for cat, items in sorted(categories.items()):
            n_cat = len(items)
            cat_summary: Dict[str, Any] = {"n": n_cat}
            for cfg in configs:
                h8 = sum(1 for it in items if it["metrics"][cfg]["hit@8"] == 1.0)
                cat_summary[cfg] = {
                    "k_hit@8": h8,
                    "hit@8": h8 / n_cat if n_cat else 0.0,
                }
            breakdown[cat] = cat_summary
        return breakdown

    breakdown_company = compute_dimension_breakdown("company")
    breakdown_section = compute_dimension_breakdown("section")
    breakdown_style = compute_dimension_breakdown("question_style")

    # 7. Descriptive Extra: Config B Misses in Top-50 Pool
    b_misses_info = []
    for it in per_item_results:
        m_b = it["metrics"]["B"]
        if m_b["hit@8"] == 0.0:
            first_rank = m_b["first_gold_rank"]
            b_misses_info.append({
                "id": it["id"],
                "question": it["question"],
                "company": it["company"],
                "fiscal_year": it["fiscal_year"],
                "is_multi_year": it["is_multi_year"],
                "gold_chunk_ids": it["gold_chunk_ids"],
                "rank_in_top50": first_rank,  # 1..50 or None if >50
            })

    # =========================================================================
    # 8. Plain Text Tables to Stdout
    # =========================================================================
    config_labels = {
        "A": "Config A (legacy retrieve)",
        "B": "Config B (clean_v3_rrf10)",
        "C": "Config C (raw BM25-only)",
    }

    def print_sub_benchmark_table(title: str, agg_dict: Dict[str, Any], note: Optional[str] = None) -> None:
        print("\n" + "=" * 90)
        print(title)
        if note:
            print(f">>> {note}")
        print("=" * 90)
        hdr = (
            f"{'Configuration':32s} | {'Hit@8':13s} | {'Wilson 95% CI':17s} | "
            f"{'Recall@8':10s} | {'MRR@8':8s} | {'Hit@20':13s}"
        )
        print(hdr)
        print("-" * len(hdr))
        for cfg in configs:
            d = agg_dict[cfg]
            ci_str = f"[{d['hit@8_ci'][0]:.3f}, {d['hit@8_ci'][1]:.3f}]"
            row = (
                f"{config_labels[cfg]:32s} | "
                f"{d['k_hit@8']:2d}/{d['n']:2d} ({d['hit@8']:5.1%}) | "
                f"{ci_str:17s} | "
                f"{d['recall@8']:8.4f}   | "
                f"{d['mrr']:6.4f}   | "
                f"{d['k_hit@20']:2d}/{d['n']:2d} ({d['hit@20']:5.1%})"
            )
            print(row)

        mcn_a = agg_dict["mcnemar_B_vs_A"]
        mcn_c = agg_dict["mcnemar_B_vs_C"]
        print("-" * len(hdr))
        print("Paired Exact McNemar Tests on Hit@8:")
        print(
            f"  * Config B vs. Config A: b={mcn_a['only_test']}, c={mcn_a['only_base']}, "
            f"discordant={mcn_a['discordant']}, two-sided p={mcn_a['p_value']:.4f} "
            f"{'(Significant at p<0.05)' if mcn_a['p_value'] < 0.05 else '(Not statistically significant at p<0.05)'}"
        )
        print(
            f"  * Config B vs. Config C: b={mcn_c['only_test']}, c={mcn_c['only_base']}, "
            f"discordant={mcn_c['discordant']}, two-sided p={mcn_c['p_value']:.4f} "
            f"{'(Significant at p<0.05)' if mcn_c['p_value'] < 0.05 else '(Not statistically significant at p<0.05)'}"
        )

    # 1. All Answerable Items
    print_sub_benchmark_table(
        f"1. ALL ANSWERABLE ITEMS (n={n_ans})",
        agg_all,
    )

    # 2. Single-Year Items
    print_sub_benchmark_table(
        f"2. SINGLE-YEAR ITEMS (n={n_sy})",
        agg_sy,
    )

    # 3. Multi-Year Items (Descriptive Only)
    print_sub_benchmark_table(
        f"3. MULTI-YEAR ITEMS (n={n_my})",
        agg_my,
        note="MULTI-YEAR n=6: DESCRIPTIVE ONLY (under-powered for statistical inference)",
    )

    # 4. Single-Year Hit@8 Breakdowns
    def print_breakdown_table(title: str, bd_data: Dict[str, Dict[str, Any]]) -> None:
        print("\n" + "-" * 85)
        print(f"SINGLE-YEAR HIT@8 BREAKDOWN: {title}")
        print("-" * 85)
        hdr = f"{'Slice':50s} | {'n':3s} | {'Config A':14s} | {'Config B':14s} | {'Config C':14s}"
        print(hdr)
        print("-" * len(hdr))
        for cat, val in bd_data.items():
            row = (
                f"{cat:50s} | {val['n']:3d} | "
                f"{val['A']['k_hit@8']:2d}/{val['n']:2d} ({val['A']['hit@8']:5.1%}) | "
                f"{val['B']['k_hit@8']:2d}/{val['n']:2d} ({val['B']['hit@8']:5.1%}) | "
                f"{val['C']['k_hit@8']:2d}/{val['n']:2d} ({val['C']['hit@8']:5.1%})"
            )
            print(row)

    print_breakdown_table("BY COMPANY", breakdown_company)
    print_breakdown_table("BY SECTION", breakdown_section)
    print_breakdown_table("BY QUESTION STYLE", breakdown_style)

    # 5. Descriptive Extras: Extractor Accuracy
    print("\n" + "=" * 80)
    print("DESCRIPTIVE EXTRA: EXTRACTOR ACCURACY ON ROUND 2 (n=68 answerable)")
    print("=" * 80)
    print(
        f"Company Extraction Accuracy:    {extractor_summary['company_correct']}/{n_ans} "
        f"({extractor_summary['company_accuracy']:.1%})"
    )
    print(
        f"Fiscal Year Extraction Accuracy: {extractor_summary['year_correct']}/{n_ans} "
        f"({extractor_summary['year_accuracy']:.1%})"
    )
    print(
        f"Both Company & Year Accurate:    {extractor_summary['both_correct']}/{n_ans} "
        f"({extractor_summary['both_accuracy']:.1%})"
    )
    print(f"Total Extraction Mismatches:     {len(extractor_mismatches)}")
    if extractor_mismatches:
        for m in extractor_mismatches:
            print(
                f"  - [{m['id']}] Q: \"{m['question']}\" | "
                f"Gold: ({m['gold_comp']}, {m['gold_fy']}) vs Extracted: ({m['ext_comp']}, {m['ext_fy']}, all={m['all_ext_fys']})"
            )

    # 6. Descriptive Extra: Config B Misses in Top-50 Pool
    print("\n" + "=" * 80)
    print(f"DESCRIPTIVE EXTRA: CONFIG B HIT@8 MISSES (n={len(b_misses_info)} misses out of {n_ans})")
    print("Rank of Gold Chunk in Config B Top-50 Candidate Pool:")
    print("=" * 80)
    hdr_miss = f"{'Item ID':12s} | {'Slice':11s} | {'Company':10s} | {'FY':6s} | {'Gold Rank in Pool':18s} | {'Question'}"
    print(hdr_miss)
    print("-" * 95)
    for m in b_misses_info:
        slice_label = "Multi-year" if m["is_multi_year"] else "Single-year"
        rank_str = f"Rank #{m['rank_in_top50']}" if m["rank_in_top50"] is not None else ">50 (Not in pool)"
        print(
            f"{m['id']:12s} | {slice_label:11s} | {str(m['company']):10s} | {str(m['fiscal_year']):6s} | "
            f"{rank_str:18s} | {m['question']}"
        )

    # =========================================================================
    # 9. Write Output Files
    # =========================================================================
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # A. JSON Output
    output_json_data = {
        "metadata": {
            "evaluation": "Round 2 Benchmark Evaluation (Pre-Registered)",
            "preregistration_file": "backend/eval/round2_preregistration.md",
            "golden_round2_sha256_crlf_normalized": sha256_hash,
            "git_commit": commit_hash,
            "timestamp": timestamp_iso,
            "total_items": len(all_raw_items),
            "answerable_items": n_ans,
            "single_year_items": n_sy,
            "multi_year_items": n_my,
            "unanswerable_excluded": len(unanswerable_items),
        },
        "summary": {
            "all_answerable": agg_all,
            "single_year": agg_sy,
            "multi_year_descriptive": agg_my,
        },
        "breakdowns_single_year": {
            "by_company": breakdown_company,
            "by_section": breakdown_section,
            "by_question_style": breakdown_style,
        },
        "descriptive_extras": {
            "extractor_accuracy": extractor_summary,
            "config_b_misses_in_top50": b_misses_info,
        },
        "per_item_results": per_item_results,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_json_data, f, indent=2)
    print(f"\n[OK] Wrote JSON results to: {OUTPUT_JSON_PATH}")

    # B. CSV Per-Item Table
    # Columns: id, question, gold chunk ids, rank per config
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "gold_chunk_ids", "rank_A", "rank_B", "rank_C"])
        for it in per_item_results:
            qid = it["id"]
            q = it["question"]
            gold_str = ";".join(it["gold_chunk_ids"])
            r_a = it["metrics"]["A"]["first_gold_rank"]
            r_b = it["metrics"]["B"]["first_gold_rank"]
            r_c = it["metrics"]["C"]["first_gold_rank"]

            r_a_str = str(r_a) if r_a is not None else ">20"
            r_b_str = str(r_b) if r_b is not None else ">50"
            r_c_str = str(r_c) if r_c is not None else ">50"
            writer.writerow([qid, q, gold_str, r_a_str, r_b_str, r_c_str])
    print(f"[OK] Wrote per-item CSV table to: {OUTPUT_CSV_PATH}")

    # C. Markdown Artifact
    def make_md_table(title: str, d_agg: Dict[str, Any], note: Optional[str] = None) -> str:
        lines = [f"### {title}"]
        if note:
            lines.append(f"> **Note:** {note}\n")
        lines.append(
            "| Configuration | Hit@8 | Wilson 95% CI | Recall@8 | MRR@8 | Hit@20 |\n"
            "|---|---|---|---|---|---|"
        )
        for cfg in configs:
            d = d_agg[cfg]
            ci_str = f"[{d['hit@8_ci'][0]:.3f}, {d['hit@8_ci'][1]:.3f}]"
            lines.append(
                f"| **{config_labels[cfg]}** | {d['k_hit@8']}/{d['n']} ({d['hit@8']:.1%}) | "
                f"`{ci_str}` | {d['recall@8']:.4f} | {d['mrr']:.4f} | "
                f"{d['k_hit@20']}/{d['n']} ({d['hit@20']:.1%}) |"
            )

        mcn_a = d_agg["mcnemar_B_vs_A"]
        mcn_c = d_agg["mcnemar_B_vs_C"]
        sig_a = "Statistically Significant (p < 0.05)" if mcn_a["p_value"] < 0.05 else "Not Statistically Significant"
        sig_c = "Statistically Significant (p < 0.05)" if mcn_c["p_value"] < 0.05 else "Not Statistically Significant"

        lines.append("\n**Paired Exact McNemar Tests on Hit@8:**")
        lines.append(
            f"- **Config B vs. Config A:** b={mcn_a['only_test']}, c={mcn_a['only_base']}, "
            f"discordant={mcn_a['discordant']}, two-sided $p = {mcn_a['p_value']:.4f}$ ({sig_a})"
        )
        lines.append(
            f"- **Config B vs. Config C:** b={mcn_c['only_test']}, c={mcn_c['only_base']}, "
            f"discordant={mcn_c['discordant']}, two-sided $p = {mcn_c['p_value']:.4f}$ ({sig_c})\n"
        )
        return "\n".join(lines)

    md_content = f"""# Pre-Registered Round 2 Benchmark Evaluation Results

## Pre-Registration & Metadata Stamping
- **Date & Time:** `{timestamp_readable}`
- **Git Commit Hash:** `{commit_hash}`
- **Dataset:** `backend/eval/golden_round2.json`
- **CRLF-Normalized SHA-256:** `{sha256_hash}`
- **Pre-Registration Protocol:** [round2_preregistration.md](file:///c:/Users/ASUS/OneDrive/Desktop/verity/backend/eval/round2_preregistration.md)
- **Scoring Protocol:** Scored strictly once without configuration or query-cleaning adjustments.

---

## 1. Overall Answerable Sub-Benchmark (n={n_ans})
{make_md_table(f"All Answerable Items (n={n_ans})", agg_all)}

---

## 2. Single-Year Sub-Benchmark (n={n_sy})
{make_md_table(f"Single-Year Answerable Items (n={n_sy})", agg_sy)}

---

## 3. Multi-Year Sub-Benchmark (n={n_my})
{make_md_table(f"Multi-Year Answerable Items (n={n_my})", agg_my, note="Multi-year sample size is n=6; metrics are purely descriptive and under-powered for statistical hypothesis testing.")}

---

## 4. Single-Year Hit@8 Slices

### By Target Company
| Company | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
"""
    for comp, val in breakdown_company.items():
        md_content += (
            f"| {comp} | {val['n']} | "
            f"{val['A']['k_hit@8']}/{val['n']} ({val['A']['hit@8']:.1%}) | "
            f"{val['B']['k_hit@8']}/{val['n']} ({val['B']['hit@8']:.1%}) | "
            f"{val['C']['k_hit@8']}/{val['n']} ({val['C']['hit@8']:.1%}) |\n"
        )

    md_content += """
### By 10-K Filing Section
| Section | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
"""
    for sec, val in breakdown_section.items():
        md_content += (
            f"| {sec} | {val['n']} | "
            f"{val['A']['k_hit@8']}/{val['n']} ({val['A']['hit@8']:.1%}) | "
            f"{val['B']['k_hit@8']}/{val['n']} ({val['B']['hit@8']:.1%}) | "
            f"{val['C']['k_hit@8']}/{val['n']} ({val['C']['hit@8']:.1%}) |\n"
        )

    md_content += """
### By Question Formulation Style
| Question Style | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
"""
    for sty, val in breakdown_style.items():
        md_content += (
            f"| {sty} | {val['n']} | "
            f"{val['A']['k_hit@8']}/{val['n']} ({val['A']['hit@8']:.1%}) | "
            f"{val['B']['k_hit@8']}/{val['n']} ({val['B']['hit@8']:.1%}) | "
            f"{val['C']['k_hit@8']}/{val['n']} ({val['C']['hit@8']:.1%}) |\n"
        )

    md_content += f"""
---

## 5. Descriptive Extras (Not Acted Upon)

### Extractor Accuracy on Round 2
- **Company Extraction Accuracy:** {extractor_summary['company_correct']}/{n_ans} ({extractor_summary['company_accuracy']:.1%})
- **Fiscal Year Extraction Accuracy:** {extractor_summary['year_correct']}/{n_ans} ({extractor_summary['year_accuracy']:.1%})
- **Both Accurate:** {extractor_summary['both_correct']}/{n_ans} ({extractor_summary['both_accuracy']:.1%})
- **Total Mismatches:** {len(extractor_mismatches)}

### Config B Hit@8 Misses & Top-50 Pool Ranks
Total Config B Misses on Hit@8: **{len(b_misses_info)} / {n_ans}**

| Item ID | Sub-Benchmark | Company | Year | Rank in Top-50 Pool | Question |
|---|---|---|---|---|---|
"""
    for m in b_misses_info:
        slice_label = "Multi-year" if m["is_multi_year"] else "Single-year"
        rank_str = f"Rank #{m['rank_in_top50']}" if m["rank_in_top50"] is not None else ">50 (Not in pool)"
        md_content += (
            f"| `{m['id']}` | {slice_label} | {m['company']} | {m['fiscal_year']} | "
            f"**{rank_str}** | {m['question']} |\n"
        )

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Wrote Markdown report to: {OUTPUT_MD_PATH}")

    return output_json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Round 2 Pre-Registered Benchmark")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing round2_scoring.json (breaks pre-registration guard)",
    )
    args = parser.parse_args()

    score_round2(force=args.force)
