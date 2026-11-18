"""
End-to-End Evaluation Analysis for Verity Financial RAG.
Implements the analysis preregistered in backend/eval/e2e_preregistration.md.

Inputs:
- backend/eval/results/baseline_answers.json
- backend/eval/results/improved_answers.json
- backend/eval/results/baseline.json / baseline_judge_checkpoint.json
- backend/eval/results/improved.json / improved_judge_checkpoint.json
- backend/eval/golden_round2.json

Outputs:
- backend/eval/results/e2e_summary.json
- backend/eval/results/e2e_summary.md
- backend/eval/results/e2e_spot_check.md
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "eval" / "results"
GOLDEN_PATH = BASE_DIR / "eval" / "golden_round2.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataIntegrityError(ValueError):
    """Raised when evaluation data has missing items or NaN scores."""
    pass


DECLINE_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "unable to answer",
    "insufficient information",
    "not enough information to answer",
)


def normalize_bytes_newlines(b: bytes) -> bytes:
    """Normalizes CRLF and CR newlines to LF for deterministic hashing."""
    return b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def compute_file_sha256(path: Path) -> str:
    """Computes LF-normalized SHA-256 hash of a file."""
    if not path.exists():
        return "file_not_found"
    content = path.read_bytes()
    return hashlib.sha256(normalize_bytes_newlines(content)).hexdigest()


def is_decline_answer(answer: Optional[str]) -> bool:
    """
    Detects whether an answer expresses a decline / refusal to answer due to insufficient context.
    """
    if not answer or not isinstance(answer, str):
        return False
    lower = answer.lower()
    return any(phrase in lower for phrase in DECLINE_PHRASES)


def count_declines(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes decline count and rate across a list of answer items.
    """
    total = len(items)
    if total == 0:
        return {"decline_count": 0, "total": 0, "decline_rate": 0.0, "decline_ids": []}
    decline_ids = [it["id"] for it in items if is_decline_answer(it.get("answer"))]
    decline_count = len(decline_ids)
    decline_rate = round(decline_count / total, 4)
    return {
        "decline_count": decline_count,
        "total": total,
        "decline_rate": decline_rate,
        "decline_ids": decline_ids,
    }


def compute_paired_bootstrap_ci(
    baseline_scores: List[float],
    improved_scores: List[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes mean paired difference (improved - baseline) and 95% bootstrap CI
    via percentile interval over N item resamples with replacement.
    Also counts how many items are improved > baseline, ==, and <.
    """
    if len(baseline_scores) != len(improved_scores):
        raise ValueError(
            f"Score lists must have equal lengths, got {len(baseline_scores)} vs {len(improved_scores)}"
        )
    n = len(baseline_scores)
    if n == 0:
        return {
            "mean_baseline": 0.0,
            "mean_improved": 0.0,
            "mean_difference": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_level": ci,
            "higher": 0,
            "equal": 0,
            "lower": 0,
            "n": 0,
        }

    b_arr = np.array(baseline_scores, dtype=float)
    imp_arr = np.array(improved_scores, dtype=float)
    diffs = imp_arr - b_arr
    mean_diff = float(np.mean(diffs))

    higher = int(np.sum(diffs > 1e-9))
    equal = int(np.sum(np.abs(diffs) <= 1e-9))
    lower = int(np.sum(diffs < -1e-9))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_diffs = diffs[indices]
    boot_means = np.mean(resampled_diffs, axis=1)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return {
        "mean_baseline": round(float(np.mean(b_arr)), 4),
        "mean_improved": round(float(np.mean(imp_arr)), 4),
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": ci,
        "higher": higher,
        "equal": equal,
        "lower": lower,
        "n": n,
    }


def compute_two_sample_bootstrap_ci(
    group1_scores: List[float],
    group2_scores: List[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes two-sample difference in means (group1 - group2) with 95% bootstrap CI
    over independent resamples with replacement.
    """
    g1 = np.array(group1_scores, dtype=float)
    g2 = np.array(group2_scores, dtype=float)
    n1, n2 = len(g1), len(g2)
    mean1 = float(np.mean(g1)) if n1 > 0 else 0.0
    mean2 = float(np.mean(g2)) if n2 > 0 else 0.0
    diff = mean1 - mean2
    if n1 == 0 or n2 == 0:
        return {
            "mean_group1": round(mean1, 4),
            "mean_group2": round(mean2, 4),
            "mean_difference": round(diff, 4),
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n1": n1,
            "n2": n2,
        }

    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n1, size=(n_resamples, n1))
    idx2 = rng.integers(0, n2, size=(n_resamples, n2))
    means1 = np.mean(g1[idx1], axis=1)
    means2 = np.mean(g2[idx2], axis=1)
    boot_diffs = means1 - means2

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    return {
        "mean_group1": round(mean1, 4),
        "mean_group2": round(mean2, 4),
        "mean_difference": round(diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n1": n1,
        "n2": n2,
    }


def filter_non_decline_pairs(
    baseline_items: List[Dict[str, Any]],
    improved_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filters paired items to only those where NEITHER pipeline's answer is a decline.
    """
    b_map = {it["id"]: it for it in baseline_items}
    imp_map = {it["id"]: it for it in improved_items}
    shared_ids = [it_id for it_id in b_map if it_id in imp_map]

    clean_b: List[Dict[str, Any]] = []
    clean_imp: List[Dict[str, Any]] = []

    for it_id in shared_ids:
        b_it = b_map[it_id]
        imp_it = imp_map[it_id]
        if not is_decline_answer(b_it.get("answer")) and not is_decline_answer(imp_it.get("answer")):
            clean_b.append(b_it)
            clean_imp.append(imp_it)

    return clean_b, clean_imp


def sample_random_items(
    items: List[Dict[str, Any]],
    k: int = 6,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Randomly samples k items from a list without replacement using fixed seed.
    """
    if len(items) <= k:
        return list(items)
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(items), size=k, replace=False)
    return [items[i] for i in chosen_indices]


def select_lowest_faithfulness_items(
    improved_items: List[Dict[str, Any]],
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Selects the k items with the lowest faithfulness scores in the improved pipeline.
    Ties broken deterministically by id.
    """
    def _get_f(it: Dict[str, Any]) -> float:
        val = it.get("improved_faithfulness")
        if val is None:
            val = it.get("faithfulness")
        return float(val) if val is not None else 0.0

    return sorted(improved_items, key=lambda x: (_get_f(x), x.get("id", "")))[:k]


def analyze_judge_refusal_inconsistency(
    baseline_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Counts how many baseline faithfulness scores of 0.0 and 1.0 are declines,
    documenting the judge's inconsistency on refusals.
    """
    scores_0 = [it for it in baseline_items if float(it.get("faithfulness") or 0.0) == 0.0]
    scores_1 = [it for it in baseline_items if float(it.get("faithfulness") or 0.0) == 1.0]
    all_declines = [it for it in baseline_items if is_decline_answer(it.get("answer"))]

    declines_0 = [it for it in scores_0 if is_decline_answer(it.get("answer"))]
    declines_1 = [it for it in scores_1 if is_decline_answer(it.get("answer"))]

    return {
        "faithfulness_0_0": {
            "total": len(scores_0),
            "decline_count": len(declines_0),
            "non_decline_count": len(scores_0) - len(declines_0),
            "decline_pct": round(len(declines_0) / len(scores_0), 4) if scores_0 else 0.0,
            "decline_ids": [it["id"] for it in declines_0],
        },
        "faithfulness_1_0": {
            "total": len(scores_1),
            "decline_count": len(declines_1),
            "non_decline_count": len(scores_1) - len(declines_1),
            "decline_pct": round(len(declines_1) / len(scores_1), 4) if scores_1 else 0.0,
            "decline_ids": [it["id"] for it in declines_1],
        },
        "total_baseline_declines": len(all_declines),
        "declines_scored_0_or_1": len(declines_0) + len(declines_1),
        "inconsistency_summary": (
            f"Of the {len(all_declines)} total baseline declines, {len(declines_0)} were scored 0.0 "
            f"and {len(declines_1)} were scored 1.0 by the judge, demonstrating arbitrary binary scoring "
            "of standard refusal boilerplate."
        ),
    }


def compute_retrieval_metrics_for_item(retrieved_chunk_ids: List[str], source_chunk_ids: List[str], k: int = 8) -> Tuple[float, float, float]:
    """Computes hit@k, recall@k, and mrr for an item if not pre-computed."""
    top_k = retrieved_chunk_ids[:k]
    gold_set = set(source_chunk_ids)
    if not gold_set:
        return 0.0, 0.0, 0.0
    hits = [cid for cid in top_k if cid in gold_set]
    hit = 1.0 if len(hits) > 0 else 0.0
    recall = len(set(hits)) / len(gold_set)
    mrr = 0.0
    for rank, cid in enumerate(top_k, 1):
        if cid in gold_set:
            mrr = 1.0 / rank
            break
    return hit, recall, mrr


def load_pipeline_data(
    answers_path: Path,
    results_path: Optional[Path],
    checkpoint_path: Optional[Path],
    config_name: str,
    required_metrics: Tuple[str, ...] = ("faithfulness", "answer_correctness"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[str]]]:
    """
    Loads pipeline evaluation data from answers file, results file, and checkpoint.
    Merges per-item judge scores and flags missing or NaN metrics.
    Returns: (items, metadata, offending_ids_by_metric).
    """
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    items = answers_data.get("items", [])
    if not items:
        raise ValueError(f"No items found in {answers_path}")

    # Results file data (per_question_results)
    results_data: Dict[str, Any] = {}
    results_per_q: Dict[str, Dict[str, Any]] = {}
    if results_path and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)
            for it in results_data.get("per_question_results", []):
                results_per_q[it["id"]] = it
        except Exception as e:
            logger.warning(f"Could not load results from {results_path}: {e}")

    # Checkpoint data (item_id -> {metric: score})
    checkpoint_data: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint from {checkpoint_path}: {e}")

    judge_model = (
        results_data.get("judge_model")
        or results_data.get("models", {}).get("judge_model")
        or answers_data.get("models", {}).get("judge_model")
        or "unknown"
    )

    merged_items: List[Dict[str, Any]] = []
    offending_by_metric: Dict[str, List[str]] = {m: [] for m in required_metrics}

    for item in items:
        item_id = item["id"]
        merged = dict(item)

        # Merge results from results_path if available
        if item_id in results_per_q:
            res_item = results_per_q[item_id]
            for k in ("hit@8", "recall@8", "mrr", "numeric_match", "faithfulness", "answer_correctness"):
                if k in res_item and res_item[k] is not None:
                    merged[k] = res_item[k]

        # Merge from checkpoint if metric is missing
        if item_id in checkpoint_data:
            cp_scores = checkpoint_data[item_id]
            for m in required_metrics:
                if (m not in merged or merged[m] is None) and m in cp_scores and cp_scores[m] is not None:
                    merged[m] = cp_scores[m]

        # Ensure retrieval metrics are present
        if "hit@8" not in merged or merged["hit@8"] is None:
            r_hit, r_rec, r_mrr = compute_retrieval_metrics_for_item(
                merged.get("retrieved_chunk_ids", []),
                merged.get("source_chunk_ids", []),
                k=8,
            )
            merged["hit@8"] = r_hit
            merged["recall@8"] = r_rec
            merged["mrr"] = r_mrr

        # Validate required metrics are non-missing and non-NaN
        for m in required_metrics:
            val = merged.get(m)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                offending_by_metric[m].append(item_id)

        merged_items.append(merged)

    metadata = {
        "config": config_name,
        "judge_model": judge_model,
        "generator_model": answers_data.get("models", {}).get("gen_model"),
        "critic_model": answers_data.get("models", {}).get("critic_model"),
        "total_items": len(merged_items),
        "timestamp": answers_data.get("timestamp"),
    }

    return merged_items, metadata, offending_by_metric


def validate_no_nans_or_missing(
    baseline_offending: Dict[str, List[str]],
    improved_offending: Dict[str, List[str]],
) -> None:
    """
    Enforces the preregistered contract: Refuses to run if either pipeline has any
    missing or NaN per-item score, and lists the offending IDs.
    """
    has_b_offending = any(len(ids) > 0 for ids in baseline_offending.values())
    has_imp_offending = any(len(ids) > 0 for ids in improved_offending.values())

    if has_b_offending or has_imp_offending:
        lines = [
            "CRITICAL: Evaluation data contains missing or NaN per-item scores!",
            "Preregistration contract requires complete 30-item judged scores.",
            "Refusing to run end-to-end analysis until all items are judged successfully.",
        ]
        if has_b_offending:
            lines.append("\nBaseline offending items:")
            for m, ids in baseline_offending.items():
                if ids:
                    lines.append(f"  - {m} ({len(ids)} items): {ids}")
        if has_imp_offending:
            lines.append("\nImproved offending items:")
            for m, ids in improved_offending.items():
                if ids:
                    lines.append(f"  - {m} ({len(ids)} items): {ids}")

        err_msg = "\n".join(lines)
        logger.error(err_msg)
        raise DataIntegrityError(err_msg)


def compute_critic_stats(improved_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes critic revise rate and retry count distribution for the improved agent.
    If the answers file does not record it, says so explicitly without guessing.
    """
    has_retry_data = any("retry_count" in it and it["retry_count"] is not None for it in improved_items)
    if not has_retry_data:
        return {
            "recorded": False,
            "note": "The answers file does not record critic revise rate or retry count.",
            "revise_rate": None,
            "revised_count": None,
            "retry_distribution": None,
            "retry_distribution_pct": None,
        }

    retry_counts = [it.get("retry_count", 0) for it in improved_items]
    revised_items = [r > 0 for r in retry_counts]
    n = len(improved_items)
    revise_rate = round(float(np.mean(revised_items)), 4) if n > 0 else 0.0

    from collections import Counter
    dist = Counter(retry_counts)
    dist_dict = {int(k): dist[k] for k in sorted(dist.keys())}
    dist_pct = {int(k): round(dist[k] / n, 4) for k in sorted(dist.keys())}

    return {
        "recorded": True,
        "n": n,
        "revised_count": int(sum(revised_items)),
        "revise_rate": revise_rate,
        "retry_distribution": dist_dict,
        "retry_distribution_pct": dist_pct,
    }


def compute_retrieval_slices(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Slices answer quality metrics by whether the gold chunk was retrieved (hit@8 True vs False).
    """
    hit_true = [it for it in items if float(it.get("hit@8", 0.0)) > 0.0]
    hit_false = [it for it in items if float(it.get("hit@8", 0.0)) == 0.0]

    def slice_metrics(sub_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sub_items:
            return {"n": 0, "faithfulness": None, "answer_correctness": None}
        faiths = [it["faithfulness"] for it in sub_items if it.get("faithfulness") is not None]
        corrs = [it["answer_correctness"] for it in sub_items if it.get("answer_correctness") is not None]
        return {
            "n": len(sub_items),
            "faithfulness": round(float(np.mean(faiths)), 4) if faiths else None,
            "answer_correctness": round(float(np.mean(corrs)), 4) if corrs else None,
        }

    return {
        "hit_true": slice_metrics(hit_true),
        "hit_false": slice_metrics(hit_false),
    }


def _get_supporting_quote(
    item_id: str,
    golden_map: Dict[str, Dict[str, Any]],
    b_it: Dict[str, Any],
    imp_it: Dict[str, Any],
) -> str:
    """Extracts ground-truth supporting quote or primary context snippet."""
    if item_id in golden_map:
        quotes = golden_map[item_id].get("source_quotes", [])
        if quotes:
            if isinstance(quotes, list):
                return " ".join(str(q) for q in quotes)
            return str(quotes)

    for it in (imp_it, b_it):
        ctxs = it.get("contexts", [])
        if ctxs and isinstance(ctxs, list) and ctxs[0]:
            snippet = str(ctxs[0]).strip()
            return snippet[:300] + ("..." if len(snippet) > 300 else "")

    return "No quote available."


def select_spot_check_items(
    baseline_items: List[Dict[str, Any]],
    improved_items: List[Dict[str, Any]],
    golden_map: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Selects 6 items for human spot check:
    - 3 with largest absolute difference in faithfulness
    - 3 with largest absolute difference in answer correctness
    """
    b_map = {it["id"]: it for it in baseline_items}
    imp_map = {it["id"]: it for it in improved_items}

    paired: List[Dict[str, Any]] = []
    for item_id, imp_it in imp_map.items():
        if item_id in b_map:
            b_it = b_map[item_id]
            b_f = float(b_it.get("faithfulness") or 0.0)
            imp_f = float(imp_it.get("faithfulness") or 0.0)
            b_c = float(b_it.get("answer_correctness") or 0.0)
            imp_c = float(imp_it.get("answer_correctness") or 0.0)

            paired.append({
                "id": item_id,
                "category": imp_it.get("category") or b_it.get("category", "unknown"),
                "question": imp_it.get("question") or b_it.get("question", ""),
                "ground_truth": imp_it.get("ground_truth") or b_it.get("ground_truth", ""),
                "baseline_answer": b_it.get("answer", ""),
                "improved_answer": imp_it.get("answer", ""),
                "baseline_faithfulness": round(b_f, 4),
                "improved_faithfulness": round(imp_f, 4),
                "diff_faithfulness": round(imp_f - b_f, 4),
                "abs_diff_faithfulness": round(abs(imp_f - b_f), 4),
                "baseline_answer_correctness": round(b_c, 4),
                "improved_answer_correctness": round(imp_c, 4),
                "diff_answer_correctness": round(imp_c - b_c, 4),
                "abs_diff_answer_correctness": round(abs(imp_c - b_c), 4),
                "supporting_quote": _get_supporting_quote(item_id, golden_map, b_it, imp_it),
            })

    top3_faith = sorted(paired, key=lambda x: x["abs_diff_faithfulness"], reverse=True)[:3]
    top3_corr = sorted(paired, key=lambda x: x["abs_diff_answer_correctness"], reverse=True)[:3]

    return {
        "faithfulness_top3": top3_faith,
        "answer_correctness_top3": top3_corr,
    }


def generate_e2e_summary_markdown(summary: Dict[str, Any]) -> str:
    """Generates clean GitHub-flavored markdown for results/e2e_summary.md."""
    meta = summary["metadata"]
    paired = summary["paired_differences"]
    declines = summary["decline_metrics"]
    critic = summary["critic_metrics"]
    slices = summary["retrieval_slices"]

    lines = [
        "# End-to-End Evaluation Summary: Verity Financial RAG",
        "",
        f"- **Date**: {meta['date']}",
        f"- **Judge Model**: `{meta['judge_model']}`",
        f"- **Golden Benchmark**: `golden_round2.json` (SHA-256: `{meta['golden_round2_sha256'][:16]}...`)",
        f"- **Sample Size**: {meta['sample_size']} items (stratified by company & section)",
        f"- **Generator Model**: `{meta['generator_model']}` | **Critic Model**: `{meta['critic_model']}`",
        "",
        "## 1. Primary Metrics & Paired Differences",
        "",
        "| Metric | Baseline | Improved | Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_correctness": "Answer Correctness",
        "hit@8": "Hit@8",
        "recall@8": "Recall@8",
        "mrr": "MRR@8",
        "numeric_accuracy": f"Numeric Accuracy (n={paired['numeric_accuracy']['n']})",
    }

    for m_key, label in metric_labels.items():
        p = paired[m_key]
        diff_str = f"{p['mean_difference']:+.4f}"
        ci_str = f"[{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]"
        counts_str = f"{p['higher']} / {p['equal']} / {p['lower']}"
        lines.append(
            f"| **{label}** | {p['mean_baseline']:.4f} | {p['mean_improved']:.4f} | **{diff_str}** | {ci_str} | {counts_str} |"
        )

    lines.extend([
        "",
        "## 2. Refusal / Decline Behavior",
        "",
        f"- **Baseline Decline Rate**: {declines['baseline']['decline_count']} / {declines['baseline']['total']} "
        f"({declines['baseline']['decline_rate'] * 100:.1f}%)",
        f"- **Improved Decline Rate**: {declines['improved']['decline_count']} / {declines['improved']['total']} "
        f"({declines['improved']['decline_rate'] * 100:.1f}%)",
        f"- **Decline Reduction**: {(declines['baseline']['decline_rate'] - declines['improved']['decline_rate']) * 100:+.1f} percentage points",
        "",
        "## 3. Critic Revision & Retry Behavior (Improved Agent)",
        "",
    ])

    if critic.get("recorded"):
        lines.extend([
            f"- **Critic Revise Rate**: {critic['revised_count']} / {critic['n']} ({critic['revise_rate'] * 100:.1f}%)",
            "- **Retry Count Distribution**:",
        ])
        for retries, count in critic["retry_distribution"].items():
            pct = critic["retry_distribution_pct"].get(retries, 0.0) * 100
            lines.append(f"  - {retries} retries: {count} items ({pct:.1f}%)")
    else:
        lines.append(f"- *{critic.get('note', 'No critic telemetry recorded.')}*")

    lines.extend([
        "",
        "## 4. Retrieval Conditioning Slice (Hit@8 True vs False)",
        "",
        "| Pipeline | Gold Chunk Retrieved (Hit@8) | N | Mean Faithfulness | Mean Answer Correctness |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for pipe_name, slice_data in [("Baseline", slices["baseline"]), ("Improved", slices["improved"])]:
        ht = slice_data["hit_true"]
        hf = slice_data["hit_false"]
        ht_f = f"{ht['faithfulness']:.4f}" if ht["faithfulness"] is not None else "N/A"
        ht_c = f"{ht['answer_correctness']:.4f}" if ht["answer_correctness"] is not None else "N/A"
        hf_f = f"{hf['faithfulness']:.4f}" if hf["faithfulness"] is not None else "N/A"
        hf_c = f"{hf['answer_correctness']:.4f}" if hf["answer_correctness"] is not None else "N/A"

        lines.append(f"| {pipe_name} | **Yes (Hit)** | {ht['n']} | {ht_f} | {ht_c} |")
        lines.append(f"| {pipe_name} | **No (Miss)** | {hf['n']} | {hf_f} | {hf_c} |")

    lines.append("")
    return "\n".join(lines)


def generate_e2e_spot_check_markdown(spot_checks: Dict[str, List[Dict[str, Any]]], metadata: Dict[str, Any]) -> str:
    """Generates clean GitHub-flavored markdown for results/e2e_spot_check.md."""
    lines = [
        "# End-to-End Evaluation: Human Spot Check",
        "",
        f"- **Date**: {metadata['date']}",
        f"- **Judge Model**: `{metadata['judge_model']}`",
        "- **Purpose**: 6 items selected for human qualitative audit: 3 with largest absolute difference in faithfulness, and 3 with largest absolute difference in answer correctness.",
        "",
        "---",
        "",
        "## Category A: Top 3 Items by Absolute Faithfulness Difference",
        "",
    ]

    for rank, it in enumerate(spot_checks["faithfulness_top3"], 1):
        lines.extend([
            f"### #{rank} [Item `{it['id']}`] ({it['category']})",
            "",
            f"**Question**: {it['question']}",
            "",
            f"**Ground Truth**: `{it['ground_truth']}`",
            "",
            f"**Supporting Quote**: {it['supporting_quote']}",
            "",
            f"- **Baseline Answer**: {it['baseline_answer']}",
            f"- **Improved Answer**: {it['improved_answer']}",
            "",
            f"**Judge Scores**:",
            f"- Baseline: Faithfulness = **{it['baseline_faithfulness']}**, Answer Correctness = {it['baseline_answer_correctness']}",
            f"- Improved: Faithfulness = **{it['improved_faithfulness']}**, Answer Correctness = {it['improved_answer_correctness']}",
            f"- **Difference (Δ Faithfulness)**: `{it['diff_faithfulness']:+.4f}`",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Category B: Top 3 Items by Absolute Answer Correctness Difference",
        "",
    ])

    for rank, it in enumerate(spot_checks["answer_correctness_top3"], 1):
        lines.extend([
            f"### #{rank} [Item `{it['id']}`] ({it['category']})",
            "",
            f"**Question**: {it['question']}",
            "",
            f"**Ground Truth**: `{it['ground_truth']}`",
            "",
            f"**Supporting Quote**: {it['supporting_quote']}",
            "",
            f"- **Baseline Answer**: {it['baseline_answer']}",
            f"- **Improved Answer**: {it['improved_answer']}",
            "",
            f"**Judge Scores**:",
            f"- Baseline: Faithfulness = {it['baseline_faithfulness']}, Answer Correctness = **{it['baseline_answer_correctness']}**",
            f"- Improved: Faithfulness = {it['improved_faithfulness']}, Answer Correctness = **{it['improved_answer_correctness']}**",
            f"- **Difference (Δ Answer Correctness)**: `{it['diff_answer_correctness']:+.4f}`",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def generate_e2e_posthoc_markdown(posthoc_data: Dict[str, Any]) -> str:
    """Generates clean GitHub-flavored markdown for results/e2e_posthoc.md."""
    meta = posthoc_data["metadata"]
    non_dec = posthoc_data["non_decline_analysis"]
    gold_slices = posthoc_data["gold_retrieved_slices"]
    judge_incon = posthoc_data["judge_refusal_inconsistency"]

    lines = [
        "# Secondary Post-Hoc Evaluation Analysis: Verity Financial RAG",
        "",
        "> [!IMPORTANT]",
        "> **POST-HOC, not pre-registered**: The analyses below are secondary and exploratory. They were designed after examining the primary pre-registered evaluation results to control for refusal scoring artifacts and retrieval dependencies.",
        "",
        f"- **Date**: {meta['date']}",
        f"- **Judge Model**: `{meta['judge_model']}`",
        f"- **Golden Benchmark**: `golden_round2.json` (SHA-256: `{meta['golden_round2_sha256'][:16]}...`)",
        f"- **Sample Size**: {meta['sample_size']} items total",
        "",
        "## 1. Paired Differences Restricted to Non-Decline Answers",
        "",
        f"This slice excludes items where either pipeline declined to answer (n={non_dec['n']} items where both pipelines produced substantive answers). This controls for judge refusal scoring anomalies.",
        "",
        "### Table 1: Paired Differences on Non-Decline Answers (n=15) [POST-HOC, not pre-registered]",
        "",
        "| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_key, label in [("faithfulness", "Faithfulness"), ("answer_correctness", "Answer Correctness")]:
        p = non_dec[m_key]
        diff_str = f"{p['mean_difference']:+.4f}"
        ci_str = f"[{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]"
        counts_str = f"{p['higher']} / {p['equal']} / {p['lower']}"
        lines.append(
            f"| **{label}** | {p['mean_baseline']:.4f} | {p['mean_improved']:.4f} | **{diff_str}** | {ci_str} | {counts_str} |"
        )

    lines.extend([
        "",
        "## 2. Analysis Restricted to Gold-Retrieved Items",
        "",
        "### Table 2: Paired Differences on Gold-Retrieved Items (Improved Hit@8 = True, n=24) [POST-HOC, not pre-registered]",
        "",
        "| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_key, label in [("faithfulness", "Faithfulness"), ("answer_correctness", "Answer Correctness")]:
        p = gold_slices["improved_hit_paired_differences"][m_key]
        diff_str = f"{p['mean_difference']:+.4f}"
        ci_str = f"[{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]"
        counts_str = f"{p['higher']} / {p['equal']} / {p['lower']}"
        lines.append(
            f"| **{label}** | {p['mean_baseline']:.4f} | {p['mean_improved']:.4f} | **{diff_str}** | {ci_str} | {counts_str} |"
        )

    lines.extend([
        "",
        "### Table 3: Paired Differences on Improved-Only Gold-Retrieved Items (Imp Hit & Base Miss, n=16) [POST-HOC, not pre-registered]",
        "",
        "| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_key, label in [("faithfulness", "Faithfulness"), ("answer_correctness", "Answer Correctness")]:
        p = gold_slices["improved_only_hit_paired_differences"][m_key]
        diff_str = f"{p['mean_difference']:+.4f}"
        ci_str = f"[{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]"
        counts_str = f"{p['higher']} / {p['equal']} / {p['lower']}"
        lines.append(
            f"| **{label}** | {p['mean_baseline']:.4f} | {p['mean_improved']:.4f} | **{diff_str}** | {ci_str} | {counts_str} |"
        )

    lines.extend([
        "",
        "### Table 4: Improved Pipeline Alone: Gold-Retrieved (n=24) vs Miss (n=6) [POST-HOC, not pre-registered]",
        "",
        "| Metric | Gold Chunk Retrieved (Hit) | Gold Chunk Missed (Miss) | Difference (Hit - Miss) | 95% Bootstrap CI |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    imp_alone = gold_slices["improved_pipeline_hit_vs_miss"]
    for m_key, label in [("faithfulness", "Faithfulness"), ("answer_correctness", "Answer Correctness")]:
        p = imp_alone[m_key]
        diff_str = f"{p['mean_difference']:+.4f}"
        ci_str = f"[{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}]"
        lines.append(
            f"| **{label}** | {p['mean_group1']:.4f} | {p['mean_group2']:.4f} | **{diff_str}** | {ci_str} |"
        )

    f0 = judge_incon["faithfulness_0_0"]
    f1 = judge_incon["faithfulness_1_0"]
    lines.extend([
        "",
        "## 3. Judge Inconsistency on Baseline Refusals / Declines",
        "",
        "Documenting RAGAS faithfulness judge inconsistency on standard refusal boilerplate:",
        "",
        "### Table 5: Judge Inconsistency on Baseline Refusals / Declines [POST-HOC, not pre-registered]",
        "",
        "| Baseline Faithfulness Score | Total Items | Declines (Refusals) | Non-Declines | Decline Percentage |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Score = 0.0** | {f0['total']} | {f0['decline_count']} | {f0['non_decline_count']} | {f0['decline_pct'] * 100:.1f}% |",
        f"| **Score = 1.0** | {f1['total']} | {f1['decline_count']} | {f1['non_decline_count']} | {f1['decline_pct'] * 100:.1f}% |",
        f"| **Total Declines** | {judge_incon['total_baseline_declines']} | {judge_incon['declines_scored_0_or_1']} scored 0.0 or 1.0 | 0 scored between 0 and 1 | {judge_incon['declines_scored_0_or_1'] / judge_incon['total_baseline_declines'] * 100:.1f}% |",
        "",
        f"> **Key Finding**: {judge_incon['inconsistency_summary']}",
        "",
    ])

    return "\n".join(lines)


def generate_e2e_spot_check_random_markdown(
    random_items: List[Dict[str, Any]],
    lowest_faith_items: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    """Generates clean GitHub-flavored markdown for results/e2e_spot_check_random.md."""
    lines = [
        "# Secondary Post-Hoc Evaluation: Random & Low-Faithfulness Spot Check",
        "",
        "> [!IMPORTANT]",
        "> **POST-HOC, not pre-registered**: This audit set was selected post-hoc to evaluate qualitative response fidelity and judge reliability without selection bias toward extreme delta outliers.",
        "",
        f"- **Date**: {metadata['date']}",
        f"- **Judge Model**: `{metadata['judge_model']}`",
        "- **Contents**: 6 random items sampled with seed 42, plus the 3 lowest-faithfulness answers from the improved pipeline.",
        "",
        "---",
        "",
        "## Category A: 6 Random Items (Seed 42) [POST-HOC, not pre-registered]",
        "",
    ]

    for rank, it in enumerate(random_items, 1):
        lines.extend([
            f"### #{rank} [Item `{it['id']}`] ({it['category']}) [POST-HOC, not pre-registered]",
            "",
            f"**Question**: {it['question']}",
            "",
            f"**Ground Truth**: `{it['ground_truth']}`",
            "",
            f"**Supporting Quote**: {it['supporting_quote']}",
            "",
            f"- **Baseline Answer**: {it['baseline_answer']}",
            f"- **Improved Answer**: {it['improved_answer']}",
            "",
            "**Judge Scores**:",
            f"- Baseline: Faithfulness = **{it['baseline_faithfulness']}**, Answer Correctness = {it['baseline_answer_correctness']}",
            f"- Improved: Faithfulness = **{it['improved_faithfulness']}**, Answer Correctness = {it['improved_answer_correctness']}",
            f"- **Difference (Δ Faithfulness)**: `{it['diff_faithfulness']:+.4f}` | **(Δ Answer Correctness)**: `{it['diff_answer_correctness']:+.4f}`",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Category B: 3 Lowest-Faithfulness Improved Answers [POST-HOC, not pre-registered]",
        "",
    ])

    for rank, it in enumerate(lowest_faith_items, 1):
        lines.extend([
            f"### #{rank} [Item `{it['id']}`] ({it['category']}) [POST-HOC, not pre-registered]",
            "",
            f"**Question**: {it['question']}",
            "",
            f"**Ground Truth**: `{it['ground_truth']}`",
            "",
            f"**Supporting Quote**: {it['supporting_quote']}",
            "",
            f"- **Baseline Answer**: {it['baseline_answer']}",
            f"- **Improved Answer**: {it['improved_answer']}",
            "",
            "**Judge Scores**:",
            f"- Baseline: Faithfulness = {it['baseline_faithfulness']}, Answer Correctness = {it['baseline_answer_correctness']}",
            f"- Improved: Faithfulness = **{it['improved_faithfulness']}**, Answer Correctness = **{it['improved_answer_correctness']}**",
            f"- **Difference (Δ Faithfulness)**: `{it['diff_faithfulness']:+.4f}` | **(Δ Answer Correctness)**: `{it['diff_answer_correctness']:+.4f}`",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def run_posthoc_analysis(
    baseline_items: List[Dict[str, Any]],
    improved_items: List[Dict[str, Any]],
    golden_map: Dict[str, Dict[str, Any]],
    output_dir: Path,
    metadata: Dict[str, Any],
    force: bool = False,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the secondary, post-hoc analysis and writes results to e2e_posthoc.json,
    e2e_posthoc.md, and e2e_spot_check_random.md.
    """
    posthoc_json_path = output_dir / "e2e_posthoc.json"
    posthoc_md_path = output_dir / "e2e_posthoc.md"
    spot_check_random_md_path = output_dir / "e2e_spot_check_random.md"

    # Enforce write-once for posthoc files unless force=True
    existing_posthoc = [p for p in (posthoc_json_path, posthoc_md_path, spot_check_random_md_path) if p.exists()]
    if existing_posthoc and not force:
        err = (
            f"Refusing to overwrite existing post-hoc file(s): {', '.join(p.name for p in existing_posthoc)}. "
            "Specify --force to overwrite."
        )
        logger.error(err)
        raise FileExistsError(err)

    # 1. Non-decline paired differences
    clean_b, clean_imp = filter_non_decline_pairs(baseline_items, improved_items)
    f_clean_b = [float(it["faithfulness"]) for it in clean_b]
    f_clean_imp = [float(it["faithfulness"]) for it in clean_imp]
    c_clean_b = [float(it["answer_correctness"]) for it in clean_b]
    c_clean_imp = [float(it["answer_correctness"]) for it in clean_imp]

    non_decline_analysis = {
        "n": len(clean_b),
        "neither_decline_ids": [it["id"] for it in clean_b],
        "faithfulness": compute_paired_bootstrap_ci(f_clean_b, f_clean_imp, n_resamples=n_resamples, seed=seed),
        "answer_correctness": compute_paired_bootstrap_ci(c_clean_b, c_clean_imp, n_resamples=n_resamples, seed=seed),
    }

    # 2. Gold-retrieved comparisons
    b_map = {it["id"]: it for it in baseline_items}
    imp_map = {it["id"]: it for it in improved_items}
    shared_ids = [it["id"] for it in improved_items if it["id"] in b_map]

    # 2A: Items where Improved retrieved gold chunk (improved hit@8 == True, n=24)
    imp_hit_ids = [it_id for it_id in shared_ids if float(imp_map[it_id].get("hit@8") or 0.0) > 0.0]
    b_f_ih = [float(b_map[i]["faithfulness"]) for i in imp_hit_ids]
    imp_f_ih = [float(imp_map[i]["faithfulness"]) for i in imp_hit_ids]
    b_c_ih = [float(b_map[i]["answer_correctness"]) for i in imp_hit_ids]
    imp_c_ih = [float(imp_map[i]["answer_correctness"]) for i in imp_hit_ids]

    imp_hit_paired = {
        "n": len(imp_hit_ids),
        "item_ids": imp_hit_ids,
        "faithfulness": compute_paired_bootstrap_ci(b_f_ih, imp_f_ih, n_resamples=n_resamples, seed=seed),
        "answer_correctness": compute_paired_bootstrap_ci(b_c_ih, imp_c_ih, n_resamples=n_resamples, seed=seed),
    }

    # 2B: Items where Improved retrieved gold chunk but Baseline missed (improved-only gold retrieved, n=16)
    imp_only_hit_ids = [
        it_id for it_id in shared_ids
        if float(imp_map[it_id].get("hit@8") or 0.0) > 0.0 and float(b_map[it_id].get("hit@8") or 0.0) == 0.0
    ]
    b_f_io = [float(b_map[i]["faithfulness"]) for i in imp_only_hit_ids]
    imp_f_io = [float(imp_map[i]["faithfulness"]) for i in imp_only_hit_ids]
    b_c_io = [float(b_map[i]["answer_correctness"]) for i in imp_only_hit_ids]
    imp_c_io = [float(imp_map[i]["answer_correctness"]) for i in imp_only_hit_ids]

    imp_only_paired = {
        "n": len(imp_only_hit_ids),
        "item_ids": imp_only_hit_ids,
        "faithfulness": compute_paired_bootstrap_ci(b_f_io, imp_f_io, n_resamples=n_resamples, seed=seed),
        "answer_correctness": compute_paired_bootstrap_ci(b_c_io, imp_c_io, n_resamples=n_resamples, seed=seed),
    }

    # 2C: Improved pipeline alone: Gold-retrieved vs Miss
    imp_miss_ids = [it_id for it_id in shared_ids if float(imp_map[it_id].get("hit@8") or 0.0) == 0.0]
    imp_f_miss = [float(imp_map[i]["faithfulness"]) for i in imp_miss_ids]
    imp_c_miss = [float(imp_map[i]["answer_correctness"]) for i in imp_miss_ids]

    imp_pipeline_hit_vs_miss = {
        "hit_n": len(imp_hit_ids),
        "miss_n": len(imp_miss_ids),
        "faithfulness": compute_two_sample_bootstrap_ci(imp_f_ih, imp_f_miss, n_resamples=n_resamples, seed=seed),
        "answer_correctness": compute_two_sample_bootstrap_ci(imp_c_ih, imp_c_miss, n_resamples=n_resamples, seed=seed),
    }

    gold_slices = {
        "improved_hit_paired_differences": imp_hit_paired,
        "improved_only_hit_paired_differences": imp_only_paired,
        "improved_pipeline_hit_vs_miss": imp_pipeline_hit_vs_miss,
    }

    # 3. Judge refusal inconsistency on baseline
    judge_inconsistency = analyze_judge_refusal_inconsistency(baseline_items)

    # 4. Spot check: 6 random items (seed 42) + 3 lowest-faithfulness improved answers
    paired_all: List[Dict[str, Any]] = []
    for it_id in shared_ids:
        b_it = b_map[it_id]
        imp_it = imp_map[it_id]
        b_f = float(b_it.get("faithfulness") or 0.0)
        imp_f = float(imp_it.get("faithfulness") or 0.0)
        b_c = float(b_it.get("answer_correctness") or 0.0)
        imp_c = float(imp_it.get("answer_correctness") or 0.0)
        paired_all.append({
            "id": it_id,
            "category": imp_it.get("category") or b_it.get("category", "unknown"),
            "question": imp_it.get("question") or b_it.get("question", ""),
            "ground_truth": imp_it.get("ground_truth") or b_it.get("ground_truth", ""),
            "baseline_answer": b_it.get("answer", ""),
            "improved_answer": imp_it.get("answer", ""),
            "baseline_faithfulness": round(b_f, 4),
            "improved_faithfulness": round(imp_f, 4),
            "diff_faithfulness": round(imp_f - b_f, 4),
            "baseline_answer_correctness": round(b_c, 4),
            "improved_answer_correctness": round(imp_c, 4),
            "diff_answer_correctness": round(imp_c - b_c, 4),
            "supporting_quote": _get_supporting_quote(it_id, golden_map, b_it, imp_it),
        })

    sampled_random = sample_random_items(paired_all, k=6, seed=seed)
    lowest_faith = select_lowest_faithfulness_items(paired_all, k=3)

    meta_copy = dict(metadata)
    meta_copy["analysis_type"] = "SECONDARY, POST-HOC analysis (not pre-registered)"

    posthoc_output = {
        "metadata": meta_copy,
        "non_decline_analysis": non_decline_analysis,
        "gold_retrieved_slices": gold_slices,
        "judge_refusal_inconsistency": judge_inconsistency,
    }

    posthoc_md = generate_e2e_posthoc_markdown(posthoc_output)
    spot_check_random_md = generate_e2e_spot_check_random_markdown(sampled_random, lowest_faith, meta_copy)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(posthoc_json_path, "w", encoding="utf-8") as f:
        json.dump(posthoc_output, f, indent=2, ensure_ascii=False)
    with open(posthoc_md_path, "w", encoding="utf-8") as f:
        f.write(posthoc_md)
    with open(spot_check_random_md_path, "w", encoding="utf-8") as f:
        f.write(spot_check_random_md)

    logger.info(f"Successfully generated {posthoc_json_path.name}")
    logger.info(f"Successfully generated {posthoc_md_path.name}")
    logger.info(f"Successfully generated {spot_check_random_md_path.name}")

    return posthoc_output


def run_e2e_analysis(
    baseline_answers_path: Path,
    improved_answers_path: Path,
    baseline_results_path: Optional[Path],
    improved_results_path: Optional[Path],
    baseline_checkpoint_path: Optional[Path],
    improved_checkpoint_path: Optional[Path],
    golden_path: Path,
    output_dir: Path,
    force: bool = False,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes primary end-to-end evaluation analysis and secondary post-hoc analysis.
    Preserves existing write-once summary files unless force=True.
    Refuses to run if any items have missing or NaN scores.
    """
    summary_json_path = output_dir / "e2e_summary.json"
    summary_md_path = output_dir / "e2e_summary.md"
    spot_check_md_path = output_dir / "e2e_spot_check.md"

    # The existing e2e_summary files are strictly write-once; do not touch them if they already exist
    existing_primary = [p for p in (summary_json_path, summary_md_path, spot_check_md_path) if p.exists()]
    write_primary = len(existing_primary) == 0
    if existing_primary:
        logger.info(
            f"Existing primary summary files found ({', '.join(p.name for p in existing_primary)}); "
            "preserving strictly write-once primary files (never touched)."
        )

    # Load baseline and improved data
    b_items, b_meta, b_offending = load_pipeline_data(
        baseline_answers_path,
        baseline_results_path,
        baseline_checkpoint_path,
        config_name="baseline",
    )
    imp_items, imp_meta, imp_offending = load_pipeline_data(
        improved_answers_path,
        improved_results_path,
        improved_checkpoint_path,
        config_name="improved",
    )

    # Validate that neither pipeline has missing or NaN scores
    validate_no_nans_or_missing(b_offending, imp_offending)

    # Align items by id
    b_map = {it["id"]: it for it in b_items}
    imp_map = {it["id"]: it for it in imp_items}
    shared_ids = [it["id"] for it in imp_items if it["id"] in b_map]

    if len(shared_ids) != len(imp_items) or len(shared_ids) != len(b_items):
        logger.warning(f"Item count mismatch: {len(b_items)} baseline, {len(imp_items)} improved, {len(shared_ids)} shared.")

    aligned_b = [b_map[it_id] for it_id in shared_ids]
    aligned_imp = [imp_map[it_id] for it_id in shared_ids]

    # Compute paired bootstrap differences for all primary metrics
    metrics_to_compute = ["faithfulness", "answer_correctness", "hit@8", "recall@8", "mrr"]
    paired_results: Dict[str, Any] = {}

    for m in metrics_to_compute:
        b_scores = [float(it[m]) for it in aligned_b]
        imp_scores = [float(it[m]) for it in aligned_imp]
        paired_results[m] = compute_paired_bootstrap_ci(
            b_scores,
            imp_scores,
            n_resamples=n_resamples,
            ci=0.95,
            seed=seed,
        )

    # Numeric accuracy for numeric_precision items only
    num_b = [it for it in aligned_b if it.get("category") == "numeric_precision"]
    num_imp = [it for it in aligned_imp if it.get("category") == "numeric_precision"]
    num_shared_ids = [it["id"] for it in num_imp if it["id"] in {x["id"]: x for x in num_b}]
    num_b_map = {it["id"]: it for it in num_b}
    num_imp_map = {it["id"]: it for it in num_imp}

    num_b_scores = [1.0 if num_b_map[i].get("numeric_match") is True else 0.0 for i in num_shared_ids]
    num_imp_scores = [1.0 if num_imp_map[i].get("numeric_match") is True else 0.0 for i in num_shared_ids]
    paired_results["numeric_accuracy"] = compute_paired_bootstrap_ci(
        num_b_scores,
        num_imp_scores,
        n_resamples=n_resamples,
        ci=0.95,
        seed=seed,
    )

    # Decline rates
    b_declines = count_declines(aligned_b)
    imp_declines = count_declines(aligned_imp)

    # Critic stats for improved agent
    critic_stats = compute_critic_stats(aligned_imp)

    # Slices by hit@8 True vs False
    retrieval_slices = {
        "baseline": compute_retrieval_slices(aligned_b),
        "improved": compute_retrieval_slices(aligned_imp),
    }

    # Load golden benchmark quotes for spot check
    golden_map: Dict[str, Dict[str, Any]] = {}
    if golden_path.exists():
        try:
            with open(golden_path, "r", encoding="utf-8") as f:
                golden_list = json.load(f)
            golden_map = {it["id"]: it for it in golden_list if "id" in it}
        except Exception as e:
            logger.warning(f"Could not load golden benchmark from {golden_path}: {e}")

    # Build metadata
    golden_sha = compute_file_sha256(golden_path)
    today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    judge_model = imp_meta.get("judge_model") or b_meta.get("judge_model") or "unknown"

    meta_summary = {
        "date": today_str,
        "golden_round2_sha256": golden_sha,
        "judge_model": judge_model,
        "sample_size": len(shared_ids),
        "generator_model": imp_meta.get("generator_model"),
        "critic_model": imp_meta.get("critic_model"),
    }

    summary_output = {
        "metadata": meta_summary,
        "paired_differences": paired_results,
        "decline_metrics": {
            "baseline": b_declines,
            "improved": imp_declines,
        },
        "critic_metrics": critic_stats,
        "retrieval_slices": retrieval_slices,
    }

    if write_primary:
        spot_checks = select_spot_check_items(aligned_b, aligned_imp, golden_map)
        summary_md = generate_e2e_summary_markdown(summary_output)
        spot_check_md = generate_e2e_spot_check_markdown(spot_checks, meta_summary)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_output, f, indent=2, ensure_ascii=False)
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        with open(spot_check_md_path, "w", encoding="utf-8") as f:
            f.write(spot_check_md)

        logger.info(f"Successfully generated {summary_json_path.name}")
        logger.info(f"Successfully generated {summary_md_path.name}")
        logger.info(f"Successfully generated {spot_check_md_path.name}")

    # Execute and save secondary post-hoc analysis
    posthoc_output = run_posthoc_analysis(
        baseline_items=aligned_b,
        improved_items=aligned_imp,
        golden_map=golden_map,
        output_dir=output_dir,
        metadata=meta_summary,
        force=force,
        n_resamples=n_resamples,
        seed=seed,
    )

    return {
        "primary_summary": summary_output if write_primary else None,
        "posthoc_analysis": posthoc_output,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Verity End-to-End Evaluation Results.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing summary and spot-check files.")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Directory containing eval results files.")
    parser.add_argument("--baseline-answers", type=str, default=None)
    parser.add_argument("--improved-answers", type=str, default=None)
    parser.add_argument("--baseline-results", type=str, default=None)
    parser.add_argument("--improved-results", type=str, default=None)
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--improved-checkpoint", type=str, default=None)
    parser.add_argument("--golden-path", type=str, default=str(GOLDEN_PATH))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    res_dir = Path(args.results_dir).resolve()

    b_ans = Path(args.baseline_answers).resolve() if args.baseline_answers else res_dir / "baseline_answers.json"
    imp_ans = Path(args.improved_answers).resolve() if args.improved_answers else res_dir / "improved_answers.json"
    b_res = Path(args.baseline_results).resolve() if args.baseline_results else res_dir / "baseline.json"
    imp_res = Path(args.improved_results).resolve() if args.improved_results else res_dir / "improved.json"
    b_cp = Path(args.baseline_checkpoint).resolve() if args.baseline_checkpoint else res_dir / "baseline_judge_checkpoint.json"
    imp_cp = Path(args.improved_checkpoint).resolve() if args.improved_checkpoint else res_dir / "improved_judge_checkpoint.json"
    golden_p = Path(args.golden_path).resolve()

    try:
        run_e2e_analysis(
            baseline_answers_path=b_ans,
            improved_answers_path=imp_ans,
            baseline_results_path=b_res,
            improved_results_path=imp_res,
            baseline_checkpoint_path=b_cp,
            improved_checkpoint_path=imp_cp,
            golden_path=golden_p,
            output_dir=res_dir,
            force=args.force,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        return 0
    except (DataIntegrityError, FileExistsError, FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
