"""
Category Breakdown Analyzer for Verity Evaluation Set.

Groups evaluation results from baseline.json and improved.json by question category
(simple_lookup, multi_hop, numeric_precision, unanswerable) using the stable 'id'
field from golden_set.json.

Features:
- Computes mean RAGAS metrics per category.
- Calculates bootstrap 95% confidence intervals (1000 resamples) for aggregate metrics
  and paired per-category deltas (improved - baseline).
- Formats side-by-side comparison tables with 95% CIs.
- Performs qualitative comparison of unanswerable queries.
- Analyzes Critic retries & score impact.
- Generates failure-case dump to backend/eval/results/failures.md for all questions with faithfulness < 0.8.
- Saves results to backend/eval/results/category_breakdown.json.
"""

import datetime
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Paths setup
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
IMPROVED_PATH = RESULTS_DIR / "improved.json"
OUTPUT_BREAKDOWN_PATH = RESULTS_DIR / "category_breakdown.json"
FAILURES_PATH = RESULTS_DIR / "failures.md"


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes a non-parametric bootstrap confidence interval for the sample mean."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    boot_means: List[float] = []

    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lower_idx = int(((1.0 - ci) / 2.0) * n_resamples)
    upper_idx = int((1.0 - (1.0 - ci) / 2.0) * n_resamples)
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))

    return (round(boot_means[lower_idx], 4), round(boot_means[upper_idx], 4))


def bootstrap_delta_ci(
    baseline_vals: List[float],
    improved_vals: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Computes a paired bootstrap confidence interval for the mean delta (improved - baseline)."""
    if not baseline_vals or not improved_vals or len(baseline_vals) != len(improved_vals):
        return (0.0, 0.0)
    n = len(baseline_vals)
    if n == 1:
        d = improved_vals[0] - baseline_vals[0]
        return (round(d, 4), round(d, 4))

    rng = random.Random(seed)
    pairs = list(zip(baseline_vals, improved_vals))
    boot_deltas: List[float] = []

    for _ in range(n_resamples):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        mean_b = sum(p[0] for p in sample) / n
        mean_i = sum(p[1] for p in sample) / n
        boot_deltas.append(mean_i - mean_b)

    boot_deltas.sort()
    lower_idx = int(((1.0 - ci) / 2.0) * n_resamples)
    upper_idx = int((1.0 - (1.0 - ci) / 2.0) * n_resamples)
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))

    return (round(boot_deltas[lower_idx], 4), round(boot_deltas[upper_idx], 4))


def map_results_to_golden_id(
    results: List[Dict[str, Any]], golden_set: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Maps result items to golden set items using the stable 'id' field.
    If result items already contain 'id', matches directly on 'id'.
    For legacy results lacking 'id', matches 1:1 by sequential index to golden_set items.
    Returns: Dict[golden_id -> result_item]
    """
    id_map: Dict[str, Dict[str, Any]] = {}
    has_ids = any("id" in res for res in results)

    if has_ids:
        for res in results:
            res_id = res.get("id")
            if res_id:
                id_map[res_id] = res
    else:
        # Fallback for legacy result files lacking the 'id' field
        for idx, g_item in enumerate(golden_set):
            if idx < len(results):
                id_map[g_item["id"]] = results[idx]

    return id_map


def generate_failure_cases_dump(
    cat_grouped: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """
    Generates eval/results/failures.md listing every question where faithfulness < 0.8,
    including the question, answer, ground truth, and retrieved chunk IDs.
    """
    failure_entries: List[Dict[str, Any]] = []

    for cat, items in cat_grouped.items():
        for item in items:
            g_id = item["id"]
            question = item["question"]
            ground_truth = item["ground_truth"]
            company = item.get("company", "Unknown")
            fiscal_year = item.get("fiscal_year", "Unknown")
            b_res = item.get("baseline") or {}
            i_res = item.get("improved") or {}

            b_faith = b_res.get("faithfulness", 1.0)
            i_faith = i_res.get("faithfulness", 1.0)

            # Check if either baseline or improved has faithfulness < 0.8
            if (b_res and b_faith < 0.8) or (i_res and i_faith < 0.8):
                failure_entries.append({
                    "id": g_id,
                    "category": cat,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "target_chunk_id": item.get("target_chunk_id"),
                    "question": question,
                    "ground_truth": ground_truth,
                    "baseline": {
                        "faithfulness": b_faith,
                        "answer": b_res.get("answer", "N/A"),
                        "retrieved_chunk_ids": b_res.get("retrieved_chunk_ids", []),
                        "contexts": b_res.get("contexts", []),
                    } if b_res else None,
                    "improved": {
                        "faithfulness": i_faith,
                        "answer": i_res.get("answer", "N/A"),
                        "retrieved_chunk_ids": i_res.get("retrieved_chunk_ids", []),
                        "contexts": i_res.get("contexts", []),
                    } if i_res else None,
                })

    # Sort failures by id
    failure_entries.sort(key=lambda x: x["id"])

    # Write failures.md
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Verity Financial RAG - Evaluation Failure Cases (Faithfulness < 0.8)",
        "",
        f"- **Generated At**: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"- **Faithfulness Threshold**: < 0.80",
        f"- **Total Questions Flagged**: {len(failure_entries)}",
        "",
        "---",
        "",
    ]

    if not failure_entries:
        lines.append("No questions failed the faithfulness threshold (all questions scored >= 0.80).")
    else:
        for idx, entry in enumerate(failure_entries, start=1):
            lines.append(f"## {idx}. [{entry['id']}] {entry['question']}")
            lines.append(f"- **Category**: `{entry['category']}`")
            lines.append(f"- **Company**: `{entry['company']}` (FY {entry['fiscal_year']})")
            target_cid = entry.get("target_chunk_id")
            if isinstance(target_cid, list):
                target_str = ", ".join([f"`{c}`" for c in target_cid])
            elif target_cid:
                target_str = f"`{target_cid}`"
            else:
                target_str = "*None (unanswerable/unspecified)*"
            lines.append(f"- **Target Chunk ID (Golden Set)**: {target_str}")
            lines.append("")

            # Improved Details
            if entry["improved"]:
                lines.append(f"### Improved Pipeline (Faithfulness: {entry['improved']['faithfulness']:.4f})")
                chunk_ids = entry["improved"]["retrieved_chunk_ids"]
                if chunk_ids:
                    chunk_str = ", ".join([f"`{cid}`" for cid in chunk_ids])
                else:
                    chunk_str = "*None recorded*"
                lines.append(f"- **Retrieved Chunk IDs**: {chunk_str}")
                lines.append("")
                lines.append("**Generated Answer:**")
                lines.append("```")
                lines.append(entry["improved"]["answer"])
                lines.append("```")
                lines.append("")

            # Baseline Details
            if entry["baseline"]:
                lines.append(f"### Baseline Pipeline (Faithfulness: {entry['baseline']['faithfulness']:.4f})")
                chunk_ids = entry["baseline"]["retrieved_chunk_ids"]
                if chunk_ids:
                    chunk_str = ", ".join([f"`{cid}`" for cid in chunk_ids])
                else:
                    chunk_str = "*None recorded*"
                lines.append(f"- **Retrieved Chunk IDs**: {chunk_str}")
                lines.append("")
                lines.append("**Generated Answer:**")
                lines.append("```")
                lines.append(entry["baseline"]["answer"])
                lines.append("```")
                lines.append("")

            # Ground Truth
            lines.append("### Ground Truth Reference")
            lines.append("```")
            lines.append(entry["ground_truth"])
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return failure_entries


def main():
    print("=" * 80)
    print("VERITY RAGAS EVALUATION - CATEGORY BREAKDOWN ANALYSIS")
    print("=" * 80)

    # 1. Load golden set
    if not GOLDEN_SET_PATH.exists():
        print(f"Error: Golden set file not found at {GOLDEN_SET_PATH}")
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set: List[Dict[str, Any]] = json.load(f)

    print(f"Loaded {len(golden_set)} items from {GOLDEN_SET_PATH.name}")

    # Build golden_by_id lookup
    golden_by_id: Dict[str, Dict[str, Any]] = {
        item.get("id", f"golden_{i+1:02d}"): item for i, item in enumerate(golden_set)
    }

    # 2. Load baseline and improved results
    if not BASELINE_PATH.exists() or not IMPROVED_PATH.exists():
        print(f"Error: Evaluation results missing ({BASELINE_PATH.name} or {IMPROVED_PATH.name})")
        sys.exit(1)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(IMPROVED_PATH, "r", encoding="utf-8") as f:
        improved_data = json.load(f)

    baseline_results = baseline_data.get("per_question_results", [])
    improved_results = improved_data.get("per_question_results", [])

    # Map results by stable 'id'
    baseline_by_id = map_results_to_golden_id(baseline_results, golden_set)
    improved_by_id = map_results_to_golden_id(improved_results, golden_set)

    # 3. Group by category
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cat_grouped = defaultdict(list)

    for idx, g_item in enumerate(golden_set):
        g_id = g_item.get("id", f"golden_{idx+1:02d}")
        cat = g_item.get("category", "simple_lookup")

        b_res = baseline_by_id.get(g_id)
        i_res = improved_by_id.get(g_id)

        cat_grouped[cat].append({
            "id": g_id,
            "golden_index": idx,
            "company": g_item.get("company", ""),
            "fiscal_year": g_item.get("fiscal_year", ""),
            "target_chunk_id": g_item.get("chunk_id"),
            "question": g_item["question"],
            "ground_truth": g_item["ground_truth"],
            "baseline": b_res,
            "improved": i_res,
        })

    # 4. Compute metrics and bootstrap 95% CIs per category
    category_summary = {}
    for cat, items in cat_grouped.items():
        count = len(items)
        cat_metrics = {}

        for m in metrics:
            b_vals = [item["baseline"].get(m, 0.0) for item in items if item["baseline"]]
            i_vals = [item["improved"].get(m, 0.0) for item in items if item["improved"]]

            b_mean = sum(b_vals) / len(b_vals) if b_vals else 0.0
            i_mean = sum(i_vals) / len(i_vals) if i_vals else 0.0
            delta = i_mean - b_mean

            b_ci = bootstrap_ci(b_vals, n_resamples=1000, ci=0.95)
            i_ci = bootstrap_ci(i_vals, n_resamples=1000, ci=0.95)
            d_ci = bootstrap_delta_ci(b_vals, i_vals, n_resamples=1000, ci=0.95)

            cat_metrics[m] = {
                "baseline": round(b_mean, 4),
                "baseline_ci_95": [b_ci[0], b_ci[1]],
                "improved": round(i_mean, 4),
                "improved_ci_95": [i_ci[0], i_ci[1]],
                "delta": round(delta, 4),
                "delta_ci_95": [d_ci[0], d_ci[1]],
            }

        category_summary[cat] = {
            "count": count,
            "metrics": cat_metrics,
        }

    # Print Category Tables with Bootstrap 95% CIs
    print("\n" + "=" * 80)
    print("1. SUMMARY METRICS BY QUESTION CATEGORY (WITH 95% BOOTSTRAP CIs)")
    print("=" * 80)

    for m in metrics:
        metric_title = m.replace("_", " ").title()
        print(f"\n--- {metric_title} ---")
        print(f"{'Category':<18} | {'Count':<5} | {'Baseline (95% CI)':<26} | {'Improved (95% CI)':<26} | {'Delta (95% CI)':<26}")
        print("-" * 110)
        for cat in ["simple_lookup", "multi_hop", "numeric_precision", "unanswerable"]:
            if cat in category_summary:
                c_info = category_summary[cat]
                m_info = c_info["metrics"][m]
                b_str = f"{m_info['baseline']:.4f} [{m_info['baseline_ci_95'][0]:.4f}, {m_info['baseline_ci_95'][1]:.4f}]"
                i_str = f"{m_info['improved']:.4f} [{m_info['improved_ci_95'][0]:.4f}, {m_info['improved_ci_95'][1]:.4f}]"
                d_str = f"{m_info['delta']:+.4f} [{m_info['delta_ci_95'][0]:+.4f}, {m_info['delta_ci_95'][1]:+.4f}]"
                print(f"{cat:<18} | {c_info['count']:<5} | {b_str:<26} | {i_str:<26} | {d_str:<26}")

    # Print Detailed Per-Category Breakdown
    print("\n" + "=" * 80)
    print("2. DETAILED METRIC BREAKDOWN PER CATEGORY WITH 95% CIs")
    print("=" * 80)
    for cat in ["simple_lookup", "multi_hop", "numeric_precision", "unanswerable"]:
        if cat not in category_summary:
            continue
        c_info = category_summary[cat]
        print(f"\nCategory: {cat.upper()} ({c_info['count']} questions)")
        print(f"{'Metric':<20} | {'Baseline (95% CI)':<26} | {'Improved (95% CI)':<26} | {'Delta (95% CI)':<26}")
        print("-" * 105)
        for m in metrics:
            m_info = c_info["metrics"][m]
            b_str = f"{m_info['baseline']:.4f} [{m_info['baseline_ci_95'][0]:.4f}, {m_info['baseline_ci_95'][1]:.4f}]"
            i_str = f"{m_info['improved']:.4f} [{m_info['improved_ci_95'][0]:.4f}, {m_info['improved_ci_95'][1]:.4f}]"
            d_str = f"{m_info['delta']:+.4f} [{m_info['delta_ci_95'][0]:+.4f}, {m_info['delta_ci_95'][1]:+.4f}]"
            print(f"{m.replace('_', ' ').title():<20} | {b_str:<26} | {i_str:<26} | {d_str:<26}")

    # 5. Qualitative evaluation of Unanswerable questions
    print("\n" + "=" * 80)
    print("3. QUALITATIVE EVALUATION: UNANSWERABLE QUESTIONS")
    print("=" * 80)

    unanswerable_items = cat_grouped.get("unanswerable", [])
    unanswerable_eval_data = []

    for item in unanswerable_items:
        g_id = item["id"]
        q_idx = item["golden_index"]
        q_text = item["question"]
        gt = item["ground_truth"]
        b_ans = item["baseline"].get("answer", "") if item["baseline"] else "N/A"
        i_ans = item["improved"].get("answer", "") if item["improved"] else "N/A"

        print(f"\n[{g_id}] (Index {q_idx})")
        print(f"  Question:      {q_text}")
        print(f"  Ground Truth:  {gt}")
        print(f"  Baseline Ans:  {b_ans}")
        print(f"  Improved Ans:  {i_ans}")

        # Check behavior
        b_declined = "don't have enough information" in b_ans.lower() or "insufficient" in b_ans.lower()
        i_declined = "don't have enough information" in i_ans.lower() or "insufficient" in i_ans.lower()

        status = "BOTH_CORRECTLY_DECLINED" if (b_declined and i_declined) else "BEHAVIOR_MISMATCH"
        print(f"  Behavior Assessment: {status}")

        unanswerable_eval_data.append({
            "id": g_id,
            "golden_index": q_idx,
            "company": item.get("company", ""),
            "fiscal_year": item.get("fiscal_year", ""),
            "question": q_text,
            "ground_truth": gt,
            "baseline_answer": b_ans,
            "improved_answer": i_ans,
            "baseline_declined": b_declined,
            "improved_declined": i_declined,
            "behavior_status": status,
        })

    # 6. Graph Execution State (retry_count / critic_verdict) Analysis
    print("\n" + "=" * 80)
    print("4. CRITIC RETRY ANALYSIS & SCORE IMPACT BY CATEGORY")
    print("=" * 80)

    has_retry_data = False
    retries_by_category = defaultdict(int)
    retried_questions_detail = []

    for item in improved_results:
        if "retry_count" in item or "critic_verdict" in item or "initial_critic_verdict" in item:
            has_retry_data = True
            r_count = item.get("retry_count", 0)
            initial_verdict = item.get("initial_critic_verdict", {})
            final_verdict = item.get("critic_verdict", {})

            # Locate matching golden item by id
            res_id = item.get("id")
            g_item = None
            if res_id and res_id in golden_by_id:
                g_item = golden_by_id[res_id]
            else:
                for gid, mapped_res in improved_by_id.items():
                    if mapped_res is item:
                        g_item = golden_by_id.get(gid)
                        break

            g_id = g_item.get("id", "unknown") if g_item else "unknown"
            cat = g_item.get("category", "unknown") if g_item else "unknown"

            # Check if a retry occurred (retry_count > 0 or initial grounded == False)
            if r_count > 0 or (initial_verdict and not initial_verdict.get("grounded", True)):
                retries_by_category[cat] += 1

                # Find matching baseline item for score comparison
                b_res = baseline_by_id.get(g_id, {})

                b_faith = b_res.get("faithfulness", 0.0) if b_res else 0.0
                i_faith = item.get("faithfulness", 0.0)
                b_rel = b_res.get("answer_relevancy", 0.0) if b_res else 0.0
                i_rel = item.get("answer_relevancy", 0.0)

                unsupported = initial_verdict.get("unsupported_claims", []) or final_verdict.get("unsupported_claims", [])

                retried_questions_detail.append({
                    "id": g_id,
                    "question": item.get("question", ""),
                    "category": cat,
                    "retry_count": r_count,
                    "initial_unsupported_claims": unsupported,
                    "final_grounded": final_verdict.get("grounded", False),
                    "baseline_faithfulness": round(b_faith, 4),
                    "improved_faithfulness": round(i_faith, 4),
                    "faithfulness_delta": round(i_faith - b_faith, 4),
                    "baseline_answer_relevancy": round(b_rel, 4),
                    "improved_answer_relevancy": round(i_rel, 4),
                    "answer_relevancy_delta": round(i_rel - b_rel, 4),
                })

    if has_retry_data:
        total_retries = len(retried_questions_detail)
        print(f"\nTotal Questions Triggering Retries: {total_retries} / {len(golden_set)}")
        print("\n--- Retry Concentration by Category ---")
        print(f"{'Category':<20} | {'Retried Questions Count':<25}")
        print("-" * 50)
        for cat in ["simple_lookup", "multi_hop", "numeric_precision", "unanswerable"]:
            cnt = retries_by_category.get(cat, 0)
            print(f"{cat:<20} | {cnt:<25}")

        if retried_questions_detail:
            print("\n--- Detailed Breakdown of Retried Questions & Score Impact ---")
            for q_det in retried_questions_detail:
                print(f"\n[{q_det['id']}] Category: {q_det['category']} | Retries: {q_det['retry_count']}")
                print(f"  Question:      {q_det['question']}")
                print(f"  Unsupported Claims Flagged (Pass 1): {q_det['initial_unsupported_claims']}")
                print(f"  Final Grounded Verdict:            {q_det['final_grounded']}")
                print(f"  Faithfulness:      Baseline={q_det['baseline_faithfulness']:.4f} -> Improved={q_det['improved_faithfulness']:.4f} (Delta: {q_det['faithfulness_delta']:+.4f})")
                print(f"  Answer Relevancy:  Baseline={q_det['baseline_answer_relevancy']:.4f} -> Improved={q_det['improved_answer_relevancy']:.4f} (Delta: {q_det['answer_relevancy_delta']:+.4f})")
        else:
            print("\nNo questions triggered a Critic retry (all passed on first evaluation pass).")
    else:
        print("[EVAL LOGGING GAP NOTICE]")
        print("Graph execution state logging (retry_count / critic_verdict) was NOT captured in backend/eval/results/improved.json.")

    # 7. Generate Failure-Case Dump (faithfulness < 0.8)
    print("\n" + "=" * 80)
    print("5. FAILURE-CASE DUMP (FAITHFULNESS < 0.8)")
    print("=" * 80)
    failures = generate_failure_cases_dump(cat_grouped, FAILURES_PATH)
    print(f"Flagged {len(failures)} question(s) with faithfulness < 0.80.")
    print(f"Saved complete failure-case analysis to: {FAILURES_PATH.name}")

    # 8. Save category breakdown JSON with bootstrap CIs
    output_data = {
        "category_summary": category_summary,
        "unanswerable_qualitative_evaluation": unanswerable_eval_data,
        "critic_retry_logging": {
            "retry_data_available": has_retry_data,
            "total_retried_questions": len(retried_questions_detail),
            "retries_by_category": dict(retries_by_category),
            "retried_questions_detail": retried_questions_detail,
        },
        "failure_cases": {
            "threshold": 0.8,
            "total_failures": len(failures),
            "failures_dump_path": str(FAILURES_PATH.name),
        },
    }

    with open(OUTPUT_BREAKDOWN_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Saved complete category breakdown to: {OUTPUT_BREAKDOWN_PATH.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
