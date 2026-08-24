"""
Category Breakdown Analyzer for Verity Evaluation Set.

Groups evaluation results from baseline.json and improved.json by question category
(simple_lookup, multi_hop, numeric_precision, unanswerable) from golden_set.json.

Computes mean RAGAS metrics per category, formats side-by-side comparison tables,
performs qualitative comparison of unanswerable queries, analyzes Critic retries & score impact,
and saves results to backend/eval/results/category_breakdown.json.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Paths setup
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
IMPROVED_PATH = RESULTS_DIR / "improved.json"
OUTPUT_BREAKDOWN_PATH = RESULTS_DIR / "category_breakdown.json"


def map_questions_to_indices(results, golden_set):
    """Maps a list of result items to their 0-indexed position in golden_set."""
    golden_questions = [g["question"] for g in golden_set]
    used_indices = set()
    indices = []

    for res in results:
        q_text = res.get("question", "")
        matched_idx = None
        for idx, g_q in enumerate(golden_questions):
            if g_q == q_text and idx not in used_indices:
                matched_idx = idx
                used_indices.add(idx)
                break
        if matched_idx is None:
            for idx, g_q in enumerate(golden_questions):
                if g_q == q_text:
                    matched_idx = idx
                    break
        indices.append(matched_idx if matched_idx is not None else -1)
    return indices


def main():
    print("=" * 80)
    print("VERITY RAGAS EVALUATION - CATEGORY BREAKDOWN ANALYSIS")
    print("=" * 80)

    # 1. Load golden set
    if not GOLDEN_SET_PATH.exists():
        print(f"Error: Golden set file not found at {GOLDEN_SET_PATH}")
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    # 2. Build index -> category mapping
    categories = [item.get("category", "simple_lookup") for item in golden_set]

    # 3. Load baseline and improved results
    if not BASELINE_PATH.exists() or not IMPROVED_PATH.exists():
        print(f"Error: Evaluation results missing ({BASELINE_PATH.name} or {IMPROVED_PATH.name})")
        sys.exit(1)

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(IMPROVED_PATH, "r", encoding="utf-8") as f:
        improved_data = json.load(f)

    baseline_results = baseline_data.get("per_question_results", [])
    improved_results = improved_data.get("per_question_results", [])

    # Map to golden set indices
    b_indices = map_questions_to_indices(baseline_results, golden_set)
    i_indices = map_questions_to_indices(improved_results, golden_set)

    # Group by category
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    cat_grouped = defaultdict(list)

    for idx, g_item in enumerate(golden_set):
        cat = g_item.get("category", "simple_lookup")

        # Find corresponding baseline & improved results
        b_res = None
        if idx in b_indices:
            b_res = baseline_results[b_indices.index(idx)]

        i_res = None
        if idx in i_indices:
            i_res = improved_results[i_indices.index(idx)]

        cat_grouped[cat].append({
            "golden_index": idx,
            "question": g_item["question"],
            "ground_truth": g_item["ground_truth"],
            "baseline": b_res,
            "improved": i_res,
        })

    # 4. Compute metrics per category
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

            cat_metrics[m] = {
                "baseline": round(b_mean, 4),
                "improved": round(i_mean, 4),
                "delta": round(delta, 4),
            }

        category_summary[cat] = {
            "count": count,
            "metrics": cat_metrics,
        }

    # Print Category Tables
    print("\n" + "=" * 80)
    print("1. SUMMARY METRICS BY QUESTION CATEGORY")
    print("=" * 80)

    for m in metrics:
        metric_title = m.replace("_", " ").title()
        print(f"\n--- {metric_title} ---")
        print(f"{'Category':<20} | {'Count':<6} | {'Baseline':<10} | {'Improved':<10} | {'Delta':<10}")
        print("-" * 65)
        for cat in ["simple_lookup", "multi_hop", "numeric_precision", "unanswerable"]:
            if cat in category_summary:
                c_info = category_summary[cat]
                m_info = c_info["metrics"][m]
                b_str = f"{m_info['baseline']:.4f}"
                i_str = f"{m_info['improved']:.4f}"
                d_str = f"{m_info['delta']:+.4f}"
                print(f"{cat:<20} | {c_info['count']:<6} | {b_str:<10} | {i_str:<10} | {d_str:<10}")

    # Print Detailed Per-Category Breakdown
    print("\n" + "=" * 80)
    print("2. DETAILED METRIC BREAKDOWN PER CATEGORY")
    print("=" * 80)
    for cat in ["simple_lookup", "multi_hop", "numeric_precision", "unanswerable"]:
        if cat not in category_summary:
            continue
        c_info = category_summary[cat]
        print(f"\nCategory: {cat.upper()} ({c_info['count']} questions)")
        print(f"{'Metric':<22} | {'Baseline':<10} | {'Improved':<10} | {'Delta':<10}")
        print("-" * 60)
        for m in metrics:
            m_info = c_info["metrics"][m]
            print(f"{m.replace('_', ' ').title():<22} | {m_info['baseline']:<10.4f} | {m_info['improved']:<10.4f} | {m_info['delta']:+.4f}")

    # 5. Qualitative evaluation of Unanswerable questions
    print("\n" + "=" * 80)
    print("3. QUALITATIVE EVALUATION: UNANSWERABLE QUESTIONS")
    print("=" * 80)

    unanswerable_items = cat_grouped.get("unanswerable", [])
    unanswerable_eval_data = []

    for item in unanswerable_items:
        q_idx = item["golden_index"]
        q_text = item["question"]
        gt = item["ground_truth"]
        b_ans = item["baseline"].get("answer", "") if item["baseline"] else "N/A"
        i_ans = item["improved"].get("answer", "") if item["improved"] else "N/A"

        print(f"\n[Question Index {q_idx}]")
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
            "golden_index": q_idx,
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

            idx = map_questions_to_indices([item], golden_set)[0]
            cat = golden_set[idx].get("category", "simple_lookup") if idx != -1 else "unknown"

            # Check if a retry occurred (retry_count > 0 or initial grounded == False)
            if r_count > 0 or (initial_verdict and not initial_verdict.get("grounded", True)):
                retries_by_category[cat] += 1

                # Find matching baseline item for score comparison
                b_res = baseline_results[b_indices.index(idx)] if idx in b_indices else {}

                b_faith = b_res.get("faithfulness", 0.0) if b_res else 0.0
                i_faith = item.get("faithfulness", 0.0)
                b_rel = b_res.get("answer_relevancy", 0.0) if b_res else 0.0
                i_rel = item.get("answer_relevancy", 0.0)

                unsupported = initial_verdict.get("unsupported_claims", []) or final_verdict.get("unsupported_claims", [])

                retried_questions_detail.append({
                    "golden_index": idx,
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
                print(f"\n[Q{q_det['golden_index']}] Category: {q_det['category']} | Retries: {q_det['retry_count']}")
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

    # 7. Save category breakdown JSON
    output_data = {
        "category_summary": category_summary,
        "unanswerable_qualitative_evaluation": unanswerable_eval_data,
        "critic_retry_logging": {
            "retry_data_available": has_retry_data,
            "total_retried_questions": len(retried_questions_detail),
            "retries_by_category": dict(retries_by_category),
            "retried_questions_detail": retried_questions_detail,
        },
    }

    with open(OUTPUT_BREAKDOWN_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Saved complete category breakdown to: {OUTPUT_BREAKDOWN_PATH.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
