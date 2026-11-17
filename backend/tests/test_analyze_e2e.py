"""
Unit tests for backend/eval/analyze_e2e.py.

Tests:
- Bootstrap helper: deterministic results, known answer, fixed seed 42.
- NaN guard: detects missing and NaN scores, reports offending IDs, raises DataIntegrityError.
- Decline counter: detects decline/refusal phrases accurately and ignores valid answers.
- Write-once & force flag behavior.
- Full end-to-end analysis execution on synthetic complete data.
"""

import json
import math
from pathlib import Path
import pytest

from backend.eval.analyze_e2e import (
    DataIntegrityError,
    compute_critic_stats,
    compute_paired_bootstrap_ci,
    compute_retrieval_slices,
    count_declines,
    is_decline_answer,
    load_pipeline_data,
    run_e2e_analysis,
    select_spot_check_items,
    validate_no_nans_or_missing,
)


def test_bootstrap_helper_known_answer_constant_shift():
    """Verifies bootstrap CI on a constant paired difference where all resamples must equal the shift."""
    b = [0.1, 0.2, 0.3, 0.4, 0.5]
    imp = [0.3, 0.4, 0.5, 0.6, 0.7]  # Constant +0.2 shift
    res = compute_paired_bootstrap_ci(b, imp, n_resamples=1000, ci=0.95, seed=42)

    assert res["mean_baseline"] == pytest.approx(0.3, abs=1e-4)
    assert res["mean_improved"] == pytest.approx(0.5, abs=1e-4)
    assert res["mean_difference"] == pytest.approx(0.2, abs=1e-4)
    assert res["ci_lower"] == pytest.approx(0.2, abs=1e-4)
    assert res["ci_upper"] == pytest.approx(0.2, abs=1e-4)
    assert res["higher"] == 5
    assert res["equal"] == 0
    assert res["lower"] == 0
    assert res["n"] == 5


def test_bootstrap_helper_fixed_seed_reproducibility():
    """Verifies that the bootstrap helper is strictly deterministic with fixed seed 42."""
    b = [0.2, 0.5, 0.7, 0.3, 0.9, 0.4]
    imp = [0.4, 0.5, 0.8, 0.6, 0.7, 0.6]

    res1 = compute_paired_bootstrap_ci(b, imp, n_resamples=5000, ci=0.95, seed=42)
    res2 = compute_paired_bootstrap_ci(b, imp, n_resamples=5000, ci=0.95, seed=42)

    assert res1["mean_difference"] == res2["mean_difference"]
    assert res1["ci_lower"] == res2["ci_lower"]
    assert res1["ci_upper"] == res2["ci_upper"]
    assert res1["higher"] == res2["higher"]
    assert res1["equal"] == res2["equal"]
    assert res1["lower"] == res2["lower"]

    # Verify counts: imp - b = [+0.2, 0.0, +0.1, +0.3, -0.2, +0.2] -> 4 higher, 1 equal, 1 lower
    assert res1["higher"] == 4
    assert res1["equal"] == 1
    assert res1["lower"] == 1
    assert res1["mean_difference"] == pytest.approx(0.1, abs=1e-4)


def test_bootstrap_helper_unequal_lengths():
    """Verifies that mismatched list lengths raise a ValueError."""
    with pytest.raises(ValueError, match="equal lengths"):
        compute_paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_bootstrap_helper_empty():
    """Verifies handling of empty input lists."""
    res = compute_paired_bootstrap_ci([], [], n_resamples=100)
    assert res["n"] == 0
    assert res["mean_difference"] == 0.0


def test_nan_guard_raises_on_offending_ids():
    """Verifies that validate_no_nans_or_missing raises DataIntegrityError with offending IDs."""
    b_offending = {"faithfulness": ["item_1", "item_3"], "answer_correctness": []}
    imp_offending = {"faithfulness": [], "answer_correctness": ["item_2"]}

    with pytest.raises(DataIntegrityError) as exc_info:
        validate_no_nans_or_missing(b_offending, imp_offending)

    err = str(exc_info.value)
    assert "CRITICAL: Evaluation data contains missing or NaN per-item scores!" in err
    assert "item_1" in err
    assert "item_3" in err
    assert "item_2" in err
    assert "faithfulness" in err
    assert "answer_correctness" in err


def test_nan_guard_passes_when_clean():
    """Verifies that validate_no_nans_or_missing does not raise when no offending IDs exist."""
    b_offending = {"faithfulness": [], "answer_correctness": []}
    imp_offending = {"faithfulness": [], "answer_correctness": []}
    # Should complete without error
    validate_no_nans_or_missing(b_offending, imp_offending)


def test_load_pipeline_data_detects_nans_and_merges_checkpoints(tmp_path):
    """Verifies that load_pipeline_data identifies NaNs, None, and uses checkpoint fallback when valid."""
    answers_path = tmp_path / "test_answers.json"
    results_path = tmp_path / "test.json"
    checkpoint_path = tmp_path / "test_checkpoint.json"

    items = [
        {"id": "q1", "answer": "Ans 1", "contexts": ["C1"], "retrieved_chunk_ids": ["c1"], "source_chunk_ids": ["c1"]},
        {"id": "q2", "answer": "Ans 2", "contexts": ["C2"], "retrieved_chunk_ids": ["c2"], "source_chunk_ids": ["c2"]},
        {"id": "q3", "answer": "Ans 3", "contexts": ["C3"], "retrieved_chunk_ids": ["c3"], "source_chunk_ids": ["c3"]},
    ]
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump({"items": items, "models": {"judge_model": "test-judge"}}, f)

    # results has NaN for q1 faithfulness, missing q2 entirely, q3 valid
    results_content = {
        "judge_model": "test-judge",
        "per_question_results": [
            {"id": "q1", "faithfulness": float("nan"), "answer_correctness": 0.8},
            {"id": "q3", "faithfulness": 0.9, "answer_correctness": 0.85},
        ],
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_content, f)

    # checkpoint has valid score for q2, but none for q1
    checkpoint_content = {
        "q2": {"faithfulness": 0.95, "answer_correctness": 0.9},
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_content, f)

    merged_items, meta, offending = load_pipeline_data(
        answers_path=answers_path,
        results_path=results_path,
        checkpoint_path=checkpoint_path,
        config_name="test_config",
    )

    assert len(merged_items) == 3
    assert meta["judge_model"] == "test-judge"

    # q1 has NaN faithfulness -> flagged in offending
    assert "q1" in offending["faithfulness"]
    assert "q1" not in offending["answer_correctness"]

    # q2 was merged from checkpoint -> valid, not offending
    assert "q2" not in offending["faithfulness"]
    assert "q2" not in offending["answer_correctness"]

    # q3 was valid in results -> not offending
    assert "q3" not in offending["faithfulness"]
    assert "q3" not in offending["answer_correctness"]


def test_decline_detection_and_counting():
    """Verifies that is_decline_answer and count_declines accurately identify refusals."""
    positive_cases = [
        "I don't have enough information to answer this question based on the retrieved SEC filings.",
        "Based on the documents, we do not have enough information to determine the 2024 revenue.",
        "The retrieved chunks contain insufficient information regarding the executive compensation.",
        "I cannot answer this question as the required 10-K sections are not present in the context.",
        "Unable to answer due to missing fiscal year data.",
        "There is not enough information to answer the user's query.",
    ]
    negative_cases = [
        "Alphabet's Google Cloud revenue was $33,088 million in 2023 [Chunk 1].",
        "Based on Microsoft's 2024 Form 10-K, capital expenditures were $55.7 billion [2].",
        "Apple reported diluted earnings per share of $6.08 in fiscal year 2024.",
        "Amazon current assets totaled $184,888 million as of December 31, 2024.",
        "",
        None,
    ]

    for text in positive_cases:
        assert is_decline_answer(text) is True, f"Failed to identify decline: {text}"

    for text in negative_cases:
        assert is_decline_answer(text) is False, f"False positive decline: {text}"

    test_items = [
        {"id": f"q_{i}", "answer": positive_cases[i % len(positive_cases)]} for i in range(4)
    ] + [
        {"id": f"q_{i+4}", "answer": negative_cases[i % len(negative_cases)]} for i in range(6)
    ]

    res = count_declines(test_items)
    assert res["total"] == 10
    assert res["decline_count"] == 4
    assert res["decline_rate"] == 0.4
    assert res["decline_ids"] == ["q_0", "q_1", "q_2", "q_3"]


def test_critic_stats_computation():
    """Verifies critic revise rate and distribution when recorded, and explicit note when missing."""
    recorded_items = [
        {"id": "q1", "retry_count": 0},
        {"id": "q2", "retry_count": 1},
        {"id": "q3", "retry_count": 0},
        {"id": "q4", "retry_count": 2},
    ]
    stats = compute_critic_stats(recorded_items)
    assert stats["recorded"] is True
    assert stats["n"] == 4
    assert stats["revised_count"] == 2  # q2 (1) and q4 (2)
    assert stats["revise_rate"] == 0.5
    assert stats["retry_distribution"] == {0: 2, 1: 1, 2: 1}
    assert stats["retry_distribution_pct"] == {0: 0.5, 1: 0.25, 2: 0.25}

    unrecorded_items = [
        {"id": "q1", "answer": "A1"},
        {"id": "q2", "answer": "A2"},
    ]
    missing_stats = compute_critic_stats(unrecorded_items)
    assert missing_stats["recorded"] is False
    assert missing_stats["revise_rate"] is None
    assert "does not record" in missing_stats["note"]


def test_retrieval_slices_computation():
    """Verifies slicing answer quality metrics by hit@8 True vs False."""
    items = [
        {"id": "q1", "hit@8": 1.0, "faithfulness": 1.0, "answer_correctness": 0.9},
        {"id": "q2", "hit@8": 1.0, "faithfulness": 0.8, "answer_correctness": 0.7},
        {"id": "q3", "hit@8": 0.0, "faithfulness": 0.2, "answer_correctness": 0.1},
    ]
    slices = compute_retrieval_slices(items)

    assert slices["hit_true"]["n"] == 2
    assert slices["hit_true"]["faithfulness"] == pytest.approx(0.9, abs=1e-4)
    assert slices["hit_true"]["answer_correctness"] == pytest.approx(0.8, abs=1e-4)

    assert slices["hit_false"]["n"] == 1
    assert slices["hit_false"]["faithfulness"] == pytest.approx(0.2, abs=1e-4)
    assert slices["hit_false"]["answer_correctness"] == pytest.approx(0.1, abs=1e-4)


def test_spot_check_selection():
    """Verifies selection of top 3 absolute differences in faithfulness and answer correctness."""
    baseline = [
        {"id": "q1", "question": "Q1", "ground_truth": "T1", "answer": "B1", "faithfulness": 0.1, "answer_correctness": 0.5},
        {"id": "q2", "question": "Q2", "ground_truth": "T2", "answer": "B2", "faithfulness": 0.9, "answer_correctness": 0.2},
        {"id": "q3", "question": "Q3", "ground_truth": "T3", "answer": "B3", "faithfulness": 0.5, "answer_correctness": 0.9},
        {"id": "q4", "question": "Q4", "ground_truth": "T4", "answer": "B4", "faithfulness": 0.6, "answer_correctness": 0.3},
    ]
    improved = [
        {"id": "q1", "question": "Q1", "ground_truth": "T1", "answer": "I1", "faithfulness": 0.9, "answer_correctness": 0.6},  # diff f=0.8, corr=0.1
        {"id": "q2", "question": "Q2", "ground_truth": "T2", "answer": "I2", "faithfulness": 0.9, "answer_correctness": 0.9},  # diff f=0.0, corr=0.7
        {"id": "q3", "question": "Q3", "ground_truth": "T3", "answer": "I3", "faithfulness": 1.0, "answer_correctness": 0.1},  # diff f=0.5, corr=0.8
        {"id": "q4", "question": "Q4", "ground_truth": "T4", "answer": "I4", "faithfulness": 0.2, "answer_correctness": 0.8},  # diff f=0.4, corr=0.5
    ]
    golden_map = {
        "q1": {"source_quotes": ["Quote 1"]},
        "q2": {"source_quotes": ["Quote 2"]},
        "q3": {"source_quotes": ["Quote 3"]},
        "q4": {"source_quotes": ["Quote 4"]},
    }

    spot = select_spot_check_items(baseline, improved, golden_map)
    f_ids = [it["id"] for it in spot["faithfulness_top3"]]
    c_ids = [it["id"] for it in spot["answer_correctness_top3"]]

    # Faithfulness abs diffs: q1: 0.8, q3: 0.5, q4: 0.4, q2: 0.0 -> top 3: q1, q3, q4
    assert f_ids == ["q1", "q3", "q4"]

    # Answer correctness abs diffs: q3: 0.8, q2: 0.7, q4: 0.5, q1: 0.1 -> top 3: q3, q2, q4
    assert c_ids == ["q3", "q2", "q4"]


def test_full_analysis_pipeline_and_write_once(tmp_path):
    """Verifies complete end-to-end analysis output generation and write-once refusal without --force."""
    out_dir = tmp_path / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    b_ans_path = tmp_path / "b_answers.json"
    imp_ans_path = tmp_path / "imp_answers.json"
    b_res_path = tmp_path / "b_results.json"
    imp_res_path = tmp_path / "imp_results.json"
    golden_path = tmp_path / "golden_round2.json"

    # Create synthetic 4-item dataset (2 numeric_precision, 2 simple_lookup)
    items_base = []
    items_imp = []
    golden_items = []

    for i in range(4):
        item_id = f"q_{i}"
        cat = "numeric_precision" if i < 2 else "simple_lookup"
        golden_items.append({
            "id": item_id,
            "category": cat,
            "question": f"Question {i}?",
            "ground_truth": f"Truth {i}",
            "source_quotes": [f"Quote {i}"],
        })
        items_base.append({
            "id": item_id,
            "category": cat,
            "question": f"Question {i}?",
            "answer": "I don't have enough information" if i == 0 else f"Baseline answer {i}",
            "contexts": [f"Context base {i}"],
            "retrieved_chunk_ids": [f"cb_{i}"],
            "source_chunk_ids": [f"cg_{i}"],
            "hit@8": 0.0 if i == 0 else 1.0,
            "recall@8": 0.0 if i == 0 else 1.0,
            "mrr": 0.0 if i == 0 else 1.0,
            "numeric_match": False if i == 0 else True,
            "faithfulness": 0.5 + 0.1 * i,
            "answer_correctness": 0.4 + 0.1 * i,
        })
        items_imp.append({
            "id": item_id,
            "category": cat,
            "question": f"Question {i}?",
            "answer": f"Improved answer {i}",
            "contexts": [f"Context imp {i}"],
            "retrieved_chunk_ids": [f"cg_{i}"],
            "source_chunk_ids": [f"cg_{i}"],
            "hit@8": 1.0,
            "recall@8": 1.0,
            "mrr": 1.0,
            "numeric_match": True,
            "faithfulness": 0.8 + 0.05 * i,
            "answer_correctness": 0.85 + 0.03 * i,
            "retry_count": 1 if i == 1 else 0,
        })

    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden_items, f)
    with open(b_ans_path, "w", encoding="utf-8") as f:
        json.dump({"items": items_base, "models": {"judge_model": "test-judge-31b"}}, f)
    with open(imp_ans_path, "w", encoding="utf-8") as f:
        json.dump({"items": items_imp, "models": {"judge_model": "test-judge-31b"}}, f)
    with open(b_res_path, "w", encoding="utf-8") as f:
        json.dump({"per_question_results": items_base, "judge_model": "test-judge-31b"}, f)
    with open(imp_res_path, "w", encoding="utf-8") as f:
        json.dump({"per_question_results": items_imp, "judge_model": "test-judge-31b"}, f)

    # First run: should succeed and create all 3 files
    res = run_e2e_analysis(
        baseline_answers_path=b_ans_path,
        improved_answers_path=imp_ans_path,
        baseline_results_path=b_res_path,
        improved_results_path=imp_res_path,
        baseline_checkpoint_path=None,
        improved_checkpoint_path=None,
        golden_path=golden_path,
        output_dir=out_dir,
        force=False,
        n_resamples=1000,
        seed=42,
    )

    summary_json_file = out_dir / "e2e_summary.json"
    summary_md_file = out_dir / "e2e_summary.md"
    spot_check_md_file = out_dir / "e2e_spot_check.md"

    assert summary_json_file.exists()
    assert summary_md_file.exists()
    assert spot_check_md_file.exists()

    with open(summary_json_file, "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert saved_json["metadata"]["judge_model"] == "test-judge-31b"
    assert saved_json["metadata"]["sample_size"] == 4
    assert "paired_differences" in saved_json
    assert saved_json["decline_metrics"]["baseline"]["decline_count"] == 1
    assert saved_json["decline_metrics"]["improved"]["decline_count"] == 0
    assert saved_json["critic_metrics"]["revised_count"] == 1

    md_content = summary_md_file.read_text(encoding="utf-8")
    assert "# End-to-End Evaluation Summary" in md_content
    assert "Faithfulness" in md_content
    assert "Numeric Accuracy" in md_content

    spot_content = spot_check_md_file.read_text(encoding="utf-8")
    assert "# End-to-End Evaluation: Human Spot Check" in spot_content
    assert "Category A: Top 3 Items by Absolute Faithfulness Difference" in spot_content
    assert "Category B: Top 3 Items by Absolute Answer Correctness Difference" in spot_content

    # Second run without force: should raise FileExistsError
    with pytest.raises(FileExistsError, match="Refusing to overwrite existing file"):
        run_e2e_analysis(
            baseline_answers_path=b_ans_path,
            improved_answers_path=imp_ans_path,
            baseline_results_path=b_res_path,
            improved_results_path=imp_res_path,
            baseline_checkpoint_path=None,
            improved_checkpoint_path=None,
            golden_path=golden_path,
            output_dir=out_dir,
            force=False,
            n_resamples=1000,
            seed=42,
        )

    # Third run with force=True: should succeed
    res_forced = run_e2e_analysis(
        baseline_answers_path=b_ans_path,
        improved_answers_path=imp_ans_path,
        baseline_results_path=b_res_path,
        improved_results_path=imp_res_path,
        baseline_checkpoint_path=None,
        improved_checkpoint_path=None,
        golden_path=golden_path,
        output_dir=out_dir,
        force=True,
        n_resamples=1000,
        seed=42,
    )
    assert res_forced is not None
