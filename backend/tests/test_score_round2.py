"""
Tests for Round 2 Pre-Registered Benchmark Scoring (backend/eval/score_round2.py)
================================================================================
Validates:
  1. Deterministic retrieval metrics (Hit@8, Recall@8, MRR@8, Hit@20, rank tracking).
  2. Exact McNemar paired test calculation.
  3. Wilson 95% confidence interval computation.
  4. Write-once pre-registration guard (refusal without --force, override with --force).
  5. CRLF-normalized SHA-256 computation consistency across line-ending variants.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from backend.eval.score_round2 import (
    check_write_once_guard,
    compute_crlf_normalized_sha256,
    compute_item_metrics,
    exact_mcnemar_test,
    find_first_gold_rank,
    wilson_ci,
)


def test_compute_item_metrics_fixed_examples():
    """Validates Hit@8, Recall@8, MRR, and Hit@20 on fixed canonical examples."""
    source_chunk_ids = ["gold_chunk_1", "gold_chunk_2"]

    # 1. Target at rank 1, second target at rank 15
    retrieved_case1 = [
        "gold_chunk_1",  # rank 1
        "other_2",
        "other_3",
        "other_4",
        "other_5",
        "other_6",
        "other_7",
        "other_8",
        "other_9",
        "other_10",
        "other_11",
        "other_12",
        "other_13",
        "other_14",
        "gold_chunk_2",  # rank 15
    ]
    m1 = compute_item_metrics(retrieved_case1, source_chunk_ids)
    assert m1["hit@8"] == 1.0
    assert m1["recall@8"] == 0.5  # 1 of 2 in top 8
    assert m1["mrr"] == 1.0  # 1 / 1
    assert m1["hit@20"] == 1.0
    assert m1["first_gold_rank"] == 1

    # 2. Target at rank 4, no other target
    retrieved_case2 = ["o1", "o2", "o3", "gold_chunk_1", "o5", "o6", "o7", "o8"]
    m2 = compute_item_metrics(retrieved_case2, ["gold_chunk_1"])
    assert m2["hit@8"] == 1.0
    assert m2["recall@8"] == 1.0  # 1 of 1 in top 8
    assert m2["mrr"] == 0.25  # 1 / 4
    assert m2["hit@20"] == 1.0
    assert m2["first_gold_rank"] == 4

    # 3. Target at rank 10 (Miss in top 8, Hit in top 20)
    retrieved_case3 = [f"o_{i}" for i in range(1, 10)] + ["gold_chunk_1"]  # rank 10
    m3 = compute_item_metrics(retrieved_case3, ["gold_chunk_1"])
    assert m3["hit@8"] == 0.0
    assert m3["recall@8"] == 0.0
    assert m3["mrr"] == 0.0  # MRR@8 is 0.0 when not in top 8
    assert m3["hit@20"] == 1.0
    assert m3["first_gold_rank"] == 10

    # 4. Target at rank 35 (Miss in top 8 and top 20, present in top 50)
    retrieved_case4 = [f"o_{i}" for i in range(1, 35)] + ["gold_chunk_1"]
    m4 = compute_item_metrics(retrieved_case4, ["gold_chunk_1"])
    assert m4["hit@8"] == 0.0
    assert m4["recall@8"] == 0.0
    assert m4["mrr"] == 0.0
    assert m4["hit@20"] == 0.0
    assert m4["first_gold_rank"] == 35

    # 5. Complete miss (target not in candidate pool)
    retrieved_case5 = [f"o_{i}" for i in range(1, 51)]
    m5 = compute_item_metrics(retrieved_case5, ["gold_chunk_1"])
    assert m5["hit@8"] == 0.0
    assert m5["recall@8"] == 0.0
    assert m5["mrr"] == 0.0
    assert m5["hit@20"] == 0.0
    assert m5["first_gold_rank"] is None

    # 6. Empty retrieved list
    m6 = compute_item_metrics([], ["gold_chunk_1"])
    assert m6["hit@8"] == 0.0
    assert m6["first_gold_rank"] is None


def test_find_first_gold_rank():
    """Tests first gold chunk rank retrieval helper."""
    assert find_first_gold_rank(["a", "b", "c"], {"b"}) == 2
    assert find_first_gold_rank(["a", "b", "c"], {"z"}) is None
    assert find_first_gold_rank(["a", "b", "c"], {"c", "a"}) == 1
    assert find_first_gold_rank([], {"a"}) is None


def test_exact_mcnemar_helper():
    """Tests exact paired two-sided binomial McNemar test against known analytic values."""
    # 1. Identical outputs -> discordant=0, p=1.0
    res_ident = exact_mcnemar_test([True, False, True], [True, False, True])
    assert res_ident["discordant"] == 0
    assert res_ident["p_value"] == 1.0

    # 2. Perfectly symmetric discordance -> b=4, c=4, n_disc=8, p=1.0
    res_symm = exact_mcnemar_test([True] * 4 + [False] * 4, [False] * 4 + [True] * 4)
    assert res_symm["only_test"] == 4
    assert res_symm["only_base"] == 4
    assert res_symm["discordant"] == 8
    assert res_symm["p_value"] == 1.0

    # 3. Completely one-sided discordance -> b=10, c=0
    # Two-sided p = 2 * (0.5^10) = 2 / 1024 = 0.001953125
    res_skew = exact_mcnemar_test([True] * 10, [False] * 10)
    assert res_skew["only_test"] == 10
    assert res_skew["only_base"] == 0
    assert res_skew["discordant"] == 10
    assert pytest.approx(res_skew["p_value"], abs=1e-6) == 0.001953125


def test_wilson_ci():
    """Tests Wilson score 95% confidence interval boundaries."""
    # Empty set
    assert wilson_ci(0, 0) == (0.0, 0.0)

    # 0 / 10 hits
    low_0, high_0 = wilson_ci(0, 10)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.35

    # 10 / 10 hits
    low_1, high_1 = wilson_ci(10, 10)
    assert 0.65 < low_1 < 1.0
    assert pytest.approx(high_1, abs=1e-6) == 1.0

    # 50 / 100 hits (symmetric around 0.5)
    low_50, high_50 = wilson_ci(50, 100)
    assert low_50 < 0.5 < high_50
    assert pytest.approx(0.5 - low_50, abs=1e-3) == high_50 - 0.5


def test_write_once_guard(tmp_path: Path):
    """Verifies that check_write_once_guard halts when target exists unless force=True."""
    target_file = tmp_path / "round2_scoring.json"

    # 1. Does not exist: succeeds without exception
    check_write_once_guard(target_file, force=False)

    # 2. Exists and force=False: raises FileExistsError
    target_file.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError) as exc_info:
        check_write_once_guard(target_file, force=False)
    assert "already exists" in str(exc_info.value)

    # 3. Exists and force=True: succeeds (prints warning)
    check_write_once_guard(target_file, force=True)


def test_crlf_normalized_sha256(tmp_path: Path):
    """Verifies SHA-256 newline normalization (CRLF and LF produce the same hash)."""
    file_crlf = tmp_path / "file_crlf.json"
    file_lf = tmp_path / "file_lf.json"

    file_crlf.write_bytes(b'{"key": "value"}\r\n{"item": 1}\r\n')
    file_lf.write_bytes(b'{"key": "value"}\n{"item": 1}\n')

    hash_crlf = compute_crlf_normalized_sha256(file_crlf)
    hash_lf = compute_crlf_normalized_sha256(file_lf)

    assert hash_crlf == hash_lf
    assert len(hash_crlf) == 64
