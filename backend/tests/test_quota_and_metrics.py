"""
Unit tests for Quota Ledger and LLM-Free Retrieval/Numeric Metrics (no network).

Tests:
1. QuotaLedger tracking per model per Pacific day.
2. Budget exhaustion and --max-calls graceful stop with exit code 2.
3. IST reset time formatting.
4. compute_retrieval_metrics: hit@8, recall@8, MRR@8.
5. check_numeric_precision_match with $, commas, %, 'million/billion'.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from backend.eval.quota_ledger import QuotaLedger, get_next_reset_time_ist, get_pacific_today
from backend.eval.retrieval_metrics import (
    check_numeric_precision_match,
    compute_retrieval_metrics,
)


def test_quota_ledger_records_and_computes_remaining(tmp_path):
    limits_path = tmp_path / "quota_limits.json"
    ledger_path = tmp_path / "quota_ledger.json"

    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({"gemini-3.5-flash": 100, "gemini-2.5-pro": 10}, f)

    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path)

    # Initial budget (0.8 safety factor applied: 100 -> 80, 10 -> 8)
    assert ledger.get_limit("gemini-3.5-flash") == 80
    assert ledger.get_limit("gemini-2.5-pro") == 8
    assert ledger.get_used_today("gemini-3.5-flash") == 0
    assert ledger.get_remaining_budget("gemini-3.5-flash") == 80

    # Record 40 calls
    ledger.record_call("gemini-3.5-flash", count=40)
    assert ledger.get_used_today("gemini-3.5-flash") == 40
    assert ledger.get_remaining_budget("gemini-3.5-flash") == 40

    # Record 40 more calls -> budget 0
    ledger.record_call("gemini-3.5-flash", count=40)
    assert ledger.get_used_today("gemini-3.5-flash") == 80
    assert ledger.get_remaining_budget("gemini-3.5-flash") == 0


def test_quota_ledger_stops_gracefully_when_budget_hits_zero(tmp_path):
    limits_path = tmp_path / "quota_limits.json"
    ledger_path = tmp_path / "quota_ledger.json"

    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({"gemini-3.5-flash": 5}, f)

    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path)
    # 5 * 0.8 = 4 limit
    ledger.record_call("gemini-3.5-flash", count=4)

    saved_called = []
    def mock_save():
        saved_called.append(True)

    with pytest.raises(SystemExit) as exc_info:
        ledger.check_budget_or_stop(
            model="gemini-3.5-flash",
            save_partial_callback=mock_save,
            next_index=15,
        )

    assert exc_info.value.code == 2
    assert len(saved_called) == 1


def test_quota_ledger_max_calls_limit(tmp_path):
    limits_path = tmp_path / "quota_limits.json"
    ledger_path = tmp_path / "quota_ledger.json"

    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({"gemini-3.5-flash": 1000}, f)

    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path, max_calls=3)
    ledger.record_call("gemini-3.5-flash", count=3)

    saved_called = []
    with pytest.raises(SystemExit) as exc_info:
        ledger.check_budget_or_stop(
            model="gemini-3.5-flash",
            save_partial_callback=lambda: saved_called.append(True),
            next_index=4,
        )

    assert exc_info.value.code == 2
    assert len(saved_called) == 1


def test_get_next_reset_time_ist():
    reset_str = get_next_reset_time_ist()
    assert "IST" in reset_str
    assert ":" in reset_str


def test_compute_retrieval_metrics():
    # Hit at rank 1
    res = compute_retrieval_metrics(
        retrieved_chunk_ids=["c1", "c2", "c3"],
        source_chunk_ids=["c1"],
        k=8,
    )
    assert res["hit@k"] == 1.0
    assert res["recall@k"] == 1.0
    assert res["mrr"] == 1.0

    # Hit at rank 2
    res = compute_retrieval_metrics(
        retrieved_chunk_ids=["c9", "c2", "c3"],
        source_chunk_ids=["c2", "c4"],
        k=8,
    )
    assert res["hit@k"] == 1.0
    assert res["recall@k"] == 0.5  # 1 of 2 targets found
    assert res["mrr"] == 0.5  # 1 / 2

    # Miss (no targets in top-8)
    res = compute_retrieval_metrics(
        retrieved_chunk_ids=["c9", "c8", "c7"],
        source_chunk_ids=["c1"],
        k=8,
    )
    assert res["hit@k"] == 0.0
    assert res["recall@k"] == 0.0
    assert res["mrr"] == 0.0

    # Unanswerable / out-of-corpus (empty target set)
    res = compute_retrieval_metrics(
        retrieved_chunk_ids=["c1", "c2"],
        source_chunk_ids=[],
        k=8,
    )
    assert res["is_unanswerable"] is True
    assert res["hit@k"] == 1.0


def test_check_numeric_precision_match():
    # Percentage matches
    gt = "Apple net sales accounted for 38% and 62%, respectively."
    ans = "Apple generated 38% through direct distribution and 62% through indirect distribution."
    assert check_numeric_precision_match(ans, gt) is True

    # Missing percentage
    ans_missing = "Apple generated 38% through direct distribution."
    assert check_numeric_precision_match(ans_missing, gt) is False

    # Scale conversions: million to billion
    gt_scale = "Total revenue reached $391,035 million in fiscal year 2024."
    ans_scale = "Apple reported total revenue of $391.035 billion in 2024."
    assert check_numeric_precision_match(ans_scale, gt_scale) is True

    # Comma stripping match
    gt_commas = "Operating income was $114,301 million."
    ans_commas = "Operating income reached 114301 million."
    assert check_numeric_precision_match(ans_commas, gt_commas) is True

