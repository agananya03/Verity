"""
Unit tests for hardened backend/eval/run_ragas_eval.py.

Verifies:
1. Model independence enforcement (JUDGE_MODEL vs GEN_MODEL and CRITIC_MODEL).
2. Transient error detection (503, timeouts) vs quota exhaustion (429).
3. Transient retry loop with exponential backoff and status='error' tagging.
4. Exclusion of status='error' items from aggregate scores.
5. Role models metadata recorded in results JSON.
6. API call estimation and --dry-run exit.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.agent.llm_client import AllModelsExhaustedError
from backend.eval.quota_ledger import QuotaLedger, get_pacific_today
from backend.eval.run_ragas_eval import (
    check_judge_model_independence,
    compute_heuristic_smoke_metrics,
    generate_single_question_with_retries,
    is_transient_error,
    print_api_call_estimate_and_budget,
    resolve_role_models,
    run_judge_stage,
)


def test_is_transient_error():
    assert is_transient_error(Exception("503 Service Unavailable")) is True
    assert is_transient_error(Exception("Connection reset by peer")) is True
    assert is_transient_error(Exception("Request timed out after 30.0s")) is True
    assert is_transient_error(Exception("Deadline Exceeded")) is True
    assert is_transient_error(Exception("500 Internal Server Error")) is True

    # 429/quota and client errors are NOT transient
    assert is_transient_error(Exception("429 Resource Exhausted")) is False
    assert is_transient_error(Exception("404 Not Found")) is False
    assert is_transient_error(Exception("Invalid prompt syntax")) is False


def test_judge_model_independence_aborts_without_flag():
    with pytest.raises(SystemExit) as exc_info:
        check_judge_model_independence(
            gen_model="gemini-3.5-flash",
            critic_model="gemini-3.5-flash-lite",
            judge_model="gemini-3.5-flash",
            config="baseline",
            allow_same_judge=False,
        )
    assert exc_info.value.code == 1

    with pytest.raises(SystemExit) as exc_info:
        check_judge_model_independence(
            gen_model="gemini-3.5-flash",
            critic_model="gemini-3.5-flash-lite",
            judge_model="gemini-3.5-flash-lite",
            config="improved",
            allow_same_judge=False,
        )
    assert exc_info.value.code == 1


def test_judge_model_independence_allows_with_flag():
    check_judge_model_independence(
        gen_model="gemini-3.5-flash",
        critic_model="gemini-3.5-flash-lite",
        judge_model="gemini-3.5-flash-lite",
        config="improved",
        allow_same_judge=True,
    )

    check_judge_model_independence(
        gen_model="gemini-3.5-flash",
        critic_model="gemini-3.5-flash-lite",
        judge_model="gemini-2.5-pro",
        config="improved",
        allow_same_judge=False,
    )


def test_transient_retries_recovers_on_second_attempt(monkeypatch):
    calls = []

    def mock_baseline(query, gen_model):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise Exception("503 Service Unavailable")
        return "Answer text", ["Context chunk 1"], ["chunk_01"], {"generator_model": "gemini-3.5-flash"}

    monkeypatch.setattr("backend.eval.run_ragas_eval.run_baseline_pipeline", mock_baseline)
    monkeypatch.setattr("time.sleep", lambda s: None)

    role_models = resolve_role_models()
    ans, ctxs, c_ids, _, status, err = generate_single_question_with_retries(
        config="baseline",
        query="Test query",
        role_models=role_models,
        max_transient_retries=5,
    )

    assert status == "success"
    assert err is None
    assert ans == "Answer text"
    assert len(calls) == 2


def test_transient_retries_marks_error_after_5_failures(monkeypatch):
    calls = []

    def mock_baseline(query, gen_model):
        calls.append(len(calls) + 1)
        raise Exception("503 Service Unavailable")

    monkeypatch.setattr("backend.eval.run_ragas_eval.run_baseline_pipeline", mock_baseline)
    monkeypatch.setattr("time.sleep", lambda s: None)

    role_models = resolve_role_models()
    ans, ctxs, c_ids, _, status, err = generate_single_question_with_retries(
        config="baseline",
        query="Test query",
        role_models=role_models,
        max_transient_retries=5,
    )

    assert status == "error"
    assert "503 Service Unavailable" in err
    assert ans == ""
    assert len(calls) == 5


def test_quota_error_stops_immediately_without_retrying(monkeypatch):
    calls = []

    def mock_baseline(query, gen_model):
        calls.append(1)
        raise AllModelsExhaustedError(attempted_models=[gen_model], last_error=Exception("429 Quota Exceeded"))

    monkeypatch.setattr("backend.eval.run_ragas_eval.run_baseline_pipeline", mock_baseline)

    role_models = resolve_role_models()
    with pytest.raises(AllModelsExhaustedError):
        generate_single_question_with_retries(
            config="baseline",
            query="Test query",
            role_models=role_models,
            max_transient_retries=5,
        )

    assert len(calls) == 1


def test_run_judge_stage_excludes_errors_and_records_metadata(tmp_path):
    answers_path = tmp_path / "baseline_answers.json"
    results_path = tmp_path / "baseline.json"

    answers_payload = {
        "config": "baseline",
        "stage": "generate",
        "items": [
            {
                "id": "golden_01",
                "category": "simple_lookup",
                "question": "What was Apple revenue in 2024?",
                "answer": "Apple revenue was $391B in 2024.",
                "contexts": ["Revenue reached $391B."],
                "retrieved_chunk_ids": ["c1"],
                "source_chunk_ids": ["c1"],
                "ground_truth": "Apple revenue was $391 billion.",
                "status": "success",
                "error": None,
                "generator_model": "gemini-3.5-flash",
            },
            {
                "id": "golden_02",
                "category": "simple_lookup",
                "question": "What was Microsoft cloud revenue in 2024?",
                "answer": "",
                "contexts": [],
                "retrieved_chunk_ids": [],
                "source_chunk_ids": ["c2"],
                "ground_truth": "Microsoft cloud revenue was $140 billion.",
                "status": "error",
                "error": "503 Service Unavailable after 5 retries",
                "generator_model": "gemini-3.5-flash",
            },
            {
                "id": "golden_03",
                "category": "numeric_precision",
                "question": "What was Amazon net income in 2024?",
                "answer": "Amazon net income was $44 billion in 2024.",
                "contexts": ["Net income was $44 billion."],
                "retrieved_chunk_ids": ["c3"],
                "source_chunk_ids": ["c3"],
                "ground_truth": "Amazon net income was $44 billion.",
                "status": "success",
                "error": None,
                "generator_model": "gemini-3.5-flash",
            },
        ],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers_payload, f)

    role_models = {
        "rewriter_model": "gemini-3.5-flash-lite",
        "classifier_model": "gemini-3.5-flash-lite",
        "gen_model": "gemini-3.5-flash",
        "critic_model": "gemini-3.5-flash-lite",
        "judge_model": "gemini-2.5-pro",
        "bench_model": "gemini-3.5-flash",
    }
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    results = run_judge_stage(
        config="baseline",
        answers_path=answers_path,
        results_path=results_path,
        role_models=role_models,
        golden_set_size=3,
        metrics_to_run=["faithfulness"],
        ledger=ledger,
        fast_eval=True,
    )

    assert results["n_scored"] == 2
    assert results["n_errors"] == 1
    assert results["total_evaluated"] == 3
    assert results["models"]["gen_model"] == "gemini-3.5-flash"
    assert results["models"]["judge_model"] == "gemini-2.5-pro"
    assert "hit@8" in results["aggregate_scores"]
    assert "recall@8" in results["aggregate_scores"]
    assert "mrr" in results["aggregate_scores"]

    pq = results["per_question_results"]
    assert len(pq) == 3
    assert pq[1]["status"] == "error"
    assert pq[1]["hit@8"] is None


def test_print_api_call_estimate_dry_run_exits_cleanly(tmp_path):
    role_models = resolve_role_models()
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    with pytest.raises(SystemExit) as exc_info:
        print_api_call_estimate_and_budget(
            config="improved",
            stage="all",
            num_questions=32,
            role_models=role_models,
            metrics_to_run=["faithfulness", "answer_correctness"],
            fast_eval=False,
            dry_run=True,
            ledger=ledger,
        )
    assert exc_info.value.code == 0


def test_pipeline_fingerprint_deterministic_and_matches():
    from backend.eval.run_ragas_eval import compute_pipeline_fingerprint
    fp1 = compute_pipeline_fingerprint()
    fp2 = compute_pipeline_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64


def test_fingerprint_ignores_commit_hash_change_and_allows_resume(tmp_path, monkeypatch):
    """
    Committing without code changes modifies the git commit hash, but the pipeline fingerprint
    must depend only on backend/agent/**/*.py and golden_set.json.
    A mismatch in git commit hash alone must NOT block resume.
    """
    from backend.eval.run_ragas_eval import compute_pipeline_fingerprint, run_generation_stage
    fp = compute_pipeline_fingerprint()

    # Answers file generated under commit_AAA
    answers_path = tmp_path / "baseline_answers.json"
    payload = {
        "config": "baseline",
        "stage": "generate",
        "git_commit_hash": "commit_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "pipeline_fingerprint": fp,
        "items": [
            {
                "id": "g01",
                "question": "Q1",
                "answer": "A1",
                "contexts": ["C1"],
                "retrieved_chunk_ids": ["c1"],
                "source_chunk_ids": ["c1"],
                "ground_truth": "GT1",
                "status": "success",
            }
        ],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Current working tree is under commit_BBB (different commit hash)
    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.get_git_commit_hash",
        lambda: "commit_BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    )

    golden_set = [
        {"id": "g01", "question": "Q1", "ground_truth": "GT1"},
        {"id": "g02", "question": "Q2", "ground_truth": "GT2"},
    ]
    role_models = resolve_role_models()
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    def mock_generate_question(*args, **kwargs):
        return "A2", ["C2"], ["c2"], {"generator_model": "gemini-3.5-flash"}, "success", None

    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.generate_single_question_with_retries",
        mock_generate_question,
    )

    # Resume from question 2: must succeed without SystemExit
    run_generation_stage(
        config="baseline",
        golden_set=golden_set,
        role_models=role_models,
        answers_path=answers_path,
        ledger=ledger,
        start_from=2,
        restart=False,
    )

    with open(answers_path, "r", encoding="utf-8") as f:
        updated = json.load(f)
    assert len(updated["items"]) == 2
    assert updated["items"][0]["answer"] == "A1"
    assert updated["items"][1]["answer"] == "A2"


def test_fingerprint_changes_when_agent_file_modified(tmp_path, monkeypatch):
    """
    Changing any agent file in backend/agent/**/*.py must produce a different fingerprint
    and block resume across versions.
    """
    from backend.eval.run_ragas_eval import compute_pipeline_fingerprint, run_generation_stage
    fp_orig = compute_pipeline_fingerprint()

    # Answers file saved with original fingerprint
    answers_path = tmp_path / "baseline_answers.json"
    payload = {
        "config": "baseline",
        "stage": "generate",
        "pipeline_fingerprint": fp_orig,
        "items": [{"id": "g01", "question": "Q1", "answer": "A1"}],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Simulate modifying draft.py content
    real_read_bytes = Path.read_bytes
    def fake_read_bytes(path_obj):
        if "draft.py" in str(path_obj):
            return b"# MODIFIED DRAFT NODE CONTENT FOR TEST"
        return real_read_bytes(path_obj)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    fp_modified = compute_pipeline_fingerprint()
    assert fp_orig != fp_modified

    golden_set = [
        {"id": "g01", "question": "Q1", "ground_truth": "GT1"},
        {"id": "g02", "question": "Q2", "ground_truth": "GT2"},
    ]
    role_models = resolve_role_models()
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    with pytest.raises(SystemExit) as exc_info:
        run_generation_stage(
            config="baseline",
            golden_set=golden_set,
            role_models=role_models,
            answers_path=answers_path,
            ledger=ledger,
            start_from=2,
            restart=False,
        )
    assert exc_info.value.code == 1


def test_fingerprint_changes_when_golden_set_modified(tmp_path, monkeypatch):
    """
    Changing golden_set.json must produce a different fingerprint and block resume.
    """
    from backend.eval.run_ragas_eval import compute_pipeline_fingerprint, run_generation_stage
    fp_orig = compute_pipeline_fingerprint()

    answers_path = tmp_path / "baseline_answers.json"
    payload = {
        "config": "baseline",
        "stage": "generate",
        "pipeline_fingerprint": fp_orig,
        "items": [{"id": "g01", "question": "Q1", "answer": "A1"}],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Simulate modifying golden_set.json content
    real_read_bytes = Path.read_bytes
    def fake_read_bytes(path_obj):
        if "golden_set.json" in str(path_obj):
            return b'[{"id": "modified_golden_set"}]'
        return real_read_bytes(path_obj)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    fp_modified = compute_pipeline_fingerprint()
    assert fp_orig != fp_modified

    golden_set = [
        {"id": "g01", "question": "Q1", "ground_truth": "GT1"},
        {"id": "g02", "question": "Q2", "ground_truth": "GT2"},
    ]
    role_models = resolve_role_models()
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    with pytest.raises(SystemExit) as exc_info:
        run_generation_stage(
            config="baseline",
            golden_set=golden_set,
            role_models=role_models,
            answers_path=answers_path,
            ledger=ledger,
            start_from=2,
            restart=False,
        )
    assert exc_info.value.code == 1


def test_restart_archives_file_and_starts_clean(tmp_path, monkeypatch):
    from backend.eval.run_ragas_eval import run_generation_stage
    answers_path = tmp_path / "baseline_answers.json"
    old_payload = {
        "config": "baseline",
        "stage": "generate",
        "pipeline_fingerprint": "different_hash",
        "items": [{"id": "g01", "question": "Q1", "answer": "Old answer"}],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(old_payload, f)

    golden_set = [{"id": "g01", "question": "Q1", "ground_truth": "GT1"}]
    role_models = resolve_role_models()
    ledger = QuotaLedger(limits_path=tmp_path / "limits.json", ledger_path=tmp_path / "ledger.json")

    def mock_generate_question(*args, **kwargs):
        return "New answer", ["ctx"], ["c1"], {"generator_model": "gemini-3.5-flash"}, "success", None

    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.generate_single_question_with_retries",
        mock_generate_question,
    )

    run_generation_stage(
        config="baseline",
        golden_set=golden_set,
        role_models=role_models,
        answers_path=answers_path,
        ledger=ledger,
        start_from=1,
        restart=True,
    )

    # Check archive file created
    archived_files = list(tmp_path.glob("baseline_answers_archived_*.json"))
    assert len(archived_files) == 1
    with open(archived_files[0], "r", encoding="utf-8") as f:
        archived_data = json.load(f)
    assert archived_data["items"][0]["answer"] == "Old answer"

    # Check new answers file created with recorded model and fingerprint
    assert answers_path.exists()
    with open(answers_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    assert new_data["items"][0]["answer"] == "New answer"
    assert new_data["items"][0]["generator_model"] == "gemini-3.5-flash"
    assert "pipeline_fingerprint" in new_data


def test_judge_quota_tracking_counts_calls_with_mocked_judge(tmp_path, monkeypatch):
    """
    Verifies that RAGAS judge LLM calls count in the quota ledger via the registered call hook.
    """
    from backend.eval.run_ragas_eval import run_judge_stage
    import pandas as pd

    answers_path = tmp_path / "baseline_answers.json"
    results_path = tmp_path / "baseline.json"
    answers_payload = {
        "config": "baseline",
        "stage": "generate",
        "items": [
            {
                "id": "g01",
                "category": "simple_lookup",
                "question": "What was Apple revenue in 2024?",
                "answer": "Apple revenue was $391B.",
                "contexts": ["Apple revenue was $391,035M."],
                "retrieved_chunk_ids": ["c1"],
                "source_chunk_ids": ["c1"],
                "ground_truth": "Apple revenue was $391B.",
                "status": "success",
            },
            {
                "id": "g02",
                "category": "simple_lookup",
                "question": "What was Microsoft revenue in 2024?",
                "answer": "Microsoft revenue was $245B.",
                "contexts": ["Microsoft revenue was $245,123M."],
                "retrieved_chunk_ids": ["c2"],
                "source_chunk_ids": ["c2"],
                "ground_truth": "Microsoft revenue was $245B.",
                "status": "success",
            },
        ],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers_payload, f)

    role_models = resolve_role_models()
    judge_model = role_models["judge_model"]

    limits_path = tmp_path / "limits.json"
    ledger_path = tmp_path / "ledger.json"
    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({judge_model: 100}, f)

    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path, max_calls=10)

    monkeypatch.setenv("GOOGLE_API_KEY", "fake_test_key")
    def make_mock_judge(**kwargs):
        judge = MagicMock(
            model=kwargs.get("model", "gemini-2.5-pro"),
            model_name=kwargs.get("model", "gemini-2.5-pro"),
            callbacks=kwargs.get("callbacks", []),
        )
        judge.generate_prompt.return_value = MagicMock(
            generations=[[MagicMock(text="mock judge evaluation response")]]
        )
        return judge

    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.ChatGoogleGenerativeAI",
        make_mock_judge,
    )
    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.GoogleGenerativeAIEmbeddings",
        lambda **kwargs: MagicMock(),
    )

    def mock_eval(dataset, metrics, llm, embeddings, run_config=None):
        # Simulate RAGAS invoking the judge LLM once per question
        for q in dataset["question"]:
            llm.generate_text(f"Judge prompt for: {q}")
        return MagicMock(to_pandas=lambda: pd.DataFrame({"faithfulness": [1.0, 1.0]}))

    monkeypatch.setattr("backend.eval.run_ragas_eval.evaluate", mock_eval)

    results = run_judge_stage(
        config="baseline",
        answers_path=answers_path,
        results_path=results_path,
        role_models=role_models,
        golden_set_size=2,
        metrics_to_run=["faithfulness"],
        ledger=ledger,
        fast_eval=False,
    )

    assert results_path.exists()
    assert results["n_scored"] == 2
    # Both judge LLM invocations must be recorded in the quota ledger
    assert ledger.run_calls == 2
    assert ledger.get_used_today(judge_model) == 2


def test_judge_quota_tracking_respects_max_calls_with_mocked_judge(tmp_path, monkeypatch):
    """
    Verifies that judge LLM invocations respect --max-calls and stop gracefully with exit code 2.
    """
    from backend.eval.run_ragas_eval import run_judge_stage
    import pandas as pd

    answers_path = tmp_path / "baseline_answers.json"
    results_path = tmp_path / "baseline.json"
    answers_payload = {
        "config": "baseline",
        "stage": "generate",
        "items": [
            {
                "id": f"g0{i}",
                "category": "simple_lookup",
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "contexts": [f"Context {i}"],
                "retrieved_chunk_ids": [f"c{i}"],
                "source_chunk_ids": [f"c{i}"],
                "ground_truth": f"GT {i}",
                "status": "success",
            }
            for i in range(1, 4)
        ],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers_payload, f)

    role_models = resolve_role_models()
    judge_model = role_models["judge_model"]

    limits_path = tmp_path / "limits.json"
    ledger_path = tmp_path / "ledger.json"
    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({judge_model: 100}, f)

    # Set max_calls to 1 so the 2nd judge call trips the limit
    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path, max_calls=1)

    monkeypatch.setenv("GOOGLE_API_KEY", "fake_test_key")
    def make_mock_judge(**kwargs):
        judge = MagicMock(
            model=kwargs.get("model", "gemini-2.5-pro"),
            model_name=kwargs.get("model", "gemini-2.5-pro"),
            callbacks=kwargs.get("callbacks", []),
        )
        judge.generate_prompt.return_value = MagicMock(
            generations=[[MagicMock(text="mock judge evaluation response")]]
        )
        return judge

    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.ChatGoogleGenerativeAI",
        make_mock_judge,
    )
    monkeypatch.setattr(
        "backend.eval.run_ragas_eval.GoogleGenerativeAIEmbeddings",
        lambda **kwargs: MagicMock(),
    )

    def mock_eval(dataset, metrics, llm, embeddings, run_config=None):
        for q in dataset["question"]:
            llm.generate_text(f"Judge prompt for: {q}")
        return MagicMock(to_pandas=lambda: pd.DataFrame({"faithfulness": [1.0] * len(dataset["question"])}))

    monkeypatch.setattr("backend.eval.run_ragas_eval.evaluate", mock_eval)

    with pytest.raises(SystemExit) as exc_info:
        run_judge_stage(
            config="baseline",
            answers_path=answers_path,
            results_path=results_path,
            role_models=role_models,
            golden_set_size=3,
            metrics_to_run=["faithfulness"],
            ledger=ledger,
            fast_eval=False,
        )

    # Must stop gracefully with exit code 2 when max-calls is reached
    assert exc_info.value.code == 2
    assert ledger.run_calls == 1


def test_judge_quota_tracking_respects_daily_budget_with_mocked_judge(tmp_path, monkeypatch):
    """
    Verifies that judge LLM invocations respect the daily budget and stop gracefully with exit code 2.
    """
    from backend.eval.run_ragas_eval import run_judge_stage

    answers_path = tmp_path / "baseline_answers.json"
    results_path = tmp_path / "baseline.json"
    answers_payload = {
        "config": "baseline",
        "stage": "generate",
        "items": [
            {
                "id": "g01",
                "category": "simple_lookup",
                "question": "Question 1",
                "answer": "Answer 1",
                "contexts": ["Context 1"],
                "retrieved_chunk_ids": ["c1"],
                "source_chunk_ids": ["c1"],
                "ground_truth": "GT 1",
                "status": "success",
            }
        ],
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers_payload, f)

    role_models = resolve_role_models()
    judge_model = role_models["judge_model"]

    limits_path = tmp_path / "limits.json"
    ledger_path = tmp_path / "ledger.json"
    # Daily limit of 1 request, but already 1 request used today -> 0 budget remaining
    with open(limits_path, "w", encoding="utf-8") as f:
        json.dump({judge_model: 1}, f)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump({get_pacific_today(): {judge_model: 10}}, f)

    ledger = QuotaLedger(limits_path=limits_path, ledger_path=ledger_path)

    monkeypatch.setenv("GOOGLE_API_KEY", "fake_test_key")

    with pytest.raises(SystemExit) as exc_info:
        run_judge_stage(
            config="baseline",
            answers_path=answers_path,
            results_path=results_path,
            role_models=role_models,
            golden_set_size=1,
            metrics_to_run=["faithfulness"],
            ledger=ledger,
            fast_eval=False,
        )

    # Must stop gracefully with exit code 2 when daily budget is exhausted
    assert exc_info.value.code == 2
