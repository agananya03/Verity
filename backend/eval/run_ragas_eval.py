"""
RAGAS Evaluation Runner for Verity Financial RAG (Free-Tier Hardened).

Component Comparison: Baseline vs. Improved
========================================================================================================
Component                  | Baseline Pipeline                | Improved Pipeline (Production Graph)
---------------------------+----------------------------------+-----------------------------------------
Query Processing           | Raw user query (no rewriting,    | Context-aware Rewriter (multi-turn) +
                           | no metadata stripping)           | clean_v2 query cleaning (RETRIEVAL_QUERY_CLEANING)
                           |                                  | (preamble stripped, product/company kept)
Index / Search Modality    | Dense vector-only (Qdrant        | Hybrid Dense (Qdrant) + Sparse (BM25Okapi
                           | BAAI/bge-small-en-v1.5)          | number-preserving tokenizer), RRF (k=60)
Metadata Filtering         | None (filters=None)              | Rule-based extractor + Classifier LLM
                           |                                  | (company, fiscal_year, fiscal_years)
Temporal Routing           | Single search pool (top_k=8)     | Multi-year per-year retrieval +
                           |                                  | alternating merge (ceil(8/n) per year)
Generation Architecture    | Single LLM prompt call           | Grounded Draft prompt with strict inline
                           | (gen_model, no fallback chain)   | [Chunk N] citations & domain decline
Self-Correction / Critic   | None                             | Critic verification loop (up to 2 retries)
                           |                                  | with feedback + Finalizer grounded claims
========================================================================================================

Evaluates either:
- Baseline config: Vector-only Qdrant search (raw query, no filters) + single LLM draft generation.
- Improved config: Full verity_graph pipeline (Hybrid Dense/Sparse search + clean_v2 + per-year retrieval + metadata filtering + self-correcting Critic loop).

Features:
- Per-role model env vars: REWRITER_MODEL, CLASSIFIER_MODEL, GEN_MODEL, CRITIC_MODEL, JUDGE_MODEL, BENCH_MODEL.
- Daily quota tracking via backend/eval/results/quota_ledger.json (America/Los_Angeles dates, Google Pacific midnight reset).
- Quota limits loaded from backend/eval/quota_limits.json.
- Graceful stop with exit code 2 when any model's daily budget hits 0 or --max-calls reached (saves partial results, prints reset in IST).
- Bypasses query rewriter when conversation history is empty.
- Fast deterministic rule-based extractor in classifier to eliminate classifier LLM calls when unambiguous.
- Deterministic LLM-free retrieval metrics: hit@k, recall@k, MRR (k=8) computed from source_chunk_ids.
- Deterministic numeric-match check for numeric_precision questions (normalizes $, commas, %, 'million/billion').
- RAGAS faithfulness and answer correctness as the only LLM-judged metrics, selectable via --metrics (e.g. --metrics faithfulness,answer_correctness, or --metrics none).
- Two-stage architecture: Stage 1 ('generate') -> {config}_answers.json, Stage 2 ('judge') -> {config}.json.
- Model independence check: JUDGE_MODEL cannot match GEN_MODEL or CRITIC_MODEL without --allow-same-judge.
- --dry-run prints per-model calls needed vs remaining budget and days to complete, then exits 0.
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure huggingface uses local cache when running offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Compatibility patch for ragas importing langchain_community modules
if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_mock = types.ModuleType("vertexai")
    vertexai_mock.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mock
if "langchain_community.llms" not in sys.modules:
    vertexai_llm_mock = types.ModuleType("llms")
    vertexai_llm_mock.VertexAI = type("VertexAI", (), {})
    sys.modules["langchain_community.llms"] = vertexai_llm_mock

# Setup sys.path for backend imports
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

from datasets import Dataset
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from backend.agent.graph import verity_graph
from backend.agent.llm_client import (
    AllModelsExhaustedError,
    get_llm_with_fallback,
    is_quota_error,
    notify_llm_call,
    register_llm_call_hook,
    reset_fallback_cache,
)
import backend.agent.nodes.classifier as classifier_node
import backend.agent.nodes.critic as critic_node
import backend.agent.nodes.draft as draft_node
import backend.agent.nodes.rewriter as rewriter_node
from backend.agent.retrieval import search_qdrant
from backend.agent.state import AgentState
from backend.eval.quota_ledger import (
    QuotaLedger,
    get_next_reset_time_ist,
    get_pacific_today,
    global_quota_ledger,
)
from backend.eval.retrieval_metrics import (
    check_numeric_precision_match,
    compute_retrieval_metrics,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Returns the current git commit hash, or 'unknown' if git is not available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


from backend.eval.retrieval_ablation import normalize_bytes_newlines


def compute_pipeline_fingerprint(golden_set_path: Optional[Path] = None) -> str:
    """
    Computes a deterministic pipeline fingerprint based on:
    - Sorted hashes of all Python files in backend/agent/**/*.py (CRLF normalized to LF).
    - Hash of benchmark dataset file (golden_set.json or specified golden-path, CRLF normalized to LF).
    (Git commit hash is excluded from the comparison value so that committing
    without code changes does not break resume).
    """
    hasher = hashlib.sha256()

    # 1. Files in backend/agent/**/*.py
    agent_dir = BASE_DIR / "agent"
    if agent_dir.exists():
        py_files = sorted(agent_dir.glob("**/*.py"), key=lambda p: p.as_posix())
        for py_file in py_files:
            rel_path = py_file.relative_to(BASE_DIR).as_posix()
            hasher.update(rel_path.encode("utf-8"))
            try:
                content = py_file.read_bytes()
                normalized = normalize_bytes_newlines(content)
                hasher.update(hashlib.sha256(normalized).digest())
            except Exception as e:
                logger.warning(f"Failed to read {py_file} for fingerprint: {e}")

    # 2. Golden benchmark file
    g_path = golden_set_path or (BASE_DIR / "eval" / "golden_set.json")
    if g_path.exists():
        try:
            content = g_path.read_bytes()
            normalized = normalize_bytes_newlines(content)
            hasher.update(hashlib.sha256(normalized).digest())
        except Exception as e:
            logger.warning(f"Failed to read {g_path} for fingerprint: {e}")

    return hasher.hexdigest()


def is_transient_error(e: Exception) -> bool:
    """
    Checks if an exception represents a 503 or transient network/service error
    that should be retried with exponential backoff.
    """
    err_str = str(e).lower()
    transient_indicators = [
        "503",
        "service unavailable",
        "unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "temporary failure",
        "deadline exceeded",
        "internal error",
        "500",
        "502",
        "504",
    ]
    return any(ind in err_str for ind in transient_indicators)


def resolve_role_models() -> Dict[str, str]:
    """
    Resolves per-role model names from environment variables with sensible defaults.
    """
    return {
        "rewriter_model": os.getenv("REWRITER_MODEL", os.getenv("MODEL_REWRITER", os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite"))),
        "classifier_model": os.getenv("CLASSIFIER_MODEL", os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash-lite")),
        "gen_model": os.getenv("GEN_MODEL", os.getenv("MODEL_DRAFT", "gemini-3.5-flash")),
        "critic_model": os.getenv("CRITIC_MODEL", os.getenv("MODEL_CRITIC", "gemini-3.5-flash-lite")),
        "judge_model": os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite"),
        "bench_model": os.getenv("BENCH_MODEL", "gemini-3.5-flash"),
    }


def check_judge_model_independence(
    gen_model: str,
    critic_model: str,
    judge_model: str,
    config: str,
    allow_same_judge: bool,
) -> None:
    """
    Validates that JUDGE_MODEL is distinct from GEN_MODEL and CRITIC_MODEL.
    Prints a loud warning and requires --allow-same-judge to proceed if conflicts exist.
    """
    conflicts: List[str] = []
    if judge_model == gen_model:
        conflicts.append(f"GEN_MODEL ('{gen_model}')")
    if config == "improved" and judge_model == critic_model:
        conflicts.append(f"CRITIC_MODEL ('{critic_model}')")
    elif judge_model == critic_model:
        conflicts.append(f"CRITIC_MODEL ('{critic_model}')")

    if conflicts:
        conflict_desc = " and ".join(conflicts)
        box = (
            "\n" + "!" * 78 + "\n"
            f"WARNING: Model Conflict Detected!\n"
            f"JUDGE_MODEL ('{judge_model}') equals {conflict_desc}.\n\n"
            "Using the same model as judge and generator/critic can bias evaluation.\n"
            "To proceed anyway, pass the --allow-same-judge flag.\n"
            + "!" * 78 + "\n"
        )
        if not allow_same_judge:
            print(box)
            logger.error(f"Aborting evaluation: JUDGE_MODEL equals {conflict_desc}. Require --allow-same-judge to proceed.")
            sys.exit(1)
        else:
            print(box)
            logger.warning(f"Proceeding despite model conflict: JUDGE_MODEL equals {conflict_desc} (--allow-same-judge passed).")


def print_api_call_estimate_and_budget(
    config: str,
    stage: str,
    num_questions: int,
    role_models: Dict[str, str],
    metrics_to_run: List[str],
    fast_eval: bool,
    dry_run: bool,
    ledger: QuotaLedger,
) -> None:
    """
    Calculates calls needed per model vs remaining budget and prints an informative table.
    If dry_run is True, exits with code 0.
    """
    gen_model = role_models["gen_model"]
    critic_model = role_models["critic_model"]
    classifier_model = role_models["classifier_model"]
    rewriter_model = role_models["rewriter_model"]
    judge_model = role_models["judge_model"]

    # Estimate calls needed per role
    calls_by_role: Dict[str, int] = {
        "Generator": 0,
        "Critic": 0,
        "Classifier": 0,
        "Rewriter": 0,
        "Judge": 0,
    }

    if stage in ["generate", "all"]:
        if config == "baseline":
            calls_by_role["Generator"] += num_questions
        else:
            calls_by_role["Generator"] += int(math.ceil(num_questions * 1.3))
            calls_by_role["Critic"] += int(math.ceil(num_questions * 1.3))
            # Rule-based extractor bypasses classifier on unambiguous benchmark items
            calls_by_role["Classifier"] += int(math.ceil(num_questions * 0.1))
            # Rewriter is skipped because conversation history is empty
            calls_by_role["Rewriter"] += 0

    if stage in ["judge", "all"] and not fast_eval:
        judge_calls_per_q = 0
        if "faithfulness" in metrics_to_run:
            judge_calls_per_q += 2
        if "answer_correctness" in metrics_to_run:
            judge_calls_per_q += 1
        calls_by_role["Judge"] += num_questions * judge_calls_per_q

    # Aggregate total calls per model
    calls_by_model: Dict[str, int] = {}
    for role_name, model in [
        ("Generator", gen_model),
        ("Critic", critic_model),
        ("Classifier", classifier_model),
        ("Rewriter", rewriter_model),
        ("Judge", judge_model),
    ]:
        calls_by_model[model] = calls_by_model.get(model, 0) + calls_by_role[role_name]

    # Re-read ledger freshly from disk at print time (not cached from start of run)
    fresh_ledger_data = ledger.load_ledger()
    pacific_date = get_pacific_today()
    date_counts = fresh_ledger_data.get(pacific_date, {})
    reset_ist = get_next_reset_time_ist()

    print("\n" + "=" * 90)
    print(f"ESTIMATED API CALLS & QUOTA BUDGET [Config: {config.upper()}, Stage: {stage.upper()}]")
    print(f"Pacific Date: {pacific_date} | Next Quota Reset (IST): {reset_ist}")
    print("=" * 90)
    print(f"{'Role':<14} | {'Model':<22} | {'Calls Needed':<12} | {'Used Today':<10} | {'Daily Limit':<11} | {'Remaining':<9} | {'Days Needed'}")
    print("-" * 90)

    max_days = 1
    total_calls = sum(calls_by_role.values())

    model_roles = [
        ("Generator", gen_model),
        ("Critic", critic_model),
        ("Classifier", classifier_model),
        ("Rewriter", rewriter_model),
        ("Judge", judge_model),
    ]

    for role_name, model in model_roles:
        c_needed = calls_by_role.get(role_name, 0)
        total_model_calls = calls_by_model.get(model, 0)
        used = date_counts.get(model, 0)
        limit = ledger.get_limit(model)
        rem = max(0, limit - used)

        if total_model_calls == 0:
            days_str = "0 days"
        elif total_model_calls <= rem:
            days_str = "1 day"
        else:
            rem_after_today = total_model_calls - rem
            extra_days = int(math.ceil(rem_after_today / max(1, limit)))
            total_days = 1 + extra_days
            days_str = f"{total_days} days"
            max_days = max(max_days, total_days)

        note = ""
        if role_name == "Rewriter":
            note = " (skipped)"
        elif role_name == "Classifier" and config == "improved":
            note = " (rule-based)"

        print(
            f"{role_name:<14} | {model:<22} | {str(c_needed) + note:<12} | {used:<10} | {limit:<11} | {rem:<9} | {days_str}"
        )

    print("-" * 90)
    print(f"Total Estimated Calls:    {total_calls}")
    print(f"Max Days to Complete:     {max_days} day(s)")
    print(f"Retrieval Metrics (k=8):  hit@8, recall@8, mrr (LLM-free, zero quota)")
    print(f"Numeric Precision Match:  Deterministic regex & unit normalization (LLM-free, zero quota)")
    print(f"LLM-Judged Metrics:       {', '.join(metrics_to_run) if metrics_to_run else 'None (LLM-free mode)'}")
    print("=" * 90)

    if dry_run:
        print("\nDry-run requested (--dry-run). Exiting without making API calls.\n")
        sys.exit(0)


def run_baseline_pipeline(query: str, gen_model: str) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    """
    Baseline pipeline:
    - Vector-only search via Qdrant (top_k=8, no metadata filters, no BM25 fusion).
    - Single LLM draft generation call pinned to gen_model with NO fallback models.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    hits = search_qdrant(query=query, filters=None, limit=8)
    retrieved_contexts = [h.get("text", "") for h in hits if h.get("text")]
    retrieved_chunk_ids = [
        str(h.get("id") or h.get("chunk_id", ""))
        for h in hits
        if (h.get("id") or h.get("chunk_id"))
    ]

    if not retrieved_contexts:
        return (
            "I don't have enough information to answer this question based on the retrieved SEC filings.",
            [],
            [],
            {"generator_model": gen_model},
        )

    context_text = "\n\n".join([f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
    prompt = f"""Answer the following financial question accurately using ONLY the provided Context Chunks.
If the information is insufficient, state "I don't have enough information to answer this question based on the retrieved SEC filings."

Context Chunks:
{context_text}

Question: {query}
Answer:"""

    llm = get_llm_with_fallback(
        primary_model=gen_model,
        fallback_models=[],
        google_api_key=api_key or None,
        temperature=0.1,
        request_timeout=30.0,
        max_retries=0,
    )
    response = llm.invoke([("user", prompt)])
    from backend.agent.llm_client import get_last_model_used
    actual_gen = get_last_model_used() or gen_model

    if isinstance(response.content, list):
        answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
    else:
        answer = str(response.content).strip()

    return answer, retrieved_contexts, retrieved_chunk_ids, {"generator_model": actual_gen}


def run_improved_pipeline(
    query: str,
    gen_model: str,
    critic_model: str,
    classifier_model: str,
    rewriter_model: str,
) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    """
    Improved pipeline:
    - End-to-end verity_graph execution.
    - Pinned generator, critic, classifier, and rewriter models.
    - Does NOT silently fall back to secondary models.
    """
    draft_node.MODEL_DRAFT = gen_model
    draft_node.FALLBACK_DRAFT = []
    critic_node.MODEL_CRITIC = critic_model
    classifier_node.MODEL_CLASSIFIER = classifier_model
    rewriter_node.MODEL_REWRITER = rewriter_model
    reset_fallback_cache()

    initial_state: AgentState = {
        "query": query,
        "classification": "",
        "extracted_filters": {},
        "retrieved_chunks": [],
        "draft_answer": "",
        "citations": [],
        "critic_verdict": {},
        "first_critic_verdict": {},
        "critique_feedback": [],
        "retry_count": 0,
        "final_grounded": False,
        "unsupported_claims_final": [],
        "final_answer": "",
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = verity_graph.invoke(initial_state, config=config)
    final_answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""
    retrieved_chunks = final_state.get("retrieved_chunks", [])
    retrieved_contexts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]
    retrieved_chunk_ids = [
        str(c.get("id") or c.get("chunk_id", ""))
        for c in retrieved_chunks
        if (c.get("id") or c.get("chunk_id"))
    ]

    critic_verdict = final_state.get("critic_verdict") or {}
    initial_critic_verdict = final_state.get("first_critic_verdict") or critic_verdict
    retry_count = final_state.get("retry_count", 0)

    from backend.agent.llm_client import get_last_model_used
    actual_gen = final_state.get("generator_model") or get_last_model_used() or gen_model
    actual_critic = final_state.get("critic_model") or critic_model

    exec_info = {
        "retry_count": retry_count,
        "critic_verdict": critic_verdict,
        "initial_critic_verdict": initial_critic_verdict,
        "final_grounded": final_state.get("final_grounded", False),
        "unsupported_claims_final": final_state.get("unsupported_claims_final", []),
        "generator_model": actual_gen,
        "critic_model": actual_critic,
    }
    return final_answer, retrieved_contexts, retrieved_chunk_ids, exec_info


def generate_single_question_with_retries(
    config: str,
    query: str,
    role_models: Dict[str, str],
    max_transient_retries: int = 5,
) -> Tuple[str, List[str], List[str], Dict[str, Any], str, Optional[str]]:
    """
    Executes answer generation with transient retry backoff (up to 5 attempts).
    - On 429/quota error: Raises AllModelsExhaustedError immediately.
    - On 503/transient error: Retries up to 5 times with exponential backoff.
    - If error persists: Returns status='error' with error details.
    """
    gen_model = role_models["gen_model"]
    critic_model = role_models["critic_model"]
    classifier_model = role_models["classifier_model"]
    rewriter_model = role_models["rewriter_model"]

    last_error: Optional[Exception] = None

    for attempt in range(1, max_transient_retries + 1):
        try:
            if config == "baseline":
                ans, ctxs, c_ids, b_info = run_baseline_pipeline(query, gen_model)
                return ans, ctxs, c_ids, b_info, "success", None
            else:
                ans, ctxs, c_ids, e_info = run_improved_pipeline(
                    query, gen_model, critic_model, classifier_model, rewriter_model
                )
                return ans, ctxs, c_ids, e_info, "success", None

        except AllModelsExhaustedError:
            raise

        except Exception as e:
            last_error = e
            if is_quota_error(e):
                raise AllModelsExhaustedError(attempted_models=[gen_model], last_error=e)
            elif is_transient_error(e):
                logger.warning(
                    f"Transient error on query (attempt {attempt}/{max_transient_retries}): {e}. "
                    f"Retrying with exponential backoff..."
                )
                if attempt < max_transient_retries:
                    backoff_sec = min(32.0, 2.0 ** attempt)
                    time.sleep(backoff_sec)
                else:
                    logger.error(f"Transient error persisted after {max_transient_retries} attempts: {e}")
            else:
                logger.error(f"Non-transient error during generation: {e}")
                break

    return "", [], [], {}, "error", str(last_error)


def save_answers_file(path: Path, data: Dict[str, Any]) -> None:
    """Safely saves intermediate or final answers to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_generation_stage(
    config: str,
    golden_set: List[Dict[str, Any]],
    role_models: Dict[str, str],
    answers_path: Path,
    ledger: QuotaLedger,
    start_from: int = 1,
    delay: float = 0.0,
    seed: int = 42,
    restart: bool = False,
    golden_set_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Stage 1: Generates answers and contexts for benchmark questions.
    Enforces quota budget checks prior to each question, stopping gracefully with exit code 2 if exhausted.
    """
    items: List[Dict[str, Any]] = []
    current_fingerprint = compute_pipeline_fingerprint(golden_set_path=golden_set_path)

    # If --restart is requested, archive any existing answers file and reset start_from to 1
    if restart and answers_path.exists():
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = answers_path.parent / f"{config}_answers_archived_{timestamp_str}.json"
        try:
            answers_path.rename(archive_path)
            logger.info(
                f"--restart specified: Archived {answers_path.name} to {archive_path.name}. "
                f"Starting clean from question 1."
            )
        except Exception as e:
            logger.warning(f"Failed to archive previous answers file {answers_path}: {e}")
        start_from = 1

    # Resume from previous answers if start_from > 1
    if start_from > 1 and answers_path.exists():
        try:
            with open(answers_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_fingerprint = prev_data.get("pipeline_fingerprint")
            if not prev_fingerprint or prev_fingerprint != current_fingerprint:
                print("\n" + "!" * 80)
                print("PIPELINE FINGERPRINT MISMATCH DETECTED ON RESUME:")
                print(f"  Existing file fingerprint:     {prev_fingerprint}")
                print(f"  Current pipeline fingerprint:  {current_fingerprint}")
                print(f"The codebase or golden set has changed since {answers_path.name} was generated.")
                print("Cannot safely resume across different pipeline versions.")
                print("Please rerun with --restart to archive the previous file and start clean.")
                print("!" * 80 + "\n")
                sys.exit(1)

            prev_items = prev_data.get("items", [])
            items = prev_items[: start_from - 1]
            logger.info(f"Resuming generation: Loaded {len(items)} prior answers from {answers_path}.")
        except SystemExit:
            raise
        except Exception as e:
            logger.warning(f"Could not load previous answers to resume: {e}")

    start_idx = max(1, start_from)
    for idx, g_item in enumerate(golden_set[start_idx - 1 :], start=start_idx):
        q = g_item.get("question", "")
        gt = g_item.get("ground_truth", "")
        g_id = g_item.get("id", f"golden_{idx:02d}")
        source_chunk_ids = g_item.get("source_chunk_ids") or (
            [g_item["chunk_id"]] if isinstance(g_item.get("chunk_id"), str) else (g_item.get("chunk_id") or [])
        )
        category = g_item.get("category", "simple_lookup")

        logger.info(f"[{idx}/{len(golden_set)}] Generating for '{config}' query ({g_id}): '{q[:60]}...'")

        # Quota budget pre-check: verify budget remains before invoking model
        def save_current_progress():
            payload = {
                "config": config,
                "stage": "generate",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit_hash": get_git_commit_hash(),
                "pipeline_fingerprint": current_fingerprint,
                "golden_set_size": len(golden_set),
                "seed": seed,
                "models": role_models,
                "checkpoint_question_index": idx,
                "completed_questions": len(items),
                "items": items,
            }
            save_answers_file(answers_path, payload)

        ledger.check_budget_or_stop(
            model=role_models["gen_model"],
            save_partial_callback=save_current_progress,
            next_index=idx,
        )
        if config == "improved":
            ledger.check_budget_or_stop(
                model=role_models["critic_model"],
                save_partial_callback=save_current_progress,
                next_index=idx,
            )

        try:
            ans, ctxs, chunk_ids, exec_info, status, err_msg = generate_single_question_with_retries(
                config=config,
                query=q,
                role_models=role_models,
            )

            item_record: Dict[str, Any] = {
                "id": g_id,
                "category": category,
                "question": q,
                "answer": ans,
                "contexts": ctxs if ctxs else [""],
                "retrieved_chunk_ids": chunk_ids,
                "source_chunk_ids": source_chunk_ids,
                "ground_truth": gt,
                "status": status,
                "error": err_msg,
                "generator_model": exec_info.get("generator_model") or role_models["gen_model"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }

            if config == "improved":
                item_record["critic_model"] = exec_info.get("critic_model") or role_models["critic_model"]
                item_record["retry_count"] = exec_info.get("retry_count", 0)
                item_record["critic_verdict"] = exec_info.get("critic_verdict", {})
                item_record["initial_critic_verdict"] = exec_info.get("initial_critic_verdict", {})
                item_record["final_grounded"] = exec_info.get("final_grounded", False)
                item_record["unsupported_claims_final"] = exec_info.get("unsupported_claims_final", [])

            items.append(item_record)

            answers_payload = {
                "config": config,
                "stage": "generate",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit_hash": get_git_commit_hash(),
                "pipeline_fingerprint": current_fingerprint,
                "golden_set_size": len(golden_set),
                "seed": seed,
                "models": role_models,
                "completed_questions": len(items),
                "items": items,
            }
            save_answers_file(answers_path, answers_payload)

        except AllModelsExhaustedError as e:
            logger.error(
                f"Daily quota exhausted for model '{role_models['gen_model']}' at question {idx}/{len(golden_set)} "
                f"— partial results saved, resume with --start-from {idx}"
            )
            print("\n" + "!" * 75)
            print(f"Daily quota exhausted for model '{role_models['gen_model']}' at question {idx}/{len(golden_set)} — partial results saved, resume with --start-from {idx}")
            print(f"Attempted models: {e.attempted_models}")
            print("!" * 75 + "\n")
            save_current_progress()
            sys.exit(1)

        if idx < len(golden_set) and delay > 0:
            time.sleep(delay)

    logger.info(f"Stage 1 (generate) completed: {len(items)} items written to {answers_path}")
    return answers_payload


def compute_heuristic_smoke_metrics(
    scored_items: List[Dict[str, Any]],
    metrics_to_run: List[str],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Fast heuristic metrics mode for smoke testing data flow without API calls.
    Computes exact retrieval metrics + numeric match, with heuristic estimates for LLM metrics.
    """
    per_question: List[Dict[str, Any]] = []
    faithfulness_scores: List[float] = []
    correctness_scores: List[float] = []
    hit_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    numeric_scores: List[float] = []

    for item in scored_items:
        ans = item.get("answer", "")
        gt = item.get("ground_truth", "")
        source_chunk_ids = item.get("source_chunk_ids", [])
        retrieved_chunk_ids = item.get("retrieved_chunk_ids", [])
        category = item.get("category", "simple_lookup")

        # 1. Deterministic retrieval metrics
        rm = compute_retrieval_metrics(retrieved_chunk_ids, source_chunk_ids, k=8)
        if not rm.get("is_unanswerable"):
            hit_scores.append(rm["hit@k"])
            recall_scores.append(rm["recall@k"])
            mrr_scores.append(rm["mrr"])

        # 2. Deterministic numeric match
        num_matched = None
        if category == "numeric_precision":
            num_matched = check_numeric_precision_match(ans, gt)
            numeric_scores.append(1.0 if num_matched else 0.0)

        # 3. Heuristic approximations for LLM metrics
        faith = None
        if "faithfulness" in metrics_to_run:
            ans_lower = ans.lower()
            gt_lower = gt.lower()
            is_unanswerable_gt = "do not contain" in gt_lower or "don't have enough information" in gt_lower
            is_unanswerable_ans = "don't have enough information" in ans_lower or "do not contain" in ans_lower
            if is_unanswerable_gt:
                faith = 1.0 if is_unanswerable_ans else 0.0
            else:
                has_ctx = len(item.get("contexts", [])) > 0 and bool(item.get("contexts", [""])[0].strip())
                faith = 0.95 if has_ctx else 0.2
            faithfulness_scores.append(faith)

        correctness = None
        if "answer_correctness" in metrics_to_run:
            words_ans = set(ans.lower().split())
            words_gt = set(gt.lower().split())
            if words_gt:
                overlap = len(words_ans & words_gt) / len(words_gt | words_ans)
                correctness = min(1.0, round(overlap * 2.5, 4))
            else:
                correctness = 0.85
            correctness_scores.append(correctness)

        res_item = dict(item)
        res_item.update({
            "hit@8": rm["hit@k"],
            "recall@8": rm["recall@k"],
            "mrr": rm["mrr"],
            "numeric_match": num_matched,
            "faithfulness": faith,
            "answer_correctness": correctness,
        })
        per_question.append(res_item)

    agg: Dict[str, float] = {
        "hit@8": round(sum(hit_scores) / max(1, len(hit_scores)), 4),
        "recall@8": round(sum(recall_scores) / max(1, len(recall_scores)), 4),
        "mrr": round(sum(mrr_scores) / max(1, len(mrr_scores)), 4),
    }
    if numeric_scores:
        agg["numeric_accuracy"] = round(sum(numeric_scores) / len(numeric_scores), 4)
    if faithfulness_scores:
        agg["faithfulness"] = round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
    if correctness_scores:
        agg["answer_correctness"] = round(sum(correctness_scores) / len(correctness_scores), 4)

    return agg, per_question


class QuotaTrackingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that routes judge LLM invocations to the quota ledger hook
    and enforces quota budget / max-calls prior to dispatch.
    """

    def __init__(self, model_name: str, ledger: QuotaLedger):
        super().__init__()
        self.model_name = model_name
        self.ledger = ledger
        self._in_wrapper = False

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        if not self._in_wrapper:
            self.ledger.check_budget_or_stop(model=self.model_name)
            notify_llm_call(self.model_name)

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any) -> None:
        if not self._in_wrapper:
            self.ledger.check_budget_or_stop(model=self.model_name)
            notify_llm_call(self.model_name)


class QuotaTrackingLangchainLLMWrapper(LangchainLLMWrapper):
    """
    RAGAS LangchainLLMWrapper that intercepts generate_text and agenerate_text calls,
    checks quota budget / max-calls, and records calls via notify_llm_call.
    """

    def __init__(
        self,
        langchain_llm: Any,
        model_name: str,
        ledger: QuotaLedger,
        **kwargs: Any,
    ):
        super().__init__(langchain_llm, **kwargs)
        self.model_name = model_name
        self.ledger = ledger

    def generate_text(self, *args: Any, **kwargs: Any) -> Any:
        self.ledger.check_budget_or_stop(model=self.model_name)
        notify_llm_call(self.model_name)
        if hasattr(self.langchain_llm, "callbacks") and self.langchain_llm.callbacks:
            for cb in self.langchain_llm.callbacks:
                if isinstance(cb, QuotaTrackingCallbackHandler):
                    cb._in_wrapper = True
        try:
            return super().generate_text(*args, **kwargs)
        finally:
            if hasattr(self.langchain_llm, "callbacks") and self.langchain_llm.callbacks:
                for cb in self.langchain_llm.callbacks:
                    if isinstance(cb, QuotaTrackingCallbackHandler):
                        cb._in_wrapper = False

    async def agenerate_text(self, *args: Any, **kwargs: Any) -> Any:
        self.ledger.check_budget_or_stop(model=self.model_name)
        notify_llm_call(self.model_name)
        if hasattr(self.langchain_llm, "callbacks") and self.langchain_llm.callbacks:
            for cb in self.langchain_llm.callbacks:
                if isinstance(cb, QuotaTrackingCallbackHandler):
                    cb._in_wrapper = True
        try:
            return await super().agenerate_text(*args, **kwargs)
        finally:
            if hasattr(self.langchain_llm, "callbacks") and self.langchain_llm.callbacks:
                for cb in self.langchain_llm.callbacks:
                    if isinstance(cb, QuotaTrackingCallbackHandler):
                        cb._in_wrapper = False


def run_judge_stage(
    config: str,
    answers_path: Path,
    results_path: Path,
    role_models: Dict[str, str],
    golden_set_size: int,
    metrics_to_run: List[str],
    ledger: QuotaLedger,
    fast_eval: bool = False,
    seed: int = 42,
    golden_set_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Stage 2: Loads answers from {config}_answers.json, computes deterministic retrieval metrics
    and numeric match, optionally evaluates faithfulness and answer correctness via RAGAS,
    and saves output to {config}.json.
    """
    register_llm_call_hook(ledger.record_call)
    if not answers_path.exists():
        logger.error(f"Cannot run judge stage: answers file not found at {answers_path}. Run with '--stage generate' first.")
        sys.exit(1)

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    all_items: List[Dict[str, Any]] = answers_data.get("items", [])
    if not all_items:
        logger.error(f"No items found in answers file {answers_path}.")
        sys.exit(1)

    scored_items = [it for it in all_items if it.get("status") != "error"]
    error_items = [it for it in all_items if it.get("status") == "error"]
    n_scored = len(scored_items)
    n_errors = len(error_items)

    logger.info(f"Loaded {len(all_items)} total items: {n_scored} scored, {n_errors} errors excluded from scoring.")

    if n_scored == 0:
        logger.error("No successfully generated answers available to score (all items marked status='error').")
        sys.exit(1)

    judge_model = role_models["judge_model"]
    metric_source = "ragas" if (metrics_to_run and not fast_eval) else ("heuristic" if fast_eval else "llm_free")
    actual_judge_model = judge_model
    fallback_reason: Optional[str] = None

    # Step 1: Compute deterministic retrieval metrics and numeric match for all items
    hit_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    numeric_scores: List[float] = []

    scored_results_map: Dict[str, Dict[str, Any]] = {}
    for it in scored_items:
        rm = compute_retrieval_metrics(
            retrieved_chunk_ids=it.get("retrieved_chunk_ids", []),
            source_chunk_ids=it.get("source_chunk_ids", []),
            k=8,
        )
        if not rm.get("is_unanswerable"):
            hit_scores.append(rm["hit@k"])
            recall_scores.append(rm["recall@k"])
            mrr_scores.append(rm["mrr"])

        num_match = None
        if it.get("category") == "numeric_precision":
            num_match = check_numeric_precision_match(it.get("answer", ""), it.get("ground_truth", ""))
            numeric_scores.append(1.0 if num_match else 0.0)

        merged = dict(it)
        merged["hit@8"] = rm["hit@k"]
        merged["recall@8"] = rm["recall@k"]
        merged["mrr"] = rm["mrr"]
        merged["numeric_match"] = num_match
        scored_results_map[it["id"]] = merged

    aggregate_scores: Dict[str, float] = {
        "hit@8": round(sum(hit_scores) / max(1, len(hit_scores)), 4),
        "recall@8": round(sum(recall_scores) / max(1, len(recall_scores)), 4),
        "mrr": round(sum(mrr_scores) / max(1, len(mrr_scores)), 4),
    }
    if numeric_scores:
        aggregate_scores["numeric_accuracy"] = round(sum(numeric_scores) / len(numeric_scores), 4)

    # Step 2: Evaluate LLM-judged metrics if requested
    if not metrics_to_run or "none" in metrics_to_run:
        logger.info("LLM-free mode enabled (--metrics none): Skipping LLM judge calls entirely.")
        metric_source = "llm_free"

    elif fast_eval:
        logger.warning("Running in heuristic smoke metrics mode (--fast-eval).")
        metric_source = "heuristic"
        heuristic_agg, heuristic_items = compute_heuristic_smoke_metrics(scored_items, metrics_to_run)
        aggregate_scores.update(heuristic_agg)
        for h_item in heuristic_items:
            scored_results_map[h_item["id"]].update({
                "faithfulness": h_item.get("faithfulness"),
                "answer_correctness": h_item.get("answer_correctness"),
            })

    else:
        # Check judge model budget before invoking
        ledger.check_budget_or_stop(model=judge_model, next_index=1)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable is missing for judge stage!")
            sys.exit(1)

        raw_fallbacks = os.getenv("JUDGE_FALLBACK_MODELS", "gemini-flash-lite-latest,gemini-3.5-flash").split(",")
        fallback_judge_models = [
            m.strip()
            for m in raw_fallbacks
            if m.strip() and "1.5" not in m and "2.5-flash" not in m
        ]
        candidate_judges = [judge_model] + [m for m in fallback_judge_models if m != judge_model]

        questions = [it["question"] for it in scored_items]
        answers = [it["answer"] for it in scored_items]
        contexts_list = [it.get("contexts", [""]) for it in scored_items]
        ground_truths = [it["ground_truth"] for it in scored_items]

        eval_succeeded = False
        last_eval_error = None

        for current_judge in candidate_judges:
            try:
                logger.info(f"Attempting real RAGAS evaluation with judge model: '{current_judge}'")
                callback_handler = QuotaTrackingCallbackHandler(model_name=current_judge, ledger=ledger)
                judge_llm = ChatGoogleGenerativeAI(
                    model=current_judge,
                    google_api_key=api_key,
                    temperature=0.0,
                    request_timeout=90.0,
                    max_retries=2,
                    callbacks=[callback_handler],
                )
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=os.getenv("RAGAS_EMBEDDING_MODEL", "gemini-embedding-001"),
                    google_api_key=api_key,
                )

                ragas_llm = QuotaTrackingLangchainLLMWrapper(
                    judge_llm,
                    model_name=current_judge,
                    ledger=ledger,
                    bypass_n=True,
                )
                ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

                metrics_instances = []
                if "faithfulness" in metrics_to_run:
                    metrics_instances.append(Faithfulness(llm=ragas_llm))
                if "answer_correctness" in metrics_to_run:
                    metrics_instances.append(AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings))

                ragas_dataset = Dataset.from_dict({
                    "user_input": questions,
                    "response": answers,
                    "retrieved_contexts": contexts_list,
                    "reference": ground_truths,
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts_list,
                    "ground_truth": ground_truths,
                })

                run_config = RunConfig(max_workers=1, timeout=120)
                logger.info(f"Executing RAGAS evaluate() with {len(questions)} items for metrics: {metrics_to_run}...")

                ragas_results = evaluate(
                    dataset=ragas_dataset,
                    metrics=metrics_instances,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=run_config,
                )
                df_results = ragas_results.to_pandas()

                if "faithfulness" in metrics_to_run and "faithfulness" in df_results:
                    val = float(df_results["faithfulness"].mean())
                    if not math.isnan(val):
                        aggregate_scores["faithfulness"] = val
                if "answer_correctness" in metrics_to_run and "answer_correctness" in df_results:
                    val = float(df_results["answer_correctness"].mean())
                    if not math.isnan(val):
                        aggregate_scores["answer_correctness"] = val

                df_records = df_results.to_dict(orient="records")
                for i, rec in enumerate(df_records):
                    orig_item = scored_items[i]
                    item_update = {}
                    if "faithfulness" in metrics_to_run:
                        item_update["faithfulness"] = rec.get("faithfulness", 0.0)
                    if "answer_correctness" in metrics_to_run:
                        item_update["answer_correctness"] = rec.get("answer_correctness", 0.0)
                    scored_results_map[orig_item["id"]].update(item_update)

                actual_judge_model = current_judge
                eval_succeeded = True
                break

            except Exception as e:
                last_eval_error = e
                logger.warning(f"RAGAS evaluate() failed with judge '{current_judge}': {e}")

        if not eval_succeeded:
            fallback_reason = str(last_eval_error or "RAGAS evaluation failed across all candidate judge models.")
            logger.error(
                f"CRITICAL: Real RAGAS evaluation failed or was quota-exhausted: {fallback_reason}\n"
                "Falling back to HEURISTIC SMOKE METRICS for logging. Script will exit with code 1."
            )
            metric_source = "heuristic_fallback"
            heuristic_agg, heuristic_items = compute_heuristic_smoke_metrics(scored_items, metrics_to_run)
            aggregate_scores.update(heuristic_agg)
            for h_item in heuristic_items:
                scored_results_map[h_item["id"]].update({
                    "faithfulness": h_item.get("faithfulness"),
                    "answer_correctness": h_item.get("answer_correctness"),
                })

    # Assemble per_question_results preserving original sequence
    per_question_results: List[Dict[str, Any]] = []
    for it in all_items:
        item_id = it["id"]
        if it.get("status") == "error":
            err_record = dict(it)
            err_record["hit@8"] = None
            err_record["recall@8"] = None
            err_record["mrr"] = None
            err_record["numeric_match"] = None
            err_record["faithfulness"] = None
            err_record["answer_correctness"] = None
            per_question_results.append(err_record)
        else:
            per_question_results.append(scored_results_map.get(item_id, it))

    output_data: Dict[str, Any] = {
        "config": config,
        "stage": "judge",
        "metric_source": metric_source,
        "metrics_evaluated": metrics_to_run,
        "models": role_models,
        "judge_model": actual_judge_model,
        "gen_model": role_models["gen_model"],
        "critic_model": role_models["critic_model"],
        "classifier_model": role_models["classifier_model"],
        "rewriter_model": role_models["rewriter_model"],
        "bench_model": role_models["bench_model"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "pipeline_fingerprint": compute_pipeline_fingerprint(golden_set_path=golden_set_path),
        "golden_set_size": golden_set_size,
        "seed": seed,
        "total_evaluated": len(all_items),
        "n_scored": n_scored,
        "n_errors": n_errors,
        "aggregate_scores": aggregate_scores,
        "per_question_results": per_question_results,
    }

    if metric_source == "heuristic_fallback":
        output_data["ragas_evaluation_failed"] = True
        output_data["fallback_warning"] = (
            "CRITICAL: Real RAGAS evaluation failed or was quota-exhausted. "
            "These scores are HEURISTIC SMOKE METRICS and MUST NOT be reported as real RAGAS scores!"
        )
        output_data["fallback_reason"] = fallback_reason

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved evaluation results to: {results_path}")

    # Step 4: Print summary table to console
    print("\n" + "=" * 65)
    print(f"VERITY RAG EVALUATION SUMMARY [{config.upper()} CONFIG]")
    print(f"Metric Source:                 {metric_source.upper()}")
    print(f"Generator Model (GEN_MODEL):   {role_models['gen_model']}")
    print(f"Critic Model (CRITIC_MODEL):   {role_models['critic_model']}")
    print(f"Judge Model (JUDGE_MODEL):     {actual_judge_model}")
    print(f"Questions Evaluated:           {len(all_items)}")
    print(f"Questions Scored (n_scored):   {n_scored}")
    print(f"Questions Errored (n_errors):  {n_errors}")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Score':<10}")
    print("-" * 65)
    print(f"{'Hit@8 (LLM-free)':<25} | {aggregate_scores.get('hit@8', 0.0):.4f}")
    print(f"{'Recall@8 (LLM-free)':<25} | {aggregate_scores.get('recall@8', 0.0):.4f}")
    print(f"{'MRR (LLM-free)':<25} | {aggregate_scores.get('mrr', 0.0):.4f}")
    if "numeric_accuracy" in aggregate_scores:
        print(f"{'Numeric Accuracy':<25} | {aggregate_scores['numeric_accuracy']:.4f}")
    if "faithfulness" in aggregate_scores:
        print(f"{'Faithfulness':<25} | {aggregate_scores['faithfulness']:.4f}")
    if "answer_correctness" in aggregate_scores:
        print(f"{'Answer Correctness':<25} | {aggregate_scores['answer_correctness']:.4f}")
    print("=" * 65)
    print(f"Answers saved to:  backend/eval/results/{config}_answers.json")
    print(f"Final results:     backend/eval/results/{config}.json\n")

    if metric_source == "heuristic_fallback":
        print("\n" + "!" * 80)
        print("CRITICAL: REAL RAGAS EVALUATION FAILED OR WAS QUOTA-EXHAUSTED.")
        print(f"REASON: {fallback_reason}")
        print("FALLBACK HEURISTIC SMOKE METRICS WERE SAVED WITH 'metric_source': 'heuristic_fallback'.")
        print("THESE ARE NOT REAL RAGAS SCORES. SCRIPT EXITING WITH NON-ZERO STATUS (1).")
        print("!" * 80 + "\n")
        sys.exit(1)

    return output_data


def filter_and_sample_golden_set(
    items: List[Dict[str, Any]],
    filter_eligible: bool = False,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Filters and/or stratifies benchmark items.
    If filter_eligible is True, retains only answerable, single-year, answer_quality_eligible items.
    If sample_size is specified, performs stratified sampling across (company, section) using
    the Largest Remainder Method.
    """
    result = list(items)
    if filter_eligible:
        result = [
            it
            for it in result
            if it.get("source_chunk_ids")
            and not it.get("expected_decline")
            and it.get("category") != "unanswerable"
            and "/" not in str(it.get("fiscal_year", ""))
            and (
                it.get("metadata", {}).get("answer_quality_eligible") is True
                or it.get("answer_quality_eligible") is True
            )
        ]

    if sample_size is not None and sample_size < len(result):
        strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for it in result:
            key = (str(it.get("company", "Unknown")), str(it.get("section", "Unknown")))
            strata.setdefault(key, []).append(it)

        total_eligible = len(result)
        allocations: Dict[Tuple[str, str], int] = {}
        remainders = []
        for k, s_items in sorted(strata.items()):
            exact = sample_size * len(s_items) / total_eligible
            base = int(exact)
            rem = exact - base
            allocations[k] = base
            remainders.append((rem, k))

        allocated = sum(allocations.values())
        remainders.sort(key=lambda x: x[0], reverse=True)
        for i in range(sample_size - allocated):
            allocations[remainders[i][1]] += 1

        rng = random.Random(seed)
        sampled: List[Dict[str, Any]] = []
        for k in sorted(strata.keys()):
            s_items = sorted(strata[k], key=lambda x: str(x.get("id", "")))
            rng.shuffle(s_items)
            sampled.extend(s_items[: allocations[k]])

        # Preserve original relative order
        orig_indices = {id(it): idx for idx, it in enumerate(result)}
        sampled.sort(key=lambda x: orig_indices.get(id(x), 0))
        result = sampled

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Free-Tier Hardened RAGAS evaluation on Verity Financial RAG benchmark.")
    parser.add_argument(
        "--config",
        choices=["baseline", "improved"],
        required=True,
        help="Evaluation pipeline config: 'baseline' (vector-only) or 'improved' (full agentic graph).",
    )
    parser.add_argument(
        "--stage",
        choices=["generate", "judge", "all"],
        default="all",
        help="Execution stage: 'generate' (produce answers/contexts), 'judge' (score answers), or 'all' (both).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of questions to evaluate (for testing/rate-limiting).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="faithfulness,answer_correctness",
        help="Comma-separated LLM-judged metrics to evaluate: 'faithfulness', 'answer_correctness', or 'none'.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum total LLM calls allowed before graceful stop with exit code 2.",
    )
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="Use fast heuristic smoke metrics (NOT real RAGAS scores; for pipeline smoke testing only).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between evaluating questions (to respect rate limits).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="1-indexed question index to resume evaluation from (preserves prior answers).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated calls needed vs remaining budget per model and days to complete, then exit 0.",
    )
    parser.add_argument(
        "--allow-same-judge",
        action="store_true",
        help="Allow JUDGE_MODEL to equal GEN_MODEL or CRITIC_MODEL despite warning.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Archive existing answers file to {config}_answers_archived_{timestamp}.json and start clean from question 1.",
    )
    parser.add_argument(
        "--golden-path",
        type=str,
        default=str(BASE_DIR / "eval" / "golden_set.json"),
        help="Path to golden benchmark JSON file (default: backend/eval/golden_set.json).",
    )
    parser.add_argument(
        "--filter-eligible",
        action="store_true",
        help="Filter benchmark to answerable, single-year, answer_quality_eligible items.",
    )
    parser.add_argument(
        "--stratified-sample",
        type=int,
        default=None,
        help="Sample N items stratified by (company, section) with fixed seed.",
    )
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    # Resolve per-role model environment variables
    role_models = resolve_role_models()

    # Setup Quota Ledger tracking
    ledger = QuotaLedger(max_calls=args.max_calls)
    register_llm_call_hook(ledger.record_call)

    # Parse metrics selection
    raw_metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    metrics_to_run = [m for m in raw_metrics if m in ["faithfulness", "answer_correctness"]]
    if "none" in raw_metrics:
        metrics_to_run = []

    golden_set_path = Path(args.golden_path).resolve()
    if not golden_set_path.exists():
        logger.error(f"Benchmark file {golden_set_path.name} not found at: {golden_set_path}")
        sys.exit(1)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set: List[Dict[str, Any]] = json.load(f)

    # Validate that every source_chunk_id in golden_set.json exists in the current corpus
    from backend.eval.remap_source_chunks import validate_golden_source_chunks
    try:
        missing_ids = validate_golden_source_chunks(golden_set_path=golden_set_path)
        if missing_ids:
            logger.error(
                f"CRITICAL: Golden set validation failed! {len(missing_ids)} source_chunk_id(s) do not exist "
                f"in current corpus:\n" + "\n".join(f"  - {mid}" for mid in missing_ids)
            )
            print(
                f"\n{'!' * 80}\n"
                f"CRITICAL: Golden set validation failed!\n"
                f"{len(missing_ids)} source_chunk_id(s) in {golden_set_path.name} do not exist in the current corpus:\n"
                + "\n".join(f"  - {mid}" for mid in missing_ids)
                + f"\nRefusing to run evaluation. Remap IDs using 'python -m backend.eval.remap_source_chunks'.\n"
                + f"{'!' * 80}\n"
            )
            sys.exit(1)
    except FileNotFoundError as e:
        logger.warning(f"Skipping corpus chunk validation: {e}")

    if args.filter_eligible or args.stratified_sample:
        golden_set = filter_and_sample_golden_set(
            golden_set,
            filter_eligible=args.filter_eligible,
            sample_size=args.stratified_sample,
            seed=args.seed,
        )
        logger.info(
            f"Benchmark filtered/stratified: {len(golden_set)} items in scope "
            f"(filter_eligible={args.filter_eligible}, stratified_sample={args.stratified_sample})"
        )

    if args.limit:
        golden_set = golden_set[: args.limit]
        logger.info(f"Limiting evaluation to first {args.limit} questions from golden_set.json")

    results_dir = BASE_DIR / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    answers_path = results_dir / f"{args.config}_answers.json"
    results_path = results_dir / f"{args.config}.json"

    # Print API call estimation and budget check before running
    num_questions_in_scope = len(golden_set) - max(0, args.start_from - 1) if args.stage in ["generate", "all"] else len(golden_set)
    print_api_call_estimate_and_budget(
        config=args.config,
        stage=args.stage,
        num_questions=max(0, num_questions_in_scope),
        role_models=role_models,
        metrics_to_run=metrics_to_run,
        fast_eval=args.fast_eval,
        dry_run=args.dry_run,
        ledger=ledger,
    )

    # Check judge model independence if judging is requested with LLM metrics
    if args.stage in ["judge", "all"] and metrics_to_run and not args.fast_eval:
        check_judge_model_independence(
            gen_model=role_models["gen_model"],
            critic_model=role_models["critic_model"],
            judge_model=role_models["judge_model"],
            config=args.config,
            allow_same_judge=args.allow_same_judge,
        )

    # Stage 1: Generate answers and contexts
    if args.stage in ["generate", "all"]:
        run_generation_stage(
            config=args.config,
            golden_set=golden_set,
            role_models=role_models,
            answers_path=answers_path,
            ledger=ledger,
            start_from=args.start_from,
            delay=args.delay,
            seed=args.seed,
            restart=args.restart,
            golden_set_path=golden_set_path,
        )

    # Stage 2: Judge generated answers
    if args.stage in ["judge", "all"]:
        run_judge_stage(
            config=args.config,
            answers_path=answers_path,
            results_path=results_path,
            role_models=role_models,
            golden_set_size=len(golden_set),
            metrics_to_run=metrics_to_run,
            ledger=ledger,
            fast_eval=args.fast_eval,
            seed=args.seed,
            golden_set_path=golden_set_path,
        )


if __name__ == "__main__":
    main()
