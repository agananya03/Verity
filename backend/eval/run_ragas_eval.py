"""
RAGAS Evaluation Runner for Verity Financial RAG.

Evaluates either:
- Baseline config: Vector-only Qdrant search + single LLM draft generation.
- Improved config: Full verity_graph pipeline (Hybrid Dense/Sparse search + metadata filtering + self-correcting Critic loop).

Features:
- Real RAGAS evaluation mode with configurable judge model (JUDGE_MODEL env var) and fallback chain.
- Explicit metric_source tracking ("ragas" vs "heuristic" vs "heuristic_fallback").
- Heuristic smoke metrics code path (compute_heuristic_smoke_metrics) clearly isolated for fast pipeline smoke testing.
- Automatic non-zero exit (sys.exit(1)) and loud warnings if real RAGAS fails or is quota-exhausted and falls back to heuristics.
- Logs metadata: judge_model, generator_models, UTC timestamp, git commit hash, golden_set_size, and random seed.
- Resumes evaluation from checkpoint using --start-from and enforces rate limiting via --delay.
- Saves aggregate scores and per-question breakdowns to:
  - backend/eval/results/baseline.json
  - backend/eval/results/improved.json
"""

import argparse
import datetime
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

# Compatibility patch for ragas importing langchain_community.chat_models.vertexai
vertexai_mock = types.ModuleType("vertexai")
vertexai_mock.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mock

# Setup sys.path for backend imports
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
try:
    from ragas.metrics.collections import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError:
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
from ragas.run_config import RunConfig

from backend.agent.graph import verity_graph
from backend.agent.llm_client import AllModelsExhaustedError, get_llm_with_fallback
from backend.agent.retrieval import search_qdrant
from backend.agent.state import AgentState

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


def run_baseline_pipeline(query: str) -> Tuple[str, List[str], List[str]]:
    """
    Baseline pipeline:
    - Vector-only search via Qdrant (top_k=8, no metadata filters, no BM25 fusion).
    - Single LLM draft generation call using get_llm_with_fallback with timeout.
    Returns:
        (answer, retrieved_contexts, retrieved_chunk_ids)
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
        )

    context_text = "\n\n".join([f"[Chunk {i+1}]\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
    prompt = f"""Answer the following financial question accurately using ONLY the provided Context Chunks.
If the information is insufficient, state "I don't have enough information to answer this question based on the retrieved SEC filings."

Context Chunks:
{context_text}

Question: {query}
Answer:"""

    baseline_model = os.getenv("MODEL_BASELINE", os.getenv("MODEL_DRAFT", "gemini-3.5-flash"))
    llm = get_llm_with_fallback(
        primary_model=baseline_model,
        fallback_models=["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"],
        google_api_key=api_key or None,
        temperature=0.1,
        request_timeout=30.0,
        max_retries=2,
    )
    response = llm.invoke([("user", prompt)])
    if isinstance(response.content, list):
        answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content]).strip()
    else:
        answer = str(response.content).strip()

    return answer, retrieved_contexts, retrieved_chunk_ids


def run_improved_pipeline(query: str) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    """
    Improved pipeline:
    - End-to-end verity_graph execution.
    - Includes metadata classifier, hybrid dense+sparse retrieval with RRF, draft generation, and self-correcting Critic loop.
    - Includes retry logic with backoff for rate limits.
    Returns:
        (final_answer, retrieved_contexts, retrieved_chunk_ids, exec_info)
    """
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

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
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

            exec_info = {
                "retry_count": retry_count,
                "critic_verdict": critic_verdict,
                "initial_critic_verdict": initial_critic_verdict,
                "final_grounded": final_state.get("final_grounded", False),
                "unsupported_claims_final": final_state.get("unsupported_claims_final", []),
            }
            return final_answer, retrieved_contexts, retrieved_chunk_ids, exec_info
        except AllModelsExhaustedError:
            raise
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.warning(f"Rate limit hit on attempt {attempt}/{max_attempts}. Reloading .env and sleeping 6 seconds before retry...")
                load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
                time.sleep(6)
            else:
                logger.warning(f"Improved pipeline execution failed: {e}")
                break

    # Fallback if retries exhausted
    hits = search_qdrant(query=query, filters=None, limit=8)
    retrieved_contexts = [h.get("text", "") for h in hits if h.get("text")]
    retrieved_chunk_ids = [
        str(h.get("id") or h.get("chunk_id", ""))
        for h in hits
        if (h.get("id") or h.get("chunk_id"))
    ]
    final_answer = f"Based on retrieved SEC filings: {retrieved_contexts[0][:250]}..." if retrieved_contexts else "I don't have enough information to answer this question based on the retrieved SEC filings."
    fallback_exec_info = {
        "retry_count": 0,
        "critic_verdict": {"grounded": False, "unsupported_claims": ["Pipeline execution fallback"]},
        "initial_critic_verdict": {"grounded": False, "unsupported_claims": ["Pipeline execution fallback"]},
        "final_grounded": False,
        "unsupported_claims_final": ["Pipeline execution fallback"],
    }
    return final_answer, retrieved_contexts, retrieved_chunk_ids, fallback_exec_info


def compute_heuristic_smoke_metrics(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str],
    golden_set_items: List[Dict[str, Any]],
    chunk_ids_list: Optional[List[List[str]]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    HEURISTIC SMOKE METRICS ONLY.

    WARNING: These are heuristic, deterministic approximations (e.g. hardcoded
    faithfulness=0.95 and lexical word overlap for relevancy).
    These MUST NEVER be reported as real RAGAS scores. They exist solely for
    smoke-testing the pipeline and ensuring end-to-end data flow when LLM judge
    APIs are unavailable.
    """
    per_question: List[Dict[str, Any]] = []
    faithfulness_scores: List[float] = []
    answer_relevancy_scores: List[float] = []
    context_precision_scores: List[float] = []
    context_recall_scores: List[float] = []

    for i in range(len(questions)):
        q = questions[i]
        ans = answers[i]
        ctxs = contexts_list[i]
        gt = ground_truths[i]
        g_item = golden_set_items[i] if i < len(golden_set_items) else {}
        g_id = g_item.get("id", f"golden_{i+1:02d}")
        chunk_ids = chunk_ids_list[i] if (chunk_ids_list and i < len(chunk_ids_list)) else []

        ans_lower = ans.lower()
        gt_lower = gt.lower()

        is_unanswerable_gt = "do not contain" in gt_lower or "don't have enough information" in gt_lower
        is_unanswerable_ans = "don't have enough information" in ans_lower or "do not contain" in ans_lower or "insufficient" in ans_lower

        if is_unanswerable_gt:
            if is_unanswerable_ans:
                faith = 1.0
                relevancy = 1.0
                c_prec = 1.0
                c_rec = 1.0
            else:
                faith = 0.0
                relevancy = 0.2
                c_prec = 0.0
                c_rec = 0.0
        else:
            has_context = len(ctxs) > 0 and bool(ctxs[0].strip())
            c_prec = 1.0 if has_context else 0.0
            c_rec = 1.0 if has_context else 0.0

            if is_unanswerable_ans:
                faith = 0.5
                relevancy = 0.3
            else:
                faith = 0.95 if has_context else 0.2

                words_ans = set(ans_lower.split())
                words_gt = set(gt_lower.split())
                if words_gt:
                    overlap = len(words_ans & words_gt) / len(words_gt | words_ans)
                    relevancy = min(1.0, round(overlap * 2.8, 4))
                else:
                    relevancy = 0.85

        faithfulness_scores.append(faith)
        answer_relevancy_scores.append(relevancy)
        context_precision_scores.append(c_prec)
        context_recall_scores.append(c_rec)

        per_question.append({
            "id": g_id,
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "retrieved_chunk_ids": chunk_ids,
            "ground_truth": gt,
            "faithfulness": faith,
            "answer_relevancy": relevancy,
            "context_precision": c_prec,
            "context_recall": c_rec,
        })

    avg_f = sum(faithfulness_scores) / max(1, len(faithfulness_scores))
    avg_ar = sum(answer_relevancy_scores) / max(1, len(answer_relevancy_scores))
    avg_cp = sum(context_precision_scores) / max(1, len(context_precision_scores))
    avg_cr = sum(context_recall_scores) / max(1, len(context_recall_scores))

    return {
        "faithfulness": round(avg_f, 4),
        "answer_relevancy": round(avg_ar, 4),
        "context_precision": round(avg_cp, 4),
        "context_recall": round(avg_cr, 4),
    }, per_question


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Verity Financial RAG benchmark.")
    parser.add_argument(
        "--config",
        choices=["baseline", "improved"],
        required=True,
        help="Evaluation pipeline config: 'baseline' (vector-only) or 'improved' (full agentic graph).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of questions to evaluate (for testing/rate-limiting).",
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
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is missing!")
        sys.exit(1)

    judge_model = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
    logger.info(f"Configured RAGAS Judge Model: '{judge_model}' (via env JUDGE_MODEL)")

    # Define generator models metadata
    if args.config == "improved":
        generator_models = {
            "draft": os.getenv("MODEL_DRAFT", "gemini-3.5-flash"),
            "critic": os.getenv("MODEL_CRITIC", "gemini-3.5-flash"),
            "classifier": os.getenv("MODEL_CLASSIFIER", "gemini-3.5-flash"),
            "rewriter": os.getenv("MODEL_REWRITER", "gemini-3.5-flash"),
        }
    else:
        generator_models = {
            "baseline": os.getenv("MODEL_BASELINE", os.getenv("MODEL_DRAFT", "gemini-3.5-flash")),
        }

    golden_set_path = BASE_DIR / "eval" / "golden_set.json"
    if not golden_set_path.exists():
        logger.error(f"Benchmark file golden_set.json not found at: {golden_set_path}")
        sys.exit(1)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set: List[Dict[str, Any]] = json.load(f)

    if args.limit:
        golden_set = golden_set[: args.limit]
        logger.info(f"Limiting evaluation to first {args.limit} questions from golden_set.json")

    logger.info(f"Loaded {len(golden_set)} evaluation benchmark questions for config '{args.config}'.")

    questions: List[str] = []
    answers: List[str] = []
    contexts_list: List[List[str]] = []
    chunk_ids_list: List[List[str]] = []
    ground_truths: List[str] = []
    execution_infos: List[Dict[str, Any]] = []

    # Check for existing results to resume if start_from > 1
    results_dir = BASE_DIR / "eval" / "results"
    results_path = results_dir / f"{args.config}.json"
    if args.start_from > 1 and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                prev_questions = prev_data.get("per_question_results", [])
                for item in prev_questions[: args.start_from - 1]:
                    questions.append(item.get("question", ""))
                    answers.append(item.get("answer", ""))
                    contexts_list.append(item.get("contexts", [""]))
                    chunk_ids_list.append(item.get("retrieved_chunk_ids", []))
                    ground_truths.append(item.get("ground_truth", ""))
                    if args.config == "improved":
                        execution_infos.append({
                            "retry_count": item.get("retry_count", 0),
                            "critic_verdict": item.get("critic_verdict", {}),
                            "initial_critic_verdict": item.get("initial_critic_verdict", {}),
                            "final_grounded": item.get("final_grounded", False),
                            "unsupported_claims_final": item.get("unsupported_claims_final", []),
                        })
                logger.info(f"Resuming evaluation: Loaded {len(questions)} existing answers prior to index {args.start_from}.")
        except Exception as e:
            logger.warning(f"Could not load previous evaluation results to resume: {e}")

    # Step 1: Run pipeline execution for each question starting from args.start_from
    start_idx = max(1, args.start_from)
    for idx, item in enumerate(golden_set[start_idx - 1 :], start=start_idx):
        q = item.get("question", "")
        gt = item.get("ground_truth", "")

        logger.info(f"[{idx}/{len(golden_set)}] Running '{args.config}' pipeline for query: '{q[:60]}...'")

        try:
            if args.config == "baseline":
                answer, contexts, chunk_ids = run_baseline_pipeline(q)
            else:
                answer, contexts, chunk_ids, exec_info = run_improved_pipeline(q)
                execution_infos.append(exec_info)

            questions.append(q)
            answers.append(answer)
            contexts_list.append(contexts if contexts else [""])
            chunk_ids_list.append(chunk_ids)
            ground_truths.append(gt)

        except AllModelsExhaustedError as e:
            logger.error(
                f"Daily quota exhausted across all fallback models at question {idx}/{len(golden_set)} "
                f"— partial results saved, resume with --start-from {idx}"
            )
            print("\n" + "!" * 75)
            print(f"Daily quota exhausted across all fallback models at question {idx}/{len(golden_set)} — partial results saved, resume with --start-from {idx}")
            print(f"Attempted models: {e.attempted_models}")
            print("!" * 75 + "\n")

            if questions:
                partial_per_question = []
                for i in range(len(questions)):
                    g_id = golden_set[i].get("id", f"golden_{i+1:02d}") if i < len(golden_set) else f"q_{i+1}"
                    pq_item = {
                        "id": g_id,
                        "question": questions[i],
                        "answer": answers[i],
                        "contexts": contexts_list[i],
                        "retrieved_chunk_ids": chunk_ids_list[i] if i < len(chunk_ids_list) else [],
                        "ground_truth": ground_truths[i],
                    }
                    if args.config == "improved" and i < len(execution_infos):
                        pq_item["retry_count"] = execution_infos[i].get("retry_count", 0)
                        pq_item["critic_verdict"] = execution_infos[i].get("critic_verdict", {})
                        pq_item["initial_critic_verdict"] = execution_infos[i].get("initial_critic_verdict", {})
                        pq_item["final_grounded"] = execution_infos[i].get("final_grounded", False)
                        pq_item["unsupported_claims_final"] = execution_infos[i].get("unsupported_claims_final", [])
                    partial_per_question.append(pq_item)

                partial_output = {
                    "config": args.config,
                    "metric_source": "partial_unscored",
                    "judge_model": judge_model,
                    "generator_models": generator_models,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "git_commit_hash": get_git_commit_hash(),
                    "golden_set_size": len(golden_set),
                    "seed": args.seed,
                    "total_evaluated": len(questions),
                    "checkpoint_question_index": idx,
                    "per_question_results": partial_per_question,
                }
                results_dir = BASE_DIR / "eval" / "results"
                results_dir.mkdir(parents=True, exist_ok=True)
                partial_path = results_dir / f"{args.config}_partial.json"
                results_path = results_dir / f"{args.config}.json"

                with open(partial_path, "w", encoding="utf-8") as f:
                    json.dump(partial_output, f, indent=2, ensure_ascii=False)
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(partial_output, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved partial results to {partial_path} and {results_path}")

            sys.exit(1)

        except Exception as e:
            logger.error(f"Error running pipeline for question '{q}': {e}")
            questions.append(q)
            answers.append("Error generating answer.")
            contexts_list.append([""])
            chunk_ids_list.append([])
            ground_truths.append(gt)
            if args.config == "improved":
                execution_infos.append({
                    "retry_count": 0,
                    "critic_verdict": {"grounded": False, "unsupported_claims": [str(e)]},
                    "initial_critic_verdict": {"grounded": False, "unsupported_claims": [str(e)]},
                    "final_grounded": False,
                    "unsupported_claims_final": [str(e)],
                })

        if idx < len(golden_set) and args.delay > 0:
            time.sleep(args.delay)

    # Step 2: Metric Evaluation
    logger.info("Initializing evaluation metrics...")

    aggregate_scores: Dict[str, float] = {}
    per_question_breakdown: List[Dict[str, Any]] = []
    metric_source = "ragas"
    fallback_reason: Optional[str] = None

    if args.fast_eval:
        logger.warning("Running in heuristic smoke metrics mode (--fast-eval). Metric source: 'heuristic'.")
        metric_source = "heuristic"
        aggregate_scores, per_question_breakdown = compute_heuristic_smoke_metrics(
            questions, answers, contexts_list, ground_truths, golden_set, chunk_ids_list
        )
    else:
        try:
            fallback_judge_models = [
                m.strip()
                for m in os.getenv("JUDGE_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-1.5-flash").split(",")
                if m.strip()
            ]
            logger.info(f"Using RAGAS LLM judge: primary='{judge_model}', fallbacks={fallback_judge_models}")

            llm = get_llm_with_fallback(
                primary_model=judge_model,
                fallback_models=fallback_judge_models,
                google_api_key=api_key or None,
                temperature=0.0,
                request_timeout=30.0,
                max_retries=2,
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )

            ragas_llm = LangchainLLMWrapper(llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            for metric in metrics:
                metric.llm = ragas_llm
                if hasattr(metric, "embeddings"):
                    metric.embeddings = ragas_embeddings

            ragas_dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths,
            })

            run_config = RunConfig(max_workers=1, timeout=20)

            logger.info("Executing real RAGAS evaluate()...")
            ragas_results = evaluate(
                dataset=ragas_dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
            )
            df_results = ragas_results.to_pandas()

            aggregate_scores = {
                "faithfulness": float(df_results["faithfulness"].mean()) if "faithfulness" in df_results else 0.0,
                "answer_relevancy": float(df_results["answer_relevancy"].mean()) if "answer_relevancy" in df_results else 0.0,
                "context_precision": float(df_results["context_precision"].mean()) if "context_precision" in df_results else 0.0,
                "context_recall": float(df_results["context_recall"].mean()) if "context_recall" in df_results else 0.0,
            }

            if math.isnan(aggregate_scores.get("faithfulness", float("nan"))):
                raise ValueError("RAGAS evaluate() returned NaN scores (API rate limit or quota exhaustion).")

            per_question_breakdown = df_results.to_dict(orient="records")
            # Ensure stable id and chunk_ids on each result
            for i, res in enumerate(per_question_breakdown):
                g_id = golden_set[i].get("id", f"golden_{i+1:02d}") if i < len(golden_set) else f"q_{i+1}"
                res["id"] = g_id
                if i < len(chunk_ids_list):
                    res["retrieved_chunk_ids"] = chunk_ids_list[i]

        except Exception as e:
            fallback_reason = str(e)
            logger.error(
                f"CRITICAL: Real RAGAS evaluation failed or was quota-exhausted: {e}\n"
                "Falling back to HEURISTIC SMOKE METRICS for logging. Script will exit with code 1."
            )
            metric_source = "heuristic_fallback"
            aggregate_scores, per_question_breakdown = compute_heuristic_smoke_metrics(
                questions, answers, contexts_list, ground_truths, golden_set, chunk_ids_list
            )

    # Attach graph execution info to per_question_breakdown for improved config
    if args.config == "improved" and execution_infos:
        for i, res in enumerate(per_question_breakdown):
            if i < len(execution_infos):
                res["retry_count"] = execution_infos[i].get("retry_count", 0)
                res["critic_verdict"] = execution_infos[i].get("critic_verdict", {})
                res["initial_critic_verdict"] = execution_infos[i].get("initial_critic_verdict", {})
                res["final_grounded"] = execution_infos[i].get("final_grounded", False)
                res["unsupported_claims_final"] = execution_infos[i].get("unsupported_claims_final", [])

    # Step 3: Save results to backend/eval/results/{config}.json
    output_data: Dict[str, Any] = {
        "config": args.config,
        "metric_source": metric_source,
        "judge_model": judge_model,
        "generator_models": generator_models,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "golden_set_size": len(golden_set),
        "seed": args.seed,
        "total_evaluated": len(questions),
        "aggregate_scores": aggregate_scores,
        "per_question_results": per_question_breakdown,
    }

    if metric_source == "heuristic_fallback":
        output_data["ragas_evaluation_failed"] = True
        output_data["fallback_warning"] = (
            "CRITICAL: Real RAGAS evaluation failed or was quota-exhausted. "
            "These scores are HEURISTIC SMOKE METRICS and MUST NOT be reported as real RAGAS scores!"
        )
        output_data["fallback_reason"] = fallback_reason

    results_dir = BASE_DIR / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{args.config}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved evaluation results to: {results_path}")

    # Step 4: Print summary table to console
    print("\n" + "=" * 60)
    print(f"RAGAS EVALUATION SUMMARY RESULTS [{args.config.upper()} CONFIG]")
    print(f"Metric Source: {metric_source.upper()}")
    print(f"Judge Model:   {judge_model}")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Score':<10}")
    print("-" * 60)
    print(f"{'Faithfulness':<25} | {aggregate_scores['faithfulness']:.4f}")
    print(f"{'Answer Relevancy':<25} | {aggregate_scores['answer_relevancy']:.4f}")
    print(f"{'Context Precision':<25} | {aggregate_scores['context_precision']:.4f}")
    print(f"{'Context Recall':<25} | {aggregate_scores['context_recall']:.4f}")
    print("=" * 60)
    print(f"Full details saved to: backend/eval/results/{args.config}.json\n")

    # If RAGAS fell back to heuristic smoke metrics, loudly notify and exit non-zero
    if metric_source == "heuristic_fallback":
        print("\n" + "!" * 80)
        print("CRITICAL: REAL RAGAS EVALUATION FAILED OR WAS QUOTA-EXHAUSTED.")
        print(f"REASON: {fallback_reason}")
        print("FALLBACK HEURISTIC SMOKE METRICS WERE SAVED WITH 'metric_source': 'heuristic_fallback'.")
        print("THESE ARE NOT REAL RAGAS SCORES. SCRIPT EXITING WITH NON-ZERO STATUS (1).")
        print("!" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
