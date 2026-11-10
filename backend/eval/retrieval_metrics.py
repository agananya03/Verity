"""
LLM-Free Retrieval & Numeric Evaluation Metrics for Verity Financial RAG.

Provides deterministic, zero-cost evaluation:
1. Retrieval metrics computed against golden source_chunk_ids:
   - hit@k: whether at least one relevant chunk is in top-k
   - recall@k: fraction of relevant chunks captured in top-k
   - mrr (MRR@k): reciprocal rank of the first relevant chunk in top-k
2. Numeric precision check:
   - Normalizes currency ($), commas, percentages (%), and scales ('million', 'billion', 'thousand')
   - Validates whether numbers asserted in ground truth are preserved in the generated answer
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

SCALE_FACTORS = {
    "billion": 1e9,
    "b": 1e9,
    "million": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
    "%": 1.0,
    "percent": 1.0,
}


def compute_retrieval_metrics(
    retrieved_chunk_ids: List[str],
    source_chunk_ids: List[str],
    k: int = 8,
) -> Dict[str, Any]:
    """
    Computes hit@k, recall@k, and MRR@k for a question without any LLM calls.

    Args:
        retrieved_chunk_ids: Ranked list of chunk IDs returned by retriever.
        source_chunk_ids: List of ground-truth target chunk IDs.
        k: Maximum rank depth to evaluate (default 8).

    Returns:
        dict with 'hit@k', 'recall@k', 'mrr', and 'is_unanswerable' flag.
    """
    # Clean and normalize IDs
    top_k = [str(cid).strip() for cid in retrieved_chunk_ids[:k] if str(cid).strip()]
    target_set = set(str(cid).strip() for cid in source_chunk_ids if str(cid).strip())

    if not target_set:
        # Out-of-corpus / unanswerable item (no target chunk exists)
        return {
            "hit@k": 1.0,
            "recall@k": 1.0,
            "mrr": 1.0,
            "is_unanswerable": True,
        }

    # Identify hits within top-k
    matching_ranks = [
        rank for rank, cid in enumerate(top_k, start=1) if cid in target_set
    ]

    hit_at_k = 1.0 if len(matching_ranks) > 0 else 0.0
    unique_hits = set(top_k) & target_set
    recall_at_k = len(unique_hits) / max(1, len(target_set))
    mrr = 1.0 / matching_ranks[0] if matching_ranks else 0.0

    return {
        "hit@k": round(hit_at_k, 4),
        "recall@k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "is_unanswerable": False,
    }


def extract_numbers_with_scales(text: str) -> List[Tuple[float, str, str]]:
    """
    Extracts numerical figures along with their unit/scale from text.
    Handles currency stripping ($), comma removal, percentages, and scale suffixes.
    Returns: list of (scaled_value, raw_scale, raw_token)
    """
    if not text:
        return []

    # Strip currency symbols and extraneous chars, preserve numbers and scales
    cleaned = text.replace(",", "").replace("$", " ").replace("€", " ").replace("£", " ")
    pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(billion|million|thousand|percent|%|b|m|k)?\b",
        re.IGNORECASE,
    )

    results: List[Tuple[float, str, str]] = []
    for match in pattern.finditer(cleaned):
        num_str = match.group(1)
        scale_str = (match.group(2) or "").lower()
        try:
            raw_val = float(num_str)
        except ValueError:
            continue

        # Skip typical 4-digit fiscal calendar years (2015-2035) when not explicitly scaled
        if 2015 <= raw_val <= 2035 and not scale_str:
            continue

        multiplier = SCALE_FACTORS.get(scale_str, 1.0)
        scaled_val = raw_val * multiplier
        results.append((scaled_val, scale_str, match.group(0)))

    return results


def check_numeric_precision_match(answer: str, ground_truth: str, rel_tol: float = 0.01) -> bool:
    """
    Validates whether the key numeric figures asserted in ground_truth
    appear in the answer (matching numerically or via normalized scale).

    Args:
        answer: Model generated answer text.
        ground_truth: Ground truth answer text.
        rel_tol: Relative numerical tolerance (default 1%).

    Returns:
        True if all significant numbers in ground_truth have a match in answer.
    """
    gt_nums = extract_numbers_with_scales(ground_truth)
    if not gt_nums:
        # If no numeric figures in ground truth, cannot evaluate numeric precision
        return True

    ans_nums = extract_numbers_with_scales(answer)
    if not ans_nums:
        return False

    ans_scaled_vals = [item[0] for item in ans_nums]
    ans_raw_tokens = {item[2].lower().strip() for item in ans_nums}

    for gt_val, gt_scale, gt_token in gt_nums:
        matched = False

        # 1. Exact raw token match (e.g. "38%" in "38%")
        if gt_token.lower().strip() in ans_raw_tokens:
            matched = True
            continue

        # 2. Scaled numerical value match with tolerance
        for a_val in ans_scaled_vals:
            if math.isclose(gt_val, a_val, rel_tol=rel_tol, abs_tol=1e-5):
                matched = True
                break

        # 3. Direct substring match of normalized number in answer text
        raw_num = f"{gt_val:g}"
        if raw_num in answer.replace(",", ""):
            matched = True
            continue

        if not matched:
            return False

    return True

