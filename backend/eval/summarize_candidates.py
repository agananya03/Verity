"""
Summary and Audit Script for Held-Out Evaluation Benchmark Candidates.

Zero LLM calls. Computes and displays:
1. Counts by final review status (Accepted vs Rejected).
2. Counts by reject reason.
3. Counts by question style (Accepted vs Rejected).
4. Counts by company and section.
5. Check: How many accepted items still lack a company name (should be 0).
6. 10 sample accepted items as text, mixed across companies, sections, and styles,
   displaying question, ground truth, supporting quote, and any rewrite applied.
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_PATH = BASE_DIR / "eval" / "results" / "candidates.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def summarize_candidates(candidates_path: Path = DEFAULT_CANDIDATES_PATH) -> Dict[str, Any]:
    """
    Computes candidate statistics and prints formatted summary tables.
    """
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates: List[Dict[str, Any]] = json.load(f)

    total_candidates = len(candidates)
    accepted_items = [c for c in candidates if not c.get("default_reject")]
    rejected_items = [c for c in candidates if c.get("default_reject")]

    # 1. Reject reasons breakdown
    reject_reasons: Counter[str] = Counter()
    for c in rejected_items:
        flags = c.get("validator_flags", [])
        reasons_found = False
        for f in flags:
            if any(f.startswith(p) for p in ["REJECT_", "AMBIGUOUS_", "CHUNK1_", "CHUNK2_", "FLAG_"]):
                # Clean prefix for display
                main_reason = f.split("(")[0].split(":")[0].strip()
                reject_reasons[main_reason] += 1
                reasons_found = True
        if not reasons_found:
            reject_reasons["OTHER_REJECTION"] += 1

    # 2. Style breakdown
    styles_total: Counter[str] = Counter()
    styles_accepted: Counter[str] = Counter()
    styles_rejected: Counter[str] = Counter()
    for c in candidates:
        st = c.get("question_style", "unspecified")
        styles_total[st] += 1
        if c.get("default_reject"):
            styles_rejected[st] += 1
        else:
            styles_accepted[st] += 1

    # 3. Company and Section breakdown
    comp_sec_total: Dict[str, Counter[str]] = defaultdict(Counter)
    comp_sec_accepted: Dict[str, Counter[str]] = defaultdict(Counter)
    for c in candidates:
        comp = c.get("company", "Unknown")
        sec = c.get("section", "None")
        comp_sec_total[comp][sec] += 1
        if not c.get("default_reject"):
            comp_sec_accepted[comp][sec] += 1

    # 4. Accepted items lacking company name check
    accepted_lacking_comp = []
    for c in accepted_items:
        comp = c.get("company", "")
        q = c.get("question", "")
        if comp and comp.lower() not in q.lower():
            accepted_lacking_comp.append(c)

    # =========================================================================
    # Console Output Formatting
    # =========================================================================
    print("\n" + "=" * 80)
    print("HELD-OUT BENCHMARK CANDIDATES AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Candidates:         {total_candidates}")
    print(f"Accepted Candidates:      {len(accepted_items)} ({len(accepted_items)/total_candidates*100:.1f}%)")
    print(f"Rejected Candidates:      {len(rejected_items)} ({len(rejected_items)/total_candidates*100:.1f}%)")
    print(f"Accepted Lacking Company: {len(accepted_lacking_comp)} (Goal: exactly 0)")
    print("-" * 80)

    print("\n1. COUNTS BY FINAL STATUS:")
    print(f"  - ACCEPT: {len(accepted_items):>3} candidates")
    print(f"  - REJECT: {len(rejected_items):>3} candidates")

    print("\n2. COUNTS BY REJECT REASON:")
    if reject_reasons:
        for r, cnt in reject_reasons.most_common():
            print(f"  - {r:<35}: {cnt:>3}")
    else:
        print("  (No rejected candidates)")

    print("\n3. COUNTS BY QUESTION STYLE:")
    print(f"  {'Question Style':<55} | {'Total':>5} | {'Accept':>6} | {'Reject':>6}")
    print("  " + "-" * 78)
    for st in sorted(styles_total.keys()):
        tot = styles_total[st]
        acc = styles_accepted[st]
        rej = styles_rejected[st]
        print(f"  {st:<55} | {tot:>5} | {acc:>6} | {rej:>6}")

    print("\n4. COUNTS BY COMPANY AND SECTION (Accepted / Total):")
    core_companies = ["Alphabet", "Amazon", "Apple", "Microsoft", "NVIDIA"]
    other_companies = [c for c in sorted(comp_sec_total.keys()) if c not in core_companies]
    for comp in core_companies + other_companies:
        sec_strs = []
        for sec in sorted(comp_sec_total[comp].keys()):
            a = comp_sec_accepted[comp][sec]
            t = comp_sec_total[comp][sec]
            sec_strs.append(f"{sec}: {a}/{t}")
        tot_a = sum(comp_sec_accepted[comp].values())
        tot_t = sum(comp_sec_total[comp].values())
        print(f"  - {comp:<10} ({tot_a:>2}/{tot_t:>2} total) -> {', '.join(sec_strs)}")

    # =========================================================================
    # 5. Print 10 Mixed Accepted Items
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAMPLE OF 10 ACCEPTED ITEMS (Mixed across companies, sections & styles)")
    print("=" * 80)

    sample_picks: List[Dict[str, Any]] = []
    # Try picking 2 items per core company with varied sections and styles
    for comp in core_companies:
        comp_items = [c for c in accepted_items if c.get("company") == comp]
        # Sort by section then style to pick diverse items
        comp_items_sorted = sorted(comp_items, key=lambda x: (x.get("section", ""), x.get("question_style", "")))
        if len(comp_items_sorted) >= 2:
            sample_picks.append(comp_items_sorted[0])
            sample_picks.append(comp_items_sorted[-1])
        elif comp_items_sorted:
            sample_picks.extend(comp_items_sorted)

    # Pad or slice to exactly 10
    if len(sample_picks) < 10:
        remaining = [c for c in accepted_items if c not in sample_picks]
        sample_picks.extend(remaining[: 10 - len(sample_picks)])
    sample_picks = sample_picks[:10]

    for idx, item in enumerate(sample_picks, start=1):
        item_id = item.get("id", f"heldout_{idx:02d}")
        comp = item.get("company", "")
        yr = item.get("fiscal_year", "")
        sec = item.get("section", "")
        style = item.get("question_style", "unspecified")
        cat = item.get("category", "")
        rewrite = item.get("rewrite_applied") or (
            f"Original: '{item.get('original_question')}'" if item.get("original_question") else "None"
        )

        print(f"\n--- [{idx}/10] {item_id}: {comp} FY{yr} ({sec}) [{style}] ---")
        print(f"Question:         {item.get('question')}")
        print(f"Ground Truth:     {item.get('ground_truth')}")
        print(f"Supporting Quote: {item.get('supporting_quote')}")
        print(f"Rewrite Applied:  {rewrite}")

    print("\n" + "=" * 80 + "\n")

    return {
        "total_candidates": total_candidates,
        "accepted_count": len(accepted_items),
        "rejected_count": len(rejected_items),
        "accepted_lacking_company_count": len(accepted_lacking_comp),
        "reject_reasons": dict(reject_reasons),
        "styles_total": dict(styles_total),
        "comp_sec_accepted": {c: dict(secs) for c, secs in comp_sec_accepted.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize and audit benchmark candidate items.")
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH, help="Path to candidates.json")
    args = parser.parse_args()
    summarize_candidates(args.candidates_path)


if __name__ == "__main__":
    main()
