"""
Clean Golden Set Evaluation Dataset for Verity Financial RAG.

Operations performed:
1. Backs up backend/eval/golden_set.json to backend/eval/golden_set.pre_clean.json.
2. Removes duplicate questions (specifically the 3 duplicate unanswerable questions at the end).
3. Assigns a stable 'id' field (golden_01, golden_02, ...) to every item.
4. Categorizes every item missing a category (simple_lookup, multi_hop, numeric_precision, unanswerable)
   and prints assignments for review.
5. Normalizes the company field (e.g., 'Apple/Apple' -> 'Apple') and ensures 'fiscal_year' and 'expected_decline' (bool) are set.
6. Flags any 'unanswerable' questions whose subject company is in our indexed corpus
   (Apple, Microsoft, Alphabet, Amazon, NVIDIA), specifically highlighting questions that may now be answerable.
7. Saves the cleaned dataset back to golden_set.json.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
BACKUP_PATH = EVAL_DIR / "golden_set.pre_clean.json"

INDEXED_COMPANIES: Set[str] = {
    "apple",
    "microsoft",
    "alphabet",
    "google",
    "amazon",
    "nvidia",
}


def normalize_company(company: str) -> str:
    """Normalizes company names (e.g. 'Apple/Apple' -> 'Apple')."""
    if not company:
        return "Unknown"
    company = company.strip()
    if "/" in company:
        parts = [p.strip() for p in company.split("/") if p.strip()]
        if len(set(parts)) == 1:
            return parts[0]
        # Default to first company if multi-company comparison
        return parts[0]
    return company


def classify_item(item: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Classifies an item into one of:
    - simple_lookup
    - multi_hop
    - numeric_precision
    - unanswerable

    Returns:
        (category, is_newly_assigned)
    """
    existing_cat = item.get("category")
    if existing_cat and existing_cat in {"simple_lookup", "multi_hop", "numeric_precision", "unanswerable"}:
        return existing_cat, False

    q = item.get("question", "").lower().strip()
    gt = item.get("ground_truth", "").lower().strip()
    ctxs = item.get("contexts", [])

    # 1. Unanswerable check
    if (
        not ctxs
        or "don't have enough information" in gt
        or "do not contain" in gt
        or "not contain this information" in gt
        or "cannot answer" in gt
    ):
        return "unanswerable", True

    # 2. Multi-hop check
    if (
        any(k in q for k in ["comparing", "compare", "chunk a", "chunk b", "across"])
        or ("fiscal year 2023" in q and "fiscal year 2024" in q)
        or len(ctxs) > 1
    ):
        return "multi_hop", True

    # 3. Numeric precision check
    numeric_indicators = [
        "percentage",
        "percent",
        "%",
        "how many",
        "how much",
        "total net sales",
        "operating income",
        "revenue",
        "margin",
        "capital expenditure",
        "dollar",
        "$",
    ]
    if any(indicator in q for indicator in numeric_indicators):
        return "numeric_precision", True

    # 4. Default to simple_lookup
    return "simple_lookup", True


def clean_golden_set() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Cleans the golden set and returns:
    (cleaned_items, newly_assigned_items, flagged_unanswerables)
    """
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden set not found at: {GOLDEN_SET_PATH}")

    # 1. Create backup
    shutil.copy2(GOLDEN_SET_PATH, BACKUP_PATH)
    print(f"Backed up original file to: {BACKUP_PATH.name}")

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        raw_items: List[Dict[str, Any]] = json.load(f)

    print(f"Original items count: {len(raw_items)}")

    # 2. Deduplicate by normalized question text while preserving order
    seen_questions: Set[str] = set()
    deduped_items: List[Dict[str, Any]] = []
    dropped_duplicates: List[Tuple[int, str]] = []

    for idx, item in enumerate(raw_items):
        q_norm = " ".join(item.get("question", "").strip().split()).lower()
        if q_norm in seen_questions:
            dropped_duplicates.append((idx, item.get("question", "")))
            continue
        seen_questions.add(q_norm)
        deduped_items.append(item)

    print(f"Deduplication: removed {len(dropped_duplicates)} duplicate question(s). Remaining: {len(deduped_items)}")
    for drop_idx, drop_q in dropped_duplicates:
        print(f"  - Dropped duplicate [orig_idx={drop_idx}]: {drop_q[:80]}...")

    # 3. Process each item: assign category, id, company normalization, fiscal_year, expected_decline
    cleaned_items: List[Dict[str, Any]] = []
    newly_assigned_items: List[Dict[str, Any]] = []
    flagged_unanswerables: List[Dict[str, Any]] = []

    for i, item in enumerate(deduped_items):
        stable_id = f"golden_{i + 1:02d}"
        category, was_new = classify_item(item)

        # Company normalization
        orig_company = item.get("company", "")
        norm_company = normalize_company(orig_company)

        # Fiscal year
        fiscal_year = item.get("fiscal_year")
        if not fiscal_year:
            # Fallback heuristic if missing
            fiscal_year = "Unknown"

        # Expected decline
        expected_decline = (category == "unanswerable")

        # Build clean item with ordered keys
        clean_item = {
            "id": stable_id,
            "category": category,
            "company": norm_company,
            "fiscal_year": fiscal_year,
            "expected_decline": expected_decline,
            "question": item.get("question", "").strip(),
            "ground_truth": item.get("ground_truth", "").strip(),
            "contexts": item.get("contexts", []),
            "chunk_id": item.get("chunk_id"),
        }
        cleaned_items.append(clean_item)

        if was_new:
            newly_assigned_items.append({
                "id": stable_id,
                "category": category,
                "company": norm_company,
                "question": clean_item["question"],
            })

        # 4. Check if unanswerable question has company in our indexed corpus
        if category == "unanswerable":
            comp_lower = norm_company.lower()
            q_lower = clean_item["question"].lower()
            in_corpus = any(
                c in comp_lower or c in q_lower for c in INDEXED_COMPANIES
            )
            if in_corpus:
                flagged_unanswerables.append({
                    "id": stable_id,
                    "company": norm_company,
                    "fiscal_year": fiscal_year,
                    "question": clean_item["question"],
                })

    # 5. Write cleaned dataset back to golden_set.json
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_items, f, indent=2, ensure_ascii=False)

    print(f"Successfully wrote {len(cleaned_items)} cleaned items to {GOLDEN_SET_PATH.name}")

    return cleaned_items, newly_assigned_items, flagged_unanswerables


def main():
    print("=" * 80)
    print("VERITY GOLDEN SET CLEANING & CATEGORIZATION SCRIPT")
    print("=" * 80)

    cleaned_items, newly_assigned, flagged_unanswerables = clean_golden_set()

    # Review Category Assignments
    print("\n" + "=" * 80)
    print("CATEGORY ASSIGNMENTS FOR PREVIOUSLY UNCLASSIFIED ITEMS (FOR REVIEW):")
    print("=" * 80)
    print(f"{'ID':<11} | {'Category':<18} | {'Company':<10} | {'Question'}")
    print("-" * 80)
    for item in newly_assigned:
        print(f"{item['id']:<11} | {item['category']:<18} | {item['company']:<10} | {item['question']}")

    # Review Flagged Unanswerables
    print("\n" + "=" * 80)
    print("FLAGGED UNANSWERABLE QUESTIONS WITH INDEXED COMPANIES:")
    print("=" * 80)
    print("These questions are marked 'unanswerable' but their subject company IS in our indexed corpus")
    print("(Apple, Microsoft, Alphabet, Amazon, NVIDIA). Especially review questions from indexed years.")
    print("-" * 80)
    for item in flagged_unanswerables:
        is_amazon_2024 = "amazon" in item["company"].lower() and "2024" in item["fiscal_year"]
        note = " [!] POTENTIALLY ANSWERABLE NOW (Amazon 2024 10-K is indexed)" if is_amazon_2024 else ""
        print(f"[{item['id']}] Company: {item['company']} | FY: {item['fiscal_year']}{note}")
        print(f"     Question: {item['question']}")

    # Category Summary
    print("\n" + "=" * 80)
    print("FINAL GOLDEN SET SUMMARY BY CATEGORY:")
    print("=" * 80)
    cat_counts: Dict[str, int] = {}
    for it in cleaned_items:
        cat_counts[it["category"]] = cat_counts.get(it["category"], 0) + 1

    for cat, count in sorted(cat_counts.items()):
        print(f"  - {cat:<20}: {count} questions")
    print(f"  TOTAL                 : {len(cleaned_items)} questions")


if __name__ == "__main__":
    main()

