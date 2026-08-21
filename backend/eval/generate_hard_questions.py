"""
Generate Hard Benchmark Questions for Verity Golden Evaluation Set.

Generates hard evaluation questions in three categories:
1. Multi-hop questions (3-4 questions): Combines information from two chunks across fiscal years or companies.
2. Numeric precision questions (3 questions): Asks for specific exact figures from MD&A or financial statement chunks.
3. Unanswerable questions (2-3 questions): Plausible queries for metrics/companies not present in the SEC filings corpus.

Presents generated candidates to the user with an interactive [Y/n/e] prompt before appending to golden_set.json.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from backend.agent.llm_client import get_llm_with_fallback

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    question: str = Field(description="The generated financial evaluation question.")
    ground_truth: str = Field(description="The exact ground truth answer based strictly on the provided context(s).")


def load_chunks() -> List[Dict[str, Any]]:
    """Loads processed chunks from chunks.json."""
    chunks_path = BASE_DIR / "backend" / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        logger.error(f"Chunks file not found at: {chunks_path}")
        sys.exit(1)
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_multihop_candidates(chunks: List[Dict[str, Any]], count: int = 4) -> List[Dict[str, Any]]:
    """Generates multi-hop QA candidates combining information from 2 chunks."""
    logger.info("Generating Multi-Hop question candidates...")
    candidates = []

    chunks_by_company: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        comp = c.get("company", "Unknown")
        chunks_by_company.setdefault(comp, []).append(c)

    companies = list(chunks_by_company.keys())

    # Build chunk pairs (cross-year or cross-company)
    pairs = []
    # 1. Cross-year pairs for same company
    for comp, comp_chunks in chunks_by_company.items():
        by_year: Dict[str, List[Dict[str, Any]]] = {}
        for c in comp_chunks:
            by_year.setdefault(c.get("fiscal_year", ""), []).append(c)
        years = sorted(list(by_year.keys()))
        if len(years) >= 2:
            y1, y2 = years[0], years[1]
            for c1 in by_year[y1][:8]:
                for c2 in by_year[y2][:8]:
                    if c1.get("item_section") == c2.get("item_section"):
                        pairs.append((c1, c2))

    # 2. Cross-company pairs
    if len(companies) >= 2:
        c1_chunks = chunks_by_company[companies[0]]
        c2_chunks = chunks_by_company[companies[1]]
        for c1 in c1_chunks[:8]:
            for c2 in c2_chunks[:8]:
                if c1.get("item_section") == c2.get("item_section"):
                    pairs.append((c1, c2))

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = get_llm_with_fallback(
        primary_model="gemini-3.5-flash",
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.6-flash-lite"],
        google_api_key=api_key or None,
        temperature=0.3,
    )
    structured_llm = llm.with_structured_output(QAPair)

    prompt_template = """You are creating a MULTI-HOP financial benchmark question.
A multi-hop question REQUIRES combining or comparing information from BOTH provided SEC filing chunks.

Chunk A (Company: {comp_A}, Fiscal Year: {year_A}, Section: {sec_A}):
{text_A}

Chunk B (Company: {comp_B}, Fiscal Year: {year_B}, Section: {sec_B}):
{text_B}

Generate a clear, high-quality question that requires reading BOTH Chunk A and Chunk B to answer (e.g. comparing metrics, trends, or risks between the two years/companies).
Provide a complete ground_truth answer that explicitly references facts from both chunks.
"""

    for chunk_a, chunk_b in pairs:
        if len(candidates) >= count:
            break
        try:
            prompt = prompt_template.format(
                comp_A=chunk_a.get("company"),
                year_A=chunk_a.get("fiscal_year"),
                sec_A=chunk_a.get("item_section"),
                text_A=chunk_a.get("text", "")[:1200],
                comp_B=chunk_b.get("company"),
                year_B=chunk_b.get("fiscal_year"),
                sec_B=chunk_b.get("item_section"),
                text_B=chunk_b.get("text", "")[:1200],
            )
            result: QAPair = structured_llm.invoke([("user", prompt)])
            c1_id = chunk_a.get("chunk_id") or chunk_a.get("id")
            c2_id = chunk_b.get("chunk_id") or chunk_b.get("id")
            candidates.append({
                "question": result.question,
                "ground_truth": result.ground_truth,
                "contexts": [chunk_a.get("text", ""), chunk_b.get("text", "")],
                "chunk_id": [c1_id, c2_id],
                "company": f"{chunk_a.get('company')}/{chunk_b.get('company')}",
                "fiscal_year": f"{chunk_a.get('fiscal_year')}/{chunk_b.get('fiscal_year')}",
                "category": "multi_hop",
            })
        except Exception as e:
            logger.warning(f"Error generating multi-hop candidate: {e}")

    return candidates


def generate_numeric_precision_candidates(chunks: List[Dict[str, Any]], count: int = 3) -> List[Dict[str, Any]]:
    """Generates numeric precision QA candidates asking for exact dollar/financial figures."""
    logger.info("Generating Numeric Precision question candidates...")
    candidates = []

    numeric_chunks = [
        c for c in chunks
        if any(term in c.get("text", "").lower() for term in ["net sales", "revenue", "r&d", "operating income", "net income", "$"])
    ]

    api_key = os.getenv("GOOGLE_API_KEY")
    llm = get_llm_with_fallback(
        primary_model="gemini-3.5-flash",
        fallback_models=["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"],
        google_api_key=api_key or None,
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(QAPair)

    prompt_template = """You are creating a NUMERIC PRECISION financial benchmark question.
Your task is to ask for a specific financial dollar amount, revenue figure, R&D spend, or exact metric mentioned in the filing chunk.

Chunk (Company: {company}, Fiscal Year: {year}, Section: {section}):
{text}

CRITICAL REQUIREMENT:
The ground_truth MUST state the EXACT figure/text directly from the chunk word-for-word (e.g., "$391,035 million" or "$245.1 billion"). Do NOT round, estimate, or paraphrase numbers.
"""

    for c in numeric_chunks:
        if len(candidates) >= count:
            break
        try:
            prompt = prompt_template.format(
                company=c.get("company"),
                year=c.get("fiscal_year"),
                section=c.get("item_section"),
                text=c.get("text", "")[:1500],
            )
            result: QAPair = structured_llm.invoke([("user", prompt)])
            c_id = c.get("chunk_id") or c.get("id")
            candidates.append({
                "question": result.question,
                "ground_truth": result.ground_truth,
                "contexts": [c.get("text", "")],
                "chunk_id": [c_id],
                "company": c.get("company"),
                "fiscal_year": c.get("fiscal_year"),
                "category": "numeric_precision",
            })
        except Exception as e:
            logger.warning(f"Error generating numeric precision candidate: {e}")

    return candidates


def generate_unanswerable_candidates(count: int = 3) -> List[Dict[str, Any]]:
    """Generates unanswerable QA candidates for metrics/companies not in the corpus."""
    logger.info("Generating Unanswerable question candidates...")

    unanswerable_templates = [
        {
            "question": "What was Amazon's total net sales and AWS operating income in fiscal year 2024 according to the filings?",
            "ground_truth": "The provided documents do not contain this information.",
            "contexts": [],
            "chunk_id": [],
            "company": "Amazon",
            "fiscal_year": "2024",
            "category": "unanswerable",
        },
        {
            "question": "What was Netflix's global subscriber churn rate and ARPU in Q4 2024?",
            "ground_truth": "The provided documents do not contain this information.",
            "contexts": [],
            "chunk_id": [],
            "company": "Netflix",
            "fiscal_year": "2024",
            "category": "unanswerable",
        },
        {
            "question": "According to Apple's Form 10-K, what was the exact breakdown of R&D expenses spent on Project Titan autonomous vehicles in 2022?",
            "ground_truth": "The provided documents do not contain this information.",
            "contexts": [],
            "chunk_id": [],
            "company": "Apple",
            "fiscal_year": "2022",
            "category": "unanswerable",
        },
    ]
    return unanswerable_templates[:count]


def review_candidates_interactively(candidates: List[Dict[str, Any]], auto_approve: bool = False) -> List[Dict[str, Any]]:
    """Interactively reviews candidate questions with [Y/n/e] prompt."""
    approved = []
    print("\n" + "=" * 80)
    print("INTERACTIVE GOLDEN SET CANDIDATE QUESTION REVIEW")
    print("=" * 80)

    for idx, c in enumerate(candidates, start=1):
        print(f"\n--- Candidate #{idx} [{c.get('category', 'general').upper()}] ---")
        print(f"QUESTION:      {c['question']}")
        print(f"GROUND TRUTH:  {c['ground_truth']}")
        print(f"TARGET INFO:   Company: {c.get('company')}, Year: {c.get('fiscal_year')}")
        print(f"CHUNK IDs:     {c['chunk_id']}")
        print("-" * 80)

        if auto_approve:
            print("Auto-approving candidate (--auto-approve enabled).")
            approved.append(c)
            continue

        while True:
            choice = input("Approve this question to append to golden_set.json? [Y/n/e] (Y=approve, n=reject, e=edit): ").strip().lower()
            if choice in ["y", "yes", ""]:
                approved.append(c)
                print("-> APPROVED")
                break
            elif choice in ["n", "no"]:
                print("-> REJECTED")
                break
            elif choice in ["e", "edit"]:
                new_q = input("Enter edited Question (leave blank to keep current): ").strip()
                new_gt = input("Enter edited Ground Truth (leave blank to keep current): ").strip()
                if new_q:
                    c["question"] = new_q
                if new_gt:
                    c["ground_truth"] = new_gt
                approved.append(c)
                print("-> EDITED & APPROVED")
                break
            else:
                print("Invalid choice. Please enter 'y', 'n', or 'e'.")

    return approved


def main():
    parser = argparse.ArgumentParser(description="Generate and append hard evaluation questions to golden_set.json")
    parser.add_argument("--auto-approve", action="store_true", help="Auto-approve candidates without interactive prompting")
    args = parser.parse_args()

    chunks = load_chunks()

    # Generate candidates for each of the 3 categories
    candidates = []
    candidates.extend(generate_multihop_candidates(chunks, count=4))
    candidates.extend(generate_numeric_precision_candidates(chunks, count=3))
    candidates.extend(generate_unanswerable_candidates(count=3))

    logger.info(f"Generated {len(candidates)} candidate hard evaluation questions.")

    # Interactive review loop
    approved_questions = review_candidates_interactively(candidates, auto_approve=args.auto_approve)

    if not approved_questions:
        logger.info("No questions were approved. Golden set remains unchanged.")
        return

    # Load existing golden set, extend, and re-save
    golden_set_path = BASE_DIR / "backend" / "eval" / "golden_set.json"
    existing_golden_set: List[Dict[str, Any]] = []
    if golden_set_path.exists():
        with open(golden_set_path, "r", encoding="utf-8") as f:
            existing_golden_set = json.load(f)

    existing_count = len(existing_golden_set)
    existing_golden_set.extend(approved_questions)

    with open(golden_set_path, "w", encoding="utf-8") as f:
        json.dump(existing_golden_set, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"SUCCESSFULLY APPENDED {len(approved_questions)} APPROVED HARD QUESTIONS TO GOLDEN SET!")
    print(f"Golden set expanded from {existing_count} to {len(existing_golden_set)} total benchmark questions.")
    print(f"Saved to: {golden_set_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
