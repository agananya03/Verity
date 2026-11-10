"""
Golden Set Generation Script for Verity Financial RAG Evaluation.

Generates a RAGAS-compatible ground-truth benchmark dataset (backend/eval/golden_set_draft.json) by:
1. Stratified sampling ~100 text chunks from backend/data/processed/chunks.json across companies and sections.
2. Using Gemini (gemini-3.6-flash) with structured Pydantic output to generate (question, ground_truth) pairs for each chunk.
3. Adding ~10 plausible unanswerable questions for out-of-corpus company/year inquiries.
4. Outputting JSON in RAGAS evaluation format:
   [{"question": str, "ground_truth": str, "contexts": [str], "chunk_id": str}, ...]
"""

import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load project environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    """Pydantic model for structured question-answer pair generation."""
    question: str = Field(
        description="A clear, realistic financial question that can be answered using ONLY the provided text chunk."
    )
    ground_truth: str = Field(
        description="The precise, factual ground truth answer to the question derived strictly from the text chunk."
    )


# System prompt for LLM QA generation
QA_GEN_PROMPT = """You are an expert financial analyst creating an evaluation dataset for SEC 10-K RAG benchmarks.

Given the following SEC filing text chunk from {company} (Fiscal Year {fiscal_year}, Section: {item_section}):

---
Text Chunk:
{text_chunk}
---

Your task:
1. Formulate ONE specific, professional financial question that can be completely answered using ONLY the information in this text chunk.
2. Provide the precise ground_truth answer derived strictly from the chunk text. Do not invent or extrapolate numbers.
"""

# ~10 Plausible Unanswerable Questions (Out-of-Corpus Companies / Years)
UNANSWERABLE_QUESTIONS = [
    {
        "question": "What was Tesla's total automotive revenue in fiscal year 2022?",
        "company": "Tesla",
        "fiscal_year": "2022",
    },
    {
        "question": "What was Amazon Web Services (AWS) operating margin in Q3 2021?",
        "company": "Amazon",
        "fiscal_year": "2021",
    },
    {
        "question": "What were Meta's total Reality Labs losses in fiscal year 2020?",
        "company": "Meta",
        "fiscal_year": "2020",
    },
    {
        "question": "How many data center GPUs did Nvidia ship in fiscal year 2023?",
        "company": "Nvidia",
        "fiscal_year": "2023",
    },
    {
        "question": "What was Netflix's total paid subscriber count at the end of 2022?",
        "company": "Netflix",
        "fiscal_year": "2022",
    },
    {
        "question": "What was Microsoft's Azure cloud revenue growth rate in fiscal year 2021?",
        "company": "Microsoft",
        "fiscal_year": "2021",
    },
    {
        "question": "What were Apple's total iPhone 12 sales in fiscal year 2020?",
        "company": "Apple",
        "fiscal_year": "2020",
    },
    {
        "question": "What was Google's total YouTube ad revenue in fiscal year 2019?",
        "company": "Alphabet",
        "fiscal_year": "2019",
    },
    {
        "question": "What was Intel's capital expenditure in fiscal year 2022?",
        "company": "Intel",
        "fiscal_year": "2022",
    },
    {
        "question": "What was Salesforce's total subscription revenue in fiscal year 2023?",
        "company": "Salesforce",
        "fiscal_year": "2023",
    },
]


def load_processed_chunks() -> List[Dict[str, Any]]:
    """Loads text chunks from backend/data/processed/chunks.json."""
    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Processed chunks file not found at: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} total chunks from {chunks_path.name}")
    return chunks


def stratified_sampling(chunks: List[Dict[str, Any]], target_count: int = 100) -> List[Dict[str, Any]]:
    """
    Performs stratified sampling across (company, item_section) groups.
    Filters out very short chunks (<150 chars).
    """
    # Filter out short or header-only chunks
    valid_chunks = [c for c in chunks if len(c.get("text", "").strip()) >= 150]

    # Group by (company, item_section)
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for c in valid_chunks:
        key = (c.get("company", "Unknown"), c.get("item_section", "Other"))
        strata[key].append(c)

    logger.info(f"Grouped valid chunks into {len(strata)} (company, item_section) strata.")

    # Calculate samples per stratum
    sampled_chunks: List[Dict[str, Any]] = []
    num_strata = len(strata)
    base_per_stratum = max(1, target_count // num_strata)

    random.seed(42)  # Ensure reproducibility

    for key, stratum_chunks in strata.items():
        sample_size = min(len(stratum_chunks), base_per_stratum)
        sampled_chunks.extend(random.sample(stratum_chunks, sample_size))

    # If we need a few more chunks to reach target_count, sample randomly from remaining
    if len(sampled_chunks) < target_count:
        sampled_ids = {c["chunk_id"] for c in sampled_chunks}
        remaining = [c for c in valid_chunks if c["chunk_id"] not in sampled_ids]
        additional_needed = min(len(remaining), target_count - len(sampled_chunks))
        sampled_chunks.extend(random.sample(remaining, additional_needed))

    logger.info(f"Selected {len(sampled_chunks)} stratified chunks for question generation.")
    return sampled_chunks


def generate_golden_set():
    """Generates golden_set_draft.json using Gemini LLM and saves to backend/eval/."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable is not set!")
        sys.exit(1)

    from backend.eval.quota_ledger import global_quota_ledger
    bench_model = os.getenv("BENCH_MODEL", "gemini-3.5-flash")
    llm = ChatGoogleGenerativeAI(
        model=bench_model,
        google_api_key=api_key,
        temperature=0.2,
    )
    structured_llm = llm.with_structured_output(QAPair)

    # Step 1: Load and sample chunks
    chunks = load_processed_chunks()
    sampled_chunks = stratified_sampling(chunks, target_count=100)

    golden_set: List[Dict[str, Any]] = []

    # Step 2: Generate QA pairs for answerable chunks
    logger.info("Starting Gemini QA generation for sampled chunks...")
    for idx, chunk in enumerate(sampled_chunks, 1):
        company = chunk.get("company", "Company")
        fiscal_year = chunk.get("fiscal_year", "Year")
        item_section = chunk.get("item_section", "Section")
        text = chunk.get("text", "")

        prompt = QA_GEN_PROMPT.format(
            company=company,
            fiscal_year=fiscal_year,
            item_section=item_section,
            text_chunk=text,
        )

        try:
            global_quota_ledger.record_call(bench_model)
            qa_res: QAPair = structured_llm.invoke([("user", prompt)])
            golden_set.append({
                "question": qa_res.question.strip(),
                "ground_truth": qa_res.ground_truth.strip(),
                "contexts": [text],
                "chunk_id": chunk["chunk_id"],
                "company": company,
                "fiscal_year": fiscal_year,
            })
            if idx % 10 == 0 or idx == len(sampled_chunks):
                logger.info(f"Generated QA pairs: {idx}/{len(sampled_chunks)}")
        except Exception as e:
            logger.warning(f"Skipping chunk {chunk.get('chunk_id')} due to LLM error: {e}")

    # Step 3: Add unanswerable questions
    logger.info("Adding unanswerable out-of-corpus question benchmarks...")
    for unans in UNANSWERABLE_QUESTIONS:
        golden_set.append({
            "question": unans["question"],
            "ground_truth": "I don't have enough information to answer this question based on the retrieved SEC filings.",
            "contexts": [],
            "chunk_id": None,
            "company": unans["company"],
            "fiscal_year": unans["fiscal_year"],
        })

    # Step 4: Save output file
    eval_dir = BASE_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "golden_set_draft.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)

    # Step 5: Highlighted User Review Notice
    print("\n" + "=" * 80)
    print("GOLDEN SET DRAFT GENERATED SUCCESSFULLY!")
    print(f"Saved {len(golden_set)} items ({len(sampled_chunks)} answerable + {len(UNANSWERABLE_QUESTIONS)} unanswerable) to:")
    print(f"  -> {output_path}")
    print("=" * 80)
    print("CRITICAL MANUAL REVIEW REMINDER:")
    print("  Before using this dataset for RAGAS evaluation, please manually inspect and")
    print("  correct `backend/eval/golden_set_draft.json` for factual correctness.")
    print("  Once reviewed, rename `golden_set_draft.json` to `golden_set.json`.")
    print("  Note: RAGAS evaluation scores are only as reliable as this ground truth!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generate_golden_set()
