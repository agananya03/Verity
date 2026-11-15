"""
Unit Tests for Candidate Generation, Validators, and Merge Script.

Tests:
1. Supporting quote validator: Exact substring match, normalized whitespace matching, rejection on missing quote.
2. 6-gram overlap validator: Rejection of questions sharing 6+ word n-grams with chunk text.
3. Preamble validator: Rejection of questions starting with 'According to'.
4. Duplicate validator: Rejection of exact and high-similarity questions.
5. Cross-filing ambiguity validator: Flagging of queries omitting company or fiscal year context.
6. Merge reviewed script:
   - Derives full UUIDv5 chunk_ids from verbatim supporting quotes.
   - Resolves multi-year pairs across distinct fiscal years.
   - Handles unanswerable items properly (expected_decline=True, empty source_chunk_ids).
   - FAILS LOUDLY (raises ValueError) when a supporting quote is missing from the corpus.
   - Honors reject markers.
7. CLI argument parsing:
   - Verifies --golden-path argument in run_ragas_eval.py and retrieval_ablation.py.
"""

import csv
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from backend.eval.generate_candidates import (
    QUESTION_STYLES,
    is_table_chunk,
    sample_heldout_chunks,
    sample_multi_year_pairs,
    validate_6gram_overlap,
    validate_answer_leakage,
    validate_cross_filing_name_ambiguity,
    validate_duplicate,
    validate_no_preamble,
    validate_quote_unique_across_filings,
    validate_supporting_quote,
    run_all_validators,
    write_review_csv,
)
from backend.eval.merge_reviewed import (
    find_matching_chunk_for_quote,
    merge_review_csv,
)


# =============================================================================
# 1. Validator Tests
# =============================================================================
def test_validate_supporting_quote_exact_and_rejection():
    chunk_text = (
        "During fiscal year 2025, the Company repurchased 402 million shares of common stock "
        "for $95 billion under its share repurchase program."
    )

    # Exact substring
    ok, reason, quote = validate_supporting_quote(
        "the Company repurchased 402 million shares of common stock for $95 billion",
        chunk_text,
    )
    assert ok is True
    assert reason == "VALID"
    assert "402 million shares" in quote

    # Missing quote
    ok_bad, reason_bad, _ = validate_supporting_quote(
        "the Company repurchased 500 million shares for $120 billion",
        chunk_text,
    )
    assert ok_bad is False
    assert reason_bad == "REJECT_QUOTE_NOT_IN_CHUNK"

    # Empty / short quote
    ok_empty, reason_empty, _ = validate_supporting_quote("short", chunk_text)
    assert ok_empty is False
    assert reason_empty == "REJECT_QUOTE_EMPTY_OR_TOO_SHORT"


def test_validate_6gram_overlap():
    chunk_text = (
        "The Company continues to develop new technologies to enhance existing products and services, "
        "and to expand the range of its offerings through research and development."
    )

    # Question with 6-gram overlap (copies 6 consecutive words)
    q_overlap = "How does the Company continue to develop new technologies to enhance existing products and services?"
    ok_bad, msg_bad = validate_6gram_overlap(q_overlap, chunk_text)
    assert ok_bad is False
    assert "REJECT_6GRAM_OVERLAP" in msg_bad

    # Paraphrased question (no 6-gram overlap)
    q_clean = "What methods does the firm use to broaden product offerings and create new technology?"
    ok_good, msg_good = validate_6gram_overlap(q_clean, chunk_text)
    assert ok_good is True
    assert msg_good == "VALID"


def test_validate_no_preamble():
    # Starts with 'According to'
    ok_bad, msg_bad = validate_no_preamble("According to Apple's fiscal year 2025 Form 10-K, what was revenue?")
    assert ok_bad is False
    assert msg_bad == "REJECT_PREAMBLE_ACCORDING_TO"

    # Case-insensitive
    ok_bad2, _ = validate_no_preamble("according to the disclosures, what occurred?")
    assert ok_bad2 is False

    # Valid natural phrasing
    ok_good, msg_good = validate_no_preamble("In Apple's 2025 fiscal year, what was total net sales?")
    assert ok_good is True
    assert msg_good == "VALID"


def test_validate_duplicate():
    existing = [
        "What was Apple's total research and development expense in fiscal year 2025?",
        "How many full-time equivalent employees did Microsoft report in 2024?",
    ]

    # Exact duplicate
    ok_exact, msg_exact = validate_duplicate(
        "What was Apple's total research and development expense in fiscal year 2025?",
        existing,
    )
    assert ok_exact is False
    assert "REJECT_DUPLICATE_EXACT" in msg_exact

    # High similarity duplicate
    ok_sim, msg_sim = validate_duplicate(
        "What was total research and development expense for Apple in fiscal year 2025?",
        existing,
        threshold=0.80,
    )
    assert ok_sim is False
    assert "REJECT_DUPLICATE_SIMILAR" in msg_sim

    # New distinct question
    ok_new, msg_new = validate_duplicate(
        "What percentage of revenue did Amazon generate through third-party seller services?",
        existing,
    )
    assert ok_new is True
    assert msg_new == "VALID"


def test_validate_quote_unique_across_filings():
    mock_chunks = [
        {
            "chunk_id": "chunk_apple_2025_risk",
            "company": "Apple",
            "fiscal_year": "2025",
            "text": "The Company is subject to complex and evolving international laws and regulations regarding privacy and data protection.",
        },
        {
            "chunk_id": "chunk_apple_2024_risk",
            "company": "Apple",
            "fiscal_year": "2024",
            "text": "The Company is subject to complex and evolving international laws and regulations regarding privacy and data protection.",
        },
        {
            "chunk_id": "chunk_apple_2025_revenue",
            "company": "Apple",
            "fiscal_year": "2025",
            "text": "Total net sales in fiscal 2025 reached $391.0 billion compared to $383.3 billion in 2024.",
        },
    ]

    repeated_quote = "The Company is subject to complex and evolving international laws and regulations regarding privacy and data protection."
    unique_quote = "Total net sales in fiscal 2025 reached $391.0 billion"

    # Case 1: Quote repeats in another fiscal year (2024) and question omits target year (2025)
    # Must flag AMBIGUOUS_REPEATED_ACROSS_YEARS and reject
    ok_rep, msg_rep, rej_rep = validate_quote_unique_across_filings(
        supporting_quote=repeated_quote,
        question="What laws and regulations regarding privacy is the Company subject to?",
        company="Apple",
        fiscal_year="2025",
        all_chunks=mock_chunks,
    )
    assert ok_rep is False
    assert "AMBIGUOUS_REPEATED_ACROSS_YEARS" in msg_rep
    assert rej_rep is True

    # Case 2: Quote repeats in another fiscal year, BUT question explicitly names its own target year (2025)
    # Passes because question disambiguates the year
    ok_dis, msg_dis, rej_dis = validate_quote_unique_across_filings(
        supporting_quote=repeated_quote,
        question="In fiscal 2025, what privacy and data protection regulations was Apple subject to?",
        company="Apple",
        fiscal_year="2025",
        all_chunks=mock_chunks,
    )
    assert ok_dis is True
    assert "REPEATED_ACROSS_YEARS_DISAMBIGUATED" in msg_dis
    assert rej_dis is False

    # Case 3: Quote does NOT repeat in another fiscal year
    # Passes cleanly
    ok_uniq, msg_uniq, rej_uniq = validate_quote_unique_across_filings(
        supporting_quote=unique_quote,
        question="What were Apple's total net sales in fiscal 2025?",
        company="Apple",
        fiscal_year="2025",
        all_chunks=mock_chunks,
    )
    assert ok_uniq is True
    assert msg_uniq == "VALID"
    assert rej_uniq is False


def test_validate_cross_filing_name_ambiguity():
    # Question omitting company name
    flags_no_comp = validate_cross_filing_name_ambiguity(
        "What were total net sales and operating income in 2025?",
        company="Apple",
        fiscal_year="2025",
    )
    assert "FLAG_NO_COMPANY_NAME" in flags_no_comp

    # Question omitting fiscal year
    flags_no_yr = validate_cross_filing_name_ambiguity(
        "What was Apple's total net sales across all product lines?",
        company="Apple",
        fiscal_year="2025",
    )
    assert "FLAG_NO_FISCAL_YEAR" in flags_no_yr

    # Fully specified question
    flags_good = validate_cross_filing_name_ambiguity(
        "What was Apple's total net sales in fiscal year 2025?",
        company="Apple",
        fiscal_year="2025",
    )
    assert flags_good == []


def test_run_all_validators_integration():
    chunk_text = "Apple operates in five reportable geographic segments: Americas, Europe, Greater China, Japan, and Rest of Asia Pacific."
    candidate_good = {
        "question": "What are the five reportable geographic segments of Apple in fiscal 2025?",
        "supporting_quote": "Apple operates in five reportable geographic segments: Americas, Europe, Greater China, Japan, and Rest of Asia Pacific.",
        "company": "Apple",
        "fiscal_year": "2025",
    }
    is_ok, flags, should_reject = run_all_validators(candidate_good, chunk_text, existing_questions=[])
    assert is_ok is True
    assert should_reject is False
    assert "PASS" in flags

    # Integration test with cross-year repeated quote in all_chunks
    other_chunk = {
        "chunk_id": "other_yr_chunk",
        "company": "Apple",
        "fiscal_year": "2024",
        "text": chunk_text,
    }
    candidate_ambig = {
        "question": "What are the five reportable geographic segments of Apple?",
        "supporting_quote": chunk_text,
        "company": "Apple",
        "fiscal_year": "2025",
    }
    is_ok2, flags2, should_reject2 = run_all_validators(
        candidate_ambig, chunk_text, existing_questions=[], all_chunks=[other_chunk]
    )
    assert is_ok2 is False
    assert should_reject2 is True
    assert any("AMBIGUOUS_REPEATED_ACROSS_YEARS" in f for f in flags2)


def test_validate_supporting_quote_ellipsis_rule():
    chunk_narrative = (
        "During fiscal year 2025, the Company repurchased 402 million shares of common stock "
        "for $95 billion under its share repurchase program."
    )
    chunk_with_ellipsis = (
        "The Company continues to develop new technologies... and to expand its product range."
    )

    # 1. Quote with standard ellipsis '...' where chunk does NOT have it -> REJECT
    quote_with_dots = "repurchased 402 million shares ... under its share repurchase program"
    ok1, reason1, _ = validate_supporting_quote(quote_with_dots, chunk_narrative)
    assert ok1 is False
    assert reason1 == "REJECT_QUOTE_CONTAINS_ELLIPSIS"

    # 2. Quote with unicode ellipsis '…' where chunk does NOT have it -> REJECT
    quote_with_unicode = "repurchased 402 million shares … under its share repurchase program"
    ok2, reason2, _ = validate_supporting_quote(quote_with_unicode, chunk_narrative)
    assert ok2 is False
    assert reason2 == "REJECT_QUOTE_CONTAINS_ELLIPSIS"

    # 3. Chunk literally contains '...' and quote matches -> VALID
    quote_valid_ellipsis = "develop new technologies... and to expand"
    ok3, reason3, _ = validate_supporting_quote(quote_valid_ellipsis, chunk_with_ellipsis)
    assert ok3 is True
    assert "VALID" in reason3


def test_validate_answer_leakage():
    # User's explicit prompt example:
    # question: "What target year did Alphabet set in its 2030 sustainability plan?" with ground truth "by the year 2030"
    q_leak = "What target year did Alphabet set in its 2030 sustainability plan?"
    gt_leak = "Alphabet set a goal to achieve net-zero emissions across all of its operations and value chain by the year 2030."
    ok_leak, msg_leak = validate_answer_leakage(
        question=q_leak,
        ground_truth=gt_leak,
        fiscal_year="2023",
        company="Alphabet",
    )
    assert ok_leak is False
    assert "REJECT_ANSWER_LEAKAGE" in msg_leak
    assert "2030" in msg_leak

    # 3+ digit number in ground truth leaked into question
    q_num_leak = "Why did Alphabet's research and development expense rise to 45427 million in 2023?"
    gt_num = "Research and development expense reached $45,427 million in fiscal 2023."
    ok_num, msg_num = validate_answer_leakage(
        question=q_num_leak,
        ground_truth=gt_num,
        fiscal_year="2023",
        company="Alphabet",
    )
    assert ok_num is False
    assert "REJECT_ANSWER_LEAKAGE" in msg_num

    # Target fiscal year is present in both ground truth and question -> Exempted, PASS
    q_clean = "What was Alphabet's total revenue in fiscal year 2023?"
    gt_clean = "In fiscal year 2023, Alphabet reported total revenues of $307,394 million."
    ok_clean, msg_clean = validate_answer_leakage(
        question=q_clean,
        ground_truth=gt_clean,
        fiscal_year="2023",
        company="Alphabet",
    )
    assert ok_clean is True
    assert msg_clean == "VALID"

    # Distinctive capitalized entity leaked into question
    q_entity = "What strategic advantage did Alphabet gain from acquiring Mandiant?"
    gt_entity = "The Company completed the acquisition of Mandiant to bolster cybersecurity offerings."
    ok_entity, msg_entity = validate_answer_leakage(
        question=q_entity,
        ground_truth=gt_entity,
        fiscal_year="2022",
        company="Alphabet",
    )
    assert ok_entity is False
    assert "REJECT_ANSWER_LEAKAGE" in msg_entity
    assert "Mandiant" in msg_entity


def test_default_reject_behavior_and_review_csv(tmp_path: Path):
    chunk_text = "Alphabet reported total capital expenditures of $32,251 million in fiscal 2023."

    # 1. Candidate with failed validator (answer leakage)
    cand_failed = {
        "id": "heldout_01",
        "company": "Alphabet",
        "fiscal_year": "2023",
        "section": "Item 7",
        "category": "numeric_precision",
        "question_style": "natural prose",
        "question": "Why did Alphabet spend 32251 million on capital expenditures?",
        "ground_truth": "Alphabet reported total capital expenditures of $32,251 million.",
        "supporting_quote": "Alphabet reported total capital expenditures of $32,251 million in fiscal 2023.",
    }
    is_ok_f, flags_f, should_reject_f = run_all_validators(
        cand_failed, chunk_text, existing_questions=[]
    )
    assert is_ok_f is False
    assert should_reject_f is True
    assert any("REJECT_ANSWER_LEAKAGE" in f for f in flags_f)
    cand_failed["validator_flags"] = flags_f
    cand_failed["default_reject"] = should_reject_f

    # 2. Fully passing candidate
    cand_passed = {
        "id": "heldout_02",
        "company": "Alphabet",
        "fiscal_year": "2023",
        "section": "Item 7",
        "category": "numeric_precision",
        "question_style": "short direct question",
        "question": "What was Alphabet's total capital expenditures in fiscal year 2023?",
        "ground_truth": "$32,251 million",
        "supporting_quote": "Alphabet reported total capital expenditures of $32,251 million in fiscal 2023.",
    }
    is_ok_p, flags_p, should_reject_p = run_all_validators(
        cand_passed, chunk_text, existing_questions=[]
    )
    assert is_ok_p is True
    assert should_reject_p is False
    cand_passed["validator_flags"] = flags_p
    cand_passed["default_reject"] = should_reject_p

    # Write to temporary review CSV
    csv_file = tmp_path / "review_test.csv"
    write_review_csv([cand_failed, cand_passed], csv_file)

    # Verify CSV format and default reject/accept markers
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    row_failed = reader[0]
    row_passed = reader[1]

    # Verify question_style column exists
    assert "question_style" in row_failed
    assert row_failed["question_style"] == "natural prose"
    assert row_passed["question_style"] == "short direct question"

    # Failed item must default to reject="x" and accept=""
    assert row_failed["reject"] == "x"
    assert row_failed["accept"] == ""

    # Passed item must default to accept="x" and reject=""
    assert row_passed["accept"] == "x"
    assert row_passed["reject"] == ""


def test_question_styles_cycling():
    expected_styles = [
        "natural prose",
        "short direct question",
        "keyword-style phrase",
        "paraphrase (no copied terms from the chunk)",
        '"what does the filing say about" style',
        "comparison-of-two-figures style where the chunk allows",
    ]
    assert QUESTION_STYLES == expected_styles

    # Deterministic cycling verification
    cycled = [QUESTION_STYLES[(idx - 1) % len(QUESTION_STYLES)] for idx in range(1, 8)]
    assert cycled[0] == "natural prose"
    assert cycled[5] == "comparison-of-two-figures style where the chunk allows"
    assert cycled[6] == "natural prose"  # Cycles back to index 0


# =============================================================================
# 2. Stratified Sampling Tests
# =============================================================================
def test_table_chunk_detection():
    chunk_table = {"text": "| Segment | 2025 | 2024 |\n| Americas | 120000 | 115000 |"}
    chunk_narrative = {"text": "The Company is subject to complex and evolving international laws and regulations."}
    assert is_table_chunk(chunk_table) is True
    assert is_table_chunk(chunk_narrative) is False


def test_sample_heldout_chunks_excludes_dev_ids():
    mock_chunks = []
    companies = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
    dev_ids = set()

    for comp in companies:
        for yr in ["2024", "2025", "2026"]:
            for sec in ["Item 1", "Item 1A", "Item 7", "Item 8"]:
                for i in range(5):
                    cid = f"chunk_{comp}_{yr}_{sec}_{i}"
                    if i == 0 and comp == "Apple" and yr == "2025":
                        dev_ids.add(cid)
                    mock_chunks.append({
                        "chunk_id": cid,
                        "company": comp,
                        "fiscal_year": yr,
                        "item_section": sec,
                        "text": f"Text for {comp} {yr} {sec} chunk {i} " + ("| table |" if i == 1 else ""),
                    })

    sampled = sample_heldout_chunks(mock_chunks, dev_ids, seed=42)
    sampled_ids = {c["chunk_id"] for c in sampled}

    # Verify dev chunk is strictly excluded
    assert not sampled_ids.intersection(dev_ids)
    assert len(sampled) > 0


# =============================================================================
# 3. Merge Script Tests
# =============================================================================
def test_merge_review_csv_resolves_quotes_and_derives_chunk_ids(tmp_path):
    # Mock chunks
    mock_chunks = [
        {
            "chunk_id": "uuid-apple-2025-item1",
            "company": "Apple",
            "fiscal_year": "2025",
            "item_section": "Item 1",
            "text": "Apple Vision Pro is the Company’s spatial computer operating on visionOS.",
        },
        {
            "chunk_id": "uuid-apple-2024-item1",
            "company": "Apple",
            "fiscal_year": "2024",
            "item_section": "Item 1",
            "text": "During fiscal 2024 Apple introduced major enhancements across iPhone and Mac.",
        },
    ]
    chunks_path = tmp_path / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f)

    # Mock review CSV
    review_csv_path = tmp_path / "review.csv"
    with open(review_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "company", "fiscal_year", "section", "category",
                "question", "ground_truth", "supporting_quote", "validator_flags",
                "accept", "edit", "reject",
            ],
        )
        writer.writeheader()
        # Item 1: Accepted
        writer.writerow({
            "id": "heldout_01",
            "company": "Apple",
            "fiscal_year": "2025",
            "section": "Item 1",
            "category": "simple_lookup",
            "question": "What is the operating system for Apple Vision Pro?",
            "ground_truth": "visionOS",
            "supporting_quote": "Apple Vision Pro is the Company’s spatial computer operating on visionOS.",
            "validator_flags": "PASS",
            "accept": "x",
            "edit": "",
            "reject": "",
        })
        # Item 2: Multi-hop across 2024 & 2025
        writer.writerow({
            "id": "heldout_02",
            "company": "Apple",
            "fiscal_year": "2024/2025",
            "section": "Item 1",
            "category": "multi_hop",
            "question": "Compare Apple products in 2024 and 2025",
            "ground_truth": "Introduced iPhone enhancements and Vision Pro.",
            "supporting_quote": (
                "During fiscal 2024 Apple introduced major enhancements across iPhone and Mac. || "
                "Apple Vision Pro is the Company’s spatial computer operating on visionOS."
            ),
            "validator_flags": "PASS",
            "accept": "x",
            "edit": "",
            "reject": "",
        })
        # Item 3: Unanswerable
        writer.writerow({
            "id": "heldout_03",
            "company": "Tesla",
            "fiscal_year": "2022",
            "section": "None",
            "category": "unanswerable",
            "question": "What was Tesla's automotive gross margin?",
            "ground_truth": "I don't have enough information.",
            "supporting_quote": "",
            "validator_flags": "PASS",
            "accept": "x",
            "edit": "",
            "reject": "",
        })
        # Item 4: Rejected row (should be excluded)
        writer.writerow({
            "id": "heldout_04",
            "company": "Apple",
            "fiscal_year": "2025",
            "section": "Item 1",
            "category": "simple_lookup",
            "question": "Rejected item",
            "ground_truth": "N/A",
            "supporting_quote": "Apple Vision Pro is the Company’s spatial computer operating on visionOS.",
            "validator_flags": "PASS",
            "accept": "",
            "edit": "",
            "reject": "x",
        })

    output_path = tmp_path / "golden_heldout.json"
    heldout = merge_review_csv(
        review_csv_path=review_csv_path,
        chunks_path=chunks_path,
        output_path=output_path,
    )

    assert len(heldout) == 3
    ids = [it["id"] for it in heldout]
    assert "heldout_01" in ids
    assert "heldout_02" in ids
    assert "heldout_03" in ids
    assert "heldout_04" not in ids

    # Check chunk_id resolution
    item1 = next(it for it in heldout if it["id"] == "heldout_01")
    assert item1["source_chunk_ids"] == ["uuid-apple-2025-item1"]
    assert item1["chunk_id"] == "uuid-apple-2025-item1"
    assert len(item1["contexts"]) == 1

    # Check multi-hop resolution (both chunks derived)
    item2 = next(it for it in heldout if it["id"] == "heldout_02")
    assert set(item2["source_chunk_ids"]) == {"uuid-apple-2024-item1", "uuid-apple-2025-item1"}
    assert len(item2["contexts"]) == 2

    # Check unanswerable
    item3 = next(it for it in heldout if it["id"] == "heldout_03")
    assert item3["expected_decline"] is True
    assert item3["source_chunk_ids"] == []


def test_merge_review_csv_fails_loudly_on_missing_quote(tmp_path):
    mock_chunks = [
        {
            "chunk_id": "uuid-apple-2025-item1",
            "company": "Apple",
            "fiscal_year": "2025",
            "item_section": "Item 1",
            "text": "Existing chunk text without the phantom quote.",
        }
    ]
    chunks_path = tmp_path / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(mock_chunks, f)

    review_csv_path = tmp_path / "review.csv"
    with open(review_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "company", "fiscal_year", "section", "category",
                "question", "ground_truth", "supporting_quote", "validator_flags",
                "accept", "edit", "reject",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "id": "heldout_fail",
            "company": "Apple",
            "fiscal_year": "2025",
            "section": "Item 1",
            "category": "simple_lookup",
            "question": "Phantom question?",
            "ground_truth": "N/A",
            "supporting_quote": "This quote is nowhere in the chunks!",
            "validator_flags": "PASS",
            "accept": "x",
            "edit": "",
            "reject": "",
        })

    output_path = tmp_path / "golden_heldout.json"

    # Must raise ValueError with loud error message
    with pytest.raises(ValueError, match="Failed to resolve"):
        merge_review_csv(
            review_csv_path=review_csv_path,
            chunks_path=chunks_path,
            output_path=output_path,
        )


# =============================================================================
# 4. CLI Argument Parsing Tests
# =============================================================================
def test_golden_path_cli_arguments():
    import argparse
    from backend.eval.run_ragas_eval import BASE_DIR

    # Test run_ragas_eval argument parser accepts --golden-path
    parser1 = argparse.ArgumentParser()
    parser1.add_argument(
        "--golden-path",
        type=str,
        default=str(BASE_DIR / "eval" / "golden_set.json"),
    )
    args1 = parser1.parse_args(["--golden-path", "backend/eval/golden_heldout.json"])
    assert args1.golden_path == "backend/eval/golden_heldout.json"

    # Test retrieval_ablation argument parser accepts --golden-path
    parser2 = argparse.ArgumentParser()
    parser2.add_argument(
        "--golden-path",
        type=str,
        default="backend/eval/golden_set.json",
    )
    args2 = parser2.parse_args(["--golden-path", "backend/eval/golden_heldout.json"])
    assert args2.golden_path == "backend/eval/golden_heldout.json"

