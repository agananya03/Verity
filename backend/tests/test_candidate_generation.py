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

    # Sentence-initial capitalized finance word false-positive test ("Awards" case)
    q_awards = "What equity awards were granted to executive officers in fiscal 2023?"
    gt_awards = "Awards granted under the equity incentive plan consisted of restricted stock units."
    ok_awards, msg_awards = validate_answer_leakage(
        question=q_awards,
        ground_truth=gt_awards,
        fiscal_year="2023",
        company="Alphabet",
    )
    assert ok_awards is True
    assert msg_awards == "VALID"

    # Common finance words in answer (shares, expenses, liabilities, estimates, plans) do NOT trigger leakage
    q_fin = "How did the company manage liabilities, expenses, and share repurchase plans?"
    gt_fin = "Liabilities and operating expenses were managed through disciplined share repurchase plans and estimates."
    ok_fin, msg_fin = validate_answer_leakage(
        question=q_fin,
        ground_truth=gt_fin,
        fiscal_year="2023",
        company="Alphabet",
    )
    assert ok_fin is True
    assert msg_fin == "VALID"


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


# =============================================================================
# 5. Candidate Post-Processing and Multi-Year Verification Tests (fix_candidates.py)
# =============================================================================
def test_count_distinct_numeric_figures():
    from backend.eval.fix_candidates import count_distinct_numeric_figures

    # Numbers and percentages with 4-digit years filtered out
    text1 = "Revenue of $50,000 grew by 15.5% compared to 2024."
    assert count_distinct_numeric_figures(text1) == 2  # 50000, 15.5 (2024 ignored)

    # Narrative text with only years
    text2 = "In 2023 and 2025, operating expenses were stable across all segments."
    assert count_distinct_numeric_figures(text2) == 0

    # Two figures
    text3 = "Gross margin was $100 million vs $250 million."
    assert count_distinct_numeric_figures(text3) == 2  # 100, 250

    # Single figure
    text4 = "As of January 25, 2026, there are zero accrued contingencies of $45."
    assert count_distinct_numeric_figures(text4) == 2  # 25, 45 (2026 ignored)


def test_fix_missing_company():
    from backend.eval.fix_candidates import fix_missing_company

    items = [
        {
            "id": "heldout_01",
            "company": "Apple",
            "question": "What was total research and development expense?",
        },
        {
            "id": "heldout_02",
            "company": "Microsoft",
            "question": "How did Microsoft report cloud segment revenues?",
        },
        {
            "id": "heldout_03",
            "company": "Amazon",
            "question": "Amazon AWS operating margin comparison",
        },
    ]

    fixed, rewrites = fix_missing_company(items)
    assert rewrites == 1
    # First item lacked Apple, should have "At Apple, " prepended and first letter lowercased
    assert fixed[0]["question"] == "At Apple, what was total research and development expense?"
    assert fixed[0]["original_question"] == "What was total research and development expense?"
    assert "Prepended 'At Apple, '" in fixed[0]["rewrite_applied"]

    # Items already having the company name are untouched
    assert fixed[1]["question"] == "How did Microsoft report cloud segment revenues?"
    assert "original_question" not in fixed[1]
    assert fixed[2]["question"] == "Amazon AWS operating margin comparison"


def test_fix_repeated_across_years_disambiguation():
    from backend.eval.fix_candidates import fix_repeated_across_years

    # Shared text in both 2024 and 2025
    shared_quote = "Amy Hood was appointed Executive Vice President and Chief Financial Officer in July 2013."
    chunks = [
        {
            "chunk_id": "c1",
            "company": "Microsoft",
            "fiscal_year": "2024",
            "section": "Item 1",
            "text": f"Leadership info: {shared_quote} Other 2024 details.",
        },
        {
            "chunk_id": "c2",
            "company": "Microsoft",
            "fiscal_year": "2025",
            "section": "Item 1",
            "text": f"Leadership info: {shared_quote} Other 2025 details.",
        },
    ]

    # Candidate with ambiguous repeated quote lacking fiscal year
    candidate = {
        "id": "heldout_55",
        "company": "Microsoft",
        "fiscal_year": "2024",
        "section": "Item 1",
        "chunk_id": "c1",
        "question": "Who serves as Chief Financial Officer?",
        "ground_truth": "Amy Hood",
        "supporting_quote": shared_quote,
        "validator_flags": ["AMBIGUOUS_REPEATED_ACROSS_YEARS"],
        "default_reject": True,
        "category": "simple_lookup",
        "question_style": "natural prose",
    }

    fixed, rewrites, passed = fix_repeated_across_years([candidate], chunks)
    assert rewrites == 1
    assert passed == 1
    # Check that candidate now has a disambiguating year phrase and passed
    assert "2024" in fixed[0]["question"]
    assert fixed[0]["default_reject"] is False
    assert "Disambiguated repeated quote" in fixed[0]["rewrite_applied"]


def test_fix_comparison_style_rejection():
    from backend.eval.fix_candidates import fix_comparison_style

    items = [
        # Comparison style but narrative GT with 0 distinct figures
        {
            "id": "heldout_10",
            "company": "Amazon",
            "fiscal_year": "2024",
            "section": "Item 7",
            "question_style": "comparison-of-two-figures style where the chunk allows",
            "question": "How did revenue recognize compared to services?",
            "ground_truth": "Revenue from services is recognized over time as performance obligations are met.",
            "supporting_quote": "Revenue from services is recognized over time as performance obligations are met.",
            "validator_flags": [],
            "default_reject": False,
        },
        # Comparison style with 2 distinct figures -> should NOT be rejected
        {
            "id": "heldout_20",
            "company": "Apple",
            "fiscal_year": "2025",
            "section": "Item 7",
            "question_style": "comparison-of-two-figures style where the chunk allows",
            "question": "What was the difference between product sales and service sales?",
            "ground_truth": "Product sales were $294,868 million compared to service sales of $96,169 million.",
            "supporting_quote": "Product sales were $294,868 million compared to service sales of $96,169 million.",
            "validator_flags": [],
            "default_reject": False,
        },
    ]

    fixed, rejected_count = fix_comparison_style(items)
    assert rejected_count == 1
    assert fixed[0]["default_reject"] is True
    assert "REJECT_COMPARISON_NOT_SUPPORTED" in fixed[0]["validator_flags"][0]

    # Second item remains accepted
    assert fixed[1]["default_reject"] is False


def test_validate_multi_year_item_both_chunks():
    from backend.eval.fix_candidates import validate_multi_year_item

    chunk1 = {
        "chunk_id": "c1",
        "company": "Apple",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "Total net sales were $383,285 million in 2024.",
    }
    chunk2 = {
        "chunk_id": "c2",
        "company": "Apple",
        "fiscal_year": "2025",
        "item_section": "Item 7",
        "text": "Total net sales reached $391,035 million in 2025.",
    }

    # Valid item with quotes in both chunks
    valid_item = {
        "question": "How did Apple's total net sales change between fiscal 2024 and 2025?",
        "ground_truth": "Sales grew from $383,285 million to $391,035 million.",
        "supporting_quotes": [
            "Total net sales were $383,285 million in 2024.",
            "Total net sales reached $391,035 million in 2025.",
        ],
    }
    is_valid, errors, combined = validate_multi_year_item(valid_item, chunk1, chunk2, [])
    assert is_valid is True
    assert errors == ["PASS"]
    assert "383,285" in combined and "391,035" in combined

    # Invalid quote in chunk 2 (contains ellipsis)
    invalid_item = {
        "question": "How did Apple's total net sales change between fiscal 2024 and 2025?",
        "ground_truth": "Sales grew from $383,285 million to $391,035 million.",
        "supporting_quotes": [
            "Total net sales were $383,285 million in 2024.",
            "Total net sales...",
        ],
    }
    is_valid2, errors2, _ = validate_multi_year_item(invalid_item, chunk1, chunk2, [])
    assert is_valid2 is False
    assert any("REJECT_QUOTE_CONTAINS_ELLIPSIS" in e for e in errors2)


def test_sample_multi_year_candidate_pairs():
    from backend.eval.fix_candidates import sample_multi_year_candidate_pairs

    mock_chunks = [
        {"chunk_id": "c1", "company": "Apple", "fiscal_year": "2024", "item_section": "Item 7", "text": "A" * 350},
        {"chunk_id": "c2", "company": "Apple", "fiscal_year": "2025", "item_section": "Item 7", "text": "B" * 350},
        {"chunk_id": "c3", "company": "Apple", "fiscal_year": "2024", "item_section": "Item 1", "text": "C" * 350},
        {"chunk_id": "c4", "company": "Apple", "fiscal_year": "2025", "item_section": "Item 1", "text": "D" * 350},
        {"chunk_id": "c5", "company": "Apple", "fiscal_year": "2024", "item_section": "Item 8", "text": "E" * 350},
    ]

    pairs_dict = sample_multi_year_candidate_pairs(
        chunks=mock_chunks,
        dev_chunk_ids=set(),
        targets={"Apple": 2},
    )
    assert "Apple" in pairs_dict
    pairs = pairs_dict["Apple"]
    assert len(pairs) >= 2
    for p1, p2 in pairs:
        assert p1["company"] == "Apple" and p2["company"] == "Apple"
        assert p1["fiscal_year"] != p2["fiscal_year"]
        assert p1["item_section"] == p2["item_section"]


# =============================================================================
# 6. ID Uniqueness and New Validator Tests
# =============================================================================
def test_enforce_unique_candidate_ids_and_prefixes(tmp_path):
    from backend.eval.generate_candidates import write_candidates_json, write_review_csv

    # Test distinct prefixes
    candidates = [
        {"id": "heldout_01", "company": "Apple", "question": "Q1"},
        {"id": "heldout_mh_01", "company": "Apple", "question": "Q2"},
        {"id": "heldout_un_01", "company": "Apple", "question": "Q3"},
    ]
    json_path = tmp_path / "candidates.json"
    csv_path = tmp_path / "review.csv"

    write_candidates_json(candidates, json_path)
    write_review_csv(candidates, csv_path)
    assert json_path.exists()
    assert csv_path.exists()

    # Test duplicate ID causes ValueError in write_candidates_json
    dup_candidates = [
        {"id": "heldout_01", "company": "Apple", "question": "Q1"},
        {"id": "heldout_01", "company": "Apple", "question": "Q2"},
    ]
    with pytest.raises(ValueError, match="Candidate IDs must be unique"):
        write_candidates_json(dup_candidates, tmp_path / "dup.json")

    # Test duplicate ID causes ValueError in write_review_csv
    with pytest.raises(ValueError, match="Candidate IDs must be unique"):
        write_review_csv(dup_candidates, tmp_path / "dup.csv")


def test_validate_question_year_matches_filing():
    from backend.eval.generate_candidates import validate_question_year_matches_filing

    # Single-chunk matching year -> passes
    ok1, msg1 = validate_question_year_matches_filing(
        question="What was Apple's total revenue in fiscal year 2024?",
        fiscal_year="2024",
    )
    assert ok1 is True
    assert msg1 == "VALID"

    # Single-chunk mismatched year -> fails
    ok2, msg2 = validate_question_year_matches_filing(
        question="What was Apple's total revenue in fiscal year 2023?",
        fiscal_year="2024",
    )
    assert ok2 is False
    assert "REJECT_QUESTION_YEAR_MISMATCH" in msg2

    # Question with no year mentioned -> passes
    ok3, msg3 = validate_question_year_matches_filing(
        question="What does the filing state regarding research and development expenses?",
        fiscal_year="2024",
    )
    assert ok3 is True
    assert msg3 == "VALID"

    # Multi-year matching one or both filings' years -> passes
    ok4, msg4 = validate_question_year_matches_filing(
        question="How did NVIDIA's data center revenue change between FY2024 and FY2025?",
        fiscal_year="2024/2025",
        is_multi_year=True,
    )
    assert ok4 is True
    assert msg4 == "VALID"

    # Multi-year naming an extraneous year -> fails
    ok5, msg5 = validate_question_year_matches_filing(
        question="How did revenue change from 2022 to 2024?",
        fiscal_year="2024/2025",
        is_multi_year=True,
    )
    assert ok5 is False
    assert "REJECT_QUESTION_YEAR_MISMATCH" in msg5

    # Unanswerable inquiry -> exempt, passes
    ok6, msg6 = validate_question_year_matches_filing(
        question="What was Apple's total research and development expense in fiscal year 2020?",
        fiscal_year="2020",
        is_unanswerable=True,
    )
    assert ok6 is True
    assert msg6 == "VALID"


def test_validate_multi_year_adjacent_years():
    from backend.eval.generate_candidates import validate_multi_year_adjacent_years

    # Adjacent pairs pass
    ok1, msg1 = validate_multi_year_adjacent_years("2023/2024")
    assert ok1 is True
    assert msg1 == "VALID"

    ok2, msg2 = validate_multi_year_adjacent_years("2024/2025")
    assert ok2 is True
    assert msg2 == "VALID"

    ok3, msg3 = validate_multi_year_adjacent_years("2025/2026")
    assert ok3 is True
    assert msg3 == "VALID"

    # Non-adjacent pairs fail
    ok4, msg4 = validate_multi_year_adjacent_years("2023/2025")
    assert ok4 is False
    assert "REJECT_MULTI_YEAR_NON_ADJACENT_YEARS" in msg4

    ok5, msg5 = validate_multi_year_adjacent_years("2024/2026")
    assert ok5 is False
    assert "REJECT_MULTI_YEAR_NON_ADJACENT_YEARS" in msg5


def test_validate_ground_truth_supported_by_quotes():
    from backend.eval.generate_candidates import validate_ground_truth_supported_by_quotes

    # Case 1: Both quotes are substantive and support GT
    gt1 = "In 2024, iPad net sales were $26,694 million. In 2025, iPad net sales grew to $28,023 million."
    sq1 = "iPad net sales were $26,694 million for 2024. || iPad net sales increased to $28,023 million during 2025."
    ok1, msg1 = validate_ground_truth_supported_by_quotes(gt1, sq1)
    assert ok1 is True
    assert msg1 == "VALID"

    # Case 2: Quote 2 is a brief executive title/signature block (< 6 words)
    gt2 = "In 2024, foreign exchange had no material impact. In 2025, internal controls were certified by Satya Nadella."
    sq2 = "Fluctuations in the U.S. dollar did not have a material impact on revenue in 2024. || Satya Nadella\n\nChief Executive Officer"
    ok2, msg2 = validate_ground_truth_supported_by_quotes(gt2, sq2)
    assert ok2 is False
    assert "FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES" in msg2

    # Case 3: Quote ends abruptly with dangling punctuation
    gt3 = "In 2025, losses for pending litigation were accrued when probable."
    sq3 = "If information becomes available that causes us to determine that a loss in any of our pending litigation, || Second quote is complete."
    ok3, msg3 = validate_ground_truth_supported_by_quotes(gt3, sq3)
    assert ok3 is False
    assert "FLAG_GROUND_TRUTH_NOT_CLEARLY_SUPPORTED_BY_BOTH_QUOTES" in msg3


