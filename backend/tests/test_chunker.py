"""
Unit tests for deterministic chunk ID generation and chunk deduplication
in backend/ingestion/chunker.py.

Tests:
- Deterministic UUID5 generation: running split_section_into_chunks twice
  on identical input produces identical chunk_ids.
- UUID5 version validation: each chunk_id is a valid RFC 4122 UUIDv5.
- Chunk ID divergence: changing company, fiscal_year, or section text produces different IDs.
- Deduplication: deduplicate_chunks removes exact duplicates within the same
  company + fiscal_year + item_section, and accurately reports removed_count.
- Cross-company preservation: chunks with identical text across different companies
  or fiscal years are preserved and not incorrectly removed.
"""

import uuid
import pytest
from backend.ingestion.chunker import split_section_into_chunks, deduplicate_chunks


SAMPLE_SECTION_TEXT = """
Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations.

Overview
Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories,
and sells a variety of related services. During fiscal year 2024, total net sales were $391,035 million, an increase
of 2% compared to total net sales of $383,285 million in fiscal year 2023.

Products and Services Performance
Net sales of iPhone were $201,183 million in 2024 compared to $200,583 million in 2023.
Services net sales increased to $96,169 million in 2024 from $85,200 million in 2023, driven by growth in advertising,
cloud services, and payment services.

Research and Development
The Company continues to invest significantly in R&D to support ongoing innovation and product development.
Total R&D expense was $31,370 million in 2024 compared to $29,915 million in 2023.
"""


def test_deterministic_chunk_ids():
    """
    Running split_section_into_chunks twice on identical text must generate
    identical chunk_id values for every chunk.
    """
    chunks_run_1 = split_section_into_chunks(
        section_text=SAMPLE_SECTION_TEXT,
        company="Apple",
        fiscal_year="2024",
        item_section="Item 7",
        item_title="MD&A",
        source_path="apple_2024.html",
    )

    chunks_run_2 = split_section_into_chunks(
        section_text=SAMPLE_SECTION_TEXT,
        company="Apple",
        fiscal_year="2024",
        item_section="Item 7",
        item_title="MD&A",
        source_path="apple_2024.html",
    )

    assert len(chunks_run_1) > 0
    assert len(chunks_run_1) == len(chunks_run_2)

    for c1, c2 in zip(chunks_run_1, chunks_run_2):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["text"] == c2["text"]

        # Validate that chunk_id is a valid UUID version 5
        parsed_uuid = uuid.UUID(c1["chunk_id"])
        assert parsed_uuid.version == 5


def test_chunk_id_divergence_on_metadata_change():
    """
    Changing company or fiscal year produces completely different deterministic chunk_ids
    even if the section text is identical.
    """
    chunks_apple = split_section_into_chunks(
        section_text=SAMPLE_SECTION_TEXT,
        company="Apple",
        fiscal_year="2024",
        item_section="Item 7",
        item_title="MD&A",
        source_path="apple_2024.html",
    )

    chunks_microsoft = split_section_into_chunks(
        section_text=SAMPLE_SECTION_TEXT,
        company="Microsoft",
        fiscal_year="2024",
        item_section="Item 7",
        item_title="MD&A",
        source_path="msft_2024.html",
    )

    assert len(chunks_apple) == len(chunks_microsoft)
    for c_a, c_m in zip(chunks_apple, chunks_microsoft):
        assert c_a["chunk_id"] != c_m["chunk_id"]


def test_deduplicate_chunks_removes_exact_duplicates():
    """
    deduplicate_chunks must remove chunks that share the same company, fiscal_year,
    item_section, and text, returning (deduped_list, removed_count).
    """
    sample_chunk = {
        "chunk_id": "uuid-1",
        "company": "Apple",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "Total net sales were $391,035 million in 2024.",
    }
    duplicate_chunk = {
        "chunk_id": "uuid-2",
        "company": "Apple",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "Total net sales were $391,035 million in 2024.",
    }
    different_chunk = {
        "chunk_id": "uuid-3",
        "company": "Apple",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "Services revenue grew 13% in 2024.",
    }

    chunks = [sample_chunk, duplicate_chunk, different_chunk, duplicate_chunk]
    deduped, removed_count = deduplicate_chunks(chunks)

    assert len(deduped) == 2
    assert removed_count == 2
    assert deduped[0]["chunk_id"] == "uuid-1"
    assert deduped[1]["chunk_id"] == "uuid-3"


def test_deduplicate_chunks_preserves_cross_company_and_years():
    """
    Chunks with identical text but different company or fiscal_year are valid disclosures
    and must NOT be deduplicated.
    """
    chunk_apple = {
        "chunk_id": "uuid-a",
        "company": "Apple",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "General corporate overview disclosure.",
    }
    chunk_msft = {
        "chunk_id": "uuid-m",
        "company": "Microsoft",
        "fiscal_year": "2024",
        "item_section": "Item 7",
        "text": "General corporate overview disclosure.",
    }
    chunk_apple_2023 = {
        "chunk_id": "uuid-a-2023",
        "company": "Apple",
        "fiscal_year": "2023",
        "item_section": "Item 7",
        "text": "General corporate overview disclosure.",
    }

    chunks = [chunk_apple, chunk_msft, chunk_apple_2023]
    deduped, removed_count = deduplicate_chunks(chunks)

    assert len(deduped) == 3
    assert removed_count == 0

