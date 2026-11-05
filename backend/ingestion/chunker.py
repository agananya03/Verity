"""
SEC 10-K Filing Structural Chunker for Verity.

Why Structural Chunking (vs. Fixed-Size) Matters for Financial Retrieval (RAG):
--------------------------------------------------------------------------------
1. Semantic Integrity:
   Fixed-character chunking (e.g., splitting every 500 characters) blindly slices through
   tables, financial footnotes, and sentence boundaries. This causes fragmented context
   and distorts financial disclosures.

2. Domain-Aware Item Section Boundaries:
   SEC 10-K filings have standardized legal structures (Item 1: Business, Item 1A: Risk Factors,
   Item 7: MD&A, Item 8: Financial Statements). By chunking within structural Item boundaries,
   each chunk carries explicitly scoped domain metadata (`item_section` and `item_title`).

3. Targeted Retrieval & Precision:
   When an analyst or AI agent queries specific risk factors or financial performance commentary,
   structural metadata enables metadata filtering (e.g., retrieving only from Item 1A or Item 7),
   drastically reducing cross-section noise contamination and improving top-k recall precision.

4. Sentence-Bounded Overlap:
   Paragraph and sentence-level token budget chunking (~300-500 tokens with ~15% overlap) ensures
   that individual semantic statements remain contiguous while preserving contextual continuity across
   adjacent chunks.
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_FILE = DATA_RAW_DIR / "manifest.json"
OUTPUT_CHUNKS_FILE = DATA_PROCESSED_DIR / "chunks.json"

# Chunking budget configuration
TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 500
OVERLAP_PERCENTAGE = 0.15  # ~15% overlap between adjacent chunks

# Structural Item definitions and regex patterns
# (SEC filings contain Table of Contents; body sections are identified by content length)
SECTION_CONFIGS = [
    {
        "item_section": "Item 1",
        "item_title": "Business",
        "start_pattern": r'(?i)\bITEM\s+1[\.\s\:\–\—\t\n]+BUSINESS\b',
        "end_patterns": [
            r'(?i)\bITEM\s+1A[\.\s\:\–\—\t\n]+',
            r'(?i)\bITEM\s+1B[\.\s\:\–\—\t\n]+',
            r'(?i)\bITEM\s+2[\.\s\:\–\—\t\n]+',
        ],
    },
    {
        "item_section": "Item 1A",
        "item_title": "Risk Factors",
        "start_pattern": r'(?i)\bITEM\s+1A[\.\s\:\–\—\t\n]+RISK\s+FACTORS\b',
        "end_patterns": [
            r'(?i)\bITEM\s+1B[\.\s\:\–\—\t\n]+',
            r'(?i)\bITEM\s+2[\.\s\:\–\—\t\n]+',
        ],
    },
    {
        "item_section": "Item 7",
        "item_title": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "start_pattern": r'(?i)\bITEM\s+7[\.\s\:\–\—\t\n]+MANAGEMENT',
        "end_patterns": [
            r'(?i)\bITEM\s+7A[\.\s\:\–\—\t\n]+',
            r'(?i)\bITEM\s+8[\.\s\:\–\—\t\n]+',
        ],
    },
    {
        "item_section": "Item 8",
        "item_title": "Financial Statements and Supplementary Data",
        "start_pattern": r'(?i)\bITEM\s+8[\.\s\:\–\—\t\n]+FINANCIAL',
        "end_patterns": [
            r'(?i)\bITEM\s+9[\.\s\:\–\—\t\n]+',
            r'(?i)\bITEM\s+9A[\.\s\:\–\—\t\n]+',
        ],
    },
]


def estimate_token_count(text: str) -> int:
    """Estimates token count for a text string (approx. 1 word = 1.3 tokens)."""
    words = text.split()
    return int(len(words) * 1.3)


def extract_html_text(html_content: str) -> str:
    """Parses HTML using BeautifulSoup, stripping script/style tags and normalizing whitespace."""
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.extract()
        
    text = soup.get_text(separator="\n")
    # Clean non-breaking spaces and redundant blank lines
    text = text.replace("\xa0", " ").replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    cleaned_text = "\n".join(line for line in lines if line)
    return cleaned_text


def find_item_sections(full_text: str) -> Dict[str, Tuple[str, str]]:
    """
    Locates body text segments for Item 1, Item 1A, Item 7, and Item 8.
    Distinguishes actual body sections from Table of Contents based on section content length.
    
    Returns dict mapping item_section -> (item_title, section_text).
    """
    extracted_sections = {}
    
    for cfg in SECTION_CONFIGS:
        sec_key = cfg["item_section"]
        sec_title = cfg["item_title"]
        start_pat = cfg["start_pattern"]
        end_pats = cfg["end_patterns"]
        
        start_matches = list(re.finditer(start_pat, full_text))
        if not start_matches:
            logger.warning(f"Could not find start pattern for {sec_key}")
            continue
            
        best_section_text = ""
        
        # Iterate over matches to find the main body occurrence (skipping short TOC entries)
        for match in start_matches:
            start_pos = match.start()
            
            # Find earliest end pattern after start_pos (minimum offset 500 chars to skip cross-references)
            end_pos = len(full_text)
            for end_pat in end_pats:
                end_matches = [m for m in re.finditer(end_pat, full_text) if m.start() > start_pos + 500]
                if end_matches:
                    end_pos = min(end_pos, end_matches[0].start())
                    
            section_candidate = full_text[start_pos:end_pos].strip()
            
            # If section length > 1200 characters, it is a body section rather than TOC link
            if len(section_candidate) > len(best_section_text):
                best_section_text = section_candidate
                
        if best_section_text and len(best_section_text) > 1000:
            extracted_sections[sec_key] = (sec_title, best_section_text)
            logger.info(f"Extracted {sec_key} ({len(best_section_text)} chars)")
        else:
            logger.warning(f"Failed to find valid body section for {sec_key}")
            
    return extracted_sections


def split_section_into_chunks(
    section_text: str,
    company: str,
    fiscal_year: str,
    item_section: str,
    item_title: str,
    source_path: str,
) -> List[Dict]:
    """
    Splits section text into paragraph/sentence bounded chunks of ~300-500 tokens
    with ~15% overlap between adjacent chunks.
    Assigns deterministic UUID5 chunk_ids derived from company, year, section, chunk_index, and text prefix.
    """
    # Split section into paragraphs
    paragraphs = [p.strip() for p in section_text.split("\n") if p.strip()]
    
    # Expand paragraphs into sentences if needed to maintain sentence-level boundaries
    units = []
    for p in paragraphs:
        if estimate_token_count(p) > TARGET_MAX_TOKENS:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            units.extend([s.strip() for s in sentences if s.strip()])
        else:
            units.append(p)
            
    chunks = []
    current_units = []
    current_tokens = 0
    i = 0
    
    while i < len(units):
        unit = units[i]
        unit_tokens = estimate_token_count(unit)
        
        current_units.append(unit)
        current_tokens += unit_tokens
        
        # When target token count is reached or at end of units
        if current_tokens >= TARGET_MIN_TOKENS or i == len(units) - 1:
            chunk_text = "\n\n".join(current_units)
            chunk_index = len(chunks)
            chunk_hash_key = f"{company}|{fiscal_year}|{item_section}|{chunk_index}|{chunk_text[:200]}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_hash_key))
            
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "company": company,
                "fiscal_year": fiscal_year,
                "item_section": item_section,
                "item_title": item_title,
                "source_path": source_path,
            })
            
            # Prepare overlap for next chunk (~15% of units)
            overlap_target = int(current_tokens * OVERLAP_PERCENTAGE)
            overlap_units = []
            overlap_tokens = 0
            
            for backtrack_unit in reversed(current_units):
                bt_tokens = estimate_token_count(backtrack_unit)
                if overlap_tokens + bt_tokens <= overlap_target or not overlap_units:
                    overlap_units.insert(0, backtrack_unit)
                    overlap_tokens += bt_tokens
                else:
                    break
                    
            if i == len(units) - 1:
                break
                
            current_units = overlap_units
            current_tokens = overlap_tokens
            
        i += 1
        
    return chunks


def chunk_filing(manifest_entry: Dict) -> List[Dict]:
    """
    Parses a 10-K HTML filing from a manifest entry and splits it into structural chunks.
    
    Returns a list of chunk dictionaries.
    """
    rel_path = manifest_entry["local_path"]
    full_file_path = BASE_DIR.parent / rel_path
    
    if not full_file_path.exists():
        logger.error(f"Filing HTML file not found at {full_file_path}")
        return []
        
    logger.info(f"Processing {manifest_entry['company']} ({manifest_entry['fiscal_year']}) from {rel_path}...")
    html_content = full_file_path.read_text(encoding="utf-8")
    
    # Step 1: Clean HTML to text
    full_text = extract_html_text(html_content)
    
    # Step 2: Extract structural sections (Item 1, 1A, 7, 8)
    sections = find_item_sections(full_text)
    
    # Step 3: Chunk each section
    filing_chunks = []
    for sec_key, (sec_title, sec_text) in sections.items():
        sec_chunks = split_section_into_chunks(
            section_text=sec_text,
            company=manifest_entry["company"],
            fiscal_year=manifest_entry["fiscal_year"],
            item_section=sec_key,
            item_title=sec_title,
            source_path=rel_path,
        )
        filing_chunks.extend(sec_chunks)
        
    logger.info(f"Generated {len(filing_chunks)} chunks for {manifest_entry['company']} ({manifest_entry['fiscal_year']})")
    return filing_chunks


def deduplicate_chunks(chunks: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Dedupes chunks with identical text within the same company+year+section.
    Returns a tuple of (deduplicated_chunks, removed_count).
    """
    seen_keys = set()
    deduped = []
    removed_count = 0

    for chunk in chunks:
        dedupe_key = (
            chunk.get("company", ""),
            str(chunk.get("fiscal_year", "")),
            chunk.get("item_section", ""),
            chunk.get("text", "").strip(),
        )
        if dedupe_key in seen_keys:
            removed_count += 1
        else:
            seen_keys.add(dedupe_key)
            deduped.append(chunk)

    return deduped, removed_count


def main():
    """Main execution function to process all filings in manifest.json and write chunks.json."""
    print("=" * 70)
    print("Starting Structural 10-K Chunker")
    print("=" * 70)
    
    if not MANIFEST_FILE.exists():
        logger.error(f"Manifest file not found at {MANIFEST_FILE}")
        return
        
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    
    for entry in manifest:
        chunks = chunk_filing(entry)
        all_chunks.extend(chunks)
        
    # Deduplicate chunks with identical text within the same company+year+section before writing
    deduped_chunks, removed_count = deduplicate_chunks(all_chunks)
    logger.info(
        f"Deduplication complete: removed {removed_count} duplicate chunks with identical text within the same company+year+section."
    )
    logger.info(f"Retained {len(deduped_chunks)} unique chunks out of {len(all_chunks)} total chunks.")

    with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped_chunks, f, indent=2)
        
    print("=" * 70)
    print(f"Chunking complete. Total unique chunks generated: {len(deduped_chunks)} (removed {removed_count} duplicates)")
    print(f"Saved processed chunks to: {OUTPUT_CHUNKS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
