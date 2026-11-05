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
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# Robust patterns matching section headings while ignoring in-text cross-references
SECTION_CONFIGS = [
    {
        "item_section": "Item 1",
        "item_title": "Business",
        "start_pattern": r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+1[\.\:\–\—\s\t\n]+BUSINESS\b(?!\s+(?:of\b|under\b|relating\b|and\s+Note))',
        "end_patterns": [
            r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+1A[\.\:\–\—\s\t\n]+RISK\s+FACTORS\b(?!\s+(?:of\b|under\b|relating\b))',
            r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+1B[\.\:\–\—\s\t\n]+UNRESOLVED\b(?!\s+(?:of\b|under\b|relating\b))',
            r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+2[\.\:\–\—\s\t\n]+PROPERTIES\b(?!\s+(?:of\b|under\b|relating\b))',
        ],
    },
    {
        "item_section": "Item 1A",
        "item_title": "Risk Factors",
        "start_pattern": r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+1A[\.\:\–\—\s\t\n]+RISK\s+FACTORS\b(?!\s+(?:of\b|under\b|relating\b))',
        "end_patterns": [
            r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+1B[\.\:\–\—\s\t\n]+UNRESOLVED\b(?!\s+(?:of\b|under\b|relating\b))',
            r'(?im)(?:^|\n)\s*(?:PART\s+I[\.\s\:\–\—\t\n]+)?ITEM\s+2[\.\:\–\—\s\t\n]+PROPERTIES\b(?!\s+(?:of\b|under\b|relating\b))',
        ],
    },
    {
        "item_section": "Item 7",
        "item_title": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
        "start_pattern": r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+7[\.\:\–\—\s\t\n]+MANAGEMENT(?!\s+(?:of\b|under\b|relating\b))',
        "end_patterns": [
            r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+7A[\.\:\–\—\s\t\n]+QUANTITATIVE\b(?!\s+(?:of\b|under\b|relating\b))',
            r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+8[\.\:\–\—\s\t\n]+FINANCIAL(?!\s+(?:of\b|under\b|relating\b))',
        ],
    },
    {
        "item_section": "Item 8",
        "item_title": "Financial Statements and Supplementary Data",
        "start_pattern": r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+8[\.\:\–\—\s\t\n]+FINANCIAL(?!\s+(?:of\b|under\b|relating\b))',
        "end_patterns": [
            r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+9[\.\:\–\—\s\t\n]+CHANGES\b(?!\s+(?:of\b|under\b|relating\b))',
            r'(?im)(?:^|\n)\s*(?:PART\s+II[\.\s\:\–\—\t\n]+)?ITEM\s+9A[\.\:\–\—\s\t\n]+CONTROLS\b(?!\s+(?:of\b|under\b|relating\b))',
        ],
    },
]


def estimate_token_count(text: str) -> int:
    """Estimates token count for a text string (approx. 1 word = 1.3 tokens)."""
    words = text.split()
    return int(len(words) * 1.3)


def extract_html_text(html_content: str) -> str:
    """
    Parses HTML using BeautifulSoup, preserving block breaks while keeping inline
    styled text contiguous (prevents words like 'B\\nUSINESS' across spans).
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.extract()
        
    # Append newlines only to block elements so inline spans remain contiguous
    for block in soup.find_all(["p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "hr"]):
        block.append("\n")
        
    text = soup.get_text()
    # Clean non-breaking spaces and redundant blank lines
    text = text.replace("\xa0", " ").replace("\r", "")
    lines = [line.strip() for line in text.split("\n")]
    cleaned_text = "\n".join(line for line in lines if line)
    return cleaned_text


def derive_period_end_and_fiscal_year(entry: Dict) -> Tuple[str, str]:
    """
    Derives period_end and fiscal_year from the manifest entry and filing URL.
    For filings whose period ends in January or February (e.g., NVIDIA FY ending Jan 2025):
    - labels the calendar year the fiscal year started in inside period_end
    - keeps fiscal_year consistent with how the company names it (e.g., FY 2025)
    
    Returns (period_end, fiscal_year).
    """
    url = entry.get("source_url", "") or entry.get("filing_url", "")
    # Check date in filename (e.g. nvda-20250126.htm)
    m = re.search(r'([12][0-9]{3})([01][0-9])([0-3][0-9])\.htm', url)
    if m:
        end_year = int(m.group(1))
        end_month = int(m.group(2))
        end_day = int(m.group(3))
        iso_date = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"
    else:
        iso_date = entry.get("filing_date", "2024-12-31")
        parts = iso_date.split("-")
        end_year = int(parts[0]) if len(parts) > 0 else 2024
        end_month = int(parts[1]) if len(parts) > 1 else 12

    company_fiscal_year = str(entry.get("fiscal_year", end_year))

    # If period ends in January or February, label the calendar year the fiscal year started in
    if end_month in [1, 2]:
        cal_start_year = end_year - 1
        period_end = f"{iso_date} (FY started {cal_start_year})"
    else:
        period_end = iso_date

    return period_end, company_fiscal_year


def find_item_sections(full_text: str) -> Dict[str, Tuple[str, str]]:
    """
    Locates body text segments for Item 1, Item 1A, Item 7, and Item 8.
    Picks the longest candidate between each start match and the next Item heading,
    which safely differentiates body sections from Table of Contents links without
    skipping valid body boundaries.
    
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
        
        # Evaluate all start matches and pick the longest candidate up to the next item heading
        for match in start_matches:
            start_pos = match.start()
            end_pos = len(full_text)
            
            for end_pat in end_pats:
                end_matches = [m for m in re.finditer(end_pat, full_text) if m.start() > start_pos]
                if end_matches:
                    end_pos = min(end_pos, end_matches[0].start())
                    
            section_candidate = full_text[start_pos:end_pos].strip()
            if len(section_candidate) > len(best_section_text):
                best_section_text = section_candidate

        # Special fallback for Item 8 stubs (e.g. NVIDIA incorporates financial statements in Item 15)
        if sec_key == "Item 8" and len(best_section_text) < 1000:
            m_sched = re.search(
                r'(?i)\b(?:Item\s+15[^\n]*\n(?:[^\n]*\n)?(?:\(a\)\s*1\.?\s*)?Financial\s+Statements|\(a\)\s*1\.?\s*Financial\s+Statements)',
                full_text,
            )
            if m_sched:
                sp = m_sched.start()
                m_end = re.search(r'(?i)\b(?:Exhibit\s+Index|Item\s+16|SIGNATURES)\b', full_text[sp + 1000:])
                ep = sp + 1000 + m_end.start() if m_end else len(full_text)
                sched_candidate = full_text[sp:ep].strip()
                if len(sched_candidate) > 1000:
                    best_section_text = sched_candidate
                
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
    period_end: Optional[str] = None,
) -> List[Dict]:
    """
    Splits section text into paragraph/sentence bounded chunks of ~300-500 tokens
    with ~15% overlap between adjacent chunks.
    Assigns deterministic UUID5 chunk_ids derived from company, year, section, chunk_index, and text prefix.
    """
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
            chunk_text = "\n\n".join(current_units).strip()
            chunk_index = len(chunks)
            chunk_hash_key = f"{company}|{fiscal_year}|{item_section}|{chunk_index}|{chunk_text[:200]}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_hash_key))
            
            chunk_dict = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "company": company,
                "fiscal_year": fiscal_year,
                "item_section": item_section,
                "item_title": item_title,
                "source_path": source_path,
            }
            if period_end:
                chunk_dict["period_end"] = period_end
                
            chunks.append(chunk_dict)
            
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
                
            # Guard against exact duplication on boundary
            if overlap_units == current_units and i < len(units) - 1 and len(overlap_units) > 1:
                overlap_units = overlap_units[1:]
                overlap_tokens = sum(estimate_token_count(u) for u in overlap_units)
                
            current_units = overlap_units
            current_tokens = overlap_tokens
            
        i += 1
        
    return chunks


def chunk_filing(manifest_entry: Dict) -> List[Dict]:
    """
    Parses a 10-K HTML filing from a manifest entry and splits it into structural chunks.
    Performs validation: logs a WARNING if any section (Item 1, 1A, 7) has under 20 chunks.
    
    Returns a list of chunk dictionaries.
    """
    rel_path = manifest_entry["local_path"]
    full_file_path = BASE_DIR.parent / rel_path
    
    if not full_file_path.exists():
        logger.error(f"Filing HTML file not found at {full_file_path}")
        return []
        
    period_end, company_fiscal_year = derive_period_end_and_fiscal_year(manifest_entry)
    company = manifest_entry["company"]
    
    logger.info(f"Processing {company} (FY {company_fiscal_year}, period end: {period_end}) from {rel_path}...")
    html_content = full_file_path.read_text(encoding="utf-8")
    
    # Step 1: Clean HTML to text
    full_text = extract_html_text(html_content)
    
    # Step 2: Extract structural sections (Item 1, 1A, 7, 8)
    sections = find_item_sections(full_text)
    
    # Step 3: Chunk each section and validate chunk counts
    filing_chunks = []
    for sec_key, (sec_title, sec_text) in sections.items():
        sec_chunks = split_section_into_chunks(
            section_text=sec_text,
            company=company,
            fiscal_year=company_fiscal_year,
            item_section=sec_key,
            item_title=sec_title,
            source_path=rel_path,
            period_end=period_end,
        )
        
        # Validation step: warn if Item 1, 1A, or 7 is under 20 chunks
        if sec_key in ["Item 1", "Item 1A", "Item 7"] and len(sec_chunks) < 20:
            logger.warning(
                f"VALIDATION WARNING: Filing '{company} (FY {company_fiscal_year})' has only {len(sec_chunks)} chunks "
                f"for {sec_key} (< 20 chunks threshold)."
            )
            
        filing_chunks.extend(sec_chunks)
        
    logger.info(f"Generated {len(filing_chunks)} chunks for {company} ({company_fiscal_year})")
    return filing_chunks


def deduplicate_chunks(chunks: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Dedupes chunks with identical text within the same company+year+section,
    including identical overlapping chunk boundaries.
    Preserves input order and deterministic IDs.
    Returns a tuple of (deduplicated_chunks, removed_count).
    """
    seen_keys = set()
    deduped = []
    removed_count = 0

    for chunk in chunks:
        norm_text = re.sub(r'\s+', ' ', chunk.get("text", "").strip())
        dedupe_key = (
            chunk.get("company", ""),
            str(chunk.get("fiscal_year", "")),
            chunk.get("item_section", ""),
            norm_text,
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
