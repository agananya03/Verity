"""
BM25 Keyword Index & Corpus Artifacts Builder for Verity.

Reads processed SEC 10-K chunks from `backend/data/processed/chunks.json`,
generates production artifacts in `backend/artifacts/`:
1. `bm25_inputs.json`: Lightweight sparse retrieval inputs containing:
   (chunk_id, text, company, fiscal_year, item_section). Rebuilt in ~0.2s at startup
   without insecure pickle files.
2. `corpus_manifest.json`: Corpus metadata with total chunks, SHA-256 digest of chunks.json,
   generation timestamp, per-company/year/section counts, and filing URL mappings.

Deletes obsolete legacy files (`bm25_inputs.npz` and `bm25_index.pkl`) and reports artifact sizes.
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

CHUNKS_FILE = DATA_PROCESSED_DIR / "chunks.json"
RAW_MANIFEST_FILE = DATA_RAW_DIR / "manifest.json"

BM25_INPUTS_JSON = ARTIFACTS_DIR / "bm25_inputs.json"
CORPUS_MANIFEST_JSON = ARTIFACTS_DIR / "corpus_manifest.json"

# Obsolete artifact paths to clean up
OBSOLETE_FILES = [
    ARTIFACTS_DIR / "bm25_inputs.npz",
    ARTIFACTS_DIR / "bm25_index.pkl",
    DATA_PROCESSED_DIR / "bm25_index.pkl",
]

COMPANY_TICKERS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while block := f.read(65536):
            hasher.update(block)
    return hasher.hexdigest()


def remove_obsolete_files():
    """Checks and removes obsolete npz and legacy pkl artifact files."""
    for file_path in OBSOLETE_FILES:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Removed obsolete file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove {file_path}: {e}")


def format_size(bytes_count: int) -> str:
    """Formats bytes into human-readable size string."""
    if bytes_count < 1024:
        return f"{bytes_count} bytes"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB ({bytes_count:,} bytes)"
    else:
        return f"{bytes_count / (1024 * 1024):.2f} MB ({bytes_count:,} bytes)"


def build_artifacts():
    """Generates bm25_inputs.json and corpus_manifest.json from chunks.json."""
    print("=" * 70)
    print("Building Verity BM25 & Corpus Manifest Artifacts")
    print("=" * 70)

    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found at {CHUNKS_FILE}")
        logger.error("Please run chunker.py first to generate chunks.json.")
        sys.exit(1)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load chunks
    logger.info(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    total_chunks = len(chunks)
    logger.info(f"Loaded {total_chunks} chunks.")
    if total_chunks == 0:
        logger.error("chunks.json is empty!")
        sys.exit(1)

    # Compute sha256 of chunks.json
    chunks_sha256 = compute_file_sha256(CHUNKS_FILE)
    logger.info(f"chunks.json SHA-256: {chunks_sha256}")

    # 2. Build bm25_inputs.json (exact required fields: chunk_id, text, company, fiscal_year, item_section)
    logger.info("Constructing bm25_inputs.json records...")
    bm25_records = []
    for c in chunks:
        bm25_records.append({
            "chunk_id": str(c.get("chunk_id", "")),
            "text": c.get("text", ""),
            "company": c.get("company", ""),
            "fiscal_year": str(c.get("fiscal_year", "")),
            "item_section": c.get("item_section", ""),
        })

    with open(BM25_INPUTS_JSON, "w", encoding="utf-8") as f:
        json.dump(bm25_records, f, ensure_ascii=False)

    bm25_inputs_size = BM25_INPUTS_JSON.stat().st_size
    logger.info(f"Wrote {len(bm25_records)} items to {BM25_INPUTS_JSON} [{format_size(bm25_inputs_size)}]")

    # 3. Compute per-company, per-year, per-section breakdown
    counts_by_company_year_section: Dict[str, Dict[str, Dict[str, int]]] = {}
    company_totals: Dict[str, int] = {}
    company_years: Dict[str, set] = {}
    filing_chunk_counts: Dict[tuple, int] = {}

    for c in chunks:
        comp = c.get("company", "UNKNOWN")
        yr = str(c.get("fiscal_year", "UNKNOWN"))
        sec = c.get("item_section", "UNKNOWN")

        if comp not in counts_by_company_year_section:
            counts_by_company_year_section[comp] = {}
            company_totals[comp] = 0
            company_years[comp] = set()

        if yr not in counts_by_company_year_section[comp]:
            counts_by_company_year_section[comp][yr] = {}

        counts_by_company_year_section[comp][yr][sec] = (
            counts_by_company_year_section[comp][yr].get(sec, 0) + 1
        )
        company_totals[comp] += 1
        company_years[comp].add(yr)

        filing_key = (comp, yr)
        filing_chunk_counts[filing_key] = filing_chunk_counts.get(filing_key, 0) + 1

    # Load raw manifest if available to enrich filing details
    raw_filings = []
    if RAW_MANIFEST_FILE.exists():
        try:
            with open(RAW_MANIFEST_FILE, "r", encoding="utf-8") as f:
                raw_filings = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load raw manifest: {e}")

    # Build filings list with chunk count annotations
    filings_metadata = []
    for rf in raw_filings:
        comp = rf.get("company", "")
        yr = str(rf.get("fiscal_year", ""))
        filing_entry = dict(rf)
        filing_entry["chunk_count"] = filing_chunk_counts.get((comp, yr), 0)
        filings_metadata.append(filing_entry)

    # Build companies list
    companies_summary = []
    # Standard ordering
    company_order = ["Apple", "Microsoft", "Alphabet", "Amazon", "NVIDIA"]
    all_companies = sorted(
        counts_by_company_year_section.keys(),
        key=lambda x: company_order.index(x) if x in company_order else 99,
    )

    for comp in all_companies:
        years_dict = counts_by_company_year_section[comp]
        sorted_years = sorted(list(company_years[comp]))

        chunk_counts_per_year = {yr: sum(years_dict[yr].values()) for yr in sorted_years}
        section_counts_all = {}
        for yr in sorted_years:
            for sec, count in years_dict[yr].items():
                section_counts_all[sec] = section_counts_all.get(sec, 0) + count

        companies_summary.append({
            "ticker": COMPANY_TICKERS.get(comp, "UNKNOWN"),
            "name": comp,
            "company": comp,
            "fiscal_years": sorted_years,
            "total_chunks": company_totals[comp],
            "chunk_counts": chunk_counts_per_year,
            "section_counts": section_counts_all,
            "breakdown": years_dict,
        })

    # Construct complete corpus manifest
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest_data = {
        "total_chunks": total_chunks,
        "generated_at": generated_at,
        "sha256": chunks_sha256,
        "chunks_sha256": chunks_sha256,
        "filing_type": "10-K",
        "per_company_year_section": counts_by_company_year_section,
        "companies": companies_summary,
        "filings": filings_metadata,
    }

    with open(CORPUS_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    manifest_size = CORPUS_MANIFEST_JSON.stat().st_size
    logger.info(f"Wrote manifest to {CORPUS_MANIFEST_JSON} [{format_size(manifest_size)}]")

    # 4. Clean up obsolete files
    remove_obsolete_files()

    # 5. Output summary
    print("-" * 70)
    print("ARTIFACT GENERATION SUMMARY:")
    print(f"  • Total Chunks:           {total_chunks}")
    print(f"  • chunks.json SHA-256:    {chunks_sha256}")
    print(f"  • Generated At (UTC):     {generated_at}")
    print(f"  • bm25_inputs.json Size:  {format_size(bm25_inputs_size)}")
    print(f"  • corpus_manifest.json:   {format_size(manifest_size)}")
    print("-" * 70)
    print("CHUNK BREAKDOWN BY COMPANY & FISCAL YEAR:")
    for comp in all_companies:
        print(f"  {comp:12s} | Total: {company_totals[comp]:4d} chunks")
        for yr in sorted(counts_by_company_year_section[comp].keys()):
            sec_str = ", ".join(
                f"{sec}: {cnt}" for sec, cnt in sorted(counts_by_company_year_section[comp][yr].items())
            )
            print(f"    - FY {yr}: {sec_str}")
    print("=" * 70)


def main():
    build_artifacts()


if __name__ == "__main__":
    main()
