"""
Debug Script for SEC 10-K Section Extraction in Verity.

Iterates through all filings in manifest.json, extracts text,
runs section extraction, and prints:
- Company, fiscal year, local file path
- For each Item section (Item 1, 1A, 7, 8):
  - Total character length
  - First 150 characters of captured content
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingestion.chunker import (
    BASE_DIR,
    MANIFEST_FILE,
    extract_html_text,
    find_item_sections,
)


def debug_all_sections():
    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file not found at {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 80)
    print("VERITY 10-K SECTION EXTRACTION DIAGNOSTIC")
    print("=" * 80)

    for entry in manifest:
        company = entry["company"]
        fiscal_year = entry.get("fiscal_year", "")
        rel_path = entry["local_path"]
        full_file_path = BASE_DIR.parent / rel_path

        print(f"\n[{company} | FY {fiscal_year}] File: {rel_path}")
        print("-" * 80)

        if not full_file_path.exists():
            print(f"  [ERROR] File not found: {full_file_path}")
            continue

        html_content = full_file_path.read_text(encoding="utf-8")
        full_text = extract_html_text(html_content)
        sections = find_item_sections(full_text)

        for sec_name in ["Item 1", "Item 1A", "Item 7", "Item 8"]:
            if sec_name in sections:
                title, text = sections[sec_name]
                preview = text[:150].replace("\n", " ").strip()
                print(f"  {sec_name:8s} | Length: {len(text):7d} chars | Preview: {preview}...")
            else:
                print(f"  {sec_name:8s} | [MISSING / NOT EXTRACTED]")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    debug_all_sections()
