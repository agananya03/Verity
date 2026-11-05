"""
SEC EDGAR 10-K Filing Downloader for Verity.

Important SEC EDGAR API Requirements & Gotchas:
-----------------------------------------------
1. User-Agent Header Format:
   - SEC EDGAR explicitly requires a customized User-Agent header in the format:
     `User-Agent: <Sample Company Name> <AdminContact@domain.com>`
   - Example: `VerityResearch admin@verity.com`
   - Requests with default HTTP client user agents (e.g., `python-requests/2.x.x`)
     or improperly formatted headers will be blocked with an HTTP 403 Forbidden error.

2. Rate Limiting:
   - SEC EDGAR strictly enforces a rate limit of a maximum of 10 requests per second per IP address.
   - Exceeding this rate limit may lead to your IP being temporarily blocked.
   - This script implements explicit delay throttling (>= 0.15 seconds between requests)
     to safely remain below the SEC threshold.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SEC EDGAR Requirements & Defaults
DEFAULT_USER_AGENT = "VerityResearch admin@verity.com"
SEC_RATE_LIMIT_DELAY = 0.15  # seconds between HTTP requests (~6.6 requests/sec)

# Target Companies (Apple, Microsoft, Alphabet/Google, Amazon, NVIDIA)
COMPANIES = [
    {"name": "Apple", "ticker": "AAPL", "cik": "0000320193"},
    {"name": "Microsoft", "ticker": "MSFT", "cik": "0000789019"},
    {"name": "Alphabet", "ticker": "GOOGL", "cik": "0001652044"},
    {"name": "Amazon", "ticker": "AMZN", "cik": "0001018724"},
    {"name": "NVIDIA", "ticker": "NVDA", "cik": "0001045810"},
]

# Path configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MANIFEST_FILE = DATA_RAW_DIR / "manifest.json"


class SECFetcher:
    """Fetcher for querying SEC EDGAR submissions API and downloading filing HTML documents."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforces a minimal delay between requests to stay below SEC's 10 req/sec limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def get_company_submissions(self, cik: str) -> Optional[Dict]:
        """
        Fetches metadata for a given CIK from SEC EDGAR data API.
        
        API Endpoint: https://data.sec.gov/submissions/CIK{cik_10_digits}.json
        """
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        
        self._rate_limit()
        logger.info(f"Fetching submission metadata for CIK {cik_padded}...")
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch metadata for CIK {cik}: {e}")
            return None

    def extract_recent_10k_filings(self, submission_data: Dict, max_years: int = 3) -> List[Dict]:
        """
        Extracts recent 10-K filing metadata from submission response.
        
        Returns filing metadata for the last `max_years` fiscal filings.
        """
        cik = submission_data["cik"]
        recent = submission_data.get("filings", {}).get("recent", {})
        
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])
        
        filings = []
        for i, form in enumerate(forms):
            if form == "10-K":
                acc_num = accession_numbers[i]
                acc_num_no_dash = acc_num.replace("-", "")
                cik_unpadded = str(int(cik))  # Document archive URLs use unpadded CIK
                primary_doc = primary_docs[i]
                
                filing_date = filing_dates[i]
                report_date = report_dates[i] if i < len(report_dates) and report_dates[i] else filing_date
                
                # Derive fiscal year from report date (YYYY-MM-DD) or filing date
                fiscal_year = report_date[:4] if report_date else filing_date[:4]
                
                source_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{acc_num_no_dash}/{primary_doc}"
                )
                filing_url = source_url
                
                filings.append({
                    "accession_number": acc_num,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "fiscal_year": fiscal_year,
                    "primary_doc": primary_doc,
                    "source_url": source_url,
                    "filing_url": filing_url,
                })
                
                if len(filings) >= max_years:
                    break
                    
        return filings

    def fetch_filing_html(self, url: str) -> Optional[str]:
        """Downloads filing document HTML from SEC archives."""
        self._rate_limit()
        logger.info(f"Downloading filing document from {url}...")
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch filing document from {url}: {e}")
            return None


def fetch_and_save_filings():
    """Fetches 10-K filings for configured companies and updates manifest.json."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = SECFetcher()
    
    # Load existing manifest entries if manifest.json exists
    existing_manifest = []
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing manifest: {e}")
            
    existing_companies = {entry["ticker"] for entry in existing_manifest if "ticker" in entry}
    manifest_entries = list(existing_manifest)

    for company in COMPANIES:
        name = company["name"]
        ticker = company["ticker"]
        cik = company["cik"]
        
        # If company is already in existing manifest and local files exist, skip re-fetching
        if ticker in existing_companies:
            logger.info(f"=== Skipping re-fetch for {name} ({ticker}): already in manifest ===")
            continue

        logger.info(f"=== Processing new company {name} ({ticker}) [CIK: {cik}] ===")
        sub_data = fetcher.get_company_submissions(cik)
        if not sub_data:
            logger.warning(f"Skipping {name}: failed to retrieve submission metadata.")
            continue

        recent_10ks = fetcher.extract_recent_10k_filings(sub_data, max_years=3)
        if not recent_10ks:
            logger.warning(f"No 10-K filings found for {name}.")
            continue

        for filing in recent_10ks:
            fiscal_year = filing["fiscal_year"]
            file_name = f"{name.lower()}_{fiscal_year}.html"
            local_file_path = DATA_RAW_DIR / file_name
            
            # Check if file already exists locally
            if local_file_path.exists() and local_file_path.stat().st_size > 0:
                logger.info(f"Filing HTML file {file_name} already exists locally. Skipping download.")
                html_content = True
            else:
                html_content = fetcher.fetch_filing_html(filing["source_url"])
                if html_content:
                    local_file_path.write_text(html_content, encoding="utf-8")
                    logger.info(f"Successfully saved {file_name} ({len(html_content)} bytes)")

            if html_content:
                relative_path = str(local_file_path.relative_to(BASE_DIR.parent)).replace("\\", "/")
                entry_data = {
                    "company": name,
                    "ticker": ticker,
                    "cik": cik,
                    "fiscal_year": fiscal_year,
                    "filing_date": filing["filing_date"],
                    "source_url": filing["source_url"],
                    "filing_url": filing.get("filing_url") or filing["source_url"],
                    "local_path": relative_path,
                }
                # Add fiscal_year note for non-calendar fiscal years
                if ticker == "NVDA":
                    entry_data["fiscal_year_note"] = "Non-calendar fiscal year ends in January (e.g., FY2024 ended January 2024)"
                elif ticker == "AAPL":
                    entry_data["fiscal_year_note"] = "Non-calendar fiscal year ends in late September"
                elif ticker == "MSFT":
                    entry_data["fiscal_year_note"] = "Non-calendar fiscal year ends June 30"

                manifest_entries.append(entry_data)

    # Ensure all manifest entries (including existing) include filing_url and non-calendar fiscal_year_note
    for entry in manifest_entries:
        if "filing_url" not in entry and "source_url" in entry:
            entry["filing_url"] = entry["source_url"]
        t = entry.get("ticker", "")
        if "fiscal_year_note" not in entry:
            if t == "NVDA":
                entry["fiscal_year_note"] = "Non-calendar fiscal year ends in January (e.g., FY2024 ended January 2024)"
            elif t == "AAPL":
                entry["fiscal_year_note"] = "Non-calendar fiscal year ends in late September"
            elif t == "MSFT":
                entry["fiscal_year_note"] = "Non-calendar fiscal year ends June 30"

    # Save manifest.json
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)
    logger.info(f"Updated manifest at {MANIFEST_FILE}")


def main():
    """Main execution entry point."""
    print("=" * 70)
    print("Starting SEC EDGAR 10-K Filings Fetcher")
    print(f"Target Companies: {', '.join(c['name'] for c in COMPANIES)}")
    print("Target Filings: Last 3 Fiscal Years per company")
    print("=" * 70)
    
    fetch_and_save_filings()
    
    print("=" * 70)
    print("Fetch complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
