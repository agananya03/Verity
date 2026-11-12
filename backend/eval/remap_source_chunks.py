"""
Source Chunk Remapper & Validator for Verity Financial RAG Golden Set.

Remaps source_chunk_ids in golden_set.json to the deterministic UUIDv5 chunk IDs
in backend/data/processed/chunks.json (and backend/artifacts/bm25_inputs.json).

Features:
- Enforces strict company and fiscal_year matching (never picks a chunk from a different year).
- Ranks candidates by ground truth key phrases, numbers, and verbatim evidence.
- Stores full UUIDv5 IDs in source_chunk_ids and verbatim excerpts in source_quotes.
- Preserves empty source_chunk_ids for unanswerable items.
- Flags items without confident matches for manual fixing.
- Generates backend/eval/results/remap_report.md.
- Backs up original file to golden_set.pre_remap.json.
- Provides validate_golden_source_chunks() for pytest and runtime validation.
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = EVAL_DIR / "results"

DEFAULT_GOLDEN_PATH = EVAL_DIR / "golden_set.json"
BACKUP_GOLDEN_PATH = EVAL_DIR / "golden_set.pre_remap.json"
DEFAULT_CHUNKS_PATH = DATA_PROCESSED_DIR / "chunks.json"
DEFAULT_BM25_INPUTS_PATH = ARTIFACTS_DIR / "bm25_inputs.json"
DEFAULT_GT_CHECK_PATH = RESULTS_DIR / "ground_truth_check.md"
DEFAULT_REPORT_PATH = RESULTS_DIR / "remap_report.md"


def get_corpus_chunk_ids(
    chunks_path: Optional[Path] = None,
    bm25_inputs_path: Optional[Path] = None,
) -> Tuple[Set[str], Path]:
    """
    Returns the set of all valid chunk IDs from the available corpus file.
    Prefers chunks.json, falls back to bm25_inputs.json.
    Raises FileNotFoundError if neither file exists.
    """
    c_path = chunks_path or DEFAULT_CHUNKS_PATH
    b_path = bm25_inputs_path or DEFAULT_BM25_INPUTS_PATH

    target_path = None
    if c_path.exists():
        target_path = c_path
    elif b_path.exists():
        target_path = b_path
    else:
        raise FileNotFoundError(
            f"Corpus files not found. Looked for {c_path} and {b_path}."
        )

    with open(target_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunk_ids = {
        str(item.get("chunk_id") or item.get("id"))
        for item in corpus
        if (item.get("chunk_id") or item.get("id"))
    }
    return chunk_ids, target_path


def validate_golden_source_chunks(
    golden_set_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    bm25_inputs_path: Optional[Path] = None,
) -> List[str]:
    """
    Validates that every source_chunk_id in golden_set.json exists in the current corpus.
    Checks chunks.json if present; otherwise checks backend/artifacts/bm25_inputs.json.

    Returns:
        List of missing chunk IDs (empty if all exist).
    """
    g_path = golden_set_path or DEFAULT_GOLDEN_PATH
    if not g_path.exists():
        raise FileNotFoundError(f"Golden set file not found at: {g_path}")

    valid_chunk_ids, corpus_file = get_corpus_chunk_ids(
        chunks_path=chunks_path,
        bm25_inputs_path=bm25_inputs_path,
    )

    with open(g_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    missing_ids: List[str] = []
    for item in golden_set:
        source_ids = item.get("source_chunk_ids")
        if source_ids is None and item.get("chunk_id"):
            source_ids = [item["chunk_id"]]
        if not source_ids:
            continue

        for cid in source_ids:
            cid_str = str(cid).strip()
            if cid_str and cid_str not in valid_chunk_ids:
                missing_ids.append(cid_str)

    return missing_ids


def clean_tokens(text: str) -> List[str]:
    """Extracts alphanumeric words for lexical overlap scoring."""
    return [
        w.lower().strip(".,;:\"'()[]{}®™℠")
        for w in re.findall(r"\b[\w$%\.\,\-\/]+\b", text)
        if len(w) > 2
    ]


def extract_numbers(text: str) -> Set[str]:
    """Extracts numerical quantities and percentages from text."""
    return set(re.findall(r"\b\d+(?:[\.,]\d+)*(?:%|\b)", text))


def parse_ground_truth_check_report(report_path: Path) -> Dict[str, Dict[str, Any]]:
    """Parses ground_truth_check.md for target chunk hints and verbatim excerpts."""
    if not report_path.exists():
        return {}

    content = report_path.read_text(encoding="utf-8")
    items_raw = content.split("### Item `")
    info: Dict[str, Dict[str, Any]] = {}
    for block in items_raw[1:]:
        item_id = block.split("`")[0].strip()
        target_chunk_m = re.search(r"\*\*Target Chunk\(s\)\*\*:\s*`([^`]+)`", block)
        target_chunk = target_chunk_m.group(1) if target_chunk_m else ""

        v_match = re.search(
            r"\*\*Verbatim Text in `chunks\.json`\*\*:\s*```text\s*(.*?)\s*```",
            block,
            re.DOTALL,
        )
        v_text = v_match.group(1).strip() if v_match else ""

        rec_match = re.search(
            r"\*\*Recommended Hand-Fix Replacement\*\*:\s*>\s*\"([^\"]+)\"", block
        )
        rec_text = rec_match.group(1).strip() if rec_match else ""

        info[item_id] = {
            "target_chunk": target_chunk,
            "verbatim_text": v_text,
            "rec_text": rec_text,
        }
    return info


def extract_best_quote(chunk_text: str, ground_truth: str) -> str:
    """Extracts the most relevant sentence or clause from chunk text supporting ground truth."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text) if s.strip()]
    gt_toks = set(clean_tokens(ground_truth))

    scored_sentences = []
    for s in sentences:
        s_toks = set(clean_tokens(s))
        overlap = sum(1 for t in gt_toks if t in s_toks)
        scored_sentences.append((s, overlap))

    if scored_sentences:
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_s, best_score = scored_sentences[0]
        if best_score > 0:
            return best_s

    # Fallback to first 200 characters
    return chunk_text[:200].strip()


def remap_golden_set(
    golden_set_path: Path = DEFAULT_GOLDEN_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    gt_check_path: Path = DEFAULT_GT_CHECK_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    backup_path: Path = BACKUP_GOLDEN_PATH,
) -> List[Dict[str, Any]]:
    """
    Remaps source_chunk_ids in golden_set.json to the current chunks.json deterministic IDs.
    Writes golden_set.pre_remap.json backup and results/remap_report.md.
    """
    if not golden_set_path.exists():
        raise FileNotFoundError(f"Golden set file not found: {golden_set_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks corpus file not found: {chunks_path}")

    # Step 1: Backup original golden set if not already backed up
    if not backup_path.exists():
        shutil.copyfile(golden_set_path, backup_path)
        logger.info(f"Backed up original golden set to: {backup_path}")
    else:
        logger.info(f"Existing backup retained at: {backup_path}")

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set: List[Dict[str, Any]] = json.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    # Index chunks by (company.lower(), fiscal_year)
    chunks_by_comp_year: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for c in chunks:
        key = (c["company"].lower(), str(c["fiscal_year"]))
        chunks_by_comp_year.setdefault(key, []).append(c)

    gt_check_info = parse_ground_truth_check_report(gt_check_path)

    report_items: List[Dict[str, Any]] = []

    for item in golden_set:
        item_id = item["id"]
        category = item.get("category", "")
        company = item.get("company", "")
        fy_str = str(item.get("fiscal_year", ""))
        question = item.get("question", "")
        gt = item.get("ground_truth", "")
        contexts = item.get("contexts", [])

        # Unanswerable items get empty source_chunk_ids and source_quotes
        if category == "unanswerable":
            item["source_chunk_ids"] = []
            item["source_quotes"] = []
            item["chunk_id"] = None
            report_items.append({
                "id": item_id,
                "category": category,
                "company": company,
                "fiscal_year": fy_str,
                "status": "UNANSWERABLE",
                "match_score": 1.0,
                "chunk_ids": [],
                "source_quotes": [],
                "sections": [],
                "previews": [],
                "notes": "Unanswerable baseline outside corpus; source_chunk_ids empty.",
            })
            continue

        # Parse required fiscal years (multi-hop items list multiple years, e.g. "2023/2024")
        years = [y.strip() for y in re.split(r"[/,]", fy_str) if y.strip()]
        check_item = gt_check_info.get(item_id, {})
        check_verb = check_item.get("verbatim_text", "")
        check_rec = check_item.get("rec_text", "")
        check_cids = [
            cid.strip()
            for cid in re.findall(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                check_item.get("target_chunk", ""),
            )
        ]

        # golden_10 is known to be in raw filing TOC preface outside ingested Items 1, 1A, 7, 8
        if item_id == "golden_10":
            item["source_chunk_ids"] = []
            item["source_quotes"] = []
            item["chunk_id"] = None
            report_items.append({
                "id": item_id,
                "category": category,
                "company": company,
                "fiscal_year": fy_str,
                "status": "NO_CONFIDENT_MATCH",
                "match_score": 0.0,
                "chunk_ids": [],
                "source_quotes": [],
                "sections": [],
                "previews": [],
                "notes": "FLAGGED: Text located in raw 2023 Form 10-K preface outside ingested Items 1, 1A, 7, 8. Needs manual fix.",
            })
            continue

        chosen_chunk_ids: List[str] = []
        chosen_source_quotes: List[str] = []
        chosen_sections: List[str] = []
        chosen_previews: List[str] = []
        year_scores: List[float] = []

        # Find supporting chunk for each required fiscal year
        for yr_idx, yr in enumerate(years):
            candidates = chunks_by_comp_year.get((company.lower(), yr), [])
            if not candidates:
                logger.warning(f"No candidates for {company} FY{yr} (item {item_id})")
                continue

            best_chunk: Optional[Dict[str, Any]] = None
            best_score = -1.0
            gt_tokens = clean_tokens(gt)
            gt_nums = extract_numbers(gt)

            # For multi-hop, match against the year-specific fact and corresponding context slice
            yr_context = contexts[yr_idx] if yr_idx < len(contexts) else ""

            # Extract year-specific ground truth focus if multi-hop
            yr_gt = gt
            if "2023" in yr and "2024" in fy_str:
                if item_id == "golden_23":
                    yr_gt = "iPhone 15 Pro, iPhone 15, iPhone 14, iPhone 13 and iPhone SE. Wearables includes smartwatches and wireless headphones."
                elif item_id == "golden_24":
                    yr_gt = "smartwatches is watchOS (specifically watchOS®)"
                elif item_id == "golden_25":
                    yr_gt = "iOS for the iPhone and iPadOS for the iPad"
                elif item_id == "golden_26":
                    yr_gt = "iPhone line of smartphones is iOS, and the operating system for the smartwatch line is watchOS"
            elif "2024" in yr and "2023" in fy_str:
                if item_id == "golden_23":
                    yr_gt = "iPhone 16 Pro, iPhone 16, iPhone 15, iPhone 14 and iPhone SE. Wearables includes smartwatches, wireless headphones and spatial computers."
                elif item_id == "golden_24":
                    yr_gt = "Apple Vision Pro, is visionOS (specifically visionOS™)"
                elif item_id == "golden_25":
                    yr_gt = "five reportable geographic segments are the Americas, Europe, Greater China, Japan, and Rest of Asia Pacific"
                elif item_id == "golden_26":
                    yr_gt = "net sales through indirect distribution channels accounted for 62% of total net sales during 2024"

            gt_tokens = clean_tokens(yr_gt)
            gt_nums = extract_numbers(yr_gt)

            for c in candidates:
                c_text = c["text"]
                c_clean = " ".join(c_text.lower().split())
                c_tokens = set(clean_tokens(c_text))
                c_nums = extract_numbers(c_text)

                score = 0.0

                # 1. Verbatim clause matching from ground truth
                clauses = [cl.strip() for cl in re.split(r"[.;\n]", yr_gt) if len(cl.strip()) > 10]
                for cl in clauses:
                    cl_clean = " ".join(cl.lower().split())
                    if cl_clean in c_clean:
                        score += 60.0
                    else:
                        cl_words = clean_tokens(cl)
                        if cl_words:
                            overlap = sum(1 for w in cl_words if w in c_tokens)
                            score += 25.0 * (overlap / len(cl_words))

                # 2. Check snippet from ground_truth_check.md
                if check_verb and any(v.strip() in c_text for v in check_verb.split("\n") if len(v.strip()) > 20):
                    score += 40.0
                if check_rec and check_rec.lower() in c_clean:
                    score += 40.0

                # 3. Target chunk hint from report
                if c["chunk_id"] in check_cids:
                    score += 30.0

                # 4. Context overlap (secondary weight)
                if yr_context:
                    for ctx_line in yr_context.split("\n"):
                        ctx_line = ctx_line.strip()
                        if len(ctx_line) > 20 and ctx_line.lower() in c_clean:
                            score += 5.0

                # 5. Number and token overlap
                if gt_nums:
                    n_overlap = sum(1 for n in gt_nums if n in c_nums)
                    score += 30.0 * (n_overlap / len(gt_nums))

                if gt_tokens:
                    t_overlap = sum(1 for t in gt_tokens if t in c_tokens)
                    score += 20.0 * (t_overlap / len(gt_tokens))

                # Special key fact verification boosts
                if item_id == "golden_24" and yr == "2024" and "visionos" in c_clean:
                    score += 80.0
                if item_id == "golden_25" and yr == "2024" and "reportable segments" in c_clean and "greater china" in c_clean:
                    score += 80.0
                if item_id == "golden_26" and yr == "2024" and "62%" in c_clean and "indirect" in c_clean:
                    score += 80.0
                if item_id == "golden_27" and "americas, europe, greater china" in c_clean:
                    score += 60.0
                if item_id == "golden_28" and "40%" in c_clean and "60%" in c_clean:
                    score += 60.0
                if item_id == "golden_29" and "only one source" in c_clean:
                    score += 60.0
                if item_id == "golden_11" and "aggressive price competition" in c_clean:
                    score += 60.0

                if score > best_score:
                    best_score = score
                    best_chunk = c

            if best_chunk and best_score >= 20.0:
                chosen_chunk_ids.append(best_chunk["chunk_id"])
                quote = extract_best_quote(best_chunk["text"], gt)
                chosen_source_quotes.append(quote)
                chosen_sections.append(best_chunk.get("item_section", "Unknown"))
                preview = best_chunk["text"][:150].replace("\n", " ").strip()
                chosen_previews.append(preview)
                year_scores.append(round(min(1.0, best_score / 100.0), 2))

        # Check if all required years were matched confidently
        if len(chosen_chunk_ids) == len(years):
            overall_status = "CONFIDENT_MATCH"
            overall_score = round(sum(year_scores) / len(year_scores), 2)
            item["source_chunk_ids"] = chosen_chunk_ids
            item["source_quotes"] = chosen_source_quotes
            item["chunk_id"] = chosen_chunk_ids[0] if chosen_chunk_ids else None
        else:
            overall_status = "NO_CONFIDENT_MATCH"
            overall_score = 0.0
            item["source_chunk_ids"] = []
            item["source_quotes"] = []
            item["chunk_id"] = None

        report_items.append({
            "id": item_id,
            "category": category,
            "company": company,
            "fiscal_year": fy_str,
            "status": overall_status,
            "match_score": overall_score,
            "chunk_ids": chosen_chunk_ids,
            "source_quotes": chosen_source_quotes,
            "sections": chosen_sections,
            "previews": chosen_previews,
            "notes": "Confident match found." if overall_status == "CONFIDENT_MATCH" else "FLAGGED: No confident match found across all required years.",
        })

    # Step 2: Save updated golden_set.json
    with open(golden_set_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved remapped golden set to: {golden_set_path}")

    # Step 3: Write backend/eval/results/remap_report.md
    write_remap_report(report_items, report_path)
    return report_items


def write_remap_report(items: List[Dict[str, Any]], report_path: Path) -> None:
    """Writes detailed Markdown report of remapped chunks."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    confident_count = sum(1 for it in items if it["status"] == "CONFIDENT_MATCH")
    unanswerable_count = sum(1 for it in items if it["status"] == "UNANSWERABLE")
    flagged_count = sum(1 for it in items if it["status"] == "NO_CONFIDENT_MATCH")

    lines: List[str] = [
        "# Golden Set Source Chunk Remap Report",
        "",
        "**Target Corpus**: `backend/data/processed/chunks.json` (Deterministic UUIDv5 IDs)  ",
        f"**Total Questions Evaluated**: {len(items)}  ",
        f"- **Confident Matches**: {confident_count}  ",
        f"- **Unanswerable (Empty source_chunk_ids)**: {unanswerable_count}  ",
        f"- **Flagged (No Confident Match - Needs Manual Fix)**: {flagged_count}  ",
        "",
        "---",
        "",
        "## 1. Summary Matrix",
        "",
        "| ID | Category | Company | Fiscal Year | Status | Match Score | Chunks Found | Section(s) |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |",
    ]

    for it in items:
        status_display = it["status"]
        if it["status"] == "NO_CONFIDENT_MATCH":
            status_display = "**FLAGGED (NO MATCH)**"
        secs = ", ".join(it["sections"]) if it["sections"] else "N/A"
        cnt = len(it["chunk_ids"])
        lines.append(
            f"| `{it['id']}` | `{it['category']}` | {it['company']} | {it['fiscal_year']} | {status_display} | {it['match_score']:.2f} | {cnt} | {secs} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Item-by-Item Remap Details",
        "",
    ])

    for it in items:
        lines.append(f"### Item `{it['id']}` ({it['company']} FY{it['fiscal_year']} - `{it['category']}`)")
        lines.append(f"- **Status**: `{it['status']}` (Score: {it['match_score']:.2f})")
        lines.append(f"- **Notes**: {it['notes']}")

        if it["chunk_ids"]:
            lines.append("- **Chosen Chunk(s)**:")
            for i, cid in enumerate(it["chunk_ids"]):
                sec = it["sections"][i] if i < len(it["sections"]) else "N/A"
                prev = it["previews"][i] if i < len(it["previews"]) else "N/A"
                quote = it["source_quotes"][i] if i < len(it["source_quotes"]) else "N/A"
                lines.append(f"  - **Chunk ID**: `{cid}`")
                lines.append(f"    - **Item Section**: `{sec}`")
                lines.append(f"    - **150-char Preview**: \"{prev}...\"")
                lines.append(f"    - **Verbatim Source Quote**: \"{quote}\"")
        else:
            lines.append("- **Source Chunk IDs**: `[]` (None)")
            lines.append("- **Source Quotes**: `[]`")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. Flagged Items Requiring Manual Review",
        "",
    ])

    flagged_items = [it for it in items if it["status"] == "NO_CONFIDENT_MATCH"]
    if flagged_items:
        for fit in flagged_items:
            lines.append(f"- **`{fit['id']}`** ({fit['company']} FY{fit['fiscal_year']}): {fit['notes']}")
    else:
        lines.append("No items flagged. All answerable items matched with high confidence.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote remap report to: {report_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("\n" + "=" * 70)
    print("REMAP SOURCE CHUNKS & GENERATE REPORT")
    print("=" * 70)

    # 1. Report existing golden source_chunk_ids in chunks.json before remapping
    try:
        valid_ids, corpus_path = get_corpus_chunk_ids()
        with open(DEFAULT_GOLDEN_PATH, "r", encoding="utf-8") as f:
            current_golden = json.load(f)
        total_existing_ids = 0
        existing_matched = 0
        for it in current_golden:
            s_ids = it.get("source_chunk_ids") or ([it["chunk_id"]] if it.get("chunk_id") else [])
            for cid in s_ids:
                total_existing_ids += 1
                if cid in valid_ids:
                    existing_matched += 1
        print(f"Pre-remap status:")
        print(f"  Corpus file used: {corpus_path.name}")
        print(f"  Total golden source chunk IDs: {total_existing_ids}")
        print(f"  Matched in corpus:             {existing_matched} (expect 0)")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"Pre-remap check failed: {e}")

    # 2. Run remapping
    results = remap_golden_set()
    confident = sum(1 for it in results if it["status"] == "CONFIDENT_MATCH")
    flagged = sum(1 for it in results if it["status"] == "NO_CONFIDENT_MATCH")
    unans = sum(1 for it in results if it["status"] == "UNANSWERABLE")

    print(f"Remap completed:")
    print(f"  Confident matches:    {confident}")
    print(f"  Flagged for manual:   {flagged}")
    print(f"  Unanswerable (empty): {unans}")
    print(f"  Report saved to:      {DEFAULT_REPORT_PATH}")
    print(f"  Backup saved to:      {BACKUP_GOLDEN_PATH}\n")

    # 3. Post-remap validation check
    missing = validate_golden_source_chunks()
    print(f"Post-remap validation check: {len(missing)} missing source_chunk_ids in corpus.")
    if missing:
        print("Missing IDs:", missing)
        sys.exit(1)
    else:
        print("ALL remapped source_chunk_ids exist in current corpus!")


if __name__ == "__main__":
    main()
