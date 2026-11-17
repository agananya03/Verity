# Pre-Registration: Round-2 Benchmark Evaluation

## Commit Timestamp
`2026-11-15T18:06:54+05:30`

## Prior Sets (Used for Development)
- **Dev Set (`golden_set.json`):** 14 single-year answerable items (+ 4 multi-year items = 18 total) already used for tuning and initial ablation development.
- **Round-1 Held-Out Set (`golden_heldout.json`):** 62 single-year answerable items (+ 2 multi-year items = 64 total) already used for held-out diagnosis and the pre-registered fusion sweep.

---

## 1. Configurations to be Scored Once on Round 2
The evaluation on the Round-2 benchmark will compare exactly three pre-registered retrieval configurations:
- **Config A:** `RETRIEVAL_STRATEGY=legacy` (old production retrieve() with clean_v2, k=60, top-20 pools)
- **Config B:** `RETRIEVAL_STRATEGY=clean_v3_rrf10` (new default production retrieve() with clean_v3, k=10, top-50 dense + top-50 BM25 candidate pools)
- **Config C:** raw query, BM25-only with the extractor's filters, top-50 pool (`raw__bm25_only`)

---

## 2. Evaluation Metrics
For each configuration, the following metrics will be calculated and reported across all answerable items, and broken down separately for single-year and multi-year questions:
- **Hit@8** (primary metric)
- **Recall@8**
- **MRR** (Mean Reciprocal Rank at 8)
- **Hit@20**
- **Wilson 95% Confidence Intervals** on Hit@8
- **Exact McNemar Tests (two-sided binomial):**
  - Config B vs. Config A
  - Config B vs. Config C

---

## 3. Pre-Registration Rule
**Round 2 is scored ONCE.**
- No config changes, no query-cleaning changes, and no question edits will be permitted after seeing round-2 results.
- Any change or adjustment made after inspecting round-2 evaluation results must be explicitly designated and reported as post-hoc exploratory analysis.
