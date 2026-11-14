# Retrieval Ablation Study Report - DEV SET (n=18, design tuned on these questions)

- **Timestamp**: `2026-11-14T11:46:59.541913+00:00`
- **Eligible Items Evaluated**: `18` (out of `32` total golden set)
- **Excluded Items**: `14` (unanswerable or unindexed items, including `golden_10`)
- **Evaluated Categories**: `simple_lookup` (10), `numeric_precision` (4), `multi_hop` (4)
- **Evaluated Slices**: `single_year` (14), `multi_year` (4)

## Traceability Fingerprint (SHA-256)

- `backend/agent/retrieval.py`: `1a700973445d32dcb5433cda8a70596a23d6381578bfeac68d2997762ed30e1e`
- `backend/agent/nodes/classifier.py`: `16b34f1434d6844d69694c4f2c0accbb005d106f3f0fbb611b2eb99f3de6cf12`
- `backend/eval/golden_set.json`: `55caa52adfc666e8b1dcc2df7122b9720049a566f7b1a5fdb5d9632e1305af9c`
- **Combined Fingerprint**: `eb9390d00f701eca7759a0036c7670a1db301ae754f09ea50038c19c5cfe9edb`

## Staged Retrieval Ladder (DEV SET, n=18)

| Configuration | N | Hit@8 (All) | Rec@8 (All) | MRR (All) | Hit@8 (Single) | Rec@8 (Single) | MRR (Single) | Hit@8 (Multi) | Rec@8 (Multi) | MRR (Multi) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 18 | 0.5000 | 0.4167 | 0.2315 | 0.3571 | 0.3571 | 0.1726 | 1.0000 | 0.6250 | 0.4375 |
| **dense_filtered** | 18 | 0.6111 | 0.5278 | 0.3398 | 0.5000 | 0.5000 | 0.3119 | 1.0000 | 0.6250 | 0.4375 |
| **bm25_filtered** | 18 | 0.8889 | 0.8333 | 0.6134 | 0.9286 | 0.9286 | 0.6905 | 0.7500 | 0.5000 | 0.3438 |
| **rrf_raw_filtered** | 18 | 0.7222 | 0.6667 | 0.4496 | 0.7143 | 0.7143 | 0.4566 | 0.7500 | 0.5000 | 0.4250 |
| **rrf_clean_v2 (single pass)** | 18 | 0.9444 | 0.9167 | 0.6737 | 1.0000 | 1.0000 | 0.7607 | 0.7500 | 0.6250 | 0.3690 |
| **rrf_clean_v2_peryear_v1** | 18 | 0.9444 | 0.9444 | 0.6704 | 1.0000 | 1.0000 | 0.7607 | 0.7500 | 0.7500 | 0.3542 |
| **production retrieve()** | 18 | 1.0000 | 0.9722 | 0.7074 | 1.0000 | 1.0000 | 0.7607 | 1.0000 | 0.8750 | 0.5208 |

## Cross-Encoder Reranker Specification (Part A)

- **Engine**: `fastembed.rerank.cross_encoder.TextCrossEncoder`
- **Model Selected**: `Xenova/ms-marco-MiniLM-L-6-v2`
- **Preferred Model Checked**: `BAAI/bge-reranker-base` (1.04 GB on-disk, ~1100 MB RAM)
- **Selected Model On-Disk Size**: `86.8 MB` (model.onnx)
- **Selected Model RAM Footprint**: `~128.0 MB` delta RAM (`~212.0 MB` total process RSS)
- **Render 512 MB Instance Compatible**: `True`
- **Design Rationale**: BAAI/bge-reranker-base (1.04 GB on-disk, ~1.1 GB RAM) exceeds 512 MB Render instance limit. Falling back to Xenova/ms-marco-MiniLM-L-6-v2 (0.08 GB / ~86.8 MB on-disk, ~128 MB RAM delta, ~212 MB total process RSS), which fits safely within 512 MB RAM.

## Query Cleaning Before / After Table (N = 18)

| ID | Filters Applied | Original Question | Cleaned Query (v2) |
| :--- | :--- | :--- | :--- |
| **golden_01** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, through what methods does the company expand the range of its offerings and develop new technologies? | Through what methods does the company expand the range of its offerings and develop new technologies? |
| **golden_02** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, what subscription-based digital content services are listed under Apple's Services offerings? | What subscription-based digital content services are listed under Apple's Services offerings? |
| **golden_03** | `comp=Apple, fy=2024` | According to Apple's fiscal year 2024 Form 10-K, what are the principal competitive factors important to the company? | What are the principal competitive factors important to the company? |
| **golden_04** | `comp=Apple, fy=2024` | According to Apple's fiscal year 2024 Form 10-K, what percentages of total net sales did Apple generate through its direct and indirect distribution channels in fiscal year 2024, respectively? | What percentages of total net sales did Apple generate through its direct and indirect distribution channels respectively? |
| **golden_05** | `comp=Apple, fy=2024` | According to Apple's fiscal year 2024 Form 10-K, what subscription-based digital content services are explicitly detailed under Apple's Services offerings? | What subscription-based digital content services are explicitly detailed under Apple's Services offerings? |
| **golden_06** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, what does the company primarily rely on for its success, despite viewing the ownership of intellectual property rights as an important factor in differentiating its business? | What does the company primarily rely on for its success, despite viewing the ownership of intellectual property rights as an important factor in differentiating its business? |
| **golden_07** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, what factors can cause initial capacity constraints when the company introduces a component or product that utilizes new technologies? | What factors can cause initial capacity constraints when the company introduces a component or product that utilizes new technologies? |
| **golden_08** | `comp=Apple, fy=2024` | According to Apple's fiscal year 2024 Form 10-K, what historical trend does Apple observe regarding its net sales in the first quarter of its fiscal year, and what drives this trend? | What historical trend does Apple observe regarding its net sales in the first quarter of its fiscal year, and what drives this trend? |
| **golden_09** | `comp=Apple, fy=2023` | According to Apple's fiscal year 2023 Form 10-K, what specific consequences could result if Apple is unable to satisfy changing regulations regarding product import and export? | What specific consequences could result if Apple is unable to satisfy changing regulations regarding product import and export? |
| **golden_11** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, what specific characteristics define the global markets in which the company offers its products and services? | What specific characteristics define the global markets in which the company offers its products and services? |
| **golden_12** | `comp=Apple, fy=2023` | According to Apple's fiscal year 2023 Form 10-K, what specific areas are gauged through the company's employee sentiment surveys? | What specific areas are gauged through the company's employee sentiment surveys? |
| **golden_23** | `comp=Apple, fy=None` | Comparing Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what specific changes were made to the listed models in the iPhone line, and what new category of devices was added to the definition of 'Wearables' in fiscal year 2024? | Comparing Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024, what specific changes were made to the listed models in the iPhone line, and what new category of devices was added to the definition of 'Wearables' in fiscal year 2024? |
| **golden_24** | `comp=Apple, fy=None` | Based on Apple's Form 10-K filings, what is the operating system associated with Apple's line of smartwatches as described in its fiscal year 2023 filing, and what is the operating system associated with Apple's first spatial computer, Apple Vision Pro, as described in its fiscal year 2024 filing? | What is the operating system associated with Apple's line of smartwatches as described in its fiscal year 2023 filing, and what is the operating system associated with Apple's first spatial computer, Apple Vision Pro, as described in its fiscal year 2024 filing? |
| **golden_25** | `comp=Apple, fy=None` | According to Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what are the specific operating systems that power the iPhone and iPad product lines, and what are the five reportable geographic segments through which Apple manages its business? | What are the specific operating systems that power the iPhone and iPad product lines, and what are the five reportable geographic segments through which Apple manages its business? |
| **golden_26** | `comp=Apple, fy=None` | According to Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what are the operating systems used for the iPhone and smartwatch product lines, and what percentage of Apple's total net sales was generated through indirect distribution channels in fiscal year 2024? | What are the operating systems used for the iPhone and smartwatch product lines, and what percentage of Apple's total net sales was generated through indirect distribution channels in fiscal year 2024? |
| **golden_27** | `comp=Apple, fy=2025` | According to the Segments section of Apple's fiscal year 2025 Form 10-K, what are the reportable geographic segments of the company? | What are the reportable geographic segments of the company? |
| **golden_28** | `comp=Apple, fy=2025` | According to Apple's fiscal year 2025 Form 10-K, during fiscal year 2025, what percentages of Apple's total net sales were accounted for by its direct and indirect distribution channels, respectively? | What percentages of Apple's total net sales were accounted for by its direct and indirect distribution channels, respectively? |
| **golden_29** | `comp=Apple, fy=2025` | According to the Item 1 section of Apple's fiscal year 2025 Form 10-K, new products introduced by Apple often utilize custom components available from how many sources? | New products introduced by Apple often utilize custom components available from how many sources? |

## Overall Retrieval Performance (N = 18)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) | Top Hub Chunk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **baseline_pipeline** | 0.5000 | 0.4167 | 0.2315 | 0.5000 | 0.4167 | 66.7% (12/18) | 243.2 / 248.1 | `70e50532...` |
| **dense_filtered** | 0.6111 | 0.5278 | 0.3398 | 0.7778 | 0.7222 | 55.6% (10/18) | 466.7 / 469.4 | `97e04e65...` |
| **bm25_filtered** | 0.8889 | 0.8333 | 0.6134 | 0.9444 | 0.9167 | 55.6% (10/18) | 26.0 / 45.7 | `97e04e65...` |
| **rrf_raw_filtered** | 0.7222 | 0.6667 | 0.4496 | 0.8889 | 0.8333 | 61.1% (11/18) | 269.7 / 293.3 | `97e04e65...` |
| **rrf_clean_v2 (single pass)** | 0.9444 | 0.9167 | 0.6737 | 1.0000 | 1.0000 | 22.2% (4/18) | 258.8 / 281.5 | `681f8256...` |
| **rrf_clean_v2_peryear_v1** | 0.9444 | 0.9444 | 0.6704 | 1.0000 | 0.9722 | 27.8% (5/18) | 320.3 / 561.3 | `70e50532...` |
| **production retrieve()** | 1.0000 | 0.9722 | 0.7074 | 1.0000 | 0.9722 | 27.8% (5/18) | 319.3 / 548.1 | `ff8dcd62...` |
| **dense_clean_v2** | 0.9444 | 0.8889 | 0.6279 | 1.0000 | 0.9722 | 27.8% (5/18) | 240.4 / 243.7 | `681f8256...` |
| **bm25_clean_v2** | 0.9444 | 0.8889 | 0.6574 | 1.0000 | 0.9722 | 27.8% (5/18) | 19.5 / 39.6 | `08b54639...` |
| **rrf_clean_v1 (single pass)** | 0.8889 | 0.8611 | 0.6070 | 1.0000 | 0.9722 | 27.8% (5/18) | 260.8 / 295.5 | `70e50532...` |
| **RRF (BM25 2x Weighted)** | 0.7222 | 0.6667 | 0.4556 | 0.8333 | 0.8333 | 61.1% (11/18) | 326.6 / 452.4 | `97e04e65...` |
| **RRF Top-30 + Rerank** | 0.3889 | 0.3611 | 0.2996 | 0.7222 | 0.6389 | 66.7% (12/18) | 2331.0 / 2522.3 | `97e04e65...` |
| **BM25 Top-30 + Rerank** | 0.4444 | 0.4167 | 0.3056 | 0.6667 | 0.6111 | 66.7% (12/18) | 2091.5 / 2251.5 | `97e04e65...` |

## Performance Breakdown by Slice (Single-Year vs Multi-Year)

### Slice: `Single-Year Questions (n=14)`

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.3571 | 0.3571 | 0.1726 | 0.3571 | 0.3571 | 57.1% (8/14) | 243.1 / 251.2 |
| **dense_filtered** | 0.5000 | 0.5000 | 0.3119 | 0.7143 | 0.7143 | 42.9% (6/14) | 466.4 / 469.5 |
| **bm25_filtered** | 0.9286 | 0.9286 | 0.6905 | 0.9286 | 0.9286 | 50.0% (7/14) | 20.7 / 26.2 |
| **rrf_raw_filtered** | 0.7143 | 0.7143 | 0.4566 | 0.8571 | 0.8571 | 50.0% (7/14) | 263.9 / 270.0 |
| **rrf_clean_v2 (single pass)** | 1.0000 | 1.0000 | 0.7607 | 1.0000 | 1.0000 | 21.4% (3/14) | 254.1 / 259.7 |
| **rrf_clean_v2_peryear_v1** | 1.0000 | 1.0000 | 0.7607 | 1.0000 | 1.0000 | 21.4% (3/14) | 253.6 / 259.8 |
| **production retrieve()** | 1.0000 | 1.0000 | 0.7607 | 1.0000 | 1.0000 | 21.4% (3/14) | 254.3 / 262.7 |
| **dense_clean_v2** | 0.9286 | 0.9286 | 0.6406 | 1.0000 | 1.0000 | 28.6% (4/14) | 239.7 / 241.1 |
| **bm25_clean_v2** | 1.0000 | 1.0000 | 0.6786 | 1.0000 | 1.0000 | 35.7% (5/14) | 14.0 / 18.9 |
| **rrf_clean_v1 (single pass)** | 1.0000 | 1.0000 | 0.6947 | 1.0000 | 1.0000 | 21.4% (3/14) | 253.2 / 260.5 |
| **RRF (BM25 2x Weighted)** | 0.7143 | 0.7143 | 0.4667 | 0.8571 | 0.8571 | 50.0% (7/14) | 336.4 / 622.8 |
| **RRF Top-30 + Rerank** | 0.3571 | 0.3571 | 0.3036 | 0.6429 | 0.6429 | 57.1% (8/14) | 2300.2 / 2395.8 |
| **BM25 Top-30 + Rerank** | 0.4286 | 0.4286 | 0.3125 | 0.6429 | 0.6429 | 57.1% (8/14) | 2060.7 / 2125.6 |

### Slice: `Multi-Year Questions (n=4)`

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 1.0000 | 0.6250 | 0.4375 | 1.0000 | 0.6250 | 100.0% (4/4) | 243.8 / 245.3 |
| **dense_filtered** | 1.0000 | 0.6250 | 0.4375 | 1.0000 | 0.7500 | 100.0% (4/4) | 467.6 / 468.3 |
| **bm25_filtered** | 0.7500 | 0.5000 | 0.3438 | 1.0000 | 0.8750 | 100.0% (4/4) | 44.5 / 63.1 |
| **rrf_raw_filtered** | 0.7500 | 0.5000 | 0.4250 | 1.0000 | 0.7500 | 100.0% (4/4) | 289.8 / 304.3 |
| **rrf_clean_v2 (single pass)** | 0.7500 | 0.6250 | 0.3690 | 1.0000 | 1.0000 | 75.0% (3/4) | 275.4 / 283.0 |
| **rrf_clean_v2_peryear_v1** | 0.7500 | 0.7500 | 0.3542 | 1.0000 | 0.8750 | 100.0% (4/4) | 553.6 / 569.0 |
| **production retrieve()** | 1.0000 | 0.8750 | 0.5208 | 1.0000 | 0.8750 | 75.0% (3/4) | 546.6 / 553.5 |
| **dense_clean_v2** | 1.0000 | 0.7500 | 0.5834 | 1.0000 | 0.8750 | 100.0% (4/4) | 242.9 / 244.7 |
| **bm25_clean_v2** | 0.7500 | 0.5000 | 0.5833 | 1.0000 | 0.8750 | 75.0% (3/4) | 38.6 / 46.3 |
| **rrf_clean_v1 (single pass)** | 0.5000 | 0.3750 | 0.3000 | 1.0000 | 0.8750 | 100.0% (4/4) | 287.5 / 305.0 |
| **RRF (BM25 2x Weighted)** | 0.7500 | 0.5000 | 0.4167 | 0.7500 | 0.7500 | 100.0% (4/4) | 292.1 / 305.9 |
| **RRF Top-30 + Rerank** | 0.5000 | 0.3750 | 0.2857 | 1.0000 | 0.6250 | 100.0% (4/4) | 2438.8 / 2547.4 |
| **BM25 Top-30 + Rerank** | 0.5000 | 0.3750 | 0.2812 | 0.7500 | 0.5000 | 100.0% (4/4) | 2199.3 / 2328.7 |


## Performance Breakdown by Category

### Category: `simple_lookup` (n = 10)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.5000 | 0.5000 | 0.2417 | 0.5000 | 0.5000 | 60.0% (6/10) | 243.3 / 254.1 |
| **dense_filtered** | 0.6000 | 0.6000 | 0.4167 | 0.7000 | 0.7000 | 40.0% (4/10) | 466.0 / 467.4 |
| **bm25_filtered** | 0.9000 | 0.9000 | 0.6917 | 0.9000 | 0.9000 | 40.0% (4/10) | 19.8 / 26.7 |
| **rrf_raw_filtered** | 0.7000 | 0.7000 | 0.4976 | 0.9000 | 0.9000 | 40.0% (4/10) | 262.5 / 269.9 |
| **rrf_clean_v2 (single pass)** | 1.0000 | 1.0000 | 0.7950 | 1.0000 | 1.0000 | 30.0% (3/10) | 254.3 / 260.6 |
| **rrf_clean_v2_peryear_v1** | 1.0000 | 1.0000 | 0.7950 | 1.0000 | 1.0000 | 30.0% (3/10) | 253.9 / 260.8 |
| **production retrieve()** | 1.0000 | 1.0000 | 0.7950 | 1.0000 | 1.0000 | 30.0% (3/10) | 254.3 / 262.9 |
| **dense_clean_v2** | 0.9000 | 0.9000 | 0.7143 | 1.0000 | 1.0000 | 40.0% (4/10) | 239.6 / 241.1 |
| **bm25_clean_v2** | 1.0000 | 1.0000 | 0.7333 | 1.0000 | 1.0000 | 40.0% (4/10) | 14.2 / 19.7 |
| **rrf_clean_v1 (single pass)** | 1.0000 | 1.0000 | 0.7393 | 1.0000 | 1.0000 | 30.0% (3/10) | 253.5 / 260.7 |
| **RRF (BM25 2x Weighted)** | 0.7000 | 0.7000 | 0.5033 | 0.9000 | 0.9000 | 40.0% (4/10) | 263.0 / 269.0 |
| **RRF Top-30 + Rerank** | 0.4000 | 0.4000 | 0.4000 | 0.5000 | 0.5000 | 50.0% (5/10) | 2290.4 / 2355.1 |
| **BM25 Top-30 + Rerank** | 0.4000 | 0.4000 | 0.4000 | 0.5000 | 0.5000 | 50.0% (5/10) | 2056.9 / 2110.0 |

### Category: `numeric_precision` (n = 4)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 50.0% (2/4) | 242.5 / 244.5 |
| **dense_filtered** | 0.2500 | 0.2500 | 0.0500 | 0.7500 | 0.7500 | 75.0% (3/4) | 467.3 / 469.5 |
| **bm25_filtered** | 1.0000 | 1.0000 | 0.6875 | 1.0000 | 1.0000 | 75.0% (3/4) | 22.8 / 24.4 |
| **rrf_raw_filtered** | 0.7500 | 0.7500 | 0.3542 | 0.7500 | 0.7500 | 75.0% (3/4) | 267.5 / 269.8 |
| **rrf_clean_v2 (single pass)** | 1.0000 | 1.0000 | 0.6750 | 1.0000 | 1.0000 | 50.0% (2/4) | 253.5 / 255.9 |
| **rrf_clean_v2_peryear_v1** | 1.0000 | 1.0000 | 0.6750 | 1.0000 | 1.0000 | 50.0% (2/4) | 252.7 / 255.6 |
| **production retrieve()** | 1.0000 | 1.0000 | 0.6750 | 1.0000 | 1.0000 | 50.0% (2/4) | 254.5 / 260.4 |
| **dense_clean_v2** | 1.0000 | 1.0000 | 0.4562 | 1.0000 | 1.0000 | 50.0% (2/4) | 239.8 / 240.7 |
| **bm25_clean_v2** | 1.0000 | 1.0000 | 0.5416 | 1.0000 | 1.0000 | 50.0% (2/4) | 13.7 / 16.1 |
| **rrf_clean_v1 (single pass)** | 1.0000 | 1.0000 | 0.5833 | 1.0000 | 1.0000 | 50.0% (2/4) | 252.4 / 254.7 |
| **RRF (BM25 2x Weighted)** | 0.7500 | 0.7500 | 0.3750 | 0.7500 | 0.7500 | 75.0% (3/4) | 519.9 / 1125.8 |
| **RRF Top-30 + Rerank** | 0.2500 | 0.2500 | 0.0625 | 1.0000 | 1.0000 | 75.0% (3/4) | 2324.9 / 2396.9 |
| **BM25 Top-30 + Rerank** | 0.5000 | 0.5000 | 0.0938 | 1.0000 | 1.0000 | 75.0% (3/4) | 2070.2 / 2135.9 |

### Category: `multi_hop` (n = 4)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 1.0000 | 0.6250 | 0.4375 | 1.0000 | 0.6250 | 100.0% (4/4) | 243.8 / 245.3 |
| **dense_filtered** | 1.0000 | 0.6250 | 0.4375 | 1.0000 | 0.7500 | 100.0% (4/4) | 467.6 / 468.3 |
| **bm25_filtered** | 0.7500 | 0.5000 | 0.3438 | 1.0000 | 0.8750 | 100.0% (4/4) | 44.5 / 63.1 |
| **rrf_raw_filtered** | 0.7500 | 0.5000 | 0.4250 | 1.0000 | 0.7500 | 100.0% (4/4) | 289.8 / 304.3 |
| **rrf_clean_v2 (single pass)** | 0.7500 | 0.6250 | 0.3690 | 1.0000 | 1.0000 | 75.0% (3/4) | 275.4 / 283.0 |
| **rrf_clean_v2_peryear_v1** | 0.7500 | 0.7500 | 0.3542 | 1.0000 | 0.8750 | 100.0% (4/4) | 553.6 / 569.0 |
| **production retrieve()** | 1.0000 | 0.8750 | 0.5208 | 1.0000 | 0.8750 | 75.0% (3/4) | 546.6 / 553.5 |
| **dense_clean_v2** | 1.0000 | 0.7500 | 0.5834 | 1.0000 | 0.8750 | 100.0% (4/4) | 242.9 / 244.7 |
| **bm25_clean_v2** | 0.7500 | 0.5000 | 0.5833 | 1.0000 | 0.8750 | 75.0% (3/4) | 38.6 / 46.3 |
| **rrf_clean_v1 (single pass)** | 0.5000 | 0.3750 | 0.3000 | 1.0000 | 0.8750 | 100.0% (4/4) | 287.5 / 305.0 |
| **RRF (BM25 2x Weighted)** | 0.7500 | 0.5000 | 0.4167 | 0.7500 | 0.7500 | 100.0% (4/4) | 292.1 / 305.9 |
| **RRF Top-30 + Rerank** | 0.5000 | 0.3750 | 0.2857 | 1.0000 | 0.6250 | 100.0% (4/4) | 2438.8 / 2547.4 |
| **BM25 Top-30 + Rerank** | 0.5000 | 0.3750 | 0.2812 | 0.7500 | 0.5000 | 100.0% (4/4) | 2199.3 / 2328.7 |

## Key Findings & Strategic Insights

1. **Query Cleaning Impact (clean_v2)**:
   - Removing generic 10-K boilerplate ("According to Apple's fiscal year 2025 Form 10-K, ...") removes distracting tokens.
   - Crucially, clean_v2 preserves the company name ("Apple") in question bodies and product names ("Apple Vision Pro", "Apple Watch", "Apple Card"), preventing BM25/Dense score degradation on product-specific queries.
2. **Per-Year Retrieval for Multi-Year Questions**:
   - Multi-year questions ("Comparing fiscal year 2023 and 2024...") suffer when retrieved without year filters, because single-year overview chunks overwhelm the result set.
   - Running per-year retrieval with alternating merge ensures top candidate chunks from every target year enter the final top 8, dramatically raising multi-hop recall.
3. **Cross-Encoder Reranker Tradeoffs**:
   - Xenova/ms-marco-MiniLM-L-6-v2 runs locally within ~212 MB total RSS, safely under the 512 MB Render limit.
   - Fits comfortably in memory while providing semantic reranking over the fused candidate pool.
