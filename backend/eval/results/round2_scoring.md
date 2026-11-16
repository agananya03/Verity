# Pre-Registered Round 2 Benchmark Evaluation Results

## Pre-Registration & Metadata Stamping
- **Date & Time:** `2026-11-16 00:21:01`
- **Git Commit Hash:** `6b8855a80d22bca6684ba3171c1fed5c3534eaf3`
- **Dataset:** `backend/eval/golden_round2.json`
- **CRLF-Normalized SHA-256:** `37cc61373c8b806b84476fbe5663a591411f3b34c02b846d57f8a653ccf94645`
- **Pre-Registration Protocol:** [round2_preregistration.md](file:///c:/Users/ASUS/OneDrive/Desktop/verity/backend/eval/round2_preregistration.md)
- **Scoring Protocol:** Scored strictly once without configuration or query-cleaning adjustments.

---

## 1. Overall Answerable Sub-Benchmark (n=68)
### All Answerable Items (n=68)
| Configuration | Hit@8 | Wilson 95% CI | Recall@8 | MRR@8 | Hit@20 |
|---|---|---|---|---|---|
| **Config A (legacy retrieve)** | 47/68 (69.1%) | `[0.574, 0.788]` | 0.6691 | 0.4014 | 60/68 (88.2%) |
| **Config B (clean_v3_rrf10)** | 49/68 (72.1%) | `[0.604, 0.813]` | 0.6985 | 0.4229 | 57/68 (83.8%) |
| **Config C (raw BM25-only)** | 46/68 (67.6%) | `[0.558, 0.776]` | 0.6397 | 0.4018 | 56/68 (82.4%) |

**Paired Exact McNemar Tests on Hit@8:**
- **Config B vs. Config A:** b=5, c=3, discordant=8, two-sided $p = 0.7266$ (Not Statistically Significant)
- **Config B vs. Config C:** b=10, c=7, discordant=17, two-sided $p = 0.6291$ (Not Statistically Significant)


---

## 2. Single-Year Sub-Benchmark (n=62)
### Single-Year Answerable Items (n=62)
| Configuration | Hit@8 | Wilson 95% CI | Recall@8 | MRR@8 | Hit@20 |
|---|---|---|---|---|---|
| **Config A (legacy retrieve)** | 44/62 (71.0%) | `[0.587, 0.808]` | 0.7097 | 0.4201 | 55/62 (88.7%) |
| **Config B (clean_v3_rrf10)** | 46/62 (74.2%) | `[0.621, 0.834]` | 0.7419 | 0.4437 | 52/62 (83.9%) |
| **Config C (raw BM25-only)** | 40/62 (64.5%) | `[0.521, 0.753]` | 0.6452 | 0.3843 | 50/62 (80.6%) |

**Paired Exact McNemar Tests on Hit@8:**
- **Config B vs. Config A:** b=5, c=3, discordant=8, two-sided $p = 0.7266$ (Not Statistically Significant)
- **Config B vs. Config C:** b=10, c=4, discordant=14, two-sided $p = 0.1796$ (Not Statistically Significant)


---

## 3. Multi-Year Sub-Benchmark (n=6)
### Multi-Year Answerable Items (n=6)
> **Note:** Multi-year sample size is n=6; metrics are purely descriptive and under-powered for statistical hypothesis testing.

| Configuration | Hit@8 | Wilson 95% CI | Recall@8 | MRR@8 | Hit@20 |
|---|---|---|---|---|---|
| **Config A (legacy retrieve)** | 3/6 (50.0%) | `[0.188, 0.812]` | 0.2500 | 0.2083 | 5/6 (83.3%) |
| **Config B (clean_v3_rrf10)** | 3/6 (50.0%) | `[0.188, 0.812]` | 0.2500 | 0.2083 | 5/6 (83.3%) |
| **Config C (raw BM25-only)** | 6/6 (100.0%) | `[0.610, 1.000]` | 0.5833 | 0.5819 | 6/6 (100.0%) |

**Paired Exact McNemar Tests on Hit@8:**
- **Config B vs. Config A:** b=0, c=0, discordant=0, two-sided $p = 1.0000$ (Not Statistically Significant)
- **Config B vs. Config C:** b=0, c=3, discordant=3, two-sided $p = 0.2500$ (Not Statistically Significant)


---

## 4. Single-Year Hit@8 Slices

### By Target Company
| Company | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
| Alphabet | 13 | 12/13 (92.3%) | 12/13 (92.3%) | 11/13 (84.6%) |
| Amazon | 14 | 9/14 (64.3%) | 9/14 (64.3%) | 8/14 (57.1%) |
| Apple | 12 | 9/12 (75.0%) | 10/12 (83.3%) | 8/12 (66.7%) |
| Microsoft | 11 | 9/11 (81.8%) | 10/11 (90.9%) | 8/11 (72.7%) |
| NVIDIA | 12 | 5/12 (41.7%) | 5/12 (41.7%) | 5/12 (41.7%) |

### By 10-K Filing Section
| Section | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
| Item 1 | 6 | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) |
| Item 1A | 8 | 6/8 (75.0%) | 7/8 (87.5%) | 5/8 (62.5%) |
| Item 7 | 24 | 16/24 (66.7%) | 16/24 (66.7%) | 17/24 (70.8%) |
| Item 8 | 24 | 17/24 (70.8%) | 18/24 (75.0%) | 13/24 (54.2%) |

### By Question Formulation Style
| Question Style | n | Config A (legacy) | Config B (clean_v3_rrf10) | Config C (raw BM25) |
|---|---|---|---|---|
| "what does the filing say about" style | 9 | 5/9 (55.6%) | 5/9 (55.6%) | 5/9 (55.6%) |
| comparison-of-two-figures style where the chunk allows | 4 | 4/4 (100.0%) | 4/4 (100.0%) | 3/4 (75.0%) |
| keyword-style phrase | 15 | 9/15 (60.0%) | 12/15 (80.0%) | 10/15 (66.7%) |
| natural prose | 13 | 12/13 (92.3%) | 12/13 (92.3%) | 10/13 (76.9%) |
| paraphrase (no copied terms from the chunk) | 11 | 8/11 (72.7%) | 7/11 (63.6%) | 7/11 (63.6%) |
| short direct question | 10 | 6/10 (60.0%) | 6/10 (60.0%) | 5/10 (50.0%) |

---

## 5. Descriptive Extras (Not Acted Upon)

### Extractor Accuracy on Round 2
- **Company Extraction Accuracy:** 68/68 (100.0%)
- **Fiscal Year Extraction Accuracy:** 68/68 (100.0%)
- **Both Accurate:** 68/68 (100.0%)
- **Total Mismatches:** 0

### Config B Hit@8 Misses & Top-50 Pool Ranks
Total Config B Misses on Hit@8: **19 / 68**

| Item ID | Sub-Benchmark | Company | Year | Rank in Top-50 Pool | Question |
|---|---|---|---|---|---|
| `round2_022` | Single-year | Alphabet | 2025 | **Rank #10** | What was Alphabet's year-over-year percentage growth in diluted earnings per share for the fiscal period ending December 31, 2025? |
| `round2_033` | Single-year | Amazon | 2023 | **Rank #29** | Amazon total current assets value for the fiscal year ended December 31, 2023 in millions |
| `round2_035` | Single-year | Amazon | 2023 | **Rank #9** | What does Amazon's fiscal year 2023 filing say about the financial effect and timing of the server depreciation timeframe extension? |
| `round2_040` | Single-year | Amazon | 2024 | **Rank #37** | What was Amazon's net amount paid for global corporate income taxes during the fiscal year ending in 2024? |
| `round2_041` | Single-year | Amazon | 2024 | **Rank #13** | What does Amazon's filing say about the foreign exchange impact on technology and infrastructure expenditures for fiscal year 2024? |
| `round2_044` | Single-year | Amazon | 2024 | **Rank #9** | What was the operating income for Amazon's cloud computing division in fiscal year 2024 in millions of dollars? |
| `round2_059` | Single-year | Apple | 2023 | **Rank #39** | What does Apple's fiscal year 2023 filing say about the magnitude of interest expense incurred on long-term obligations across the three most recent fiscal years? |
| `round2_065` | Single-year | Apple | 2024 | **>50 (Not in pool)** | What does Apple's filing say about the drivers behind the shifts in digital marketplace and software hosting revenue during fiscal year 2024? |
| `round2_092` | Single-year | Microsoft | 2025 | **>50 (Not in pool)** | What was Microsoft's total operating cash generation for fiscal year 2025? |
| `round2_105` | Single-year | NVIDIA | 2024 | **Rank #10** | NVIDIA fiscal year 2024 geopolitical tensions and regulatory modifications impact on regional market access |
| `round2_109` | Single-year | NVIDIA | 2024 | **Rank #11** | In its fiscal year 2024 report, what was the magnitude of the fee NVIDIA incurred when a prominent merger was called off during the preceding fiscal year? |
| `round2_110` | Single-year | NVIDIA | 2024 | **>50 (Not in pool)** | How many personnel does NVIDIA employ in the Israeli territory as of its fiscal year 2024 report? |
| `round2_117` | Single-year | NVIDIA | 2025 | **Rank #22** | NVIDIA fiscal year 2025 research and development expenditure total in millions |
| `round2_124` | Single-year | NVIDIA | 2026 | **>50 (Not in pool)** | What was the financial write-down amount NVIDIA recorded in the opening quarter of fiscal year 2026 related to surplus supplies and supply contracts for specific semiconductor products restricted by export controls? |
| `round2_128` | Single-year | NVIDIA | 2026 | **>50 (Not in pool)** | What was NVIDIA's research and development expense for the fiscal year ended January 25, 2026, in millions? |
| `round2_130` | Single-year | NVIDIA | 2026 | **Rank #23** | What was the total capital deployed by NVIDIA for acquisitions of non-marketable equity interests or specific target entities during the fiscal year ending in January 2026? |
| `round2_mh_03` | Multi-year | Alphabet | 2024/2025 | **Rank #24** | What was Alphabet's total traffic acquisition cost in fiscal year 2024, and what were its capital expenditures for fiscal year 2025? |
| `round2_mh_06` | Multi-year | Amazon | 2023/2024 | **Rank #10** | How much did Amazon's operating cash flow grow from 2023 to 2024, and what were the figures for each of those two years? |
| `round2_mh_22` | Multi-year | NVIDIA | 2024/2025 | **Rank #10** | What was the effect on NVIDIA's operating income of the fiscal 2024 change in property, plant and equipment useful lives, and what is the total estimated future amortization expense for intangible assets as of its fiscal 2025 report? |
