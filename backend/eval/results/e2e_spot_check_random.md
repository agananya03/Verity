# Secondary Post-Hoc Evaluation: Random & Low-Faithfulness Spot Check

> [!IMPORTANT]
> **POST-HOC, not pre-registered**: This audit set was selected post-hoc to evaluate qualitative response fidelity and judge reliability without selection bias toward extreme delta outliers.

- **Date**: 2026-11-17
- **Judge Model**: `gemma-4-31b-it`
- **Contents**: 6 random items sampled with seed 42, plus the 3 lowest-faithfulness answers from the improved pipeline.

---

## Category A: 6 Random Items (Seed 42) [POST-HOC, not pre-registered]

### #1 [Item `round2_073`] (simple_lookup) [POST-HOC, not pre-registered]

**Question**: How could legal setbacks impacting search distribution partners influence Apple's financial performance in fiscal year 2025?

**Ground Truth**: `Adverse rulings or remedies against search distribution partners like Google could prevent them from offering favorable commercial terms, which would materially diminish the revenue the company generates from these licensing agreements.`

**Supporting Quote**: A reversal of the order on appeal could result in imposition of certain remedies initially proposed by the DOJ, such as those prohibiting Google from offering the Company commercial terms for search distribution. If implemented, these remedies could materially adversely affect the Company’s ability to earn revenue from such licensing arrangements.

- **Baseline Answer**: Based on the provided SEC filings, a reversal on appeal of the court's order regarding Google (or the implementation of remedies initially proposed by the DOJ, such as prohibiting Google from offering Apple commercial terms for search distribution) could materially adversely affect Apple's ability to earn revenue from such licensing arrangements, which in turn could materially adversely affect the Company's business, results of operations, financial condition, and stock price.
- **Improved Answer**: Based on Apple's 2025 Form 10-K, Apple earns revenue from licensing arrangements with companies such as Google LLC to offer search services on Apple's platforms and applications [1]. Certain of these search distribution arrangements are currently subject to government investigations and legal proceedings [1]. 

Specifically, on August 5, 2024, Google was found to have violated U.S. antitrust laws, and on September 2, 2025, the U.S. District Court for the District of Columbia ordered certain remedies [1]. The court's order is subject to appeal by both the U.S. Department of Justice (DOJ) and Google [1]. A reversal of the order on appeal could result in the imposition of remedies initially proposed by the DOJ, such as prohibiting Google from offering Apple commercial terms for search distribution [1]. If implemented, these remedies could materially adversely affect Apple's ability to earn revenue from such licensing arrangements, which could in turn materially adversely affect the company's business, results of operations, financial condition, and stock price [1].

**Judge Scores**:
- Baseline: Faithfulness = **1.0**, Answer Correctness = 0.8183
- Improved: Faithfulness = **1.0**, Answer Correctness = 0.4961
- **Difference (Δ Faithfulness)**: `+0.0000` | **(Δ Answer Correctness)**: `-0.3223`

---

### #2 [Item `round2_082`] (numeric_precision) [POST-HOC, not pre-registered]

**Question**: What was Microsoft's year-over-year percentage growth rate for diluted earnings per share in fiscal year 2024?

**Ground Truth**: `The diluted earnings per share increased by 22% year-over-year in fiscal year 2024.`

**Supporting Quote**: Diluted earnings per share

11.80

9.68

22%

- **Baseline Answer**: Based on the provided SEC filings (specifically Chunk 1), Microsoft's year-over-year percentage growth rate for diluted earnings per share in fiscal year 2024 was **22%**.
- **Improved Answer**: Microsoft's year-over-year percentage growth rate for diluted earnings per share in fiscal year 2024 was 22% [1].

**Judge Scores**:
- Baseline: Faithfulness = **0.5**, Answer Correctness = 0.7104
- Improved: Faithfulness = **1.0**, Answer Correctness = 0.9679
- **Difference (Δ Faithfulness)**: `+0.5000` | **(Δ Answer Correctness)**: `+0.2575`

---

### #3 [Item `round2_117`] (numeric_precision) [POST-HOC, not pre-registered]

**Question**: NVIDIA fiscal year 2025 research and development expenditure total in millions

**Ground Truth**: `$12,914 million`

**Supporting Quote**: Research and development expenses$12,914 $8,675 $4,239 49 %

- **Baseline Answer**: I don't have enough information to answer this question based on the retrieved SEC filings.
- **Improved Answer**: NVIDIA's research and development expenditure total for fiscal year 2025 was $12,914 million [1].

**Judge Scores**:
- Baseline: Faithfulness = **0.0**, Answer Correctness = 0.2022
- Improved: Faithfulness = **1.0**, Answer Correctness = 0.9363
- **Difference (Δ Faithfulness)**: `+1.0000` | **(Δ Answer Correctness)**: `+0.7341`

---

### #4 [Item `round2_110`] (simple_lookup) [POST-HOC, not pre-registered]

**Question**: How many personnel does NVIDIA employ in the Israeli territory as of its fiscal year 2024 report?

**Ground Truth**: `NVIDIA employs approximately 3,700 individuals in the region who are primarily engaged in research and development, operations, sales, and marketing for networking products.`

**Supporting Quote**: We are monitoring the impact of the geopolitical conflict in and around Israel on our operations, including the health and safety of our approximately 3,700 employees in the region who primarily support the research and development, operations, and sales and marketing of our networking products.

- **Baseline Answer**: Based on the provided SEC filings, NVIDIA employed approximately 3,700 employees in the Israeli region as of its fiscal year 2024 report.
- **Improved Answer**: I don't have enough information to answer this question based on the retrieved SEC filings. Verity currently indexes SEC Form 10-K filings for Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), and NVIDIA (NVDA).

**Judge Scores**:
- Baseline: Faithfulness = **1.0**, Answer Correctness = 0.5832
- Improved: Faithfulness = **0.0**, Answer Correctness = 0.1766
- **Difference (Δ Faithfulness)**: `-1.0000` | **(Δ Answer Correctness)**: `-0.4066`

---

### #5 [Item `round2_050`] (numeric_precision) [POST-HOC, not pre-registered]

**Question**: How much did Amazon pay in income taxes after deducting refunds in fiscal year 2025, in millions?

**Ground Truth**: `$8,295 million (or $8.3 billion)`

**Supporting Quote**: Total cash taxes paid, net of refunds$11,179 $12,308 $8,295

- **Baseline Answer**: Based on the provided SEC filings (specifically Chunk 1, which details the cash paid for income taxes, net of refunds for the year ended December 31, 2025), the total cash taxes paid, net of refunds, was **$8,295 million** (or $8.295 billion), broken down as follows:

* **U.S. Federal:** $2,751 million
* **U.S. State:** $2,125 million
* **International:** $3,419 million
- **Improved Answer**: In fiscal year 2025, Amazon paid $8,295 million in income taxes, net of refunds [2].

**Judge Scores**:
- Baseline: Faithfulness = **0.8333**, Answer Correctness = 0.5729
- Improved: Faithfulness = **1.0**, Answer Correctness = 0.9381
- **Difference (Δ Faithfulness)**: `+0.1667` | **(Δ Answer Correctness)**: `+0.3652`

---

### #6 [Item `round2_015`] (numeric_precision) [POST-HOC, not pre-registered]

**Question**: Alphabet effective tax rate annual comparison percentage fiscal year 2024

**Ground Truth**: `The effective tax rate was 13.9% for the year ended December 31, 2023, and increased to 16.4% for the year ended December 31, 2024.`

**Supporting Quote**: Year Ended December 31,

20232024

Income before provision for income taxes$85,717 $119,815

Provision for income taxes$11,922 $19,697

Effective tax rate13.9 %16.4 %

- **Baseline Answer**: Based on the provided SEC filings (specifically for Alphabet Inc.), the effective tax rate was **13.9%** for the year ended December 31, 2023, and increased to **16.4%** for the year ended December 31, 2024.
- **Improved Answer**: Alphabet's effective tax rate was 13.9% for the fiscal year ended December 31, 2023, and increased to 16.4% for the fiscal year ended December 31, 2024 [2, 5].

**Judge Scores**:
- Baseline: Faithfulness = **1.0**, Answer Correctness = 0.9709
- Improved: Faithfulness = **1.0**, Answer Correctness = 0.9675
- **Difference (Δ Faithfulness)**: `+0.0000` | **(Δ Answer Correctness)**: `-0.0034`

---

## Category B: 3 Lowest-Faithfulness Improved Answers [POST-HOC, not pre-registered]

### #1 [Item `round2_065`] (simple_lookup) [POST-HOC, not pre-registered]

**Question**: What does Apple's filing say about the drivers behind the shifts in digital marketplace and software hosting revenue during fiscal year 2024?

**Ground Truth**: `Services net sales increased during 2024 compared to 2023 due primarily to higher net sales from advertising, the App Store® and cloud services.`

**Supporting Quote**: Services net sales increased during 2024 compared to 2023 due primarily to higher net sales from advertising, the App Store® and cloud services.

- **Baseline Answer**: I don't have enough information to answer this question based on the retrieved SEC filings.
- **Improved Answer**: I don't have enough information to answer this question based on the retrieved SEC filings.

**Judge Scores**:
- Baseline: Faithfulness = 0.0, Answer Correctness = 0.1865
- Improved: Faithfulness = **0.0**, Answer Correctness = **0.1865**
- **Difference (Δ Faithfulness)**: `+0.0000` | **(Δ Answer Correctness)**: `+0.0000`

---

### #2 [Item `round2_110`] (simple_lookup) [POST-HOC, not pre-registered]

**Question**: How many personnel does NVIDIA employ in the Israeli territory as of its fiscal year 2024 report?

**Ground Truth**: `NVIDIA employs approximately 3,700 individuals in the region who are primarily engaged in research and development, operations, sales, and marketing for networking products.`

**Supporting Quote**: We are monitoring the impact of the geopolitical conflict in and around Israel on our operations, including the health and safety of our approximately 3,700 employees in the region who primarily support the research and development, operations, and sales and marketing of our networking products.

- **Baseline Answer**: Based on the provided SEC filings, NVIDIA employed approximately 3,700 employees in the Israeli region as of its fiscal year 2024 report.
- **Improved Answer**: I don't have enough information to answer this question based on the retrieved SEC filings. Verity currently indexes SEC Form 10-K filings for Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), and NVIDIA (NVDA).

**Judge Scores**:
- Baseline: Faithfulness = 1.0, Answer Correctness = 0.5832
- Improved: Faithfulness = **0.0**, Answer Correctness = **0.1766**
- **Difference (Δ Faithfulness)**: `-1.0000` | **(Δ Answer Correctness)**: `-0.4066`

---

### #3 [Item `round2_124`] (numeric_precision) [POST-HOC, not pre-registered]

**Question**: What was the financial write-down amount NVIDIA recorded in the opening quarter of fiscal year 2026 related to surplus supplies and supply contracts for specific semiconductor products restricted by export controls?

**Ground Truth**: `NVIDIA incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20 for excess inventory and purchase obligations.`

**Supporting Quote**: As a result of these requirements, we incurred a $4.5 billion charge in the first quarter of fiscal year 2026 associated with H20 for excess inventory and purchase obligations, as the demand for H20 products diminished.

- **Baseline Answer**: I don't have enough information to answer this question based on the retrieved SEC filings.
- **Improved Answer**: I don't have enough information to answer this question based on the retrieved SEC filings.

**Judge Scores**:
- Baseline: Faithfulness = 1.0, Answer Correctness = 0.1863
- Improved: Faithfulness = **0.0**, Answer Correctness = **0.1863**
- **Difference (Δ Faithfulness)**: `-1.0000` | **(Δ Answer Correctness)**: `+0.0000`

---
