# Secondary Post-Hoc Evaluation Analysis: Verity Financial RAG

> [!IMPORTANT]
> **POST-HOC, not pre-registered**: The analyses below are secondary and exploratory. They were designed after examining the primary pre-registered evaluation results to control for refusal scoring artifacts and retrieval dependencies.

- **Date**: 2026-11-17
- **Judge Model**: `gemma-4-31b-it`
- **Golden Benchmark**: `golden_round2.json` (SHA-256: `37cc61373c8b806b...`)
- **Sample Size**: 30 items total

## 1. Paired Differences Restricted to Non-Decline Answers

This slice excludes items where either pipeline declined to answer (n=15 items where both pipelines produced substantive answers). This controls for judge refusal scoring anomalies.

### Table 1: Paired Differences on Non-Decline Answers (n=15) [POST-HOC, not pre-registered]

| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.8400 | 0.9965 | **+0.1565** | [+0.0667, +0.2530] | 7 / 7 / 1 |
| **Answer Correctness** | 0.6835 | 0.8219 | **+0.1384** | [-0.0098, +0.2844] | 11 / 0 / 4 |

## 2. Analysis Restricted to Gold-Retrieved Items

### Table 2: Paired Differences on Gold-Retrieved Items (Improved Hit@8 = True, n=24) [POST-HOC, not pre-registered]

| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.6917 | 0.9653 | **+0.2736** | [+0.1306, +0.4306] | 12 / 11 / 1 |
| **Answer Correctness** | 0.4987 | 0.8035 | **+0.3048** | [+0.1696, +0.4336] | 20 / 0 / 4 |

### Table 3: Paired Differences on Improved-Only Gold-Retrieved Items (Imp Hit & Base Miss, n=16) [POST-HOC, not pre-registered]

| Metric | Baseline (Mean) | Improved (Mean) | Paired Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.7042 | 0.9792 | **+0.2750** | [+0.0667, +0.5042] | 6 / 9 / 1 |
| **Answer Correctness** | 0.4062 | 0.8175 | **+0.4114** | [+0.2729, +0.5479] | 14 / 0 / 2 |

### Table 4: Improved Pipeline Alone: Gold-Retrieved (n=24) vs Miss (n=6) [POST-HOC, not pre-registered]

| Metric | Gold Chunk Retrieved (Hit) | Gold Chunk Missed (Miss) | Difference (Hit - Miss) | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.9653 | 0.4579 | **+0.5074** | [+0.1481, +0.8458] |
| **Answer Correctness** | 0.8035 | 0.3616 | **+0.4418** | [+0.1804, +0.6483] |

## 3. Judge Inconsistency on Baseline Refusals / Declines

Documenting RAGAS faithfulness judge inconsistency on standard refusal boilerplate:

### Table 5: Judge Inconsistency on Baseline Refusals / Declines [POST-HOC, not pre-registered]

| Baseline Faithfulness Score | Total Items | Declines (Refusals) | Non-Declines | Decline Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Score = 0.0** | 7 | 7 | 0 | 100.0% |
| **Score = 1.0** | 16 | 7 | 9 | 43.8% |
| **Total Declines** | 14 | 14 scored 0.0 or 1.0 | 0 scored between 0 and 1 | 100.0% |

> **Key Finding**: Of the 14 total baseline declines, 7 were scored 0.0 and 7 were scored 1.0 by the judge, demonstrating arbitrary binary scoring of standard refusal boilerplate.
