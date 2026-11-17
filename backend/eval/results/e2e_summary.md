# End-to-End Evaluation Summary: Verity Financial RAG

- **Date**: 2026-11-17
- **Judge Model**: `gemma-4-31b-it`
- **Golden Benchmark**: `golden_round2.json` (SHA-256: `37cc61373c8b806b...`)
- **Sample Size**: 30 items (stratified by company & section)
- **Generator Model**: `gemini-3.5-flash-lite` | **Critic Model**: `gemini-3.1-flash-lite`

## 1. Primary Metrics & Paired Differences

| Metric | Baseline | Improved | Difference (Imp - Base) | 95% Bootstrap CI | Imp Higher / Equal / Lower |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Faithfulness** | 0.6867 | 0.8638 | **+0.1771** | [-0.0018, +0.3505] | 13 / 12 / 5 |
| **Answer Correctness** | 0.4513 | 0.7151 | **+0.2638** | [+0.1413, +0.3859] | 22 / 2 / 6 |
| **Hit@8** | 0.2667 | 0.8000 | **+0.5333** | [+0.3667, +0.7000] | 16 / 14 / 0 |
| **Recall@8** | 0.2667 | 0.8000 | **+0.5333** | [+0.3667, +0.7000] | 16 / 14 / 0 |
| **MRR@8** | 0.1397 | 0.4469 | **+0.3072** | [+0.1755, +0.4481] | 21 / 8 / 1 |
| **Numeric Accuracy (n=21)** | 0.4762 | 0.9048 | **+0.4286** | [+0.2381, +0.6190] | 9 / 12 / 0 |

## 2. Refusal / Decline Behavior

- **Baseline Decline Rate**: 14 / 30 (46.7%)
- **Improved Decline Rate**: 4 / 30 (13.3%)
- **Decline Reduction**: +33.3 percentage points

## 3. Critic Revision & Retry Behavior (Improved Agent)

- **Critic Revise Rate**: 2 / 30 (6.7%)
- **Retry Count Distribution**:
  - 0 retries: 28 items (93.3%)
  - 1 retries: 2 items (6.7%)

## 4. Retrieval Conditioning Slice (Hit@8 True vs False)

| Pipeline | Gold Chunk Retrieved (Hit@8) | N | Mean Faithfulness | Mean Answer Correctness |
| :--- | :---: | :---: | :---: | :---: |
| Baseline | **Yes (Hit)** | 8 | 0.6667 | 0.6838 |
| Baseline | **No (Miss)** | 22 | 0.6939 | 0.3667 |
| Improved | **Yes (Hit)** | 24 | 0.9653 | 0.8035 |
| Improved | **No (Miss)** | 6 | 0.4579 | 0.3616 |
