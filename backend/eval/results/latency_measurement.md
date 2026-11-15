# End-to-End Latency & Memory Measurement

**Benchmark Scope:** 30 single-year answerable questions from `golden_heldout.json` (round 1), 1 warmup query per configuration.  
**Live Backend:** Live Qdrant Cloud (`verity_chunks`, 2921 chunks) + local BM25 index.

## Latency & Process RSS Table

| Configuration | Description | Mean Latency (ms) | Median / p50 (ms) | p95 Latency (ms) | Peak Process RSS (MB) |
|---|---|---|---|---|---|
| **(1) Old production retrieve()** | Legacy clean_v2, k=60, top-20 candidate pools | 269.5 | 265.7 | 273.5 | N/A (baseline) |
| **(2) clean_v3 + rrf_k10** | clean_v3 query, k=10, top-50 dense + BM25 pools (new default) | 261.9 | 262.5 | 271.4 | 617.8 MB |
| **(3) clean_v3 + rrf_k60 + rerank** | clean_v3, k=60, top-30 reranked via ms-marco-MiniLM-L-6-v2 | 2724.1 | 2711.2 | 2805.1 | 2487.7 MB |

## Memory Comparison: Config (2) vs Config (3)

- **Config (2) `clean_v3_rrf10`:** Runs pure RRF rank math in Python without loading neural cross-encoders into process memory. Peak RSS measured at **617.8 MB**.
- **Config (3) Cross-Encoder Rerank:** Loads `Xenova/ms-marco-MiniLM-L-6-v2` ONNX model weights and runtime, increasing process RSS to **2487.7 MB** (+1869.9 MB overhead).
- **Latency Impact:** Cross-encoder adds ~2462.2ms mean latency per query to evaluate 30 candidate chunk texts.
