# End-to-end answer-quality evaluation: pre-registration

- Items: 30-item stratified sample of golden_round2.json (answerable, single-year, answer_quality_eligible), seed 42, via --filter-eligible --stratified-sample 30.
- Pipelines: baseline (raw dense, unfiltered, limit 8, single generation call) vs improved (full agent: clean_v3_rrf10 retrieval + critic loop, MAX_RETRIES=2).
- Generator: gemini-3.5-flash-lite. Critic: gemini-3.1-flash-lite.
- Judge: a different model from the generator and critic, chosen after a 3-item pilot. The choice is recorded in the results. --allow-same-judge is NOT used.
- Metrics: faithfulness and answer_correctness (RAGAS). Report the mean for each pipeline, the paired difference with a bootstrap 95% CI, n=30, and the call counts.
- No item exclusion, rerun or judge swap after seeing results. Any change is reported as post-hoc.
- The judge is an LLM and imperfect: a human check of a few judged items is reported separately.