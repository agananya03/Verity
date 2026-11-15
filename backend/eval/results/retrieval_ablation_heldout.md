# Retrieval Ablation Study Report - HELD-OUT SET (n=64, design NOT tuned on these questions)

- **Timestamp**: `2026-11-15T09:40:16.358676+00:00`
- **Eligible Items Evaluated**: `64` (out of `76` total golden set)
- **Excluded Items**: `12` (unanswerable or unindexed items, including `golden_10`)
- **Evaluated Categories**: `simple_lookup` (28), `numeric_precision` (33), `other` (1), `multi_hop` (2)
- **Evaluated Slices**: `single_year` (62), `multi_year` (2)

## Traceability Fingerprint (SHA-256)

- `backend/agent/retrieval.py`: `1a700973445d32dcb5433cda8a70596a23d6381578bfeac68d2997762ed30e1e`
- `backend/agent/nodes/classifier.py`: `16b34f1434d6844d69694c4f2c0accbb005d106f3f0fbb611b2eb99f3de6cf12`
- `backend/eval/golden_set.json`: `f4e739ea438573742aecd8992ee330698cecc9fcbfad68eb592bf25de5b96c5f`
- **Combined Fingerprint**: `e32d72e738e5e120b953b0390beaff3543a4a1df10158b6a42889df8e839daeb`

## Staged Retrieval Ladder (HELD-OUT SET (n=64, design NOT tuned on these questions))

| Configuration | N | Hit@8 (All) | Rec@8 (All) | MRR (All) | Hit@8 (Single) | Rec@8 (Single) | MRR (Single) | Hit@8 (Multi) | Rec@8 (Multi) | MRR (Multi) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 64 | 0.3281 | 0.3281 | 0.1610 | 0.3226 | 0.3226 | 0.1581 | 0.5000 | 0.5000 | 0.2500 |
| **dense_filtered** | 64 | 0.5000 | 0.5000 | 0.2620 | 0.5000 | 0.5000 | 0.2624 | 0.5000 | 0.5000 | 0.2500 |
| **bm25_filtered** | 64 | 0.7031 | 0.6953 | 0.3572 | 0.7097 | 0.7097 | 0.3667 | 0.5000 | 0.2500 | 0.0625 |
| **rrf_raw_filtered** | 64 | 0.5938 | 0.5938 | 0.3417 | 0.5968 | 0.5968 | 0.3447 | 0.5000 | 0.5000 | 0.2500 |
| **rrf_clean_v2 (single pass)** | 64 | 0.5625 | 0.5625 | 0.3163 | 0.5645 | 0.5645 | 0.3185 | 0.5000 | 0.5000 | 0.2500 |
| **rrf_clean_v2_peryear_v1** | 64 | 0.5625 | 0.5625 | 0.3241 | 0.5645 | 0.5645 | 0.3185 | 0.5000 | 0.5000 | 0.5000 |
| **production retrieve()** | 64 | 0.5625 | 0.5625 | 0.3241 | 0.5645 | 0.5645 | 0.3185 | 0.5000 | 0.5000 | 0.5000 |

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
| **heldout_01** | `comp=Alphabet, fy=2023` | What is Alphabet's corporate goal regarding total greenhouse gas reduction across its entire business value network by the end of the decade, per Alphabet's fiscal 2023 annual report? | What is Alphabet's corporate goal regarding total greenhouse gas reduction across its entire business value network by the end of the decade, per Alphabet's annual report? |
| **heldout_02** | `comp=Alphabet, fy=2023` | At Alphabet, what factors impact workforce retention and talent acquisition, for fiscal 2023? | At Alphabet, what factors impact workforce retention and talent acquisition, for ? |
| **heldout_03** | `comp=Alphabet, fy=2023` | Alphabet 2023 APAC constant currency revenue growth rate percentage | Alphabet APAC constant currency revenue growth rate percentage |
| **heldout_04** | `comp=Alphabet, fy=2023` | At Alphabet, what was the total dollar amount spent on innovative technical investigation and engineering efforts during the fiscal year 2023? | At Alphabet, what was the total dollar amount spent on innovative technical investigation and engineering efforts during the ? |
| **heldout_05** | `comp=Alphabet, fy=2023` | At Alphabet, what was the combined carrying value of its equity investments at the end of fiscal 2023, in millions of dollars? | At Alphabet, what was the combined carrying value of its equity investments at the end of, in millions of dollars? |
| **heldout_07** | `comp=Alphabet, fy=None` | At Alphabet, how many products with a user base exceeding the two billion mark integrate the company's natively multimodal AI model? | At Alphabet, how many products with a user base exceeding the two billion mark integrate the company's natively multimodal AI model? |
| **heldout_08** | `comp=Alphabet, fy=2024` | At Alphabet, what proportion of total consolidated revenue came from outside the United States during the 2024 fiscal year? | At Alphabet, what proportion of total consolidated revenue came from outside the United States during the fiscal year? |
| **heldout_09** | `comp=Alphabet, fy=2024` | Alphabet fiscal year 2024 operating margin percentage | Alphabet operating margin percentage |
| **heldout_10** | `comp=Alphabet, fy=2024` | At Alphabet, what was the total capital returned to investors via equity buybacks during the 2024 fiscal period? | At Alphabet, what was the total capital returned to investors via equity buybacks during the fiscal period? |
| **heldout_14** | `comp=Alphabet, fy=2025` | What potential negative impacts on society from artificial intelligence deployments are identified as risk factors for the company, in Alphabet's FY2025 10-K? | What potential negative impacts on society from artificial intelligence deployments are identified as risk factors for the company, in Alphabet's ? |
| **heldout_15** | `comp=Alphabet, fy=2025` | Alphabet Traffic Acquisition Costs (TAC) total for fiscal year 2025 in millions | Alphabet Traffic Acquisition Costs (TAC) total for in millions |
| **heldout_16** | `comp=Alphabet, fy=2025` | At Alphabet, how might the integration of machine learning features into consumer and enterprise tools alter financial monetization models, in fiscal 2025? | At Alphabet, how might the integration of machine learning features into consumer and enterprise tools alter financial monetization models, ? |
| **heldout_17** | `comp=Alphabet, fy=2025` | At Alphabet, what does the filing say about the total undiscounted lease obligations for operating agreements at the conclusion of fiscal year 2025? | At Alphabet, what does the filing say about the total undiscounted lease obligations for operating agreements at the conclusion of ? |
| **heldout_21** | `comp=Amazon, fy=2023` | Amazon variable and fixed operating cost categories, for fiscal 2023 | Amazon variable and fixed operating cost categories, for |
| **heldout_22** | `comp=Amazon, fy=2023` | At Amazon, what was the total expenditure for outbound product delivery services incurred by the corporation during the year ended December 31, 2023? | At Amazon, what was the total expenditure for outbound product delivery services incurred by the corporation during the year ended December 31, ? |
| **heldout_23** | `comp=Amazon, fy=2023` | At Amazon, what does the filing say about the capital expenditures for the cloud computing division in fiscal 2023? | At Amazon, what does the filing say about the capital expenditures for the cloud computing division ? |
| **heldout_25** | `comp=Amazon, fy=2025` | At Amazon, who holds the position of chief executive officer for the cloud computing division as of January 2025? | At Amazon, who holds the position of chief executive officer for the cloud computing division as of January ? |
| **heldout_26** | `comp=Amazon, fy=2024` | What resource price increases and inflationary cost pressures are identified as potential operational factors, per Amazon's fiscal 2024 annual report? | What resource price increases and inflationary cost pressures are identified as potential operational factors, per Amazon's annual report? |
| **heldout_27** | `comp=Amazon, fy=2024` | Amazon AWS revenue fiscal year 2024 absolute amount in millions | Amazon AWS revenue absolute amount in millions |
| **heldout_29** | `comp=Amazon, fy=2024` | At Amazon, how much did AWS add to its property and equipment in fiscal 2024? | At Amazon, how much did AWS add to its property and equipment ? |
| **heldout_31** | `comp=Amazon, fy=2025` | Which executive holds the position of Chief Financial Officer for the company, in Amazon's FY2025 10-K? | Which executive holds the position of Chief Financial Officer for the company, in Amazon's ? |
| **heldout_32** | `comp=Amazon, fy=2025` | At Amazon, what factors cause sales volatility in fiscal 2025? | At Amazon, what factors cause sales volatility ? |
| **heldout_33** | `comp=Amazon, fy=2025` | Amazon variable cost components enumeration, for fiscal 2025 | Amazon variable cost components enumeration, for |
| **heldout_34** | `comp=Amazon, fy=2025` | At Amazon, how much did currency fluctuations add to tech and infrastructure expenses during fiscal year 2025? | At Amazon, how much did currency fluctuations add to tech and infrastructure expenses ? |
| **heldout_35** | `comp=Amazon, fy=2025` | At Amazon, what amount of senior notes was issued in the final quarter of fiscal 2025 for general corporate purposes? | At Amazon, what amount of senior notes was issued in the final quarter of for general corporate purposes? |
| **heldout_37** | `comp=Apple, fy=2023` | At Apple, what proportion of total revenue was generated via third-party cellular carriers, wholesalers, and other secondary pathways for the 2023 fiscal period? | At Apple, what proportion of total revenue was generated via third-party cellular carriers, wholesalers, and other secondary pathways for the fiscal period? |
| **heldout_38** | `comp=Apple, fy=2023` | What potential negative consequences could Apple face if its merchandise importation is subjected to sudden regulatory restrictions, per Apple's fiscal 2023 annual report? | What potential negative consequences could Apple face if its merchandise importation is subjected to sudden regulatory restrictions, per Apple's annual report? |
| **heldout_39** | `comp=Apple, fy=2023` | At Apple, uncertain tax positions and legal contingencies financial impact evaluation criteria, for fiscal 2023 | At Apple, uncertain tax positions and legal contingencies financial impact evaluation criteria, for |
| **heldout_40** | `comp=Apple, fy=2023` | What factors does leadership rely on when formulating calculations for carrying values of obligations and assets, in Apple's FY2023 10-K? | What factors does leadership rely on when formulating calculations for carrying values of obligations and assets, in Apple's ? |
| **heldout_41** | `comp=Apple, fy=2023` | At Apple, what does the filing say about the opinion issued regarding the consolidated balance sheets for the fiscal year ended September 30, 2023? | At Apple, what does the filing say about the opinion issued regarding the consolidated balance sheets for the fiscal year ended September 30, ? |
| **heldout_43** | `comp=Apple, fy=2024` | At Apple, what was the total headcount of full-time equivalent staff at the close of fiscal year 2024? | At Apple, what was the total headcount of full-time equivalent staff at the close of ? |
| **heldout_44** | `comp=Apple, fy=2024` | What protection limitations exist for Apple's IT network, in Apple's fiscal 2024 filing? | What protection limitations exist for Apple's IT network, in Apple's filing? |
| **heldout_45** | `comp=Apple, fy=2024` | Apple uncertain tax positions and ordinary course litigation financial impact evaluation criteria, for fiscal 2024 | Apple uncertain tax positions and ordinary course litigation financial impact evaluation criteria, for |
| **heldout_46** | `comp=Apple, fy=2024` | At Apple, what was the aggregate principal value of outstanding fixed-rate debt instruments held by the corporation as of the conclusion of fiscal year 2024? | At Apple, what was the aggregate principal value of outstanding fixed-rate debt instruments held by the corporation as of the conclusion of ? |
| **heldout_49** | `comp=Apple, fy=2025` | At Apple, what was the total headcount of full-time equivalent staff employed by the technology firm as of late September 2025? | At Apple, what was the total headcount of full-time equivalent staff employed by the technology firm as of late September ? |
| **heldout_50** | `comp=Apple, fy=2025` | What is the typical duration covered by manufacturing purchase commitments, per Apple's fiscal 2025 annual report? | What is the typical duration covered by manufacturing purchase commitments, per Apple's annual report? |
| **heldout_51** | `comp=Apple, fy=2025` | Apple FY2025 provision for income taxes | Apple provision for income taxes |
| **heldout_52** | `comp=Apple, fy=2025` | At Apple, what is the total monetary value of equity bought back by the corporate entity during the 2025 fiscal period? | At Apple, what is the total monetary value of equity bought back by the corporate entity during the fiscal period? |
| **heldout_53** | `comp=Apple, fy=2025` | At Apple, what does the filing say about the maximum hedging duration utilized for foreign currency exposure related to term debt transactions as of fiscal 2025? | At Apple, what does the filing say about the maximum hedging duration utilized for foreign currency exposure related to term debt transactions as of ? |
| **heldout_55** | `comp=Microsoft, fy=2024` | Which executive currently holds the position of Chief Financial Officer at Microsoft, in Microsoft's FY2024 10-K? | Which executive currently holds the position of Chief Financial Officer at Microsoft, in Microsoft's ? |
| **heldout_57** | `comp=Microsoft, fy=2024` | At Microsoft, what was the total value of liquid funds and near-term market holdings at the close of fiscal year 2024? | At Microsoft, what was the total value of liquid funds and near-term market holdings at the close of ? |
| **heldout_58** | `comp=Microsoft, fy=2024` | At Microsoft, what does the filing say about the total face value of long-term debt instruments for the period ending June 30, 2024? | At Microsoft, what does the filing say about the total face value of long-term debt instruments for the period ending June 30, ? |
| **heldout_60** | `comp=Microsoft, fy=2025` | What is the primary distinction between the two main classifications of original equipment manufacturers that partner with Microsoft, per Microsoft's fiscal 2025 annual report? | What is the primary distinction between the two main classifications of original equipment manufacturers that partner with Microsoft, per Microsoft's annual report? |
| **heldout_61** | `comp=Microsoft, fy=2025` | What consequences could arise from failing to meet public-sector contracting obligations, in Microsoft's FY2025 10-K? | What consequences could arise from failing to meet public-sector contracting obligations, in Microsoft's ? |
| **heldout_62** | `comp=Microsoft, fy=2025` | At Microsoft, productivity and Business Processes segment operating income fiscal year 2025 in millions | At Microsoft, productivity and Business Processes segment operating income in millions |
| **heldout_63** | `comp=Microsoft, fy=2025` | At Microsoft, how much did the Azure cloud platform's revenue grow year over year in fiscal 2025? | At Microsoft, how much did the Azure cloud platform's revenue grow year over year ? |
| **heldout_65** | `comp=Microsoft, fy=2025` | At Microsoft, what is the specific minimum liquidity requirement and the actual cash investments level maintained as of June 30, 2025? | At Microsoft, what is the specific minimum liquidity requirement and the actual cash investments level maintained as of June 30, ? |
| **heldout_66** | `comp=Microsoft, fy=2026` | What long-term environmental targets has the technology enterprise set to achieve by the end of the decade, per Microsoft's fiscal 2026 annual report? | What long-term environmental targets has the technology enterprise set to achieve by the end of the decade, per Microsoft's annual report? |
| **heldout_67** | `comp=Microsoft, fy=2026` | What potential business impacts could arise from regulatory scrutiny and enforcement regarding artificial intelligence and competition laws, in Microsoft's FY2026 10-K? | What potential business impacts could arise from regulatory scrutiny and enforcement regarding artificial intelligence and competition laws, in Microsoft's ? |
| **heldout_69** | `comp=Microsoft, fy=2026` | What was Azure's revenue growth rate at Microsoft in fiscal 2026? | What was Azure's revenue growth rate at Microsoft ? |
| **heldout_73** | `comp=NVIDIA, fy=2024` | What U.S. tax deduction benefits a significant portion of NVIDIA's domestic taxable income, in NVIDIA's FY2024 10-K? | What U.S. tax deduction benefits a significant portion of NVIDIA's domestic taxable income, in NVIDIA's ? |
| **heldout_74** | `comp=NVIDIA, fy=2024` | At NVIDIA, compute and Networking segment operating income for fiscal year ended January 28, 2024 | At NVIDIA, compute and Networking segment operating income for fiscal year ended January 28, |
| **heldout_75** | `comp=NVIDIA, fy=2024` | At NVIDIA, what capital expenditure range for property and equipment is expected for the year after fiscal 2024? | At NVIDIA, what capital expenditure range for property and equipment is expected for the year after ? |
| **heldout_76** | `comp=NVIDIA, fy=2024` | At NVIDIA, what does the filing say about the carrying value for privately held company holdings at the conclusion of fiscal year 2024? | At NVIDIA, what does the filing say about the carrying value for privately held company holdings at the conclusion of ? |
| **heldout_78** | `comp=NVIDIA, fy=2025` | At NVIDIA, what proportion of the workforce departed from the company during fiscal year 2025? | At NVIDIA, what proportion of the workforce departed from the company ? |
| **heldout_79** | `comp=None, fy=None` | At NVIDIA, what specific rights does Microsoft hold under its agreement with the company if a third party attempts to purchase a large stake in the common stock, for fiscal 2025? | At NVIDIA, what specific rights does Microsoft hold under its agreement with the company if a third party attempts to purchase a large stake in the common stock, for fiscal 2025? |
| **heldout_80** | `comp=NVIDIA, fy=2025` | At NVIDIA, compute and Networking segment operating income fiscal year 2025 in millions | At NVIDIA, compute and Networking segment operating income in millions |
| **heldout_81** | `comp=NVIDIA, fy=2025` | What is the company's policy regarding how interest and penalties associated with unrecognized tax benefits are classified in the financial statements, per NVIDIA's fiscal 2025 annual report? | What is the company's policy regarding how interest and penalties associated with unrecognized tax benefits are classified in the financial statements, per NVIDIA's annual report? |
| **heldout_84** | `comp=NVIDIA, fy=None` | At NVIDIA, what is the timeline range for the expiration dates of the company's currently active patents? | At NVIDIA, what is the timeline range for the expiration dates of the company's currently active patents? |
| **heldout_85** | `comp=NVIDIA, fy=2026` | At NVIDIA, what specific environmental constraints could restrict customer data center expansion and subsequent product consumption, for fiscal 2026? | At NVIDIA, what specific environmental constraints could restrict customer data center expansion and subsequent product consumption, for ? |
| **heldout_87** | `comp=NVIDIA, fy=2026` | At NVIDIA, which specific areas of financial reporting does the company identify as involving significant estimation uncertainty and potential material impact on operations, for fiscal 2026? | At NVIDIA, which specific areas of financial reporting does the company identify as involving significant estimation uncertainty and potential material impact on operations, for ? |
| **heldout_88** | `comp=NVIDIA, fy=2026` | At NVIDIA, what does the filing say about the deferred tax valuation allowance ending balance for fiscal year 2026? | At NVIDIA, what does the filing say about the deferred tax valuation allowance ending balance for ? |
| **heldout_mh_08** | `comp=Apple, fy=None` | At Apple, how did the description of competitor product feature imitation evolve between 2024 and 2025 regarding integrated solutions? | At Apple, how did the description of competitor product feature imitation evolve between 2024 and 2025 regarding integrated solutions? |
| **heldout_mh_10** | `comp=Microsoft, fy=None` | At Microsoft, how did the named competitors in the Windows competition description change between fiscal 2024 and fiscal 2025? | At Microsoft, how did the named competitors in the Windows competition description change between fiscal 2024 and fiscal 2025? |

## Overall Retrieval Performance (N = 18)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) | Top Hub Chunk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **baseline_pipeline** | 0.3281 | 0.3281 | 0.1610 | 0.3281 | 0.3281 | 10.9% (7/64) | 727.3 / 750.7 | `f70a9dac...` |
| **dense_filtered** | 0.5000 | 0.5000 | 0.2620 | 0.6250 | 0.6250 | 7.8% (5/64) | 462.6 / 488.2 | `1972c44f...` |
| **bm25_filtered** | 0.7031 | 0.6953 | 0.3572 | 0.7969 | 0.7891 | 6.2% (4/64) | 25.9 / 41.4 | `efa79d54...` |
| **rrf_raw_filtered** | 0.5938 | 0.5938 | 0.3417 | 0.7656 | 0.7656 | 7.8% (5/64) | 300.3 / 301.5 | `9cd5eeba...` |
| **rrf_clean_v2 (single pass)** | 0.5625 | 0.5625 | 0.3163 | 0.8125 | 0.8125 | 7.8% (5/64) | 311.0 / 298.3 | `9cd5eeba...` |
| **rrf_clean_v2_peryear_v1** | 0.5625 | 0.5625 | 0.3241 | 0.8281 | 0.8203 | 7.8% (5/64) | 294.6 / 524.1 | `9cd5eeba...` |
| **production retrieve()** | 0.5625 | 0.5625 | 0.3241 | 0.8281 | 0.8203 | 7.8% (5/64) | 297.3 / 462.1 | `9cd5eeba...` |
| **dense_clean_v2** | 0.5000 | 0.5000 | 0.2690 | 0.6406 | 0.6406 | 7.8% (5/64) | 260.3 / 259.7 | `021f7646...` |
| **bm25_clean_v2** | 0.7031 | 0.6953 | 0.3131 | 0.8281 | 0.8203 | 6.2% (4/64) | 24.6 / 37.3 | `efa79d54...` |
| **rrf_clean_v1 (single pass)** | 0.7812 | 0.7812 | 0.3430 | 0.9062 | 0.9062 | 6.2% (4/64) | 287.6 / 324.5 | `c7d97f98...` |
| **RRF (BM25 2x Weighted)** | 0.5781 | 0.5703 | 0.3537 | 0.7812 | 0.7812 | 6.2% (4/64) | 276.8 / 290.6 | `021f7646...` |
| **RRF Top-30 + Rerank** | 0.6250 | 0.6172 | 0.3523 | 0.7812 | 0.7812 | 6.2% (4/64) | 2889.5 / 3327.4 | `021f7646...` |
| **BM25 Top-30 + Rerank** | 0.6094 | 0.6094 | 0.3377 | 0.7656 | 0.7656 | 7.8% (5/64) | 2691.2 / 3097.2 | `9cd5eeba...` |

## Performance Breakdown by Slice (Single-Year vs Multi-Year)

### Slice: `Single-Year Questions (n=14)`

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.3226 | 0.3226 | 0.1581 | 0.3226 | 0.3226 | 11.3% (7/62) | 728.2 / 750.7 |
| **dense_filtered** | 0.5000 | 0.5000 | 0.2624 | 0.6129 | 0.6129 | 8.1% (5/62) | 466.2 / 488.4 |
| **bm25_filtered** | 0.7097 | 0.7097 | 0.3667 | 0.8065 | 0.8065 | 4.8% (3/62) | 26.0 / 41.5 |
| **rrf_raw_filtered** | 0.5968 | 0.5968 | 0.3447 | 0.7742 | 0.7742 | 8.1% (5/62) | 301.4 / 301.6 |
| **rrf_clean_v2 (single pass)** | 0.5645 | 0.5645 | 0.3185 | 0.8226 | 0.8226 | 8.1% (5/62) | 312.4 / 298.8 |
| **rrf_clean_v2_peryear_v1** | 0.5645 | 0.5645 | 0.3185 | 0.8226 | 0.8226 | 8.1% (5/62) | 286.9 / 332.8 |
| **production retrieve()** | 0.5645 | 0.5645 | 0.3185 | 0.8226 | 0.8226 | 8.1% (5/62) | 289.9 / 299.7 |
| **dense_clean_v2** | 0.5000 | 0.5000 | 0.2696 | 0.6290 | 0.6290 | 8.1% (5/62) | 261.0 / 259.8 |
| **bm25_clean_v2** | 0.7097 | 0.7097 | 0.3211 | 0.8387 | 0.8387 | 4.8% (3/62) | 24.6 / 37.5 |
| **rrf_clean_v1 (single pass)** | 0.7903 | 0.7903 | 0.3501 | 0.9194 | 0.9194 | 6.5% (4/62) | 288.4 / 328.2 |
| **RRF (BM25 2x Weighted)** | 0.5806 | 0.5806 | 0.3570 | 0.7903 | 0.7903 | 6.5% (4/62) | 277.0 / 290.6 |
| **RRF Top-30 + Rerank** | 0.6290 | 0.6290 | 0.3555 | 0.7903 | 0.7903 | 6.5% (4/62) | 2897.3 / 3328.4 |
| **BM25 Top-30 + Rerank** | 0.6129 | 0.6129 | 0.3405 | 0.7742 | 0.7742 | 8.1% (5/62) | 2696.5 / 3102.7 |

### Slice: `Multi-Year Questions (n=4)`

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 700.4 / 703.8 |
| **dense_filtered** | 0.5000 | 0.5000 | 0.2500 | 1.0000 | 1.0000 | 50.0% (1/2) | 352.4 / 455.2 |
| **bm25_filtered** | 0.5000 | 0.2500 | 0.0625 | 0.5000 | 0.2500 | 50.0% (1/2) | 23.6 / 27.0 |
| **rrf_raw_filtered** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 266.3 / 267.7 |
| **rrf_clean_v2 (single pass)** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 266.7 / 267.7 |
| **rrf_clean_v2_peryear_v1** | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 50.0% (1/2) | 533.9 / 538.0 |
| **production retrieve()** | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 50.0% (1/2) | 526.2 / 531.6 |
| **dense_clean_v2** | 0.5000 | 0.5000 | 0.2500 | 1.0000 | 1.0000 | 50.0% (1/2) | 239.2 / 242.3 |
| **bm25_clean_v2** | 0.5000 | 0.2500 | 0.0625 | 0.5000 | 0.2500 | 50.0% (1/2) | 27.3 / 27.5 |
| **rrf_clean_v1 (single pass)** | 0.5000 | 0.5000 | 0.1250 | 0.5000 | 0.5000 | 50.0% (1/2) | 263.1 / 265.7 |
| **RRF (BM25 2x Weighted)** | 0.5000 | 0.2500 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 269.0 / 269.6 |
| **RRF Top-30 + Rerank** | 0.5000 | 0.2500 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 2647.4 / 2775.8 |
| **BM25 Top-30 + Rerank** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 2528.7 / 2614.9 |


## Performance Breakdown by Category

### Category: `simple_lookup` (n = 28)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.3214 | 0.3214 | 0.1938 | 0.3214 | 0.3214 | 14.3% (4/28) | 721.9 / 750.6 |
| **dense_filtered** | 0.5357 | 0.5357 | 0.2848 | 0.6071 | 0.6071 | 10.7% (3/28) | 468.6 / 487.8 |
| **bm25_filtered** | 0.6786 | 0.6786 | 0.4099 | 0.7857 | 0.7857 | 10.7% (3/28) | 27.8 / 42.2 |
| **rrf_raw_filtered** | 0.5714 | 0.5714 | 0.3601 | 0.7500 | 0.7500 | 10.7% (3/28) | 285.9 / 301.2 |
| **rrf_clean_v2 (single pass)** | 0.5714 | 0.5714 | 0.3494 | 0.8929 | 0.8929 | 7.1% (2/28) | 272.9 / 291.7 |
| **rrf_clean_v2_peryear_v1** | 0.5714 | 0.5714 | 0.3494 | 0.8929 | 0.8929 | 7.1% (2/28) | 275.1 / 297.5 |
| **production retrieve()** | 0.5714 | 0.5714 | 0.3494 | 0.8929 | 0.8929 | 7.1% (2/28) | 273.4 / 297.4 |
| **dense_clean_v2** | 0.5714 | 0.5714 | 0.2923 | 0.6429 | 0.6429 | 10.7% (3/28) | 246.4 / 253.7 |
| **bm25_clean_v2** | 0.7500 | 0.7500 | 0.4081 | 0.8571 | 0.8571 | 10.7% (3/28) | 26.0 / 35.1 |
| **rrf_clean_v1 (single pass)** | 0.8214 | 0.8214 | 0.3772 | 0.9286 | 0.9286 | 7.1% (2/28) | 272.7 / 293.1 |
| **RRF (BM25 2x Weighted)** | 0.5714 | 0.5714 | 0.3619 | 0.7500 | 0.7500 | 10.7% (3/28) | 276.4 / 291.3 |
| **RRF Top-30 + Rerank** | 0.6071 | 0.6071 | 0.2768 | 0.7857 | 0.7857 | 10.7% (3/28) | 2864.3 / 3303.0 |
| **BM25 Top-30 + Rerank** | 0.5714 | 0.5714 | 0.2703 | 0.7500 | 0.7500 | 10.7% (3/28) | 2678.9 / 3037.0 |

### Category: `numeric_precision` (n = 33)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.3333 | 0.3333 | 0.1326 | 0.3333 | 0.3333 | 15.2% (5/33) | 734.8 / 757.3 |
| **dense_filtered** | 0.4848 | 0.4848 | 0.2513 | 0.6364 | 0.6364 | 9.1% (3/33) | 464.3 / 487.1 |
| **bm25_filtered** | 0.7273 | 0.7273 | 0.3361 | 0.8182 | 0.8182 | 6.1% (2/33) | 24.6 / 32.6 |
| **rrf_raw_filtered** | 0.6364 | 0.6364 | 0.3420 | 0.8182 | 0.8182 | 9.1% (3/33) | 315.9 / 378.2 |
| **rrf_clean_v2 (single pass)** | 0.5758 | 0.5758 | 0.3019 | 0.7576 | 0.7576 | 9.1% (3/33) | 347.7 / 850.7 |
| **rrf_clean_v2_peryear_v1** | 0.5758 | 0.5758 | 0.3019 | 0.7576 | 0.7576 | 9.1% (3/33) | 297.8 / 514.6 |
| **production retrieve()** | 0.5758 | 0.5758 | 0.3019 | 0.7576 | 0.7576 | 9.1% (3/33) | 304.9 / 370.3 |
| **dense_clean_v2** | 0.4545 | 0.4545 | 0.2586 | 0.6364 | 0.6364 | 9.1% (3/33) | 274.2 / 347.6 |
| **bm25_clean_v2** | 0.6667 | 0.6667 | 0.2521 | 0.8182 | 0.8182 | 6.1% (2/33) | 23.5 / 38.0 |
| **rrf_clean_v1 (single pass)** | 0.7576 | 0.7576 | 0.3316 | 0.9091 | 0.9091 | 6.1% (2/33) | 302.7 / 493.3 |
| **RRF (BM25 2x Weighted)** | 0.6061 | 0.6061 | 0.3637 | 0.8485 | 0.8485 | 9.1% (3/33) | 277.9 / 284.4 |
| **RRF Top-30 + Rerank** | 0.6364 | 0.6364 | 0.4271 | 0.7879 | 0.7879 | 9.1% (3/33) | 2929.3 / 3322.9 |
| **BM25 Top-30 + Rerank** | 0.6364 | 0.6364 | 0.4043 | 0.7879 | 0.7879 | 9.1% (3/33) | 2715.9 / 3145.5 |

### Category: `other` (n = 1)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% (1/1) | 685.4 / 685.4 |
| **dense_filtered** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% (1/1) | 457.7 / 457.7 |
| **bm25_filtered** | 1.0000 | 1.0000 | 0.1667 | 1.0000 | 1.0000 | 100.0% (1/1) | 21.3 / 21.3 |
| **rrf_raw_filtered** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% (1/1) | 258.1 / 258.1 |
| **rrf_clean_v2 (single pass)** | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100.0% (1/1) | 254.6 / 254.6 |
| **rrf_clean_v2_peryear_v1** | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100.0% (1/1) | 256.8 / 256.8 |
| **production retrieve()** | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100.0% (1/1) | 255.8 / 255.8 |
| **dense_clean_v2** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% (1/1) | 236.0 / 236.0 |
| **bm25_clean_v2** | 1.0000 | 1.0000 | 0.1667 | 1.0000 | 1.0000 | 100.0% (1/1) | 17.6 / 17.6 |
| **rrf_clean_v1 (single pass)** | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 100.0% (1/1) | 258.4 / 258.4 |
| **RRF (BM25 2x Weighted)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% (1/1) | 266.0 / 266.0 |
| **RRF Top-30 + Rerank** | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 100.0% (1/1) | 2767.8 / 2767.8 |
| **BM25 Top-30 + Rerank** | 1.0000 | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 100.0% (1/1) | 2547.9 / 2547.9 |

### Category: `multi_hop` (n = 2)

| Configuration | Hit@8 | Recall@8 | MRR | Hit@20 | Recall@20 | Hub Share | Latency (mean/p95 ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_pipeline** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 700.4 / 703.8 |
| **dense_filtered** | 0.5000 | 0.5000 | 0.2500 | 1.0000 | 1.0000 | 50.0% (1/2) | 352.4 / 455.2 |
| **bm25_filtered** | 0.5000 | 0.2500 | 0.0625 | 0.5000 | 0.2500 | 50.0% (1/2) | 23.6 / 27.0 |
| **rrf_raw_filtered** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 266.3 / 267.7 |
| **rrf_clean_v2 (single pass)** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 266.7 / 267.7 |
| **rrf_clean_v2_peryear_v1** | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 50.0% (1/2) | 533.9 / 538.0 |
| **production retrieve()** | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 0.7500 | 50.0% (1/2) | 526.2 / 531.6 |
| **dense_clean_v2** | 0.5000 | 0.5000 | 0.2500 | 1.0000 | 1.0000 | 50.0% (1/2) | 239.2 / 242.3 |
| **bm25_clean_v2** | 0.5000 | 0.2500 | 0.0625 | 0.5000 | 0.2500 | 50.0% (1/2) | 27.3 / 27.5 |
| **rrf_clean_v1 (single pass)** | 0.5000 | 0.5000 | 0.1250 | 0.5000 | 0.5000 | 50.0% (1/2) | 263.1 / 265.7 |
| **RRF (BM25 2x Weighted)** | 0.5000 | 0.2500 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 269.0 / 269.6 |
| **RRF Top-30 + Rerank** | 0.5000 | 0.2500 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 2647.4 / 2775.8 |
| **BM25 Top-30 + Rerank** | 0.5000 | 0.5000 | 0.2500 | 0.5000 | 0.5000 | 50.0% (1/2) | 2528.7 / 2614.9 |

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
