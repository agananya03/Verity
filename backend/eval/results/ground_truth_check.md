# Ground Truth Audit & Verification Report (`chunks.json`)

**Date**: 2026-11-06  
**Dataset Evaluated**: `backend/eval/golden_set.json` (32 items)  
**Corpus Analyzed**: `backend/data/processed/chunks.json` (2,921 processed chunks across Apple, Microsoft, Alphabet, Amazon, NVIDIA)  

---

## 1. Executive Summary

Every item in `golden_set.json` was cross-referenced against `backend/data/processed/chunks.json` for the cited company and fiscal year(s). The audit checked for:
1. **Exact Verbatim Match**: Full ground truth text appears word-for-word in the cited chunks.
2. **Numeric Precision**: All numbers, percentages, and metrics appear identically in the cited chunks.
3. **Factual & Clause Alignment**: Individual factual statements and claims were grepped against the actual SEC 10-K filing text.
4. **Corpus Coverage**: Identification of any facts located outside the indexed Item 1, 1A, 7, and 8 sections.

### High-Level Audit Findings:
- **100% Exact Verbatim in Corpus**: 4 items (`golden_06`, `golden_27`, `golden_28`, `golden_29`).
- **100% Numeric Match in Corpus**: 18 of 19 answerable items contain all cited numerical metrics verbatim in the target chunks.
- **Substantively Present but Paraphrased / Stylized**: 14 items (`golden_01`–`golden_05`, `golden_07`–`golden_09`, `golden_11`–`golden_12`, `golden_23`–`golden_26`). Their key facts exist in the corpus, but the ground truth was written with editorial commentary, synthesized grammar, or synthetic meta-labels (e.g., 'According to Chunk A').
- **Critical Missing Chunk (Out of Ingestion Scope)**: 1 item (`golden_10`). The text was originally extracted from Apple's 2023 Form 10-K Table of Contents preface, which is outside the ingested Items 1, 1A, 7, and 8 sections.
- **Unanswerable Outside-Corpus Items**: 13 items (`golden_13`–`golden_22`, `golden_30`–`golden_32`). All 13 questions strictly target companies and topics outside the indexed corpus (Tesla, Uber, Meta, Spotify, Netflix, Airbnb, Snap, Intel, Salesforce) with zero content on Apple, Microsoft, Alphabet, Amazon, or NVIDIA.

---

## 2. Complete Golden Set Audit Matrix

| ID | Category | Company | Fiscal Year | Verbatim Match? | Numeric Match? | Status / Finding |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `golden_01` | `simple_lookup` | Apple | FY 2025 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_02` | `simple_lookup` | Apple | FY 2025 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_03` | `simple_lookup` | Apple | FY 2024 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_04` | `numeric_precision` | Apple | FY 2024 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_05` | `simple_lookup` | Apple | FY 2024 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_06` | `simple_lookup` | Apple | FY 2025 | Yes | Yes | **Exact Verbatim Match** |
| `golden_07` | `simple_lookup` | Apple | FY 2025 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_08` | `simple_lookup` | Apple | FY 2024 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_09` | `simple_lookup` | Apple | FY 2023 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_10` | `simple_lookup` | Apple | FY 2023 | No | Yes | **MISSING FROM CORPUS** (TOC preface outside Items 1, 1A, 7, 8) |
| `golden_11` | `simple_lookup` | Apple | FY 2025 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_12` | `simple_lookup` | Apple | FY 2023 | No | Yes | **Substantive Match** (Minor paraphrase/grammar reordering) |
| `golden_13` | `unanswerable` | Tesla | FY 2022 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_14` | `unanswerable` | Uber | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_15` | `unanswerable` | Meta | FY 2020 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_16` | `unanswerable` | Spotify | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_17` | `unanswerable` | Netflix | FY 2022 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_18` | `unanswerable` | Airbnb | FY 2022 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_19` | `unanswerable` | Tesla | FY 2021 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_20` | `unanswerable` | Snap | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_21` | `unanswerable` | Intel | FY 2022 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_22` | `unanswerable` | Salesforce | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_23` | `multi_hop` | Apple | FY 2023/2024 | No | Yes | **Substantive Multi-Hop** (Facts present; contains meta-labels) |
| `golden_24` | `multi_hop` | Apple | FY 2023/2024 | No | Yes | **Substantive Multi-Hop** (Facts present; contains meta-labels) |
| `golden_25` | `multi_hop` | Apple | FY 2023/2024 | No | Yes | **Substantive Multi-Hop** (Facts present; contains meta-labels) |
| `golden_26` | `multi_hop` | Apple | FY 2023/2024 | No | Yes | **Substantive Multi-Hop** (Facts present; contains meta-labels) |
| `golden_27` | `numeric_precision` | Apple | FY 2025 | Yes | Yes | **Exact Verbatim Match** |
| `golden_28` | `numeric_precision` | Apple | FY 2025 | Yes | Yes | **Exact Verbatim Match** |
| `golden_29` | `numeric_precision` | Apple | FY 2025 | Yes | Yes | **Exact Verbatim Match** |
| `golden_30` | `unanswerable` | Netflix | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_31` | `unanswerable` | Netflix | FY 2024 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |
| `golden_32` | `unanswerable` | Meta | FY 2023 | N/A | N/A | **Unanswerable Baseline** (Outside corpus company) |

---

## 3. Actionable Items: Discrepancies & Recommended Hand-Fixes

Below is the comprehensive item-by-item analysis for every item whose ground truth was not found 100% verbatim in `chunks.json`. Use these exact filing excerpts to update or standardize ground truth by hand.

### Item `golden_01` (Apple FY 2025 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2025 Form 10-K, through what methods does the company expand the range of its offerings and develop new technologies?
- **Current Ground Truth**:  
  > "The Company develops new technologies to enhance existing products and services and expands its range of offerings through research and development (“R&D”), the licensing of intellectual property, and the acquisition of third-party businesses and technology."
- **Target Chunk(s)**: `dfa8abbf-4dda-57f5-b67f-6ada36cf6dae` (2025 Item 1 (Research and Development))
- **Verbatim Text in `chunks.json`**:  
  ```text
  The Company continues to develop new technologies to enhance existing products and services, and to expand the range of its offerings through research and development (“R&D”), licensing of intellectual property and acquisition of third-party businesses and technology.
  ```
- **Discrepancy Analysis**: Minor grammatical paraphrase: GT has 'The Company develops... and expands... the licensing... and the acquisition...' whereas the 10-K text states 'The Company continues to develop... and to expand... licensing of intellectual property and acquisition...'.
- **Recommended Hand-Fix Replacement**:  
  > "The Company continues to develop new technologies to enhance existing products and services, and to expand the range of its offerings through research and development (“R&D”), licensing of intellectual property and acquisition of third-party businesses and technology."

### Item `golden_02` (Apple FY 2025 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2025 Form 10-K, what subscription-based digital content services are listed under Apple's Services offerings?
- **Current Ground Truth**:  
  > "Apple offers subscription-based services including Apple Arcade (a game service), Apple Fitness+ (a personalized fitness service), Apple Music (offering a curated listening experience with on-demand radio stations), Apple News+ (a news and magazine service), and Apple TV."
- **Target Chunk(s)**: `9ccb486b-cb47-5ebd-8f27-8a699b270c4f` (2025 Item 1 (Digital Content))
- **Verbatim Text in `chunks.json`**:  
  ```text
  The Company also offers digital content through subscription-based services, including Apple Arcade®, a game service; Apple Fitness+®, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a news and magazine service; and Apple TV®, which offers exclusive original content and live sports.
  ```
- **Discrepancy Analysis**: GT omitted registered trademark symbols (®) and trimmed the final description clause for Apple TV ('which offers exclusive original content and live sports'). In Chunk 9ccb486b, Apple TV is referred to as 'Apple TV®' rather than 'Apple TV+®'.
- **Recommended Hand-Fix Replacement**:  
  > "Apple offers digital content through subscription-based services, including Apple Arcade®, a game service; Apple Fitness+®, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a news and magazine service; and Apple TV®, which offers exclusive original content and live sports."

### Item `golden_03` (Apple FY 2024 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2024 Form 10-K, what are the principal competitive factors important to the company?
- **Current Ground Truth**:  
  > "The principal competitive factors important to the Company include price, product and service features (including security features), relative price and performance, product and service quality and reliability, design innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, and corporate reputation."
- **Target Chunk(s)**: `5105704e-c923-5e1b-b8e9-2b875814e62f` (2024 Item 1 (Competition))
- **Verbatim Text in `chunks.json`**:  
  ```text
  Principal competitive factors important to the Company include price, product and service features (including security features), relative price and performance, product and service quality and reliability, design innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, and corporate reputation.
  ```
- **Discrepancy Analysis**: Leading article discrepancy: GT begins with 'The principal competitive factors...', whereas filing sentence begins with 'Principal competitive factors...'.
- **Recommended Hand-Fix Replacement**:  
  > "Principal competitive factors important to the Company include price, product and service features (including security features), relative price and performance, product and service quality and reliability, design innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, and corporate reputation."

### Item `golden_04` (Apple FY 2024 - `numeric_precision`)
- **Rewritten Question**: According to Apple's fiscal year 2024 Form 10-K, what percentages of total net sales did Apple generate through its direct and indirect distribution channels in fiscal year 2024, respectively?
- **Current Ground Truth**:  
  > "During 2024, Apple's net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales."
- **Target Chunk(s)**: `9908f268-63bb-561c-bd95-f5f3240ff6a8` (2024 Item 1 (Markets and Distribution))
- **Verbatim Text in `chunks.json`**:  
  ```text
  During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales.
  ```
- **Discrepancy Analysis**: GT substitutes 'Apple's net sales' in place of the filing's verbatim 'the Company’s net sales'. All numbers (2024, 38%, 62%) are 100% numerically verified.
- **Recommended Hand-Fix Replacement**:  
  > "During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales (or: 38% direct and 62% indirect)."

### Item `golden_05` (Apple FY 2024 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2024 Form 10-K, what subscription-based digital content services are explicitly detailed under Apple's Services offerings?
- **Current Ground Truth**:  
  > "The subscription-based digital content services explicitly listed are Apple Arcade (a game subscription service), Apple Fitness+ (a personalized fitness service), Apple Music (which offers a curated listening experience with on-demand radio stations), and Apple News+."
- **Target Chunk(s)**: `a67656e8-ad29-550a-9d4e-e89b5e72fe40` (2024 Item 1 (Digital Content))
- **Verbatim Text in `chunks.json`**:  
  ```text
  The Company also offers digital content through subscription-based services, including Apple Arcade®, a game subscription service; Apple Fitness+SM, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a subscription news and magazine service; and Apple TV+®, which offers exclusive original content and live sports.
  ```
- **Discrepancy Analysis**: GT completely omits 'Apple TV+® (which offers exclusive original content and live sports)' and adds an artificial opening clause.
- **Recommended Hand-Fix Replacement**:  
  > "Apple offers digital content through subscription-based services, including Apple Arcade®, a game subscription service; Apple Fitness+SM, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a subscription news and magazine service; and Apple TV+®, which offers exclusive original content and live sports."

### Item `golden_07` (Apple FY 2025 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2025 Form 10-K, what factors can cause initial capacity constraints when the company introduces a component or product that utilizes new technologies?
- **Current Ground Truth**:  
  > "Initial capacity constraints may exist when a component or product uses new technologies until the suppliers' yields have matured or their manufacturing capacities have increased."
- **Target Chunk(s)**: `dfa8abbf-4dda-57f5-b67f-6ada36cf6dae` (2025 Item 1 (Manufacturing and Logistical Services))
- **Verbatim Text in `chunks.json`**:  
  ```text
  When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased.
  ```
- **Discrepancy Analysis**: Clause inversion: Filing begins with conditional clause 'When a component or product uses new technologies, initial capacity constraints may exist...', whereas GT inverts it to main clause first.
- **Recommended Hand-Fix Replacement**:  
  > "When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased."

### Item `golden_08` (Apple FY 2024 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2024 Form 10-K, what historical trend does Apple observe regarding its net sales in the first quarter of its fiscal year, and what drives this trend?
- **Current Ground Truth**:  
  > "Apple historically experiences higher net sales in its first quarter compared to other quarters in its fiscal year, which is driven in part by seasonal holiday demand."
- **Target Chunk(s)**: `447dd5eb-6650-5317-b998-0f0f90a75f22` (2024 Item 1 (Business Seasonality and Product Introductions))
- **Verbatim Text in `chunks.json`**:  
  ```text
  The Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due in part to seasonal holiday demand.
  ```
- **Discrepancy Analysis**: GT replaces 'The Company has historically experienced... due in part to...' with 'Apple historically experiences... which is driven in part by...'.
- **Recommended Hand-Fix Replacement**:  
  > "The Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due in part to seasonal holiday demand."

### Item `golden_09` (Apple FY 2023 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2023 Form 10-K, what specific consequences could result if Apple is unable to satisfy changing regulations regarding product import and export?
- **Current Ground Truth**:  
  > "If Apple fails to satisfy import and export regulatory requirements or is found in violation, its products could be banned, delayed, or prohibited from importation, which could materially adversely affect the Company's business, reputation, results of operations, and financial condition."
- **Target Chunk(s)**: `11351d57-89ad-5ddb-9f9e-89a66f1a437a` (2023 Item 1A (Risk Factors))
- **Verbatim Text in `chunks.json`**:  
  ```text
  For example, the Company is subject to changing regulations relating to the export and import of its products. Although the Company has programs, policies and procedures in place that are designed to satisfy regulatory requirements, there can be no assurance that such policies and procedures will be effective in preventing a violation or a claim of a violation. As a result, the Company’s products could be banned, delayed or prohibited from importation, which could materially adversely affect the Company’s business, reputation, results of operations and financial condition.
  ```
- **Discrepancy Analysis**: GT condenses two sentences into an artificial 'If Apple fails...' conditional sentence.
- **Recommended Hand-Fix Replacement**:  
  > "The Company's products could be banned, delayed or prohibited from importation, which could materially adversely affect the Company’s business, reputation, results of operations and financial condition."

### Item `golden_10` (Apple FY 2023 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2023 Form 10-K, in which specific sections are many of the forward-looking statements located?
- **Current Ground Truth**:  
  > "Many of the forward-looking statements are located in Part I, Item 1 under the heading "Business" and Part II, Item 7 under the heading "Management’s Discussion and Analysis of Financial Condition and Results of Operations.""
- **Target Chunk(s)**: `NOT_FOUND_IN_CHUNKS` (Form 10-K Preface / Table of Contents (Outside Ingested Items 1, 1A, 7, 8))
- **Verbatim Text in `chunks.json`**:  
  ```text
  CRITICAL MISMATCH: This paragraph appeared on page 5 of the raw filing preceding Part I Item 1 in Apple's 2023 Form 10-K: 'Many of the forward-looking statements are located in Part I, Item 1 of this Form 10-K under the heading “Business” and Part II, Item 7 of this Form 10-K under the heading “Management’s Discussion and Analysis of Financial Condition and Results of Operations.”' Because our ingestion extracts Items 1, 1A, 7, and 8, this text was discarded during section slicing and does not exist anywhere in chunks.json.
  ```
- **Discrepancy Analysis**: Fact is entirely absent from the indexed chunks. Retriever cannot ground this question from the corpus.
- **Recommended Hand-Fix Replacement**:  
  > "Replace question with an Item 1A risk disclosure question, OR update question/ground_truth to Item 1A's forward-looking statement notice (Chunk d4349ba4-7f55-581a-8ed7-e64dc56452f3): 'This discussion of risk factors contains forward-looking statements. This section should be read in conjunction with Part II, Item 7, “Management’s Discussion and Analysis of Financial Condition and Results of Operations” and Part II, Item 8, “Financial Statements and Supplementary Data”.'"

### Item `golden_11` (Apple FY 2025 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2025 Form 10-K, what specific characteristics define the global markets in which the company offers its products and services?
- **Current Ground Truth**:  
  > "The global markets for the Company's products and services are characterized by aggressive price competition, downward pressure on gross margins, continual improvement in product performance, price sensitivity on the part of consumers and businesses, frequent introduction of new products and services, short product life cycles, evolving industry standards, and rapid adoption of technological advancements."
- **Target Chunk(s)**: `31c03bf4-2c2d-5b32-9cb7-7946fb33bf9c` (2025 Item 1 (Competition))
- **Verbatim Text in `chunks.json`**:  
  ```text
  The markets for the Company’s products and services are highly competitive, and are characterized by aggressive price competition and resulting downward pressure on gross margins, frequent introduction of new products and services, short product life cycles, evolving industry standards, continual improvement in product price and performance characteristics, rapid adoption of technological advancements by competitors, and price sensitivity on the part of consumers and businesses.
  ```
- **Discrepancy Analysis**: GT adds 'global' to opening phrase and reorders items in the comma-delimited characteristics list.
- **Recommended Hand-Fix Replacement**:  
  > "The markets for the Company’s products and services are highly competitive, and are characterized by aggressive price competition and resulting downward pressure on gross margins, frequent introduction of new products and services, short product life cycles, evolving industry standards, continual improvement in product price and performance characteristics, rapid adoption of technological advancements by competitors, and price sensitivity on the part of consumers and businesses."

### Item `golden_12` (Apple FY 2023 - `simple_lookup`)
- **Rewritten Question**: According to Apple's fiscal year 2023 Form 10-K, what specific areas are gauged through the company's employee sentiment surveys?
- **Current Ground Truth**:  
  > "The Company conducts surveys that gauge employee sentiment in the areas of career development, manager performance, and inclusivity."
- **Target Chunk(s)**: `b427da43-b7b5-5cb3-9613-5d21a299c1d4` (2023 Item 1 (Human Resources))
- **Verbatim Text in `chunks.json`**:  
  ```text
  ...and the Company conducts surveys that gauge employee sentiment in areas like career development, manager performance and inclusivity.
  ```
- **Discrepancy Analysis**: GT states 'in the areas of career development...' instead of filing's verbatim 'in areas like career development...'.
- **Recommended Hand-Fix Replacement**:  
  > "The Company conducts surveys that gauge employee sentiment in areas like career development, manager performance and inclusivity."

### Item `golden_23` (Apple FY 2023/2024 - `multi_hop`)
- **Rewritten Question**: Comparing Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what specific changes were made to the listed models in the iPhone line, and what new category of devices was added to the definition of 'Wearables' in fiscal year 2024?
- **Current Ground Truth**:  
  > "Based on Apple's Item 1 filings: 1) For the iPhone line, the fiscal year 2023 filing lists the iPhone 15 Pro, iPhone 15, iPhone 14, iPhone 13, and iPhone SE, whereas the fiscal year 2024 filing lists the iPhone 16 Pro, iPhone 16, iPhone 15, iPhone 14, and iPhone SE (meaning the iPhone 16 Pro and iPhone 16 were added, while the iPhone 15 Pro and iPhone 13 were removed). 2) For the definition of 'Wearables', the fiscal year 2023 filing states that 'Wearables includes smartwatches and wireless headphones', whereas the fiscal year 2024 filing adds 'spatial computers' to this definition, stating 'Wearables includes smartwatches, wireless headphones and spatial computers'."
- **Target Chunk(s)**: `b0cf1104-bec6-5e5b-8711-cece88499c98 & ff8dcd62-5714-5dbc-a09a-f8c2da4fa3a5` (2023 Item 1 & 2024 Item 1)
- **Verbatim Text in `chunks.json`**:  
  ```text
  2023 Item 1: 'The iPhone line includes iPhone 15 Pro, iPhone 15, iPhone 14, iPhone 13 and iPhone SE®.' / 'Wearables includes smartwatches and wireless headphones.'
2024 Item 1: 'The iPhone line includes iPhone 16 Pro, iPhone 16, iPhone 15, iPhone 14 and iPhone SE®.' / 'Wearables includes smartwatches, wireless headphones and spatial computers.'
  ```
- **Discrepancy Analysis**: All model names and definitions are 100% verified in the corpus chunks, but GT contains synthesis wording ('Based on Apple's Item 1 filings... meaning the iPhone 16 Pro and iPhone 16 were added...').
- **Recommended Hand-Fix Replacement**:  
  > "For the iPhone line, the 2023 filing lists iPhone 15 Pro, iPhone 15, iPhone 14, iPhone 13 and iPhone SE, while the 2024 filing lists iPhone 16 Pro, iPhone 16, iPhone 15, iPhone 14 and iPhone SE (iPhone 16 Pro and iPhone 16 were added; iPhone 15 Pro and iPhone 13 were removed). For Wearables, the 2024 filing added 'spatial computers' to the definition (changing from 'smartwatches and wireless headphones' to 'smartwatches, wireless headphones and spatial computers')."

### Item `golden_24` (Apple FY 2023/2024 - `multi_hop`)
- **Rewritten Question**: Based on Apple's Form 10-K filings, what is the operating system associated with Apple's line of smartwatches as described in its fiscal year 2023 filing, and what is the operating system associated with Apple's first spatial computer, Apple Vision Pro, as described in its fiscal year 2024 filing?
- **Current Ground Truth**:  
  > "According to Chunk A (FY 2023), the operating system associated with Apple's line of smartwatches is watchOS (specifically watchOS®). According to Chunk B (FY 2024), the operating system associated with Apple's first spatial computer, Apple Vision Pro, is visionOS (specifically visionOS™)."
- **Target Chunk(s)**: `b0cf1104-bec6-5e5b-8711-cece88499c98 & ff8dcd62-5714-5dbc-a09a-f8c2da4fa3a5` (2023 Item 1 & 2024 Item 1)
- **Verbatim Text in `chunks.json`**:  
  ```text
  2023 Item 1: 'The Company’s line of smartwatches, based on its watchOS® operating system, includes Apple Watch Ultra™ 2...'
2024 Item 1: 'Apple Vision Pro™ is the Company’s first spatial computer based on its visionOS™ operating system.'
  ```
- **Discrepancy Analysis**: Contains prompt-engineering remnants 'According to Chunk A' and 'According to Chunk B'. The facts (watchOS® and visionOS™) are 100% verified in chunks.
- **Recommended Hand-Fix Replacement**:  
  > "The operating system associated with Apple's line of smartwatches in the fiscal year 2023 filing is watchOS (watchOS®), and the operating system associated with Apple Vision Pro in the fiscal year 2024 filing is visionOS (visionOS™)."

### Item `golden_25` (Apple FY 2023/2024 - `multi_hop`)
- **Rewritten Question**: According to Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what are the specific operating systems that power the iPhone and iPad product lines, and what are the five reportable geographic segments through which Apple manages its business?
- **Current Ground Truth**:  
  > "Based on Apple's Item 1 disclosures:
1. The operating systems powering the product lines (from Chunk A) are iOS for the iPhone and iPadOS for the iPad.
2. The five reportable geographic segments (from Chunk B) are the Americas, Europe, Greater China, Japan, and Rest of Asia Pacific."
- **Target Chunk(s)**: `b0cf1104-bec6-5e5b-8711-cece88499c98 & 9908f268-63bb-561c-bd95-f5f3240ff6a8` (2023 Item 1 & 2024 Item 1)
- **Verbatim Text in `chunks.json`**:  
  ```text
  2023 Item 1: 'iPhone® is the Company’s line of smartphones based on its iOS operating system... iPad® is the Company’s line of multipurpose tablets based on its iPadOS® operating system.'
2024 Item 1: 'The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific.'
  ```
- **Discrepancy Analysis**: Contains meta-labels '(from Chunk A)' and '(from Chunk B)'. Both facts are 100% verified in chunks.
- **Recommended Hand-Fix Replacement**:  
  > "The operating systems powering the product lines are iOS for the iPhone and iPadOS (iPadOS®) for the iPad. The five reportable geographic segments are the Americas, Europe, Greater China, Japan and Rest of Asia Pacific."

### Item `golden_26` (Apple FY 2023/2024 - `multi_hop`)
- **Rewritten Question**: According to Apple's Item 1 disclosures in its fiscal year 2023 and fiscal year 2024 Form 10-K filings, what are the operating systems used for the iPhone and smartwatch product lines, and what percentage of Apple's total net sales was generated through indirect distribution channels in fiscal year 2024?
- **Current Ground Truth**:  
  > "According to Chunk A (Apple's 2023 Item 1), the operating system for the iPhone line of smartphones is iOS, and the operating system for the smartwatch line (under Wearables, Home and Accessories) is watchOS. According to Chunk B (Apple's 2024 Item 1), the Company's net sales through indirect distribution channels accounted for 62% of total net sales during 2024."
- **Target Chunk(s)**: `b0cf1104-bec6-5e5b-8711-cece88499c98 & 9908f268-63bb-561c-bd95-f5f3240ff6a8` (2023 Item 1 & 2024 Item 1)
- **Verbatim Text in `chunks.json`**:  
  ```text
  2023 Item 1: 'iPhone® is the Company’s line of smartphones based on its iOS operating system... The Company’s line of smartwatches, based on its watchOS® operating system...'
2024 Item 1: 'During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales.'
  ```
- **Discrepancy Analysis**: Contains meta-labels 'According to Chunk A' and 'According to Chunk B'. Facts (iOS, watchOS, 62%) are 100% verified in chunks.
- **Recommended Hand-Fix Replacement**:  
  > "The operating system for the iPhone line is iOS and for the smartwatch line is watchOS. During fiscal year 2024, net sales through indirect distribution channels accounted for 62% of total net sales."

---

## 4. Unanswerable Verification (Outside-Corpus Guarantee)

The 13 unanswerable items were audited to ensure zero overlap with our indexed corpus (Apple, Microsoft, Alphabet, Amazon, NVIDIA).

| ID | Company | Fiscal Year | Topic | Verification |
| :--- | :--- | :--- | :--- | :--- |
| `golden_13` | Tesla | 2022 | Automotive revenue | Tesla not in corpus |
| `golden_14` | Uber | 2023 | Delivery segment gross bookings | Uber not in corpus |
| `golden_15` | Meta | 2020 | Reality Labs operating losses | Meta not in corpus |
| `golden_16` | Spotify | 2023 | Premium subscriber count | Spotify not in corpus |
| `golden_17` | Netflix | 2022 | Paid memberships | Netflix not in corpus |
| `golden_18` | Airbnb | 2022 | Gross booking value | Airbnb not in corpus |
| `golden_19` | Tesla | 2021 | Model 3/Y deliveries | Tesla not in corpus |
| `golden_20` | Snap | 2023 | Daily active users (DAUs) | Snap not in corpus |
| `golden_21` | Intel | 2022 | Capital expenditures | Intel not in corpus |
| `golden_22` | Salesforce | 2023 | Subscription and support revenue | Salesforce not in corpus |
| `golden_30` | Netflix | 2023 | Streaming revenues & operating income | Netflix not in corpus |
| `golden_31` | Netflix | 2024 | Q4 churn rate & ARPU | Netflix not in corpus (Quarterly metric) |
| `golden_32` | Meta | 2023 | Project Orion AR internal R&D budget | Meta not in corpus (Internal program budget) |

