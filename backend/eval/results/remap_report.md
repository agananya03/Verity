# Golden Set Source Chunk Remap Report

**Target Corpus**: `backend/data/processed/chunks.json` (Deterministic UUIDv5 IDs)  
**Total Questions Evaluated**: 32  
- **Confident Matches**: 18  
- **Unanswerable (Empty source_chunk_ids)**: 13  
- **Flagged (No Confident Match - Needs Manual Fix)**: 1  

---

## 1. Summary Matrix

| ID | Category | Company | Fiscal Year | Status | Match Score | Chunks Found | Section(s) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `golden_01` | `simple_lookup` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_02` | `simple_lookup` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_03` | `simple_lookup` | Apple | 2024 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_04` | `numeric_precision` | Apple | 2024 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_05` | `simple_lookup` | Apple | 2024 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_06` | `simple_lookup` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_07` | `simple_lookup` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_08` | `simple_lookup` | Apple | 2024 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_09` | `simple_lookup` | Apple | 2023 | CONFIDENT_MATCH | 1.00 | 1 | Item 1A |
| `golden_10` | `simple_lookup` | Apple | 2023 | **FLAGGED (NO MATCH)** | 0.00 | 0 | N/A |
| `golden_11` | `simple_lookup` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1A |
| `golden_12` | `simple_lookup` | Apple | 2023 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_13` | `unanswerable` | Tesla | 2022 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_14` | `unanswerable` | Uber | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_15` | `unanswerable` | Meta | 2020 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_16` | `unanswerable` | Spotify | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_17` | `unanswerable` | Netflix | 2022 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_18` | `unanswerable` | Airbnb | 2022 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_19` | `unanswerable` | Tesla | 2021 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_20` | `unanswerable` | Snap | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_21` | `unanswerable` | Intel | 2022 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_22` | `unanswerable` | Salesforce | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_23` | `multi_hop` | Apple | 2023/2024 | CONFIDENT_MATCH | 1.00 | 2 | Item 1, Item 1 |
| `golden_24` | `multi_hop` | Apple | 2023/2024 | CONFIDENT_MATCH | 1.00 | 2 | Item 1, Item 1 |
| `golden_25` | `multi_hop` | Apple | 2023/2024 | CONFIDENT_MATCH | 1.00 | 2 | Item 1, Item 1 |
| `golden_26` | `multi_hop` | Apple | 2023/2024 | CONFIDENT_MATCH | 1.00 | 2 | Item 1, Item 1 |
| `golden_27` | `numeric_precision` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_28` | `numeric_precision` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_29` | `numeric_precision` | Apple | 2025 | CONFIDENT_MATCH | 1.00 | 1 | Item 1 |
| `golden_30` | `unanswerable` | Netflix | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_31` | `unanswerable` | Netflix | 2024 | UNANSWERABLE | 1.00 | 0 | N/A |
| `golden_32` | `unanswerable` | Meta | 2023 | UNANSWERABLE | 1.00 | 0 | N/A |

---

## 2. Item-by-Item Remap Details

### Item `golden_01` (Apple FY2025 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `dfa8abbf-4dda-57f5-b67f-6ada36cf6dae`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom..."
    - **Verbatim Source Quote**: "The Company continues to develop new technologies to enhance existing products and services, and to expand the range of its offerings through research and development (“R&D”), licensing of intellectual property and acquisition of third-party businesses and technology."

### Item `golden_02` (Apple FY2025 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `9ccb486b-cb47-5ebd-8f27-8a699b270c4f`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Home includes Apple TV 4K®, the Company’s media streaming and gaming device based on its tvOS® operating system, and HomePod® and HomePod mini®, high-..."
    - **Verbatim Source Quote**: "The Company also offers digital content through subscription-based services, including Apple Arcade®, a game service; Apple Fitness+®, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a news and magazine service; and Apple TV®, which offers exclusive original content and live sports."

### Item `golden_03` (Apple FY2024 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `5105704e-c923-5e1b-b8e9-2b875814e62f`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The markets for the Company’s products and services are highly competitive, and are characterized by aggressive price competition and resulting downwa..."
    - **Verbatim Source Quote**: "Principal competitive factors important to the Company include price, product and service features (including security features), relative price and performance, product and service quality and reliability, design innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, and corporate reputation."

### Item `golden_04` (Apple FY2024 - `numeric_precision`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `9908f268-63bb-561c-bd95-f5f3240ff6a8`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Payment Services  The Company offers payment services, including Apple Card®, a co-branded credit card, and Apple Pay®, a cashless payment service.  S..."
    - **Verbatim Source Quote**: "During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales."

### Item `golden_05` (Apple FY2024 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `a67656e8-ad29-550a-9d4e-e89b5e72fe40`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Home includes Apple TV®, the Company’s media streaming and gaming device based on its tvOS® operating system, and HomePod® and HomePod mini®, high-fid..."
    - **Verbatim Source Quote**: "The Company also offers digital content through subscription-based services, including Apple Arcade®, a game subscription service; Apple Fitness+SM, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a subscription news and magazine service; and Apple TV+®, which offers exclusive original content and live sports."

### Item `golden_06` (Apple FY2025 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `ea2d524c-8da9-5970-99c9-9f64527663ba`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Because the industries in which the Company competes are characterized by rapid technological advances, the Company’s ability to compete successfully..."
    - **Verbatim Source Quote**: "Although the Company believes the ownership of such intellectual property rights is an important factor in differentiating its business and that its success does depend in part on such ownership, the Company relies primarily on the innovative skills, technical competence and marketing abilities of its personnel."

### Item `golden_07` (Apple FY2025 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `dfa8abbf-4dda-57f5-b67f-6ada36cf6dae`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom..."
    - **Verbatim Source Quote**: "When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased."

### Item `golden_08` (Apple FY2024 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `447dd5eb-6650-5317-b998-0f0f90a75f22`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company regularly files patent, design, copyright and trademark applications to protect innovations arising from its research, development, design..."
    - **Verbatim Source Quote**: "| 2024 Form 10-K | 3

Business Seasonality and Product Introductions

The Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due in part to seasonal holiday demand."

### Item `golden_09` (Apple FY2023 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `11351d57-89ad-5ddb-9f9e-89a66f1a437a`
    - **Item Section**: `Item 1A`
    - **150-char Preview**: "While the Company maintains insurance coverage for certain types of claims, such insurance coverage may be insufficient to cover all losses or all typ..."
    - **Verbatim Source Quote**: "As a result, the Company’s products could be banned, delayed or prohibited from importation, which could materially adversely affect the Company’s business, reputation, results of operations and financial condition."

### Item `golden_10` (Apple FY2023 - `simple_lookup`)
- **Status**: `NO_CONFIDENT_MATCH` (Score: 0.00)
- **Notes**: FLAGGED: Text located in raw 2023 Form 10-K preface outside ingested Items 1, 1A, 7, 8. Needs manual fix.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_11` (Apple FY2025 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `a5177a5b-5a57-51a9-a2e4-244eb3479af6`
    - **Item Section**: `Item 1A`
    - **150-char Preview**: "Following any interruption to its business, the Company can require substantial recovery time, incur significant expenditures to resume operations, an..."
    - **Verbatim Source Quote**: "These markets are characterized by aggressive price competition, downward pressure on gross margins, continual improvement in product performance, and price sensitivity on the part of consumers and businesses."

### Item `golden_12` (Apple FY2023 - `simple_lookup`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `b427da43-b7b5-5cb3-9613-5d21a299c1d4`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company believes that compensation should be competitive and equitable, and should enable employees to share in the Company’s success. The Company..."
    - **Verbatim Source Quote**: "Team members are encouraged to come to their managers with questions, feedback or concerns, and the Company conducts surveys that gauge employee sentiment in areas like career development, manager performance and inclusivity."

### Item `golden_13` (Tesla FY2022 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_14` (Uber FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_15` (Meta FY2020 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_16` (Spotify FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_17` (Netflix FY2022 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_18` (Airbnb FY2022 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_19` (Tesla FY2021 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_20` (Snap FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_21` (Intel FY2022 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_22` (Salesforce FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_23` (Apple FY2023/2024 - `multi_hop`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `b0cf1104-bec6-5e5b-8711-cece88499c98`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "The Company’s line of smartwatches, based on its watchOS® operating system, includes Apple Watch Ultra™ 2, Apple Watch® Series 9 and Apple Watch SE®."
  - **Chunk ID**: `ff8dcd62-5714-5dbc-a09a-f8c2da4fa3a5`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "Wearables, Home and Accessories

Wearables includes smartwatches, wireless headphones and spatial computers."

### Item `golden_24` (Apple FY2023/2024 - `multi_hop`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `b0cf1104-bec6-5e5b-8711-cece88499c98`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "The Company’s line of smartwatches, based on its watchOS® operating system, includes Apple Watch Ultra™ 2, Apple Watch® Series 9 and Apple Watch SE®."
  - **Chunk ID**: `ff8dcd62-5714-5dbc-a09a-f8c2da4fa3a5`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "Apple Vision Pro™ is the Company’s first spatial computer based on its visionOS™ operating system."

### Item `golden_25` (Apple FY2023/2024 - `multi_hop`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `b0cf1104-bec6-5e5b-8711-cece88499c98`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "Products

iPhone

iPhone® is the Company’s line of smartphones based on its iOS operating system."
  - **Chunk ID**: `9908f268-63bb-561c-bd95-f5f3240ff6a8`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Payment Services  The Company offers payment services, including Apple Card®, a co-branded credit card, and Apple Pay®, a cashless payment service.  S..."
    - **Verbatim Source Quote**: "The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific."

### Item `golden_26` (Apple FY2023/2024 - `multi_hop`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `b0cf1104-bec6-5e5b-8711-cece88499c98`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "PART I  Item 1.    Business  Company Background  The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and..."
    - **Verbatim Source Quote**: "Products

iPhone

iPhone® is the Company’s line of smartphones based on its iOS operating system."
  - **Chunk ID**: `9908f268-63bb-561c-bd95-f5f3240ff6a8`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Payment Services  The Company offers payment services, including Apple Card®, a co-branded credit card, and Apple Pay®, a cashless payment service.  S..."
    - **Verbatim Source Quote**: "During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales."

### Item `golden_27` (Apple FY2025 - `numeric_precision`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `6a029b39-3278-56bf-bed2-44b8f4c98eab`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "Payment Services  The Company offers payment services, including Apple Card®, a co-branded credit card, and Apple Pay®, a cashless payment service.  S..."
    - **Verbatim Source Quote**: "The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific."

### Item `golden_28` (Apple FY2025 - `numeric_precision`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `681f8256-617a-56b3-8afb-878e2406935e`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company’s customers are primarily in the consumer, small and mid-sized business, education, enterprise and government markets. The Company sells i..."
    - **Verbatim Source Quote**: "During 2025, the Company’s net sales through its direct and indirect distribution channels accounted for 40% and 60%, respectively, of total net sales."

### Item `golden_29` (Apple FY2025 - `numeric_precision`)
- **Status**: `CONFIDENT_MATCH` (Score: 1.00)
- **Notes**: Confident match found.
- **Chosen Chunk(s)**:
  - **Chunk ID**: `dfa8abbf-4dda-57f5-b67f-6ada36cf6dae`
    - **Item Section**: `Item 1`
    - **150-char Preview**: "The Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom..."
    - **Verbatim Source Quote**: "The Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom components available from only one source."

### Item `golden_30` (Netflix FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_31` (Netflix FY2024 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

### Item `golden_32` (Meta FY2023 - `unanswerable`)
- **Status**: `UNANSWERABLE` (Score: 1.00)
- **Notes**: Unanswerable baseline outside corpus; source_chunk_ids empty.
- **Source Chunk IDs**: `[]` (None)
- **Source Quotes**: `[]`

---

## 3. Flagged Items Requiring Manual Review

- **`golden_10`** (Apple FY2023): FLAGGED: Text located in raw 2023 Form 10-K preface outside ingested Items 1, 1A, 7, 8. Needs manual fix.