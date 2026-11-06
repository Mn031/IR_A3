# CS60075 / CS40011: Information Retrieval — Programming Assignment III
## Learning to Rank on the Cranfield Text Collection

**Student Author:** Mayash Nayak  
**Roll Number:** 22CS30064  
**Team Name:** `Generic_group_name`  
**Department:** Department of Computer Science and Engineering, IIT Kharagpur  
**Course Instructor:** Prof. Pabitra Mitra  
**Submission Date:** November 6, 2026  

---

## 1. Project Overview

This project implements an end-to-end supervised **Learning-to-Rank (LTR)** retrieval pipeline on the classic Cranfield aerodynamics text collection utilizing Python and the **PyTerrier** framework.

We systematically implement and evaluate three distinct retrieval systems:
1. **Benchmark Baseline Model:** Robertson-Spärck Jones BM25 probabilistic retrieval ($k_1=1.2, b=0.75$).
2. **Method A (Boolean Weighted Zone Scoring):** Decomposes documents into four zones (Title, Author, Bibliography, Body text) and learns optimal linear base weights via a simplex grid search ($\Delta = 0.05$, 1,771 candidate weight vectors).
3. **Method B (Linear Support Vector Machines):** Extracts 22 fine-grained query-document features via PyTerrier pipelines, learns real-valued feature weights using a soft-margin Linear SVM ($C=0.01$, balanced class weights), and re-ranks candidate documents using the continuous decision function $w \cdot x + b$.

Retrieval performance is evaluated against human relevance judgments using **Precision@k**, **Recall@k**, and **F1 score@k** across cutoffs $k \in \{5, 10, 20\}$, alongside **Mean Average Precision (MAP)** and **Mean Reciprocal Rank (MRR)**.

---

## 2. Directory Structure of Submission Package

```
22CS30064_Generic_group_name_PA3/
│
├── IR_PA3.pdf                          # Technical project report PDF (monochrome report style)
├── README.md                           # Documentation, experimental setup, and execution guide
│
├── results/                            # Test query ranking results (TREC 6-column format)
│   ├── test_results_baseline.txt       # BM25 Baseline test query rankings (4,500 lines)
│   ├── test_results_method_a.txt       # Method A Zone Scoring test query rankings (4,500 lines)
│   ├── test_results_method_b.txt       # Method B Linear SVM test query rankings (4,500 lines)
│   └── test_results_all.txt            # Consolidated metrics, optimal weights, and feature tables
│
├── main.py                             # Master orchestrator script running end-to-end pipeline
├── data_parser.py                      # Custom parser for .T, .A, .B, .W, queries, qrels, 80/20 split
├── indexer.py                          # PyTerrier IterDictIndexer and corpus statistics engine
├── baseline.py                         # BM25 baseline retriever implementation
├── method_a_zone_scoring.py            # Method A: Boolean Weighted Zone Scorer & simplex grid search
├── method_b_svm_ltr.py                 # Method B: 22-Feature extractor & Linear SVM LTR re-ranker
├── evaluation.py                       # Precision@k, Recall@k, F1@k, MAP, MRR & TREC exporter
└── run_on_unknown_queries.py           # Evaluation runner for unseen/unknown test queries
```

---

## 3. Dataset & Data Engineering Details

* **Corpus Size:** 1,400 documents, 225 natural language queries, 1,837 relevance assessments.
* **Train / Test Split Ratio:** Exactly **80% Training (180 queries)** and **20% Testing (45 queries)** using `random_state=42` for exact reproducibility.
  * Training Split: 180 queries, 1,505 positive relevance judgments.
  * Testing Split: 45 queries, 332 positive relevance judgments.
* **Query ID Normalization:** In raw Cranfield data, `cran.qry` numbers queries discontinuously (.I 001 to .I 365, with 225 total queries), whereas `cranqrel` indexes them consecutively from 1 to 225. Our parser implements an explicit 1-to-1 sequential mapping ($i$-th query in `cran.qry` maps to query ID $i$ in `cranqrel`), guaranteeing 100% judgment alignment (1,837 / 1,837 pairs).
* **Binary Relevance Formulation:** Grades in $\{-1, 1, 2, 3, 4\}$ map to $y=1$; unlisted documents map to $y=0$.

---

## 4. Empirical Evaluation Results

Evaluated on the **45 unseen test queries** (macro-averaged across all queries):

| Evaluation Metric | Pure BM25 Baseline | Method A (Weighted Zone) | Method B (Linear SVM LTR) | Relative $\Delta$ (A vs. Base) | Relative $\Delta$ (B vs. Base) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P@5** | **0.4578** | 0.4000 | 0.3867 | -12.63% | -15.53% |
| **P@10** | **0.3089** | 0.2644 | 0.2644 | -14.41% | -14.41% |
| **P@20** | **0.1956** | 0.1678 | 0.1667 | -14.21% | -14.78% |
| **R@5** | **0.3698** | 0.3207 | 0.3259 | -13.28% | -11.87% |
| **R@10** | **0.4812** | 0.4141 | 0.4272 | -13.94% | -11.22% |
| **R@20** | **0.5864** | 0.5265 | 0.5295 | -10.21% | -9.70% |
| **F1@5** | **0.3795** | 0.3338 | 0.3282 | -12.04% | -13.52% |
| **F1@10** | **0.3500** | 0.3033 | 0.3066 | -13.34% | -12.40% |
| **F1@20** | **0.2759** | 0.2416 | 0.2406 | -12.43% | -12.79% |
| **MAP (@100)** | **0.4291** | 0.3473 | 0.3926 | -19.06% | -8.51% |
| **MRR** | 0.8310 | 0.7145 | **0.8346** | -14.02% | **+0.43%** |

---

## 5. Learned Model Parameters

### Method A: Optimal Zone Weights (Simplex Grid Search)
* **Optimization Objective:** Maximize mean F1@10 on 180 training queries.
* **Best Training F1@10:** **0.2976** across 1,771 candidate vectors.
* **Learned Weights:**
  * **Title ($w_T$):** **0.5500** (55%)
  * **Text Body ($w_W$):** **0.4000** (40%)
  * **Bibliography ($w_B$):** **0.0500** (5%)
  * **Author ($w_A$):** **0.0000** (0%)

### Method B: Learned Real-Valued Weights (Linear SVM, $C=0.01$)
Trained on 18,419 query-document candidate pairs with balanced class weights and bounded cosine similarity:
* **Top Positive Drivers:**
  * `vector space model` ($w_{\text{std}} = +0.2171$, $w_{\text{raw}} = +3.169704$)
  * `BM25` ($w_{\text{std}} = +0.2059$, $w_{\text{raw}} = +0.043650$)
  * `mean of term frequency` ($w_{\text{std}} = +0.1949$, $w_{\text{raw}} = +0.323499$)
  * `variance of tf*idf` ($w_{\text{std}} = +0.1780$, $w_{\text{raw}} = +0.008587$)
  * `max norm tf` ($w_{\text{std}} = +0.1270$, $w_{\text{raw}} = +6.223214$)
  * `sum norm tf` ($w_{\text{std}} = +0.0839$, $w_{\text{raw}} = +1.660487$)
  * `min of term frequency` ($w_{\text{std}} = +0.0631$, $w_{\text{raw}} = +0.052758$)
  * `mean of tf*idf` ($w_{\text{std}} = +0.0447$, $w_{\text{raw}} = +0.033947$)
  * `stream length` ($w_{\text{std}} = +0.0369$, $w_{\text{raw}} = +0.000662$)
* **Decision Boundary Offset (Intercept):** $b_{\text{std}} = -0.1166$, $b_{\text{raw}} = +0.233543$.

---

## 6. Execution Instructions

### A. Run Full Pipeline
Executes data parsing, train/test splitting, PyTerrier indexing, BM25 baseline, Method A grid search, Method B SVM LTR, metric evaluation, and exports result text files:
```bash
python main.py --data_dir ./data --test_ratio 0.20 --grid_step 0.05 --svm_c 0.01 --output_dir ./results
```

### B. Evaluate on Arbitrary / Unknown Test Queries
Evaluates trained models on an unseen query file (e.g., for TA evaluation scripts) and saves TREC-formatted ranking results:
```bash
python run_on_unknown_queries.py --queries_file ./data/cran.qry --model all --output_file ./results/evaluation_output.txt
```

### C. Submission Package Verification
All code, result files, and the report (`IR_PA3.pdf`) are bundled into `22CS30064_Generic_group_name_PA3.zip`.

---

## 7. Submission Checklist Verification

According to `PA3.pdf`:
- [x] **Program Files:** All modular Python scripts (`data_parser.py`, `indexer.py`, `baseline.py`, `method_a_zone_scoring.py`, `method_b_svm_ltr.py`, `evaluation.py`, `main.py`, `run_on_unknown_queries.py`).
- [x] **Result Files for Test Queries:** Saved in `results/` (`test_results_baseline.txt`, `test_results_method_a.txt`, `test_results_method_b.txt`, `test_results_all.txt`).
- [x] **Report PDF File:** Titled `IR_PA3.pdf`, containing all 6 required sections:
  1. Group details (Mayash Nayak, 22CS30064, `Generic_group_name`, CS60075 / CS40011)
  2. Problem statement
  3. Implementation details
  4. Plan of experiments
  5. Results
  6. Discussion of results
- [x] **Evaluation Script Readiness:** `run_on_unknown_queries.py` supports arbitrary test queries.

