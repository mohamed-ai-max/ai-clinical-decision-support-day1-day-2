# 🏥 AI Clinical Decision Support Lite — Hackathon Project

Clinical RAG pipeline for hypertension decision support based on 2 premier WHO guidelines — evidence-grounded, citation-traceable, empirically benchmarked, and strictly schema-validated.

---

## 🚀 Executive Summary & Daily Accomplishments

### 🟢 Day 1: Document Ingestion Pipeline
* **Document Processing:** Ingested 104 pages across 2 core WHO hypertension guidelines in `data/`.
* **Chunking Strategy:** 104 pages → 190 chunks using section-aware `RecursiveCharacterTextSplitter` (500/75 token chunks, ~15% overlap) with stripped PDF header noise.
* **Embeddings & Vector Store:** Generated embeddings locally via `BAAI/bge-small-en-v1.5` (FastEmbed) into a persistent, idempotent ChromaDB collection (`clinical_guidelines`).
* **Metadata & Traceability:** Attached stable per-page 1-indexed metadata (`document_name`, `page_number`, `chunk_id`).

### 🔵 Day 2: Retrieval Optimization & Evaluation
* **Optimal Top-K Selection:** Selected **$K=3$** delivering **63.3% Precision@3**, **100% PageRecall@3**, and **100% DocHit@3** at $\approx 8\text{ ms}$ latency.
* **Ablation & Model Benchmarks:** Verified $500/75$ token configuration against small/large chunks. Validated `bge-small-en-v1.5` outperforming `bge-base` and `MiniLM-L6-v2`.
* **Anti-Overfitting & Negative Controls:** Evaluated against a held-out paraphrase query set (80% PageRecall@3) and 8 hard negative queries (100% abstain rate).

### 🟣 Day 3: Grounded Generation & Citation
* **System Prompt Constraints:** Engineered strict grounding prompt containing all 4 pillars: Role Isolation, Context Boundary Enforcement, Structured JSON Output, and Refusal Escape Hatch.
* **Live LLM Integration:** Integrated live LLM inference using **NVIDIA NIM API** (`meta/llama-3.1-8b-instruct`) via `NV_API_KEY`.
* **JSON Schema Enforcement:** Guaranteed 100% response compliance against `schema/response_schema.json` via Draft7Validator.
* **Citation Traceability:** In-scope clinical questions generate structured JSON responses with explicit chunk citations (e.g. `[chunk_15, chunk_16, chunk_17]`).
* **Abstain Guard:** Out-of-scope medical/general queries trigger automated, schema-valid refusal (`status: "abstain"`, `confidence: "insufficient"`).

---

## ⚡ Quick Start & Environment Setup

### 1. Installation
```bash
# Clone repository
git clone https://github.com/mohamed-ai-max/ai-clinical-decision-support-day1-day-2.git
cd ai-clinical-decision-support-day1-day-2

# Set up virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the root directory:
```bash
NV_API_KEY=your_nvidia_nim_api_key_here
NV_LLM_ENDPOINT=https://integrate.api.nvidia.com/v1/chat/completions
NV_MODEL=meta/llama-3.1-8b-instruct
```

### 3. Master End-to-End Run
Run the full automated verification suite covering Day 1, Day 2, and Day 3:
```bash
python run_all.py
```

---

## 📊 Empirical Benchmarks & Acceptance Criteria Summary

### 1. Primary Retrieval Benchmark (`evaluation_set.py`, $n=10$)

| K | Precision@K | PageRecall@K | DocHit@K | Latency |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 80.0% | 80.0% | 100.0% | 8.9 ms |
| **3** | **63.3%** | **100.0%** | **100.0%** | **8.3 ms** |
| 4 | 55.0% | 100.0% | 100.0% | 9.2 ms |

### 2. Robustness & Anti-Overfitting Benchmark

| Benchmark Set | Precision@3 | PageRecall@3 | DocHit@3 | Abstention Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Primary (Development)** | 63.3% | 100.0% | 100.0% | — |
| **Paraphrase (Held-out wording)** | 43.3% | **80.0%** | 100.0% | — |
| **Hard Negative Controls ($n=8$)** | — | — | — | **100.0% Refusal** |

### 3. Embedding Model Comparison

| Embedding Model | Precision@3 | PageRecall@3 | Latency | Decision |
| :--- | :---: | :---: | :---: | :--- |
| **BAAI/bge-small-en-v1.5** | **63.3%** | **100.0%** | **11.2 ms** | **Selected (Production)** |
| BAAI/bge-base-en-v1.5 | 50.0% | 80.0% | 26.6 ms | Higher latency, lower recall |
| all-MiniLM-L6-v2 | 23.3% | 60.0% | 15.7 ms | Lower precision |

---

## 📁 Repository Architecture & Deliverables

```text
├── data/                                 # Active WHO guidelines (104 pages total)
│   ├── Guideline for the pharmacological treatment of hypertension in adults.pdf
│   └── WHO-NMH-NVI-18.2-eng.pdf
├── reference_candidates/                 # Reference annexes & secondary sources
├── schema/
│   └── response_schema.json              # Strict Draft-07 JSON Response Schema
├── prompt/
│   └── grounding_prompt.txt              # Grounding system prompt template
├── notebooks/                            # Fully executed Jupyter Notebooks
│   ├── day1/Day1_Task1_Document_Ingestion.ipynb
│   ├── day2/Day2_Retrieval_Optimization.ipynb
│   └── Day3/Task3_Grounded_Generation.ipynb
├── Task3_Grounded_Generation.ipynb       # Live LLM Grounded Generation Notebook
├── DAY1_REPORT.md                        # Day 1 Technical Report
├── DAY2_REPORT.md                        # Day 2 Retrieval & Ablation Report
├── DAY3_REPORT.md                        # Day 3 Live LLM Generation & Schema Report
├── ingest.py                             # Document loading, chunking & Chroma indexer
├── query.py                              # Query & live decision support interface
├── retrieval.py                          # Citation wrapper & clinical view formatting
├── generation.py                         # Grounded generation logic & schema validation
├── evaluation_set.py                     # 10 clinical queries + negative test set
├── evaluate_retrieval.py                 # Precision@k & PageRecall@k evaluation suite
├── evaluate_embeddings.py                # Multi-model embedding benchmark
├── evaluate_retrieval_architectures.py   # Dense vs. BM25 vs. Hybrid vs. Rerank
├── evaluate_robustness.py                # Paraphrase & hard-negative evaluation suite
├── verify_dod.py                         # Day 1 Definition of Done verification (12 checks)
├── verify_day2_dod.py                    # Day 2 Definition of Done verification (21 checks)
├── verify_day3_dod.py                    # Day 3 Definition of Done verification (10 checks)
└── run_all.py                            # Master End-to-End Verification Pipeline
```

---

## 🛡️ Definition of Done Verification Suites

Run any daily verification script directly:

* **Day 1 DoD:** `python verify_dod.py` *(12/12 checks passed)*
* **Day 2 DoD:** `python verify_day2_dod.py` *(21/21 checks passed)*
* **Day 3 DoD:** `python verify_day3_dod.py` *(10/10 checks passed)*
* **Master Pipeline:** `python run_all.py` *(All 8 master steps passed)*

---
*AI Clinical Decision Support Lite — AI Max Team*
