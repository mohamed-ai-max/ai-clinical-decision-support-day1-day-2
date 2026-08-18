# 🏥 AI Clinical Decision Support Lite — Hackathon Project

Clinical **RAG** pipeline for hypertension decision support on 2 premier WHO guidelines — evidence-grounded, citation-traceable, empirically benchmarked.

---

## Key Accomplishments

### Day 1
- **104 pages → 190 chunks** (PDF header cleaning + 500/75 token chunking, ~15% overlap)
- Local embeddings: `BAAI/bge-small-en-v1.5` via FastEmbed
- Stable per-page `chunk_id`, idempotent Chroma index

### Day 2
- **K=3** selected: **63.3% Precision@3 · 100% PageRecall@3 · 100% DocHit@3 · ~8ms**
- **Anti-overfitting suite:** paraphrase held-out set → **80% PageRecall@3** (honest generalization signal)
- **8 negative controls:** 100% abstain rate (top score below threshold)
- Embedding + architecture benchmarks, explainability wrapper, automated DoD scripts

---

## Quick Start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

| Script | Purpose |
| :--- | :--- |
| `verify_dod.py` | Day 1 DoD (12 checks) |
| `verify_day2_dod.py` | Day 2 DoD (21 checks) |
| `evaluate_retrieval.py` | Top-K metrics + negatives |
| `evaluate_robustness.py` | Paraphrase + hard-negative anti-overfitting test |
| `evaluate_embeddings.py` | 3-model embedding comparison |
| `dense_reranker.py` | Dense fetch-10 → re-rank to top-3 |

---

## Verified Metrics (190 chunks, K=3)

### Primary benchmark (`evaluation_set.py`, n=10)

| K | Precision@K | PageRecall@K | DocHit@K | Latency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 80.0% | 80.0% | 100% | 8.9ms |
| **3** | **63.3%** | **100%** | **100%** | **8.3ms** |
| 4 | 55.0% | 100% | 100% | 9.2ms |

### Robustness benchmark (`evaluation_set_robustness.py`, held-out wording)

| Set | Precision@3 | PageRecall@3 | DocHit@3 |
| :--- | ---: | ---: | ---: |
| Primary (development) | 63.3% | 100% | 100% |
| **Paraphrase (anti-leakage)** | **43.3%** | **80%** | **100%** |
| Hard negatives (n=8) | — | — | 100% abstain |

> **For judges:** 100% on the primary set is a **small local benchmark** (10 queries, 190 chunks). Paraphrase recall is the generalization metric.

### Embedding benchmark

| Model | P@3 | PageRecall@3 | Latency |
| :--- | ---: | ---: | ---: |
| **bge-small (selected)** | **63.3%** | **100%** | **~11ms** |
| bge-base | 50.0% | 80% | 26.6ms |
| MiniLM-L6-v2 | 23.3% | 60% | 15.7ms |

---

## Active Corpus

| File | Pages | Chunks |
| :--- | ---: | ---: |
| Guideline for the pharmacological treatment of hypertension in adults.pdf | 61 | 114 |
| WHO-NMH-NVI-18.2-eng.pdf | 43 | 76 |
| **Total** | **104** | **190** |

---
### 📦 Day 2 Deliverables (تسليمات اليوم الثاني)

- **The Notebook:** Day2_Retrieval_Optimization.ipynb
- **The Report:** DAY2_REPORT.md
- **Evaluation Data:** evaluation_set.py
- **Validation Scripts:** evaluate_retrieval.py, chunk_ablation.py, evaluate_embeddings.py, erify_day2_dod.py

*AI Clinical Decision Support Lite — Insight Ai Team*
