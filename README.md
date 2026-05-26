# Project Nova — Movie Personalisation & LTV Engine

A hybrid movie recommendation system with an embedded A/B experiment framework and 12-month LTV model. Built to answer one product question: **does better personalisation during the first 30 days increase subscriber lifetime value?**

**Result:** +3.5pp retention lift (p=0.0001) and +$5.93 incremental LTV per user.

---

## Live Demo

**API:** https://novametrics-production.up.railway.app/docs

---

## What This Project Does

1. **Recommends movies** — hybrid content-based / collaborative filtering engine that adapts to how much a user has watched
2. **Measures impact** — retrospective A/B simulation on 10,000 users with statistical significance testing
3. **Quantifies revenue** — 12-month LTV model with CAC decomposition, payback period, and sensitivity analysis

---

## Architecture

```
Amazon Reviews 2023 (McAuley Lab, UCSD)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              DATABRICKS PIPELINE                │
│  Job 1 — Item Embeddings & FAISS Index          │
│  Job 2 — SVD Collaborative Filtering Model      │
│  Job 3 — Cohort Nova Simulation                 │
│  Job 4 — A/B Test & LTV Computation             │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│              FASTAPI SERVING LAYER              │
│  POST /recommend   — hybrid CB/CF endpoint      │
│  GET  /experiment/results — A/B + LTV results   │
│  GET  /ping        — health check               │
└─────────────────────────────────────────────────┘
```

**Recommendation logic:**
- **< 10 watched movies** → Content-based (FAISS semantic search)
- **≥ 10 watched movies** → Collaborative filtering (SVD matrix factorisation)

---

## Repo Structure

```
novametrics/
├── src/
│   ├── features.py       Embedding input construction — four-tier fallback
│   ├── model_cb.py       Content-based pipeline — FAISS build, load, query
│   ├── model_cf.py       Collaborative filtering pipeline — SVD wrapper
│   ├── experiment.py     A/B framework — power calc, assignment, z-test
│   ├── LTV.py            LTV model — survival curves, payback period
│   └── serve.py          FastAPI app — 3 endpoints
├── tests/
│   ├── test_features.py
│   ├── test_LTV.py
│   └── test_experiment.py
├── notebooks/            Databricks job notebooks (Jobs 1–4)
├── docs/
│   └── product_memo.md   One-page business write-up
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```


---

## Stack

Python · FastAPI · FAISS · sentence-transformers · scikit-surprise · Databricks · Docker · Railway · GitHub Actions

---

## Data

Amazon Reviews 2023 — Movies and TV (McAuley Lab, UCSD). Non-commercial research use only.  
TMDB API — poster images and metadata enrichment.

---

## Limitations

This is a retrospective simulation, not a live experiment. Retention is proxied by continued rating activity. Churn rate (5%) and CAC ($40) are documented assumptions. 

---

*Code: non-commercial use only. Data: Amazon Reviews 2023 (McAuley Lab, UCSD) · TMDB API.*
