"""
serve.py — FastAPI serving layer for Project Nova.

Three endpoints:
    GET  /ping                — health check / server wake
    POST /recommend           — hybrid CB/CF recommendations
    GET  /experiment/results  — pre-computed A/B + LTV results

Artifacts are downloaded from Hugging Face Hub on first startup if not
already present on disk. Subsequent restarts use the cached files.

Environment variables (set in Railway dashboard):
    HF_TOKEN              Hugging Face read token (required for private repo)
    HF_REPO               HF dataset repo id (default: mqweb/novametrics-artifacts)
    ARTIFACTS_DIR         Local dir to store artifacts (default: /app/artifacts)
    TMDB_API_KEY          TMDB API key for poster URLs
    GRAD_THRESHOLD        CB->CF graduation threshold (default: 10)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))
from model_cb import load_index, query_index
from model_cf import load_svd_model, predict_for_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_REPO       = os.getenv("HF_REPO",  "mqweb/novametrics-artifacts")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))

TMDB_API_KEY  = os.getenv("TMDB_API_KEY", "")
GRAD_THRESHOLD = int(os.getenv("GRAD_THRESHOLD", "10"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CANDIDATES    = 500

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# Files to download from HF — (hf_filename, local_filename)
HF_FILES = [
    ("faiss_index.bin",          "faiss_index.bin"),
    ("asin_index.npy",           "asin_index.npy"),
    ("embeddings.npy",           "embeddings.npy"),
    ("svd_model.pkl",            "svd_model.pkl"),
    ("meta_clean_single.parquet","meta_clean_single.parquet"),
    ("experiment_results.json",  "experiment_results.json"),
]

state: dict = {}


# ── Hugging Face download ─────────────────────────────────────────────────────
def _hf_url(filename: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"


def _download_artifact(filename: str, dest: Path) -> None:
    if dest.exists():
        log.info("Already cached: %s", dest.name)
        return

    url = _hf_url(filename)
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    log.info("Downloading %s from HF...", filename)
    t0 = time.time()

    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    log.info("  %s: %.1f%%", filename, pct)

    log.info("Downloaded %s in %.0f s", filename, time.time() - t0)


def download_all_artifacts() -> None:
    log.info("Checking artifacts in %s ...", ARTIFACTS_DIR)
    for hf_name, local_name in HF_FILES:
        _download_artifact(hf_name, ARTIFACTS_DIR / local_name)
    log.info("All artifacts ready.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_tmdb_poster(title: str) -> Optional[str]:
    if not TMDB_API_KEY or not title:
        return None
    try:
        resp = requests.get(
            TMDB_SEARCH_URL,
            params={"api_key": TMDB_API_KEY, "query": title, "language": "en-US"},
            timeout=5,
        )
        results = resp.json().get("results", [])
        if results and results[0].get("poster_path"):
            return TMDB_IMAGE_BASE + results[0]["poster_path"]
    except Exception:
        pass
    return None


def _meta_lookup(parent_asin: str) -> dict:
    if parent_asin in state["meta"].index:
        row = state["meta"].loc[parent_asin]
        return {
            "title":  str(row.get("title") or "Unknown"),
            "genres": str(row.get("genres_str") or ""),
        }
    return {"title": "Unknown", "genres": ""}


def _cb_recommend(seed_titles: list[str], n: int) -> list:
    meta = state["meta"]
    seed_indices = []
    for title in seed_titles:
        matches = meta[meta["title"].str.contains(title, case=False, na=False)]
        if not matches.empty:
            asin = matches.index[0]
            idx  = state["asin_to_idx"].get(asin)
            if idx is not None:
                seed_indices.append(idx)

    if not seed_indices:
        seed_indices = [0]

    query_vec = state["embeddings"][seed_indices].mean(axis=0).reshape(1, -1).astype(np.float32)
    dists, idxs = query_index(state["index"], query_vec, k=n + len(seed_indices) + 5)

    results = []
    seen = set(seed_indices)
    for dist, idx in zip(dists[0], idxs[0]):
        if int(idx) in seen:
            continue
        asin = state["all_asins"][int(idx)]
        m    = _meta_lookup(asin)
        results.append({
            "movie_id":   asin,
            "title":      m["title"],
            "poster_url": _get_tmdb_poster(m["title"]),
            "score":      round(float(1 / (1 + dist)), 4),
            "reason":     "Similar to your watched movies based on genre and style",
        })
        if len(results) == n:
            break
    return results


def _cf_recommend(watched_movies: list[str], n: int) -> list:
    import hashlib
    proxy_user_id = hashlib.md5(
        "|".join(sorted(watched_movies)).encode()
    ).hexdigest()

    rng        = np.random.default_rng(42)
    candidates = rng.choice(
        state["all_asins"],
        size=min(N_CANDIDATES, len(state["all_asins"])),
        replace=False
    ).tolist()

    preds = predict_for_user(state["svd"], proxy_user_id, candidates, n=n)
    results = []
    for p in preds:
        m = _meta_lookup(p["parent_asin"])
        results.append({
            "movie_id":   p["parent_asin"],
            "title":      m["title"],
            "poster_url": _get_tmdb_poster(m["title"]),
            "score":      p["score"],
            "reason":     "Recommended based on your viewing history",
        })
    return results


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up...")
    try:
        # Step 1 — download from HF if needed
        download_all_artifacts()

        # Step 2 — load into memory
        state["index"]       = load_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
        state["all_asins"]   = np.load(str(ARTIFACTS_DIR / "asin_index.npy"), allow_pickle=True).tolist()
        state["asin_to_idx"] = {a: i for i, a in enumerate(state["all_asins"])}

        log.info("Loading embeddings...")
        state["embeddings"]  = np.load(str(ARTIFACTS_DIR / "embeddings.npy")).astype(np.float32)
        log.info("Embeddings: %s", state["embeddings"].shape)

        state["meta"] = pd.read_parquet(
            str(ARTIFACTS_DIR / "meta_clean_single.parquet"),
            columns=["parent_asin", "title", "genres_str", "description_str"]
        ).set_index("parent_asin")

        state["svd"] = load_svd_model(str(ARTIFACTS_DIR / "svd_model.pkl"))

        with open(ARTIFACTS_DIR / "experiment_results.json") as f:
            state["experiment_results"] = json.load(f)

        from sentence_transformers import SentenceTransformer
        state["encoder"] = SentenceTransformer(EMBEDDING_MODEL)

        state["ready"] = True
        log.info("All artifacts loaded successfully.")

    except Exception as e:
        log.error("Startup failed: %s", e)
        state["ready"] = False

    yield
    state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Project Nova — Movie Personalisation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    watched_movies: list[str]
    n_items: Optional[int] = 5


class RecommendItem(BaseModel):
    movie_id:   str
    title:      str
    poster_url: Optional[str]
    score:      float
    reason:     str


@app.get("/ping")
def ping():
    return {"status": "ok", "ready": state.get("ready", False)}


@app.post("/recommend", response_model=list[RecommendItem])
def recommend(req: RecommendRequest):
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Artifacts not loaded yet.")
    n = req.n_items or 5
    if len(req.watched_movies) < GRAD_THRESHOLD:
        log.info("CB path: %d movies", len(req.watched_movies))
        return _cb_recommend(req.watched_movies, n)
    else:
        log.info("CF path: %d movies", len(req.watched_movies))
        return _cf_recommend(req.watched_movies, n)


@app.get("/experiment/results")
def experiment_results():
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Artifacts not loaded yet.")
    return state["experiment_results"]
