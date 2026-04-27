"""
serve.py — FastAPI serving layer for Project Nova.

Three endpoints:
    GET  /ping                — health check / server wake
    POST /recommend           — hybrid CB/CF recommendations
    GET  /experiment/results  — pre-computed A/B + LTV results

All three artifacts are loaded ONCE at startup — nothing recomputes
at inference time.

Environment variables (set in Railway dashboard):
    FAISS_INDEX_PATH      path to faiss_index.bin
    ASIN_INDEX_PATH       path to asin_index.npy
    SVD_MODEL_PATH        path to svd_model.pkl
    META_PATH             path to meta_clean.parquet
    EXPERIMENT_PATH       path to experiment_results.json
    TMDB_API_KEY          TMDB API key for poster URLs
    GRAD_THRESHOLD        CB->CF graduation threshold (default 10)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── src/ is a sibling directory ───────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__)))
from features  import build_embedding_input
from model_cb  import load_index, query_index
from model_cf  import load_svd_model, predict_for_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH",  "artifacts/faiss_index.bin")
ASIN_INDEX_PATH  = os.getenv("ASIN_INDEX_PATH",   "artifacts/asin_index.npy")
SVD_MODEL_PATH   = os.getenv("SVD_MODEL_PATH",    "artifacts/svd_model.pkl")
META_PATH        = os.getenv("META_PATH",          "artifacts/meta_clean.parquet")
EXPERIMENT_PATH  = os.getenv("EXPERIMENT_PATH",   "artifacts/experiment_results.json")
TMDB_API_KEY     = os.getenv("TMDB_API_KEY",      "")
GRAD_THRESHOLD   = int(os.getenv("GRAD_THRESHOLD", "10"))
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
N_CANDIDATES     = 500   # items fed to SVD for re-ranking (CF path)

TMDB_SEARCH_URL  = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE  = "https://image.tmdb.org/t/p/w342"

# ── Global state (loaded once at startup) ─────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all artifacts at startup."""
    log.info("Loading artifacts…")
    t0 = time.time()

    # FAISS index + ASIN lookup
    state["index"]     = load_index(FAISS_INDEX_PATH)
    state["all_asins"] = np.load(ASIN_INDEX_PATH, allow_pickle=True).tolist()
    state["asin_to_idx"] = {a: i for i, a in enumerate(state["all_asins"])}

    # Metadata for title + poster lookup
    state["meta"] = pd.read_parquet(
        META_PATH,
        columns=["parent_asin", "title_final", "genres_final",
                 "description_final", "most_helpful_review"]
    ).set_index("parent_asin")

    # SVD model
    state["svd"] = load_svd_model(SVD_MODEL_PATH)

    # Experiment results
    with open(EXPERIMENT_PATH) as f:
        state["experiment_results"] = json.load(f)

    # Sentence-transformer (lazy import — heavy)
    from sentence_transformers import SentenceTransformer
    state["encoder"] = SentenceTransformer(EMBEDDING_MODEL)

    log.info("All artifacts loaded in %.1f s", time.time() - t0)
    yield
    state.clear()


app = FastAPI(
    title="Project Nova — Movie Personalisation API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    watched_movies: list[str]   # list of movie titles the user has watched
    n_items: Optional[int] = 5  # number of recommendations to return


class RecommendItem(BaseModel):
    movie_id:   str
    title:      str
    poster_url: Optional[str]
    score:      float
    reason:     str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_tmdb_poster(title: str) -> Optional[str]:
    """Fetch poster URL from TMDB for a given title."""
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
    """Return title and genres for a parent_asin."""
    if parent_asin in state["meta"].index:
        row = state["meta"].loc[parent_asin]
        return {
            "title":       str(row.get("title_final") or "Unknown"),
            "genres":      str(row.get("genres_final") or ""),
            "description": str(row.get("description_final") or ""),
            "review":      str(row.get("most_helpful_review") or ""),
        }
    return {"title": "Unknown", "genres": "", "description": "", "review": ""}


def _cb_recommend(seed_titles: list[str], n: int) -> list[RecommendItem]:
    """
    Content-based recommendations via FAISS.

    Strategy: encode the mean of seed movie embeddings, query FAISS,
    return top-n results excluding seed items.
    """
    encoder = state["encoder"]
    index   = state["index"]
    meta    = state["meta"]

    # Find seed ASINs from titles
    seed_asins = []
    for title in seed_titles:
        matches = meta[meta["title_final"].str.contains(title, case=False, na=False)]
        if not matches.empty:
            seed_asins.append(matches.index[0])

    if not seed_asins:
        # Fall back to first item in index
        seed_asins = [state["all_asins"][0]]

    # Build query embedding: mean of seed embeddings
    seed_indices = [state["asin_to_idx"][a] for a in seed_asins
                    if a in state["asin_to_idx"]]
    if not seed_indices:
        raise HTTPException(status_code=404, detail="No matching movies found in index.")

    embeddings = np.load(  # We need embeddings for query — load from disk
        os.getenv("EMBEDDINGS_PATH", "artifacts/embeddings.npy"),
        mmap_mode="r"
    )
    query_vec = embeddings[seed_indices].mean(axis=0).reshape(1, -1).astype(np.float32)

    dists, idxs = query_index(index, query_vec, k=n + len(seed_asins) + 5)

    results = []
    seed_asin_set = set(seed_asins)
    for dist, idx in zip(dists[0], idxs[0]):
        asin = state["all_asins"][int(idx)]
        if asin in seed_asin_set:
            continue
        m = _meta_lookup(asin)
        results.append(RecommendItem(
            movie_id=asin,
            title=m["title"],
            poster_url=_get_tmdb_poster(m["title"]),
            score=round(float(1 / (1 + dist)), 4),
            reason=f"Similar to your watched movies based on genre and style",
        ))
        if len(results) == n:
            break

    return results


def _cf_recommend(user_id: str, n: int) -> list[RecommendItem]:
    """
    Collaborative filtering recommendations via SVD.

    Scores N_CANDIDATES random unseen items and returns top-n.
    """
    algo       = state["svd"]
    all_asins  = state["all_asins"]

    # Sample candidates (avoid loading full 200K per request)
    rng        = np.random.default_rng(42)
    candidates = rng.choice(all_asins, size=min(N_CANDIDATES, len(all_asins)),
                            replace=False).tolist()

    preds = predict_for_user(algo, user_id, candidates, n=n)

    results = []
    for p in preds:
        m = _meta_lookup(p["parent_asin"])
        results.append(RecommendItem(
            movie_id=p["parent_asin"],
            title=m["title"],
            poster_url=_get_tmdb_poster(m["title"]),
            score=p["score"],
            reason=f"Recommended based on your viewing history",
        ))
    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    """Health check. Called on Vercel page load to wake the Railway server."""
    return {"status": "ok"}


@app.post("/recommend", response_model=list[RecommendItem])
def recommend(req: RecommendRequest):
    """
    Hybrid CB/CF recommendation endpoint.

    Cold-start (< GRAD_THRESHOLD watched movies) → CB via FAISS.
    Warm user  (>= GRAD_THRESHOLD watched movies) → CF via SVD.
    """
    n = req.n_items or 5
    n_watched = len(req.watched_movies)

    if n_watched < GRAD_THRESHOLD:
        # Cold start — CB pipeline
        log.info("CB path: %d watched movies", n_watched)
        return _cb_recommend(req.watched_movies, n)
    else:
        # Warm user — CF pipeline
        # Use a hash of watched movies as a proxy user_id
        import hashlib
        proxy_user_id = hashlib.md5(
            "|".join(sorted(req.watched_movies)).encode()
        ).hexdigest()
        log.info("CF path: proxy_user_id=%s", proxy_user_id[:8])
        return _cf_recommend(proxy_user_id, n)


@app.get("/experiment/results")
def experiment_results():
    """
    Return pre-computed A/B test simulation results.
    Loaded at startup — no recomputation.
    """
    return state["experiment_results"]
