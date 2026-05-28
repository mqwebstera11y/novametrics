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
from fastapi.responses import HTMLResponse
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
    dists, idxs = query_index(state["index"], query_vec, k=n * 10 + len(seed_indices))

    results = []
    seen_indices = set(seed_indices)
    seen_titles  = set()

    def _norm_title(t: str) -> str:
        """Normalise title for dedup — lowercase, strip edition/format suffixes."""
        import re
        t = t.lower().strip()
        # Remove common format suffixes
        t = re.sub(r'\s*[\(\[].*?[\)\]]', '', t)   # strip (anything) and [anything]
        t = re.sub(r'\s*(dvd|blu.ray|4k|uhd|steelbook|widescreen|special edition|limited edition|bonus content|digital|theatrical)\s*', '', t, flags=re.I)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    for dist, idx in zip(dists[0], idxs[0]):
        if int(idx) in seen_indices:
            continue
        asin  = state["all_asins"][int(idx)]
        m     = _meta_lookup(asin)
        norm  = _norm_title(m["title"])

        # Skip if normalised title already shown
        if norm in seen_titles:
            continue

        seen_indices.add(int(idx))
        seen_titles.add(norm)

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


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Project Nova</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #131a28;
    --border:   #1e2d45;
    --gold:     #c9a84c;
    --gold2:    #e8c97a;
    --text:     #e8e4dc;
    --muted:    #7a8499;
    --green:    #4caf7d;
    --red:      #e05c5c;
    --blue:     #4a8fe8;
  }

  html { scroll-behavior: smooth; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── GRAIN OVERLAY ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
  }

  /* ── HERO ── */
  .hero {
    position: relative;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    background:
      radial-gradient(ellipse 80% 60% at 50% 0%, rgba(201,168,76,0.08) 0%, transparent 70%),
      radial-gradient(ellipse 60% 40% at 80% 100%, rgba(74,143,232,0.06) 0%, transparent 60%);
  }

  .badge {
    display: inline-block;
    border: 1px solid var(--gold);
    color: var(--gold);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 2px;
    margin-bottom: 2rem;
    animation: fadeUp 0.8s ease both;
  }

  h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 8vw, 7rem);
    font-weight: 900;
    line-height: 0.95;
    letter-spacing: -0.02em;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.8s 0.1s ease both;
  }

  h1 span {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 50%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .hero-sub {
    font-size: clamp(1rem, 2vw, 1.2rem);
    color: var(--muted);
    max-width: 560px;
    line-height: 1.7;
    margin-bottom: 3rem;
    animation: fadeUp 0.8s 0.2s ease both;
  }

  .stat-row {
    display: flex;
    gap: 2.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 3.5rem;
    animation: fadeUp 0.8s 0.3s ease both;
  }

  .stat {
    text-align: center;
  }

  .stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--gold);
    display: block;
    line-height: 1;
  }

  .stat-lbl {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.35rem;
  }

  .scroll-hint {
    position: absolute;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    animation: fadeUp 1s 0.6s ease both;
  }

  .scroll-hint::after {
    content: '';
    width: 1px;
    height: 40px;
    background: linear-gradient(to bottom, var(--muted), transparent);
    animation: scrollLine 1.5s ease infinite;
  }

  /* ── DEMO SECTION ── */
  .demo-section {
    position: relative;
    padding: 6rem 2rem;
    max-width: 760px;
    margin: 0 auto;
  }

  .section-label {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
  }

  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 700;
    margin-bottom: 0.75rem;
    line-height: 1.15;
  }

  .section-desc {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 2.5rem;
  }

  /* ── SEARCH ── */
  .search-wrap {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  input[type="text"] {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    padding: 0.9rem 1.2rem;
    border-radius: 4px;
    outline: none;
    transition: border-color 0.2s;
  }

  input[type="text"]:focus { border-color: var(--gold); }
  input[type="text"]::placeholder { color: var(--muted); }

  button.search-btn {
    background: var(--gold);
    color: #080c12;
    border: none;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.9rem 1.8rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    white-space: nowrap;
  }

  button.search-btn:hover { background: var(--gold2); }
  button.search-btn:active { transform: scale(0.98); }
  button.search-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .hint {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 2rem;
  }

  /* ── RESULTS ── */
  #results { min-height: 2rem; }

  .loading {
    color: var(--muted);
    font-size: 0.9rem;
    padding: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .spinner {
    width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--gold);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 1.25rem;
    align-items: flex-start;
    animation: fadeUp 0.4s ease both;
    transition: border-color 0.2s;
  }

  .result-card:hover { border-color: rgba(201,168,76,0.4); }

  .result-poster {
    width: 52px;
    height: 76px;
    object-fit: cover;
    border-radius: 3px;
    flex-shrink: 0;
    background: var(--surface);
  }

  .result-poster-placeholder {
    width: 52px;
    height: 76px;
    background: var(--surface);
    border-radius: 3px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--border);
    font-size: 1.4rem;
  }

  .result-info { flex: 1; min-width: 0; }

  .result-rank {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.3rem;
  }

  .result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 0.4rem;
    word-break: break-word;
  }

  .result-reason {
    font-size: 0.8rem;
    color: var(--muted);
  }

  .result-score {
    font-size: 0.85rem;
    color: var(--gold);
    font-weight: 500;
    white-space: nowrap;
    margin-top: 0.2rem;
  }

  .error-msg {
    color: var(--red);
    font-size: 0.9rem;
    padding: 0.75rem 0;
  }

  /* ── EXPERIMENT SECTION ── */
  .experiment-section {
    position: relative;
    padding: 6rem 2rem;
    max-width: 760px;
    margin: 0 auto;
    border-top: 1px solid var(--border);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 2rem;
  }

  .metric-cell {
    background: var(--card);
    padding: 1.5rem;
  }

  .metric-cell-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }

  .metric-cell-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--gold);
    line-height: 1;
  }

  .metric-cell-val.green { color: var(--green); }
  .metric-cell-val.blue  { color: var(--blue); }

  .metric-cell-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.35rem;
  }

  .verdict-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    padding: 1.25rem 1.5rem;
    border-radius: 4px;
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text);
    font-style: italic;
  }

  /* ── FOOTER ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
  }

  footer a { color: var(--muted); text-decoration: none; }
  footer a:hover { color: var(--gold); }

  /* ── DIVIDER LINE ── */
  .h-line {
    width: 48px;
    height: 1px;
    background: var(--gold);
    margin: 1.5rem 0;
    opacity: 0.6;
  }

  /* ── ANIMATIONS ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  @keyframes scrollLine {
    0%   { opacity: 0; transform: scaleY(0); transform-origin: top; }
    50%  { opacity: 1; transform: scaleY(1); transform-origin: top; }
    100% { opacity: 0; transform: scaleY(1); transform-origin: bottom; }
  }
</style>
</head>
<body>

<!-- ── HERO ── -->
<section class="hero">
  <div class="badge">Project Nova &nbsp;·&nbsp; Movie Personalisation Engine</div>
  <h1>Does personalisation<br><span>move the needle?</span></h1>
  <p class="hero-sub">
    A hybrid recommendation system with an embedded A/B experiment framework
    and 12-month LTV model — built to answer one product question.
  </p>

  <div class="stat-row">
    <div class="stat">
      <span class="stat-val">+3.5pp</span>
      <span class="stat-lbl">Retention lift</span>
    </div>
    <div class="stat">
      <span class="stat-val">p=0.0001</span>
      <span class="stat-lbl">Significance</span>
    </div>
    <div class="stat">
      <span class="stat-val">+$5.93</span>
      <span class="stat-lbl">LTV / user</span>
    </div>
    <div class="stat">
      <span class="stat-val">433K</span>
      <span class="stat-lbl">Items embedded</span>
    </div>
  </div>

  <div class="scroll-hint">Try it below</div>
</section>

<!-- ── DEMO ── -->
<section class="demo-section">
  <p class="section-label">Live Demo</p>
  <h2 class="section-title">Find your next watch</h2>
  <div class="h-line"></div>
  <p class="section-desc">
    Type a movie you've enjoyed. Under 10 titles triggers the content-based engine (FAISS semantic search).
    10 or more switches to collaborative filtering (SVD).
  </p>

  <div class="search-wrap">
    <input type="text" id="movieInput" placeholder="e.g. The Dark Knight, Inception, Parasite"/>
    <button class="search-btn" id="searchBtn">Recommend</button>
  </div>
  <p class="hint">Separate multiple titles with commas</p>

  <div id="results"></div>
</section>

<!-- ── EXPERIMENT ── -->
<section class="experiment-section">
  <p class="section-label">A/B Experiment Results</p>
  <h2 class="section-title">Does it work?</h2>
  <div class="h-line"></div>
  <p class="section-desc">
    Retrospective simulation on 10,000 users. Personalised recommendations
    during onboarding vs. popularity baseline. Day-30 retention measured.
  </p>

  <div class="metrics-grid">
    <div class="metric-cell">
      <div class="metric-cell-label">Control retention</div>
      <div class="metric-cell-val">32.1%</div>
      <div class="metric-cell-sub">Popularity baseline</div>
    </div>
    <div class="metric-cell">
      <div class="metric-cell-label">Treatment retention</div>
      <div class="metric-cell-val green">35.7%</div>
      <div class="metric-cell-sub">Personalised recs</div>
    </div>
    <div class="metric-cell">
      <div class="metric-cell-label">Retention lift</div>
      <div class="metric-cell-val green">+3.5pp</div>
      <div class="metric-cell-sub">p=0.0001, z=3.72</div>
    </div>
    <div class="metric-cell">
      <div class="metric-cell-label">LTV incremental</div>
      <div class="metric-cell-val">+$5.93</div>
      <div class="metric-cell-sub">Per user, 12-month</div>
    </div>
    <div class="metric-cell">
      <div class="metric-cell-label">LTV — control</div>
      <div class="metric-cell-val">$5.46</div>
      <div class="metric-cell-sub">12-month projection</div>
    </div>
    <div class="metric-cell">
      <div class="metric-cell-label">LTV — treatment</div>
      <div class="metric-cell-val green">$11.39</div>
      <div class="metric-cell-sub">12-month projection</div>
    </div>
  </div>

  <div class="verdict-box" id="verdictBox">
    "Treatment improves 30-day retention by +3.5pp (32.1% → 35.7%).
    Result is statistically significant (z=3.72, p=0.0001, α=0.05)."
  </div>
</section>

<!-- ── FOOTER ── -->
<footer>
  <p>Project Nova &nbsp;·&nbsp; Built with FastAPI · FAISS · SVD · Databricks</p>
  <p style="margin-top:0.5rem;">
    Data: <a href="https://amazon-reviews-2023.github.io/" target="_blank">Amazon Reviews 2023</a>
    (McAuley Lab, UCSD) &nbsp;·&nbsp; Non-commercial research use only
  </p>
</footer>

<script>
async function getRecommendations() {
  const input = document.getElementById('movieInput').value.trim();
  if (!input) return;

  const movies = input.split(',').map(m => m.trim()).filter(Boolean);
  const btn    = document.getElementById('searchBtn');
  const res    = document.getElementById('results');

  btn.disabled = true;
  res.innerHTML = '<div class="loading"><div class="spinner"></div>Finding recommendations...</div>';

  try {
    const resp = await fetch('/recommend', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ watched_movies: movies, n_items: 5 })
    });

    if (!resp.ok) {
      const err = await resp.json();
      res.innerHTML = '<p class="error-msg">Error: ' + (err.detail || resp.statusText) + '</p>';
      return;
    }

    const data = await resp.json();

    if (!data.length) {
      res.innerHTML = '<p class="error-msg">No results found. Try different movie titles.</p>';
      return;
    }

    const pipeline = movies.length < 10 ? 'Content-Based (FAISS)' : 'Collaborative Filtering (SVD)';
    let html = '<p class="hint" style="margin-bottom:1rem;">Pipeline: <strong style="color:var(--gold)">' + pipeline + '</strong></p>';

    data.forEach((item, i) => {
      const poster = item.poster_url
        ? '<img class="result-poster" src="' + item.poster_url + '" alt="poster" onerror="this.style.display=`none`;this.nextElementSibling.style.display=`flex`"/><div class="result-poster-placeholder" style="display:none">🎬</div>'
        : '<div class="result-poster-placeholder">🎬</div>';

      html += '<div class="result-card" style="animation-delay:' + (i * 0.07) + 's">' +
        poster +
        '<div class="result-info">' +
          '<div class="result-rank">Rank ' + (i + 1) + '</div>' +
          '<div class="result-title">' + item.title + '</div>' +
          '<div class="result-reason">' + item.reason + '</div>' +
          '<div class="result-score">Score: ' + item.score.toFixed(3) + '</div>' +
        '</div>' +
      '</div>';
    });

    res.innerHTML = html;

  } catch (e) {
    res.innerHTML = '<p class="error-msg">Network error. Is the server running?</p>';
  } finally {
    btn.disabled = false;
  }
}

// Load live experiment results
async function loadExperiment() {
  try {
    const resp = await fetch('/experiment/results');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.verdict) {
      document.getElementById('verdictBox').textContent = '"' + data.verdict + '"';
    }
  } catch(e) {}
}

loadExperiment();

document.getElementById('searchBtn').addEventListener('click', getRecommendations);
document.getElementById('movieInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') getRecommendations();
});
</script>
</body>
</html>""")