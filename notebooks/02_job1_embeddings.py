"""
02_job1_embeddings.py — Job 1: Build Content-Based Embeddings & FAISS Index

Inputs  (already on disk from EDA / Job 0):
    /Volumes/movie_recsys/data/outputs/meta_clean.parquet      748,224 rows, key=parent_asin
    /Volumes/movie_recsys/data/outputs/reviews_5core.parquet   7,569,072 rows, key=parent_asin

Outputs:
    /Volumes/movie_recsys/data/outputs/most_helpful.parquet    checkpoint — 200K most-helpful reviews
    /Volumes/movie_recsys/data/outputs/tmdb_enriched.parquet   checkpoint — Tier 4 enrichment
    /Volumes/movie_recsys/data/outputs/embeddings.npy          float32 (n_items, 384)
    /Volumes/movie_recsys/data/outputs/asin_index.npy          parent_asin lookup aligned to embeddings
    /Volumes/movie_recsys/data/outputs/faiss_index.bin         IVF-Flat index

Resumable: re-running skips any stage whose output file already exists.
"""

# Databricks notebook source
import json

# Load secrets from workspace notebook — never committed to git

_secrets = json.loads(
    dbutils.fs.head("dbfs:/Workspace/Users/mqwebster238@gmail.com/secrets.json")
)
TMDB_API_KEY = _secrets["TMDB_API_KEY"]

%pip install sentence-transformers faiss-cpu tqdm requests

import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import requests

sys.path.append('/Workspace/Users/mqwebster238@gmail.com/novametrics/src/')

from features import build_embedding_input, get_embedding_tier
from model_cb import build_faiss_index, save_index, load_index, query_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — all paths and params live here, never hardcoded below
# ---------------------------------------------------------------------------
OUTPUTS_DIR       = "/Volumes/movie_recsys/data/outputs"
META_CLEAN_PATH   = f"{OUTPUTS_DIR}/meta_clean.parquet"
REVIEWS_PATH      = f"{OUTPUTS_DIR}/reviews_5core.parquet"
MOST_HELPFUL_PATH = f"{OUTPUTS_DIR}/most_helpful.parquet"
TMDB_CHECKPOINT   = f"{OUTPUTS_DIR}/tmdb_enriched.parquet"
EMBEDDINGS_PATH   = f"{OUTPUTS_DIR}/embeddings.npy"
ASIN_INDEX_PATH   = f"{OUTPUTS_DIR}/asin_index.npy"
FAISS_INDEX_PATH  = f"{OUTPUTS_DIR}/faiss_index.bin"

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
BATCH_SIZE        = 512        # CONFIG PARAM — safe for CPU memory
N_CLUSTERS        = 256        # CONFIG PARAM — IVF cells for FAISS
MAX_REVIEW_WORDS  = 256        # CONFIG PARAM — word cap on review text
CHECKPOINT_EVERY  = 50         # save embeddings.npy every N batches
LOG_EVERY         = 10         # progress log every N batches

TMDB_SEARCH_URL   = "https://api.themoviedb.org/3/search/movie"
TMDB_SLEEP        = 1.0 / 40  # 40 req/s free-tier rate limit

SPOT_CHECK_TITLES = ["The Dark Knight", "Toy Story", "The Godfather"]
SPOT_CHECK_K      = 5

# ---------------------------------------------------------------------------
# String cleaning — applied after every load and merge
#
# Root cause of AttributeError: pandas fills missing strings with float NaN.
# bool(NaN) is True in Python (NaN != 0), so a naive `value and value.strip()`
# passes the guard and then crashes because floats have no .strip().
#
# _clean_str_cols coerces every string column to str | None immediately after
# data enters the session, before any downstream function touches it.
# ---------------------------------------------------------------------------
_STR_COLS = [
    "title", "genres_str", "description_str", "most_helpful_review",
    "tmdb_title", "tmdb_description", "tmdb_genres",
    "title_final", "genres_final", "description_final",
]


def _clean_str_cols(df: pd.DataFrame, cols: list = _STR_COLS) -> pd.DataFrame:
    """
    Coerce string columns to clean str | None in-place.

    Handles:
      - float NaN  (pandas default for missing strings)
      - the string 'nan'  (str(NaN))
      - empty / whitespace-only strings
      - non-object dtypes (warns and coerces)

    Applied after every pd.read_parquet() and every DataFrame merge.
    """
    for col in cols:
        if col not in df.columns:
            continue
        if df[col].dtype != object:
            log.warning("Column '%s' has dtype %s, expected str/object — coercing.", col, df[col].dtype)
        df[col] = df[col].apply(
            lambda v: None if (
                v is None
                or (isinstance(v, float) and pd.isna(v))
                or str(v).strip().lower() == "nan"
                or str(v).strip() == ""
            ) else str(v).strip()
        )
    return df


def _is_present(v) -> bool:
    """Return True only for non-empty, non-NaN string values."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip()
    return bool(s) and s.lower() != "nan"


def _coalesce(primary, fallback):
    """Return the first present value as a clean string, or None."""
    if _is_present(primary):
        return str(primary).strip()
    if _is_present(fallback):
        return str(fallback).strip()
    return None


# ---------------------------------------------------------------------------
# Step 1 — Load meta_clean and reconstruct meta_with_review
# ---------------------------------------------------------------------------
log.info("Loading meta_clean from %s", META_CLEAN_PATH)
meta = pd.read_parquet(META_CLEAN_PATH)
assert "parent_asin" in meta.columns, "Expected parent_asin in meta_clean.parquet"
_clean_str_cols(meta)
log.info("meta_clean: %d rows | dtypes checked and string cols normalised", len(meta))

if os.path.exists(MOST_HELPFUL_PATH):
    log.info("most_helpful checkpoint found — skipping reviews load.")
    most_helpful = pd.read_parquet(MOST_HELPFUL_PATH)
else:
    log.info("Loading reviews from %s", REVIEWS_PATH)
    reviews = pd.read_parquet(REVIEWS_PATH, columns=["parent_asin", "helpful_vote", "text"])
    log.info("reviews_5core: %d rows — building most_helpful …", len(reviews))
    most_helpful = (
        reviews
        .sort_values("helpful_vote", ascending=False)
        .groupby("parent_asin", as_index=False)
        .first()[["parent_asin", "text"]]
        .rename(columns={"text": "most_helpful_review"})
    )
    del reviews
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    most_helpful.to_parquet(MOST_HELPFUL_PATH, index=False)
    log.info("most_helpful saved to %s", MOST_HELPFUL_PATH)

_clean_str_cols(most_helpful, ["most_helpful_review"])
log.info("most_helpful: %d items, %d with review text",
         len(most_helpful), most_helpful["most_helpful_review"].notna().sum())

# Join → meta_with_review (748,224 rows, left join keeps all meta items)
meta = meta.merge(most_helpful, on="parent_asin", how="left")
_clean_str_cols(meta)
log.info("meta_with_review: %d rows | string cols normalised after merge", len(meta))


# ---------------------------------------------------------------------------
# Step 2 — TMDB enrichment for Tier 4 items
# ---------------------------------------------------------------------------
def _fetch_tmdb(title: str, api_key: str, session: requests.Session) -> dict | None:
    try:
        resp = session.get(
            TMDB_SEARCH_URL,
            params={"api_key": api_key, "query": title, "language": "en-US", "page": 1},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        top = results[0]
        genre_str = "|".join(str(g) for g in top.get("genre_ids", []))
        return {
            "title":       top.get("title", ""),
            "description": top.get("overview", ""),
            "genres":      genre_str,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("TMDB fetch failed for title '%s': %s", title, exc)
        return None


def run_tmdb_enrichment(tier4_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    records = []
    session = requests.Session()
    total   = len(tier4_df)

    for i, (_, row) in enumerate(tier4_df.iterrows()):
        if i % 500 == 0:
            log.info("TMDB enrichment: %d / %d (%.1f%%)", i, total, 100 * i / max(total, 1))
        result = _fetch_tmdb(row["title"] or "", api_key, session)
        records.append({
            "parent_asin":      row["parent_asin"],
            "tmdb_title":       result["title"]       if result else None,
            "tmdb_description": result["description"] if result else None,
            "tmdb_genres":      result["genres"]      if result else None,
        })
        time.sleep(TMDB_SLEEP)

    session.close()
    return pd.DataFrame(records)


# Identify Tier 4 before any enrichment
meta["_emb_input"] = meta.apply(
    lambda r: build_embedding_input(
        r["title"], r["genres_str"], r["description_str"], r["most_helpful_review"],
        max_review_words=MAX_REVIEW_WORDS,
    ),
    axis=1,
)
tier4_mask = meta["_emb_input"].isna()
tier4_df   = meta[tier4_mask].copy()
log.info("Tier 4 items (need TMDB): %d / %d (%.1f%%)",
         len(tier4_df), len(meta), 100 * len(tier4_df) / len(meta))

if os.path.exists(TMDB_CHECKPOINT):
    log.info("TMDB checkpoint found — skipping API loop.")
    tmdb_enriched = pd.read_parquet(TMDB_CHECKPOINT)
else:
    if not TMDB_API_KEY:
        raise EnvironmentError(
            "TMDB_API_KEY not found in secrets.json. "
            "Add it at dbfs:/Workspace/Users/mqwebsters238@gmail.com/secrets.json"
        )
    log.info("Starting TMDB enrichment for %d items …", len(tier4_df))
    tmdb_enriched = run_tmdb_enrichment(tier4_df, TMDB_API_KEY)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tmdb_enriched.to_parquet(TMDB_CHECKPOINT, index=False)
    log.info("TMDB checkpoint saved to %s", TMDB_CHECKPOINT)

_clean_str_cols(tmdb_enriched, ["tmdb_title", "tmdb_description", "tmdb_genres"])


# ---------------------------------------------------------------------------
# Step 3 — Merge TMDB enrichment, coalesce into final columns
# ---------------------------------------------------------------------------
meta = meta.merge(tmdb_enriched, on="parent_asin", how="left")
_clean_str_cols(meta)   # normalise tmdb_* cols before coalescing

meta["title_final"]       = meta.apply(lambda r: _coalesce(r["title"],           r.get("tmdb_title")),       axis=1)
meta["genres_final"]      = meta.apply(lambda r: _coalesce(r["genres_str"],      r.get("tmdb_genres")),      axis=1)
meta["description_final"] = meta.apply(lambda r: _coalesce(r["description_str"], r.get("tmdb_description")), axis=1)

# Final normalisation pass on the coalesced columns
_clean_str_cols(meta, ["title_final", "genres_final", "description_final"])

log.info(
    "Post-TMDB coverage — title: %.1f%%, genres: %.1f%%, description: %.1f%%",
    meta["title_final"].notna().mean() * 100,
    meta["genres_final"].notna().mean() * 100,
    meta["description_final"].notna().mean() * 100,
)


# ---------------------------------------------------------------------------
# Step 4 — Build embedding input strings & report tier distribution
# ---------------------------------------------------------------------------
meta["embedding_input"] = meta.apply(
    lambda r: build_embedding_input(
        r["title_final"], r["genres_final"],
        r["description_final"], r["most_helpful_review"],
        max_review_words=MAX_REVIEW_WORDS,
    ),
    axis=1,
)
meta["embedding_tier"] = meta.apply(
    lambda r: get_embedding_tier(
        r["title_final"], r["genres_final"],
        r["description_final"], r["most_helpful_review"],
    ),
    axis=1,
)

tier_counts = meta["embedding_tier"].value_counts().sort_index()
log.info("Tier distribution after TMDB enrichment:")
for tier, count in tier_counts.items():
    log.info("  Tier %d: %6d items (%5.1f%%)", tier, count, 100 * count / len(meta))

true_gaps  = meta["embedding_input"].isna()
embeddable = meta[~true_gaps].reset_index(drop=True)
log.info("True gaps after TMDB: %d — skipped. Items to embed: %d", true_gaps.sum(), len(embeddable))


# ---------------------------------------------------------------------------
# Step 5 — Generate embeddings in batches (resumable)
# ---------------------------------------------------------------------------
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(ASIN_INDEX_PATH):
    log.info("Embeddings checkpoint found — skipping embedding generation.")
    all_embeddings = np.load(EMBEDDINGS_PATH)
    all_asins      = np.load(ASIN_INDEX_PATH, allow_pickle=True)
    log.info("Loaded embeddings shape %s, %d items", all_embeddings.shape, len(all_asins))
    total_time = None
else:
    from sentence_transformers import SentenceTransformer

    log.info("Loading model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts     = embeddable["embedding_input"].tolist()
    asins     = embeddable["parent_asin"].tolist()
    n         = len(texts)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    log.info("Embedding %d items in %d batches (batch_size=%d)", n, n_batches, BATCH_SIZE)
    all_embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    t_start = time.time()

    for batch_idx in range(n_batches):
        lo = batch_idx * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, n)

        all_embeddings[lo:hi] = model.encode(
            texts[lo:hi],
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == n_batches - 1:
            elapsed  = time.time() - t_start
            rate     = hi / elapsed if elapsed > 0 else 0
            eta_secs = (n - hi) / rate if rate > 0 else 0
            log.info(
                "Batch %d/%d | items %d–%d | %.0f items/s | ETA %.0f s",
                batch_idx + 1, n_batches, lo, hi - 1, rate, eta_secs,
            )

        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            np.save(EMBEDDINGS_PATH, all_embeddings[:hi])
            np.save(ASIN_INDEX_PATH, np.array(asins[:hi], dtype=object))
            log.info("Checkpoint saved at batch %d (%d items)", batch_idx + 1, hi)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    all_asins  = np.array(asins, dtype=object)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    np.save(ASIN_INDEX_PATH, all_asins)
    total_time = time.time() - t_start
    log.info("Embedding complete: %d items in %.1f s (%.0f items/s)", n, total_time, n / total_time)


# ---------------------------------------------------------------------------
# Step 6 — Build FAISS index
# ---------------------------------------------------------------------------
n_clusters_actual = min(N_CLUSTERS, len(all_embeddings))
if n_clusters_actual < N_CLUSTERS:
    log.warning("Reducing n_clusters from %d to %d to match embedding count",
                N_CLUSTERS, n_clusters_actual)
log.info("Building FAISS IVF-Flat index: n_items=%d, dim=%d, n_clusters=%d",
         len(all_embeddings), EMBEDDING_DIM, n_clusters_actual)

index = build_faiss_index(all_embeddings, n_clusters=n_clusters_actual)
save_index(index, FAISS_INDEX_PATH)
log.info("FAISS index saved to %s | ntotal=%d", FAISS_INDEX_PATH, index.ntotal)


# ---------------------------------------------------------------------------
# Step 7 — Validation
# ---------------------------------------------------------------------------
index = load_index(FAISS_INDEX_PATH)
assert index.ntotal == len(all_embeddings), (
    f"Index ntotal ({index.ntotal}) != embeddings ({len(all_embeddings)})"
)
log.info("Index integrity check passed: ntotal=%d", index.ntotal)

asin_to_idx   = {asin: i for i, asin in enumerate(all_asins)}
asin_to_title = meta.set_index("parent_asin")["title_final"].fillna("(unknown)").to_dict()

log.info("Spot-check: top-%d neighbours", SPOT_CHECK_K)
for seed_title in SPOT_CHECK_TITLES:
    matches = meta[meta["title_final"].str.contains(seed_title, case=False, na=False)]
    if matches.empty:
        log.info("  '%s': not found in metadata — skipping", seed_title)
        continue
    seed_asin = matches.iloc[0]["parent_asin"]
    seed_idx  = asin_to_idx.get(seed_asin)
    if seed_idx is None:
        log.info("  '%s': not in embedded set — skipping", seed_title)
        continue
    distances, indices = query_index(
        index, all_embeddings[seed_idx : seed_idx + 1], k=SPOT_CHECK_K + 1
    )
    log.info("  Seed: '%s' (%s)", asin_to_title.get(seed_asin, seed_asin), seed_asin)
    for rank, (dist, neighbour_idx) in enumerate(zip(distances[0], indices[0])):
        if neighbour_idx == seed_idx:
            continue
        neighbour_asin  = all_asins[int(neighbour_idx)]
        neighbour_title = asin_to_title.get(neighbour_asin, "(unknown)")
        log.info("    %d. %s  [L2=%.4f]", rank, neighbour_title, dist)


# ---------------------------------------------------------------------------
# Step 8 — Summary
# ---------------------------------------------------------------------------
tier_labels = {
    1: "Full (title+genres+desc+review)",
    2: "Good (title+genres+desc)",
    3: "Thin (title+genres)",
    4: "Bridge (TMDB)",
}
index_size_mb        = os.path.getsize(FAISS_INDEX_PATH) / (1024 ** 2)
embedding_time_label = f"{total_time:.1f} s" if total_time else "N/A (loaded from checkpoint)"

log.info("═" * 60)
log.info("JOB 1 SUMMARY")
log.info("═" * 60)
for t, count in meta["embedding_tier"].value_counts().sort_index().items():
    log.info("  Tier %d — %-35s : %6d (%.1f%%)", t, tier_labels.get(t, ""), count, 100 * count / len(meta))
log.info("Total items embedded : %d", index.ntotal)
log.info("True gaps (skipped)  : %d", true_gaps.sum())
log.info("Embedding time       : %s", embedding_time_label)
log.info("FAISS index size     : %.1f MB", index_size_mb)
log.info("═" * 60)
print("Job 1 complete. Proceed to Job 2 (SVD training).")
