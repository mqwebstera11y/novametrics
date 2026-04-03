# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Job 1 — Build Content-Based Embeddings & FAISS Index
# MAGIC
# MAGIC **Purpose**
# MAGIC Generate sentence-transformer embeddings for all ~200K Movies & TV items
# MAGIC and build a FAISS IVF-Flat index for sub-10ms cold-start recommendation
# MAGIC queries.
# MAGIC
# MAGIC **Inputs**
# MAGIC | Path | Description |
# MAGIC |------|-------------|
# MAGIC | `/Volumes/movie_recsys/data/outputs/meta_clean.parquet` | Cleaned item metadata (200K rows, `asin` as key) |
# MAGIC
# MAGIC **Outputs**
# MAGIC | Path | Description |
# MAGIC |------|-------------|
# MAGIC | `/Volumes/movie_recsys/data/outputs/tmdb_enriched.parquet` | TMDB-enriched metadata for Tier 4 items (checkpoint) |
# MAGIC | `/Volumes/movie_recsys/data/outputs/embeddings.npy` | Float32 embedding matrix, shape (n_items, 384) |
# MAGIC | `/Volumes/movie_recsys/data/outputs/asin_index.npy` | ASIN lookup array aligned with embeddings.npy rows |
# MAGIC | `/Volumes/movie_recsys/data/outputs/faiss_index.bin` | Trained IVF-Flat FAISS index |
# MAGIC
# MAGIC **Resumability**
# MAGIC - TMDB enrichment: skipped if `tmdb_enriched.parquet` already exists.
# MAGIC - Embedding generation: skipped if `embeddings.npy` already exists (jumps to index build).
# MAGIC - Checkpoint saved every 50 batches so a cluster timeout loses at most
# MAGIC   50 × 512 = 25,600 items of work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & Config

# COMMAND ----------

import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import requests

# Make src/ importable in Databricks (not on sys.path by default)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.features import build_embedding_input, get_embedding_tier
from src.model_cb import build_faiss_index, save_index, load_index, query_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths (never hardcoded elsewhere in this notebook) ────────────────────────
OUTPUTS_DIR        = "/Volumes/movie_recsys/data/outputs"
META_CLEAN_PATH    = f"{OUTPUTS_DIR}/meta_clean.parquet"
TMDB_CHECKPOINT    = f"{OUTPUTS_DIR}/tmdb_enriched.parquet"
EMBEDDINGS_PATH    = f"{OUTPUTS_DIR}/embeddings.npy"
ASIN_INDEX_PATH    = f"{OUTPUTS_DIR}/asin_index.npy"
FAISS_INDEX_PATH   = f"{OUTPUTS_DIR}/faiss_index.bin"

# ── Model & index config (CONFIG PARAMs — edit here, not inline) ──────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # sentence-transformers model name
EMBEDDING_DIM      = 384                   # output dimension of the model above
BATCH_SIZE         = 512                   # items per encode() call (CPU-safe)
N_CLUSTERS         = 256                   # IVF cells; ~sqrt(200K) ≈ 450, 256 is conservative
MAX_REVIEW_WORDS   = 256                   # word cap on review text (CONFIG PARAM)
CHECKPOINT_EVERY   = 50                    # save embeddings.npy every N batches
LOG_EVERY          = 10                    # progress log every N batches

# ── TMDB config ───────────────────────────────────────────────────────────────
TMDB_API_KEY       = os.environ.get("TMDB_API_KEY")           # never hardcode
TMDB_SEARCH_URL    = "https://api.themoviedb.org/3/search/movie"
TMDB_RATE_LIMIT    = 40                    # requests per second (free tier)
TMDB_SLEEP         = 1.0 / TMDB_RATE_LIMIT  # seconds between requests

# ── Spot-check seeds (Cell 8) ─────────────────────────────────────────────────
SPOT_CHECK_TITLES  = ["The Dark Knight", "Toy Story", "The Godfather"]
SPOT_CHECK_K       = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load meta_clean.parquet

# COMMAND ----------

log.info("Loading metadata from %s", META_CLEAN_PATH)
meta = pd.read_parquet(META_CLEAN_PATH)
log.info("Loaded %d items. Columns: %s", len(meta), list(meta.columns))

# Normalise expected column names (defensive — adapt if upstream changes)
assert "asin" in meta.columns, "Expected 'asin' column in meta_clean.parquet (parent_asin fix applied in EDA)"

# Ensure string columns are str (not float NaN) for downstream functions
for col in ["title", "genres", "description", "review_text"]:
    if col in meta.columns:
        meta[col] = meta[col].where(meta[col].notna(), other=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — TMDB Enrichment for Tier 4 Items

# COMMAND ----------

def _fetch_tmdb(title: str, api_key: str, session: requests.Session) -> dict | None:
    """
    Query TMDB /search/movie for ``title`` and return a dict with
    ``title``, ``description``, and ``genres`` from the top result,
    or None if no result is found or the request fails.
    """
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
        # genre_ids are integers; map to names via /genre/movie/list if needed.
        # For the embedding input we use the raw genre_ids as a string — they
        # carry signal even without name mapping and avoid an extra API call.
        genre_str = "|".join(str(g) for g in top.get("genre_ids", []))
        return {
            "title": top.get("title", ""),
            "description": top.get("overview", ""),
            "genres": genre_str,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("TMDB fetch failed for title '%s': %s", title, exc)
        return None


def run_tmdb_enrichment(tier4_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Fetch TMDB metadata for all rows in ``tier4_df`` and return an enriched
    DataFrame with columns ``asin``, ``tmdb_title``, ``tmdb_description``,
    ``tmdb_genres``.
    """
    records = []
    session = requests.Session()
    total = len(tier4_df)

    for i, (_, row) in enumerate(tier4_df.iterrows()):
        if i % 500 == 0:
            log.info("TMDB enrichment: %d / %d (%.1f%%)", i, total, 100 * i / max(total, 1))

        result = _fetch_tmdb(row["title"] or "", api_key, session)
        records.append({
            "asin": row["asin"],
            "tmdb_title": result["title"] if result else None,
            "tmdb_description": result["description"] if result else None,
            "tmdb_genres": result["genres"] if result else None,
        })
        time.sleep(TMDB_SLEEP)

    session.close()
    return pd.DataFrame(records)


# Identify Tier 4 items in the current metadata
meta["_emb_input"] = meta.apply(
    lambda r: build_embedding_input(
        r.get("title"), r.get("genres"), r.get("description"), r.get("review_text"),
        max_review_words=MAX_REVIEW_WORDS,
    ),
    axis=1,
)
tier4_mask = meta["_emb_input"].isna()
tier4_df   = meta[tier4_mask].copy()
log.info("Tier 4 items (need TMDB): %d / %d (%.1f%%)", len(tier4_df), len(meta), 100 * len(tier4_df) / len(meta))

if os.path.exists(TMDB_CHECKPOINT):
    log.info("TMDB checkpoint found at %s — skipping API loop.", TMDB_CHECKPOINT)
    tmdb_enriched = pd.read_parquet(TMDB_CHECKPOINT)
else:
    if not TMDB_API_KEY:
        raise EnvironmentError(
            "TMDB_API_KEY environment variable is not set. "
            "Add it in Databricks: Compute → your cluster → Environment variables."
        )
    log.info("Starting TMDB enrichment for %d items …", len(tier4_df))
    tmdb_enriched = run_tmdb_enrichment(tier4_df, TMDB_API_KEY)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    tmdb_enriched.to_parquet(TMDB_CHECKPOINT, index=False)
    log.info("TMDB checkpoint saved to %s", TMDB_CHECKPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Merge TMDB Enrichment Back into Metadata

# COMMAND ----------

meta = meta.merge(tmdb_enriched, on="asin", how="left")


# For Tier 4 rows: fill missing fields from TMDB columns
def _is_present_val(v) -> bool:
    return bool(v and str(v).strip())


def _coalesce(primary, fallback):
    """Return primary if present, else fallback."""
    if _is_present_val(primary):
        return primary
    return fallback if _is_present_val(fallback) else None


meta["title_final"]       = meta.apply(lambda r: _coalesce(r.get("title"),       r.get("tmdb_title")),       axis=1)
meta["genres_final"]      = meta.apply(lambda r: _coalesce(r.get("genres"),      r.get("tmdb_genres")),      axis=1)
meta["description_final"] = meta.apply(lambda r: _coalesce(r.get("description"), r.get("tmdb_description")), axis=1)

log.info(
    "Post-TMDB field coverage — title: %.1f%%, genres: %.1f%%, description: %.1f%%",
    meta["title_final"].notna().mean() * 100,
    meta["genres_final"].notna().mean() * 100,
    meta["description_final"].notna().mean() * 100,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Build Embedding Input Strings & Report Tier Distribution

# COMMAND ----------

meta["embedding_input"] = meta.apply(
    lambda r: build_embedding_input(
        r.get("title_final"),
        r.get("genres_final"),
        r.get("description_final"),
        r.get("review_text"),
        max_review_words=MAX_REVIEW_WORDS,
    ),
    axis=1,
)

meta["embedding_tier"] = meta.apply(
    lambda r: get_embedding_tier(
        r.get("title_final"),
        r.get("genres_final"),
        r.get("description_final"),
        r.get("review_text"),
    ),
    axis=1,
)

tier_counts = meta["embedding_tier"].value_counts().sort_index()
log.info("Tier distribution after TMDB enrichment:")
for tier, count in tier_counts.items():
    log.info("  Tier %d: %6d items (%5.1f%%)", tier, count, 100 * count / len(meta))

true_gaps = meta["embedding_input"].isna()
log.info(
    "True gaps (still None after TMDB): %d items (%.2f%%) — these will be skipped.",
    true_gaps.sum(), 100 * true_gaps.mean(),
)

# Keep only embeddable rows for the embedding loop
embeddable = meta[~true_gaps].reset_index(drop=True)
log.info("Items to embed: %d", len(embeddable))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Generate Embeddings in Batches

# COMMAND ----------

if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(ASIN_INDEX_PATH):
    log.info(
        "Embeddings checkpoint found at %s — skipping embedding generation.",
        EMBEDDINGS_PATH,
    )
    all_embeddings = np.load(EMBEDDINGS_PATH)
    all_asins      = np.load(ASIN_INDEX_PATH, allow_pickle=True)
    log.info("Loaded embeddings: shape %s, ASINs: %d", all_embeddings.shape, len(all_asins))
else:
    from sentence_transformers import SentenceTransformer

    log.info("Loading sentence-transformer model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts     = embeddable["embedding_input"].tolist()
    asins     = embeddable["asin"].tolist()
    n         = len(texts)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(
        "Starting embedding generation: %d items, batch_size=%d, n_batches=%d",
        n, BATCH_SIZE, n_batches,
    )

    all_embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    t_start = time.time()

    for batch_idx in range(n_batches):
        lo = batch_idx * BATCH_SIZE
        hi = min(lo + BATCH_SIZE, n)
        batch_texts = texts[lo:hi]

        batch_embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        all_embeddings[lo:hi] = batch_embeddings

        if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == n_batches - 1:
            elapsed  = time.time() - t_start
            done     = hi
            rate     = done / elapsed if elapsed > 0 else 0
            eta_secs = (n - done) / rate if rate > 0 else 0
            log.info(
                "Batch %d/%d | items %d–%d | %.0f items/s | ETA %.0f s",
                batch_idx + 1, n_batches, lo, hi - 1, rate, eta_secs,
            )

        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            np.save(EMBEDDINGS_PATH, all_embeddings[:hi])
            np.save(ASIN_INDEX_PATH, np.array(asins[:hi], dtype=object))
            log.info("Checkpoint saved at batch %d (%d items)", batch_idx + 1, hi)

    # Final save
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    all_asins = np.array(asins, dtype=object)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    np.save(ASIN_INDEX_PATH, all_asins)
    total_time = time.time() - t_start
    log.info(
        "Embedding generation complete: %d items in %.1f s (%.0f items/s)",
        n, total_time, n / total_time,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Build FAISS Index

# COMMAND ----------

log.info(
    "Building FAISS IVF-Flat index: n_items=%d, dim=%d, n_clusters=%d",
    len(all_embeddings), EMBEDDING_DIM, N_CLUSTERS,
)

index = build_faiss_index(all_embeddings, n_clusters=N_CLUSTERS)
save_index(index, FAISS_INDEX_PATH)

log.info(
    "FAISS index saved to %s | ntotal=%d | nlist=%d",
    FAISS_INDEX_PATH, index.ntotal, index.nlist,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Validation

# COMMAND ----------

# Reload to prove round-trip serialisation works
index = load_index(FAISS_INDEX_PATH)
assert index.ntotal == len(all_embeddings), (
    f"Index ntotal ({index.ntotal}) != number of embeddings ({len(all_embeddings)}). "
    "This indicates embeddings were added in a different run than the current arrays."
)
log.info("Index integrity check passed: ntotal=%d", index.ntotal)

# Build ASIN → row-index lookup and title lookup for spot-checks
asin_to_idx  = {asin: i for i, asin in enumerate(all_asins)}
idx_to_title = embeddable["title_final"].fillna("(unknown)").to_dict()

log.info("\n── Spot-check: top-%d neighbours ──────────────────────────────", SPOT_CHECK_K)
for seed_title in SPOT_CHECK_TITLES:
    matches = embeddable[embeddable["title_final"].str.contains(seed_title, case=False, na=False)]
    if matches.empty:
        log.info("Seed '%s': not found in embeddable items — skipping.", seed_title)
        continue

    seed_row  = matches.iloc[0]
    seed_asin = seed_row["asin"]
    seed_idx  = asin_to_idx.get(seed_asin)
    if seed_idx is None:
        log.info("Seed '%s' (ASIN %s): index position not found — skipping.", seed_title, seed_asin)
        continue

    query_vec = all_embeddings[seed_idx : seed_idx + 1]  # shape (1, 384)
    distances, indices = query_index(index, query_vec, k=SPOT_CHECK_K + 1)

    log.info("Seed: '%s' (ASIN: %s)", seed_row["title_final"], seed_asin)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == seed_idx:
            continue  # skip the seed itself
        neighbour_title = idx_to_title.get(int(idx), "(unknown)")
        log.info("  %d. %s  [L2=%.4f]", rank, neighbour_title, dist)
    log.info("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Summary Report

# COMMAND ----------

tier_summary = meta["embedding_tier"].value_counts().sort_index().to_dict()
embedding_time_label = f"{total_time:.1f} s" if "total_time" in dir() else "N/A (loaded from checkpoint)"
index_size_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024 ** 2) if os.path.exists(FAISS_INDEX_PATH) else 0

tier_labels = {
    1: "Full (title+genres+desc+review)",
    2: "Good (title+genres+desc)",
    3: "Thin (title+genres)",
    4: "Bridge (TMDB)",
}

log.info("═" * 60)
log.info("JOB 1 SUMMARY")
log.info("═" * 60)
log.info("Items by embedding tier:")
for t, count in tier_summary.items():
    log.info("  Tier %d — %-35s : %6d (%.1f%%)", t, tier_labels.get(t, ""), count, 100 * count / len(meta))
log.info("Total items embedded : %d", index.ntotal)
log.info("True gaps (skipped)  : %d", true_gaps.sum())
log.info("Embedding time       : %s", embedding_time_label)
log.info("FAISS index size     : %.1f MB", index_size_mb)
log.info("Index path           : %s", FAISS_INDEX_PATH)
log.info("Embeddings path      : %s", EMBEDDINGS_PATH)
log.info("═" * 60)
print("Job 1 complete. Proceed to Job 2 (SVD training).")
