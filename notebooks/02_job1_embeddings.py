# Databricks notebook source
# COMMAND ----------
# MAGIC 
%pip install sentence-transformers faiss-cpu tqdm requests

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
# MAGIC
# MAGIC import os
# MAGIC import sys
# MAGIC import time
# MAGIC import logging
# MAGIC
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC import requests
# MAGIC
# MAGIC # Make src/ importable in Databricks (not on sys.path by default)
# MAGIC _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# MAGIC if _repo_root not in sys.path:
# MAGIC     sys.path.insert(0, _repo_root)
# MAGIC
# MAGIC from src.features import build_embedding_input, get_embedding_tier
# MAGIC from src.model_cb import build_faiss_index, save_index, load_index, query_index
# MAGIC
# MAGIC logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# MAGIC log = logging.getLogger(__name__)
# MAGIC
# MAGIC # ── Paths (never hardcoded elsewhere in this notebook) ────────────────────────
# MAGIC OUTPUTS_DIR        = "/Volumes/movie_recsys/data/outputs"
# MAGIC META_CLEAN_PATH    = f"{OUTPUTS_DIR}/meta_clean.parquet"
# MAGIC TMDB_CHECKPOINT    = f"{OUTPUTS_DIR}/tmdb_enriched.parquet"
# MAGIC EMBEDDINGS_PATH    = f"{OUTPUTS_DIR}/embeddings.npy"
# MAGIC ASIN_INDEX_PATH    = f"{OUTPUTS_DIR}/asin_index.npy"
# MAGIC FAISS_INDEX_PATH   = f"{OUTPUTS_DIR}/faiss_index.bin"
# MAGIC
# MAGIC # ── Model & index config (CONFIG PARAMs — edit here, not inline) ──────────────
# MAGIC EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # sentence-transformers model name
# MAGIC EMBEDDING_DIM      = 384                   # output dimension of the model above
# MAGIC BATCH_SIZE         = 512                   # items per encode() call (CPU-safe)
# MAGIC N_CLUSTERS         = 256                   # IVF cells; ~sqrt(200K) ≈ 450, 256 is conservative
# MAGIC MAX_REVIEW_WORDS   = 256                   # word cap on review text (CONFIG PARAM)
# MAGIC CHECKPOINT_EVERY   = 50                    # save embeddings.npy every N batches
# MAGIC LOG_EVERY          = 10                    # progress log every N batches
# MAGIC
# MAGIC # ── TMDB config ───────────────────────────────────────────────────────────────
# MAGIC TMDB_API_KEY       = os.environ.get("TMDB_API_KEY")           # never hardcode
# MAGIC TMDB_SEARCH_URL    = "https://api.themoviedb.org/3/search/movie"
# MAGIC TMDB_RATE_LIMIT    = 40                    # requests per second (free tier)
# MAGIC TMDB_SLEEP         = 1.0 / TMDB_RATE_LIMIT  # seconds between requests
# MAGIC
# MAGIC # ── Spot-check seeds (Cell 8) ─────────────────────────────────────────────────
# MAGIC SPOT_CHECK_TITLES  = ["The Dark Knight", "Toy Story", "The Godfather"]
# MAGIC SPOT_CHECK_K       = 5

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load meta_clean.parquet
# MAGIC
# MAGIC log.info("Loading metadata from %s", META_CLEAN_PATH)
# MAGIC meta = pd.read_parquet(META_CLEAN_PATH)
# MAGIC log.info("Loaded %d items. Columns: %s", len(meta), list(meta.columns))
# MAGIC
# MAGIC # Normalise expected column names (defensive — adapt if upstream changes)
# MAGIC assert "asin" in meta.columns, "Expected 'asin' column in meta_clean.parquet (parent_asin fix applied in EDA)"
# MAGIC
# MAGIC # Ensure string columns are str (not float NaN) for downstream functions
# MAGIC for col in ["title", "genres", "description", "review_text"]:
# MAGIC     if col in meta.columns:
# MAGIC         meta[col] = meta[col].where(meta[col].notna(), other=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — TMDB Enrichment for Tier 4 Items
# MAGIC
# MAGIC def _fetch_tmdb(title: str, api_key: str, session: requests.Session) -> dict | None:
# MAGIC     """
# MAGIC     Query TMDB /search/movie for ``title`` and return a dict with
# MAGIC     ``title``, ``description``, and ``genres`` from the top result,
# MAGIC     or None if no result is found or the request fails.
# MAGIC     """
# MAGIC     try:
# MAGIC         resp = session.get(
# MAGIC             TMDB_SEARCH_URL,
# MAGIC             params={"api_key": api_key, "query": title, "language": "en-US", "page": 1},
# MAGIC             timeout=10,
# MAGIC         )
# MAGIC         resp.raise_for_status()
# MAGIC         results = resp.json().get("results", [])
# MAGIC         if not results:
# MAGIC             return None
# MAGIC         top = results[0]
# MAGIC         # genre_ids are integers; map to names via /genre/movie/list if needed.
# MAGIC         # For the embedding input we use the raw genre_ids as a string — they
# MAGIC         # carry signal even without name mapping and avoid an extra API call.
# MAGIC         genre_str = "|".join(str(g) for g in top.get("genre_ids", []))
# MAGIC         return {
# MAGIC             "title": top.get("title", ""),
# MAGIC             "description": top.get("overview", ""),
# MAGIC             "genres": genre_str,
# MAGIC         }
# MAGIC     except Exception as exc:  # noqa: BLE001
# MAGIC         log.warning("TMDB fetch failed for title '%s': %s", title, exc)
# MAGIC         return None
# MAGIC
# MAGIC
# MAGIC def run_tmdb_enrichment(tier4_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
# MAGIC     """
# MAGIC     Fetch TMDB metadata for all rows in ``tier4_df`` and return an enriched
# MAGIC     DataFrame with columns ``asin``, ``tmdb_title``, ``tmdb_description``,
# MAGIC     ``tmdb_genres``.
# MAGIC     """
# MAGIC     records = []
# MAGIC     session = requests.Session()
# MAGIC     total = len(tier4_df)
# MAGIC
# MAGIC     for i, (_, row) in enumerate(tier4_df.iterrows()):
# MAGIC         if i % 500 == 0:
# MAGIC             log.info("TMDB enrichment: %d / %d (%.1f%%)", i, total, 100 * i / max(total, 1))
# MAGIC
# MAGIC         result = _fetch_tmdb(row["title"] or "", api_key, session)
# MAGIC         records.append({
# MAGIC             "asin": row["asin"],
# MAGIC             "tmdb_title": result["title"] if result else None,
# MAGIC             "tmdb_description": result["description"] if result else None,
# MAGIC             "tmdb_genres": result["genres"] if result else None,
# MAGIC         })
# MAGIC         time.sleep(TMDB_SLEEP)
# MAGIC
# MAGIC     session.close()
# MAGIC     return pd.DataFrame(records)
# MAGIC
# MAGIC
# MAGIC # Identify Tier 4 items in the current metadata
# MAGIC meta["_emb_input"] = meta.apply(
# MAGIC     lambda r: build_embedding_input(
# MAGIC         r.get("title"), r.get("genres"), r.get("description"), r.get("review_text"),
# MAGIC         max_review_words=MAX_REVIEW_WORDS,
# MAGIC     ),
# MAGIC     axis=1,
# MAGIC )
# MAGIC tier4_mask = meta["_emb_input"].isna()
# MAGIC tier4_df   = meta[tier4_mask].copy()
# MAGIC log.info("Tier 4 items (need TMDB): %d / %d (%.1f%%)", len(tier4_df), len(meta), 100 * len(tier4_df) / len(meta))
# MAGIC
# MAGIC if os.path.exists(TMDB_CHECKPOINT):
# MAGIC     log.info("TMDB checkpoint found at %s — skipping API loop.", TMDB_CHECKPOINT)
# MAGIC     tmdb_enriched = pd.read_parquet(TMDB_CHECKPOINT)
# MAGIC else:
# MAGIC     if not TMDB_API_KEY:
# MAGIC         raise EnvironmentError(
# MAGIC             "TMDB_API_KEY environment variable is not set. "
# MAGIC             "Add it in Databricks: Compute → your cluster → Environment variables."
# MAGIC         )
# MAGIC     log.info("Starting TMDB enrichment for %d items …", len(tier4_df))
# MAGIC     tmdb_enriched = run_tmdb_enrichment(tier4_df, TMDB_API_KEY)
# MAGIC     os.makedirs(OUTPUTS_DIR, exist_ok=True)
# MAGIC     tmdb_enriched.to_parquet(TMDB_CHECKPOINT, index=False)
# MAGIC     log.info("TMDB checkpoint saved to %s", TMDB_CHECKPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Merge TMDB Enrichment Back into Metadata
# MAGIC
# MAGIC meta = meta.merge(tmdb_enriched, on="asin", how="left")
# MAGIC
# MAGIC # For Tier 4 rows: fill missing fields from TMDB columns
# MAGIC def _coalesce(primary, fallback):
# MAGIC     """Return primary if present, else fallback."""
# MAGIC     if _is_present_val(primary):
# MAGIC         return primary
# MAGIC     return fallback if _is_present_val(fallback) else None
# MAGIC
# MAGIC def _is_present_val(v) -> bool:
# MAGIC     return bool(v and str(v).strip())
# MAGIC
# MAGIC meta["title_final"]       = meta.apply(lambda r: _coalesce(r.get("title"),       r.get("tmdb_title")),       axis=1)
# MAGIC meta["genres_final"]      = meta.apply(lambda r: _coalesce(r.get("genres"),      r.get("tmdb_genres")),      axis=1)
# MAGIC meta["description_final"] = meta.apply(lambda r: _coalesce(r.get("description"), r.get("tmdb_description")), axis=1)
# MAGIC
# MAGIC log.info(
# MAGIC     "Post-TMDB field coverage — title: %.1f%%, genres: %.1f%%, description: %.1f%%",
# MAGIC     meta["title_final"].notna().mean() * 100,
# MAGIC     meta["genres_final"].notna().mean() * 100,
# MAGIC     meta["description_final"].notna().mean() * 100,
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Build Embedding Input Strings & Report Tier Distribution
# MAGIC
# MAGIC meta["embedding_input"] = meta.apply(
# MAGIC     lambda r: build_embedding_input(
# MAGIC         r.get("title_final"),
# MAGIC         r.get("genres_final"),
# MAGIC         r.get("description_final"),
# MAGIC         r.get("review_text"),
# MAGIC         max_review_words=MAX_REVIEW_WORDS,
# MAGIC     ),
# MAGIC     axis=1,
# MAGIC )
# MAGIC
# MAGIC meta["embedding_tier"] = meta.apply(
# MAGIC     lambda r: get_embedding_tier(
# MAGIC         r.get("title_final"),
# MAGIC         r.get("genres_final"),
# MAGIC         r.get("description_final"),
# MAGIC         r.get("review_text"),
# MAGIC     ),
# MAGIC     axis=1,
# MAGIC )
# MAGIC
# MAGIC tier_counts = meta["embedding_tier"].value_counts().sort_index()
# MAGIC log.info("Tier distribution after TMDB enrichment:")
# MAGIC for tier, count in tier_counts.items():
# MAGIC     log.info("  Tier %d: %6d items (%5.1f%%)", tier, count, 100 * count / len(meta))
# MAGIC
# MAGIC true_gaps = meta["embedding_input"].isna()
# MAGIC log.info(
# MAGIC     "True gaps (still None after TMDB): %d items (%.2f%%) — these will be skipped.",
# MAGIC     true_gaps.sum(), 100 * true_gaps.mean(),
# MAGIC )
# MAGIC
# MAGIC # Keep only embeddable rows for the embedding loop
# MAGIC embeddable = meta[~true_gaps].reset_index(drop=True)
# MAGIC log.info("Items to embed: %d", len(embeddable))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Generate Embeddings in Batches
# MAGIC
# MAGIC if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(ASIN_INDEX_PATH):
# MAGIC     log.info(
# MAGIC         "Embeddings checkpoint found at %s — skipping embedding generation.",
# MAGIC         EMBEDDINGS_PATH,
# MAGIC     )
# MAGIC     all_embeddings = np.load(EMBEDDINGS_PATH)
# MAGIC     all_asins      = np.load(ASIN_INDEX_PATH, allow_pickle=True)
# MAGIC     log.info("Loaded embeddings: shape %s, ASINs: %d", all_embeddings.shape, len(all_asins))
# MAGIC else:
# MAGIC     from sentence_transformers import SentenceTransformer
# MAGIC
# MAGIC     log.info("Loading sentence-transformer model: %s", EMBEDDING_MODEL)
# MAGIC     model = SentenceTransformer(EMBEDDING_MODEL)
# MAGIC
# MAGIC     texts  = embeddable["embedding_input"].tolist()
# MAGIC     asins  = embeddable["asin"].tolist()
# MAGIC     n      = len(texts)
# MAGIC     n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
# MAGIC
# MAGIC     log.info(
# MAGIC         "Starting embedding generation: %d items, batch_size=%d, n_batches=%d",
# MAGIC         n, BATCH_SIZE, n_batches,
# MAGIC     )
# MAGIC
# MAGIC     all_embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
# MAGIC     t_start = time.time()
# MAGIC
# MAGIC     for batch_idx in range(n_batches):
# MAGIC         lo = batch_idx * BATCH_SIZE
# MAGIC         hi = min(lo + BATCH_SIZE, n)
# MAGIC         batch_texts = texts[lo:hi]
# MAGIC
# MAGIC         batch_embeddings = model.encode(
# MAGIC             batch_texts,
# MAGIC             batch_size=BATCH_SIZE,
# MAGIC             show_progress_bar=False,
# MAGIC             convert_to_numpy=True,
# MAGIC             normalize_embeddings=False,
# MAGIC         ).astype(np.float32)
# MAGIC
# MAGIC         all_embeddings[lo:hi] = batch_embeddings
# MAGIC
# MAGIC         if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == n_batches - 1:
# MAGIC             elapsed  = time.time() - t_start
# MAGIC             done     = hi
# MAGIC             rate     = done / elapsed if elapsed > 0 else 0
# MAGIC             eta_secs = (n - done) / rate if rate > 0 else 0
# MAGIC             log.info(
# MAGIC                 "Batch %d/%d | items %d–%d | %.0f items/s | ETA %.0f s",
# MAGIC                 batch_idx + 1, n_batches, lo, hi - 1, rate, eta_secs,
# MAGIC             )
# MAGIC
# MAGIC         if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
# MAGIC             os.makedirs(OUTPUTS_DIR, exist_ok=True)
# MAGIC             np.save(EMBEDDINGS_PATH, all_embeddings[:hi])
# MAGIC             np.save(ASIN_INDEX_PATH, np.array(asins[:hi], dtype=object))
# MAGIC             log.info("Checkpoint saved at batch %d (%d items)", batch_idx + 1, hi)
# MAGIC
# MAGIC     # Final save
# MAGIC     os.makedirs(OUTPUTS_DIR, exist_ok=True)
# MAGIC     all_asins = np.array(asins, dtype=object)
# MAGIC     np.save(EMBEDDINGS_PATH, all_embeddings)
# MAGIC     np.save(ASIN_INDEX_PATH, all_asins)
# MAGIC     total_time = time.time() - t_start
# MAGIC     log.info(
# MAGIC         "Embedding generation complete: %d items in %.1f s (%.0f items/s)",
# MAGIC         n, total_time, n / total_time,
# MAGIC     )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Build FAISS Index
# MAGIC
# MAGIC log.info("Building FAISS IVF-Flat index: n_items=%d, dim=%d, n_clusters=%d",
# MAGIC          len(all_embeddings), EMBEDDING_DIM, N_CLUSTERS)
# MAGIC
# MAGIC index = build_faiss_index(all_embeddings, n_clusters=N_CLUSTERS)
# MAGIC save_index(index, FAISS_INDEX_PATH)
# MAGIC
# MAGIC log.info(
# MAGIC     "FAISS index saved to %s | ntotal=%d | nlist=%d",
# MAGIC     FAISS_INDEX_PATH, index.ntotal, index.nlist,
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Validation
# MAGIC
# MAGIC # Reload to prove round-trip serialisation works
# MAGIC index = load_index(FAISS_INDEX_PATH)
# MAGIC assert index.ntotal == len(all_embeddings), (
# MAGIC     f"Index ntotal ({index.ntotal}) != number of embeddings ({len(all_embeddings)}). "
# MAGIC     "This indicates embeddings were added in a different run than the current arrays."
# MAGIC )
# MAGIC log.info("Index integrity check passed: ntotal=%d", index.ntotal)
# MAGIC
# MAGIC # Build ASIN → row-index lookup and title lookup for spot-checks
# MAGIC asin_to_idx   = {asin: i for i, asin in enumerate(all_asins)}
# MAGIC # Use the full meta for title lookup (covers both embedded and gap items)
# MAGIC idx_to_title  = embeddable["title_final"].fillna("(unknown)").to_dict()
# MAGIC
# MAGIC log.info("\n── Spot-check: top-%d neighbours ──────────────────────────────", SPOT_CHECK_K)
# MAGIC for seed_title in SPOT_CHECK_TITLES:
# MAGIC     # Find the first item whose title_final matches the seed
# MAGIC     matches = embeddable[embeddable["title_final"].str.contains(seed_title, case=False, na=False)]
# MAGIC     if matches.empty:
# MAGIC         log.info("Seed '%s': not found in embeddable items — skipping.", seed_title)
# MAGIC         continue
# MAGIC
# MAGIC     seed_row   = matches.iloc[0]
# MAGIC     seed_asin  = seed_row["asin"]
# MAGIC     seed_idx   = asin_to_idx.get(seed_asin)
# MAGIC     if seed_idx is None:
# MAGIC         log.info("Seed '%s' (ASIN %s): index position not found — skipping.", seed_title, seed_asin)
# MAGIC         continue
# MAGIC
# MAGIC     query_vec = all_embeddings[seed_idx : seed_idx + 1]  # shape (1, 384)
# MAGIC     distances, indices = query_index(index, query_vec, k=SPOT_CHECK_K + 1)
# MAGIC
# MAGIC     log.info("Seed: '%s' (ASIN: %s)", seed_row["title_final"], seed_asin)
# MAGIC     for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
# MAGIC         if idx == seed_idx:
# MAGIC             continue  # skip the seed itself
# MAGIC         neighbour_title = idx_to_title.get(int(idx), "(unknown)")
# MAGIC         log.info("  %d. %s  [L2=%.4f]", rank, neighbour_title, dist)
# MAGIC     log.info("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Summary Report
# MAGIC
# MAGIC tier_summary = meta["embedding_tier"].value_counts().sort_index().to_dict()
# MAGIC embedding_time_label = f"{total_time:.1f} s" if "total_time" in dir() else "N/A (loaded from checkpoint)"
# MAGIC index_size_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024 ** 2) if os.path.exists(FAISS_INDEX_PATH) else 0
# MAGIC
# MAGIC log.info("═" * 60)
# MAGIC log.info("JOB 1 SUMMARY")
# MAGIC log.info("═" * 60)
# MAGIC log.info("Items by embedding tier:")
# MAGIC tier_labels = {1: "Full (title+genres+desc+review)", 2: "Good (title+genres+desc)",
# MAGIC                3: "Thin (title+genres)", 4: "Bridge (TMDB)"}
# MAGIC for t, count in tier_summary.items():
# MAGIC     log.info("  Tier %d — %-35s : %6d (%.1f%%)", t, tier_labels.get(t, ""), count, 100 * count / len(meta))
# MAGIC log.info("Total items embedded : %d", index.ntotal)
# MAGIC log.info("True gaps (skipped)  : %d", true_gaps.sum())
# MAGIC log.info("Embedding time       : %s", embedding_time_label)
# MAGIC log.info("FAISS index size     : %.1f MB", index_size_mb)
# MAGIC log.info("Index path           : %s", FAISS_INDEX_PATH)
# MAGIC log.info("Embeddings path      : %s", EMBEDDINGS_PATH)
# MAGIC log.info("═" * 60)
# MAGIC print("Job 1 complete. Proceed to Job 2 (SVD training).")
