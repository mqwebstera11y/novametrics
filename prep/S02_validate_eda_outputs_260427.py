# Databricks notebook source
# MAGIC %md
# MAGIC ## After EDA and Filtering
# MAGIC ### But before Job1
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Validate EDA Outputs
# MAGIC ## CohortNova · Movie Recommendation System
# MAGIC
# MAGIC **Purpose:** Inspect every dataframe produced by Notebook 01 (EDA & Filtering).
# MAGIC For each output: print columns, row count, and key field values (asin / parent_asin).
# MAGIC Run this immediately after Notebook 01 completes to confirm outputs are correct
# MAGIC before proceeding to Job 1 (embeddings).
# MAGIC
# MAGIC **What this checks:**
# MAGIC - reviews_5core.parquet   — the filtered review dataset (join key: parent_asin)
# MAGIC - meta_clean.parquet      — the cleaned metadata (join key: parent_asin, stored as item_id then re-aliased)
# MAGIC - most_helpful (in-memory) — most helpful review per item (join key: parent_asin)
# MAGIC - meta_with_review (join) — metadata + most helpful review joined (join key: parent_asin)
# MAGIC
# MAGIC **Run order:** After Notebook 01. Before Job 1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Setup

# COMMAND ----------

from pyspark.sql import functions as F

PROCESSED_DIR = "/Volumes/movie_recsys/data/outputs"
REVIEWS_OUT   = f"{PROCESSED_DIR}/reviews_5core.parquet"
META_OUT      = f"{PROCESSED_DIR}/meta_clean.parquet"

def inspect(label, df, key_cols):
    """
    Print schema, row count, and key field diagnostics for a dataframe.
    key_cols: list of column names to check for nulls and show sample values.
    """
    print("=" * 70)
    print(f"DATAFRAME: {label}")
    print("=" * 70)

    # Columns
    print(f"\n  Columns ({len(df.columns)}):")
    for col in df.columns:
        dtype = dict(df.dtypes)[col]
        print(f"    {col:<35s} {dtype}")

    # Row count
    n = df.count()
    print(f"\n  Row count: {n:,}")

    # Key field diagnostics
    print(f"\n  Key field diagnostics:")
    for key in key_cols:
        if key not in df.columns:
            print(f"    {key:<20s} ← COLUMN MISSING")
            continue
        null_count    = df.filter(F.col(key).isNull()).count()
        distinct_count = df.select(key).distinct().count()
        print(f"    {key:<20s} nulls={null_count:,}   distinct={distinct_count:,}")

    # Sample values for key columns
    print(f"\n  Sample rows (top 3, key columns only):")
    sample_cols = [k for k in key_cols if k in df.columns]
    if sample_cols:
        df.select(sample_cols).show(3, truncate=50)

    print()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · reviews_5core.parquet
# MAGIC
# MAGIC Written in Section 9 of Notebook 01.
# MAGIC Key column: `parent_asin` (item identifier after `asin` was dropped from reviews_raw).
# MAGIC Secondary key: `user_id`.
# MAGIC
# MAGIC Selected columns written:
# MAGIC   user_id, parent_asin, rating, event_ts, event_date,
# MAGIC   event_year, event_month, helpful_vote, verified_purchase, text

# COMMAND ----------

reviews_5core = spark.read.parquet(REVIEWS_OUT)

inspect(
    label    = "reviews_5core.parquet",
    df       = reviews_5core,
    key_cols = ["parent_asin", "user_id"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · meta_clean.parquet
# MAGIC
# MAGIC Written in Section 9 of Notebook 01.
# MAGIC
# MAGIC IMPORTANT — key column is `parent_asin`.
# MAGIC In the EDA code, meta_raw.parent_asin was renamed to `item_id` during cleaning,
# MAGIC then re-aliased back to `parent_asin` via:
# MAGIC   meta = meta.withColumn("parent_asin", F.col("item_id"))
# MAGIC The write statement selects `parent_asin` explicitly.
# MAGIC
# MAGIC Selected columns written:
# MAGIC   parent_asin, title, primary_genre, genres_str, description_str,
# MAGIC   price_raw, price_float, has_price, revenue_tier,
# MAGIC   average_rating, rating_number

# COMMAND ----------

meta_clean = spark.read.parquet(META_OUT)

inspect(
    label    = "meta_clean.parquet",
    df       = meta_clean,
    key_cols = ["parent_asin"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · most_helpful (in-memory — reconstruct from parquet)
# MAGIC
# MAGIC Not written to disk separately — it is an intermediate dataframe
# MAGIC derived from reviews_5core inside Notebook 01, Section 7.
# MAGIC
# MAGIC Construction in EDA:
# MAGIC   most_helpful = (
# MAGIC       reviews_5core
# MAGIC       .orderBy(F.desc("helpful_vote"))
# MAGIC       .groupBy("parent_asin")
# MAGIC       .agg(F.first("text").alias("most_helpful_review"))
# MAGIC   )
# MAGIC
# MAGIC Key column: `parent_asin`.
# MAGIC We reconstruct it here from the saved parquet so we can inspect it.

# COMMAND ----------

most_helpful = (
    reviews_5core
    .orderBy(F.desc("helpful_vote"))
    .groupBy("parent_asin")
    .agg(F.first("text").alias("most_helpful_review"))
)

inspect(
    label    = "most_helpful (reconstructed)",
    df       = most_helpful,
    key_cols = ["parent_asin"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · meta_with_review (join — reconstruct)
# MAGIC
# MAGIC Not written to disk separately — intermediate join inside Notebook 01, Section 7.
# MAGIC
# MAGIC Construction in EDA:
# MAGIC   meta_with_review = meta.join(most_helpful, on="parent_asin", how="left")
# MAGIC
# MAGIC This is the dataframe that drives embedding coverage reporting.
# MAGIC Key question: does most_helpful_review actually populate after the ASIN fix?
# MAGIC
# MAGIC Key column: `parent_asin`.

# COMMAND ----------

meta_with_review = meta_clean.join(most_helpful, on="parent_asin", how="left")

inspect(
    label    = "meta_with_review (reconstructed join)",
    df       = meta_with_review,
    key_cols = ["parent_asin"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Join Health Check
# MAGIC
# MAGIC The single most important diagnostic: how many meta items
# MAGIC successfully joined to a most_helpful_review after the ASIN fix?
# MAGIC This is what drove the 0% → 26.7% jump in the EDA summary.

# COMMAND ----------

join_stats = meta_with_review.select(
    F.count("*").alias("total_meta_items"),
    F.sum(
        F.when(
            F.col("most_helpful_review").isNotNull() &
            (F.col("most_helpful_review") != ""),
            1
        ).otherwise(0)
    ).alias("has_review"),
    F.sum(
        F.when(F.col("title").isNotNull() & (F.col("title") != ""), 1).otherwise(0)
    ).alias("has_title"),
    F.sum(
        F.when(F.col("genres_str").isNotNull() & (F.col("genres_str") != ""), 1).otherwise(0)
    ).alias("has_genres"),
    F.sum(
        F.when(F.col("description_str").isNotNull() & (F.col("description_str") != ""), 1).otherwise(0)
    ).alias("has_description"),
).collect()[0]

n = join_stats["total_meta_items"]

print("=" * 70)
print("JOIN HEALTH CHECK — meta_with_review")
print("=" * 70)
print(f"  Total meta items       : {n:,}")
print(f"  has_title              : {join_stats['has_title']:,}  ({join_stats['has_title']/n*100:.1f}%)")
print(f"  has_genres             : {join_stats['has_genres']:,}  ({join_stats['has_genres']/n*100:.1f}%)")
print(f"  has_description        : {join_stats['has_description']:,}  ({join_stats['has_description']/n*100:.1f}%)")
print(f"  has_most_helpful_review: {join_stats['has_review']:,}  ({join_stats['has_review']/n*100:.1f}%)")
print()

# Expected from updated EDA summary:
#   Title  : 58.0%
#   Genres : 57.9%
#   Desc   : 46.5%
#   Review : 26.7%

# Embedding tier breakdown
t1 = meta_with_review.filter(
    F.col("title").isNotNull() & (F.col("title") != "") &
    F.col("genres_str").isNotNull() & (F.col("genres_str") != "") &
    F.col("description_str").isNotNull() & (F.col("description_str") != "") &
    F.col("most_helpful_review").isNotNull() & (F.col("most_helpful_review") != "")
).count()

t2 = meta_with_review.filter(
    F.col("title").isNotNull() & (F.col("title") != "") &
    F.col("genres_str").isNotNull() & (F.col("genres_str") != "") &
    F.col("description_str").isNotNull() & (F.col("description_str") != "") &
    (F.col("most_helpful_review").isNull() | (F.col("most_helpful_review") == ""))
).count()

t3 = meta_with_review.filter(
    F.col("title").isNotNull() & (F.col("title") != "") &
    F.col("genres_str").isNotNull() & (F.col("genres_str") != "") &
    (F.col("description_str").isNull() | (F.col("description_str") == "")) &
    (F.col("most_helpful_review").isNull() | (F.col("most_helpful_review") == ""))
).count()

t4 = n - t1 - t2 - t3

print("  Embedding tier breakdown (for Job 1):")
print(f"    Tier 1 — Full  (title+genres+desc+review) : {t1:,}  ({t1/n*100:.1f}%)")
print(f"    Tier 2 — Good  (title+genres+desc)        : {t2:,}  ({t2/n*100:.1f}%)")
print(f"    Tier 3 — Thin  (title+genres only)        : {t3:,}  ({t3/n*100:.1f}%)")
print(f"    Tier 4 — Bridge (TMDB needed)             : {t4:,}  ({t4/n*100:.1f}%)")
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Key Column Summary Table
# MAGIC
# MAGIC One-glance reference of every output's key and join compatibility.

# COMMAND ----------

print("=" * 70)
print("KEY COLUMN REFERENCE — all EDA outputs")
print("=" * 70)
print(f"  {'Dataset':<40s} {'Key column':<20s} {'Joins with'}")
print(f"  {'-'*38} {'-'*18} {'-'*20}")
print(f"  {'reviews_5core.parquet':<40s} {'parent_asin':<20s} meta_clean.parent_asin")
print(f"  {'reviews_5core.parquet':<40s} {'user_id':<20s} (user-level joins in A/B)")
print(f"  {'meta_clean.parquet':<40s} {'parent_asin':<20s} reviews_5core.parent_asin")
print(f"  {'most_helpful (derived)':<40s} {'parent_asin':<20s} meta_clean.parent_asin")
print(f"  {'meta_with_review (derived join)':<40s} {'parent_asin':<20s} Job 1 embedding input")
print()
print("  NOTE: `asin` was DROPPED from reviews_raw early in Notebook 01.")
print("  The item join key throughout this project is `parent_asin` only.")
print("  meta_clean does NOT have an `item_id` column — it was re-aliased")
print("  to `parent_asin` before writing.")
print()
print("  If any of the above show 'COLUMN MISSING' — stop and fix before Job 1.")
print("=" * 70)

# COMMAND ----------

## Looks fine 

# COMMAND ----------


