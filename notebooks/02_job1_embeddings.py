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
BATCH_SIZE        = 512        # CONFIG PARAM — safe for serverless CPU memory
N_CLUSTERS        = 256        # CONFIG PARAM — IVF cells for FAISS
MAX_REVIEW_WORDS  = 256        # CONFIG PARAM — word cap on review text
CHECKPOINT_EVERY  = 50         # save embeddings.npy every N batches
LOG_EVERY         = 10         # progress log every N batches

# Set to True to ignore all checkpoints and regenerate from scratch
FORCE_REGENERATE  = True  # ✅ ENABLED to bypass corrupted 51,200-item checkpoint

TMDB_SEARCH_URL   = "https://api.themoviedb.org/3/search/movie"
TMDB_SLEEP        = 1.0 / 40  # 40 req/s free-tier rate limit

SPOT_CHECK_TITLES = ["The Dark Knight", "Toy Story", "The Godfather"]
SPOT_CHECK_K      = 5