"""
model_cb.py — FAISS index build and query for the content-based engine.

FRACTION tag: this is the only file that imports faiss.
Swapping to Pinecone, Weaviate, or any other ANN backend means
changing only this file — all callers depend on the function
signatures, not the implementation.

Index type: IVF-Flat
    - IVF (Inverted File Index) partitions the vector space into
      n_clusters Voronoi cells, enabling sub-linear search.
    - Flat quantiser means exact distance within each cell (no PQ
      compression), which is appropriate for 384-dim vectors at 200K scale.
    - Search time: sub-10ms for 200K items on CPU (per blueprint spec).
"""

from __future__ import annotations

import os

import faiss
import numpy as np


def build_faiss_index(
    embeddings: np.ndarray,
    n_clusters: int = 100,
) -> faiss.Index:
    """
    Build and return a trained IVF-Flat FAISS index over ``embeddings``.

    The index is trained on the full embedding matrix and then all vectors
    are added in a single call. Callers should persist the returned index
    with :func:`save_index` before the process exits.

    Parameters
    ----------
    embeddings:
        2-D float32 array of shape ``(n_items, dim)``.  Must have at least
        ``n_clusters`` rows; FAISS requires more training points than cells.
    n_clusters:
        Number of Voronoi cells (IVF partitions).  CONFIG PARAM — never
        hardcode this in the notebook; pass it from the config cell.
        Typical rule of thumb: ``sqrt(n_items)``.  Default 100 is
        conservative for 200K items and safe for smaller smoke-test subsets.

    Returns
    -------
    faiss.Index
        Trained, populated index ready for search.

    Raises
    ------
    ValueError
        If ``embeddings`` is not a 2-D array, is not float32, or has fewer
        rows than ``n_clusters``.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n_items, dim), got shape {embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError(
            f"embeddings must be float32, got {embeddings.dtype}. "
            "Cast with embeddings.astype(np.float32) before calling."
        )
    n_items, dim = embeddings.shape
    if n_items < n_clusters:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be <= number of embeddings ({n_items}). "
            "Reduce n_clusters or provide more embeddings."
        )

    quantiser = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, n_clusters, faiss.METRIC_L2)

    index.train(embeddings)
    index.add(embeddings)

    return index


def save_index(index: faiss.Index, path: str) -> None:
    """
    Persist a FAISS index to disk, creating parent directories as needed.

    Parameters
    ----------
    index:
        A trained and populated FAISS index as returned by
        :func:`build_faiss_index`.
    path:
        Absolute or relative file path for the output file, e.g.
        ``"/Volumes/movie_recsys/data/outputs/faiss_index.bin"``.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.

    Parameters
    ----------
    path:
        Path to a file previously written by :func:`save_index`.

    Returns
    -------
    faiss.Index
        The loaded index, ready for search.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist, with a message indicating the expected
        location so operators can diagnose missing artefacts quickly.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at '{path}'. "
            "Run Job 1 (02_job1_embeddings.py) to build and save the index first."
        )
    return faiss.read_index(path)


def query_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the ``k`` nearest neighbours of a query vector in the index.

    Parameters
    ----------
    index:
        A populated FAISS index (IVF must be trained and have vectors added).
    query_embedding:
        Query vector as a 2-D float32 array of shape ``(1, dim)``.
        The explicit shape requirement prevents silent broadcasting errors
        when callers pass a 1-D vector.
    k:
        Number of neighbours to return.  Defaults to 10.

    Returns
    -------
    distances : np.ndarray, shape (1, k)
        L2 distances to each neighbour (ascending order).
    indices : np.ndarray, shape (1, k)
        Positional indices into the original embeddings array that was
        used to build the index.  Map these back to ASINs via the
        index-to-ASIN lookup array saved alongside embeddings.npy.

    Raises
    ------
    ValueError
        If ``query_embedding`` is not 2-D, not float32, or has the wrong
        embedding dimension.
    """
    if query_embedding.ndim != 2:
        raise ValueError(
            f"query_embedding must be 2-D with shape (1, dim), "
            f"got shape {query_embedding.shape}. "
            "Reshape with query_embedding.reshape(1, -1)."
        )
    if query_embedding.dtype != np.float32:
        raise ValueError(
            f"query_embedding must be float32, got {query_embedding.dtype}."
        )
    if query_embedding.shape[0] != 1:
        raise ValueError(
            f"query_index expects a single query vector (shape (1, dim)), "
            f"got {query_embedding.shape[0]} rows. "
            "Call query_index once per query."
        )
    expected_dim = index.d
    if query_embedding.shape[1] != expected_dim:
        raise ValueError(
            f"query_embedding has dim {query_embedding.shape[1]} but "
            f"index was built with dim {expected_dim}."
        )

    distances, indices = index.search(query_embedding, k)
    return distances, indices
