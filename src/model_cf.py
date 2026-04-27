"""
model_cf.py — Collaborative filtering pipeline (SVD).

Wraps the pickle-loaded Surprise SVD model.
Swapping to a Two-Tower model or any other CF backend means
changing only this file.
"""
from __future__ import annotations

import pickle
from pathlib import Path


def load_svd_model(path: str):
    """Load the SVD model from a pickle file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"SVD model not found at '{path}'. "
            "Run Job 2 (Job2_SVD.ipynb) to train and save the model first."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def predict_for_user(
    algo,
    user_id: str,
    candidate_items: list[str],
    n: int = 5,
) -> list[dict]:
    """
    Generate top-n CF predictions for a user over a list of candidate items.

    Parameters
    ----------
    algo:
        Loaded Surprise SVD model.
    user_id:
        The user's ID string. If the user was not in the training set,
        Surprise falls back to the global mean — handled gracefully.
    candidate_items:
        List of parent_asin strings to score.
    n:
        Number of top recommendations to return. Default 5.

    Returns
    -------
    list of dict, each with keys: parent_asin, score
        Sorted descending by score, length <= n.
    """
    predictions = []
    for item_id in candidate_items:
        pred = algo.predict(user_id, item_id)
        predictions.append({"parent_asin": item_id, "score": round(pred.est, 4)})

    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions[:n]
