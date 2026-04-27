"""
tests/test_features.py — Unit tests for features.py

Run with: pytest tests/test_features.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from features import build_embedding_input, get_embedding_tier


# ── get_embedding_tier ────────────────────────────────────────────────────────

def test_tier1_all_fields():
    assert get_embedding_tier("The Dark Knight", "Action|Drama", "A superhero film.", "Great movie!") == 1

def test_tier2_no_review():
    assert get_embedding_tier("The Dark Knight", "Action|Drama", "A superhero film.", None) == 2

def test_tier2_empty_review():
    assert get_embedding_tier("The Dark Knight", "Action|Drama", "A superhero film.", "") == 2

def test_tier3_title_genres_only():
    assert get_embedding_tier("The Dark Knight", "Action", None, None) == 3

def test_tier4_title_only():
    assert get_embedding_tier("The Dark Knight", None, None, None) == 4

def test_tier4_all_none():
    assert get_embedding_tier(None, None, None, None) == 4

def test_tier4_nan_float():
    """pandas fills missing string fields with float NaN — must not crash."""
    import math
    assert get_embedding_tier(float("nan"), float("nan"), float("nan"), float("nan")) == 4

def test_tier4_whitespace_only():
    assert get_embedding_tier("   ", "   ", "   ", "   ") == 4


# ── build_embedding_input ─────────────────────────────────────────────────────

def test_tier1_output_contains_all_parts():
    result = build_embedding_input("The Dark Knight", "Action|Drama", "A superhero film.", "Excellent!")
    assert "The Dark Knight" in result
    assert "Action|Drama" in result
    assert "A superhero film." in result
    assert "Excellent!" in result

# def test_tier2_output_no_review():
#     result = build_embedding_input("The Dark Knight", "Action|Drama", "A superhero film.", None)
#     assert result is not None
#     assert "A superhero film." in result
#     # Review should not be present
#     assert result.count("|") == 2  # title | genres | description
    
def test_tier2_output_no_review():
    result = build_embedding_input("The Dark Knight", "Action|Drama", "A superhero film.", None)
    assert result is not None
    assert "A superhero film." in result
    assert "The Dark Knight" in result
    assert "Action|Drama" in result
    # Tier 2 has no review — check by splitting on " | " separator
    parts = result.split(" | ")
    assert len(parts) == 3

def test_tier3_output_title_genres():
    result = build_embedding_input("The Dark Knight", "Action", None, None)
    assert result is not None
    assert "The Dark Knight" in result
    assert "Action" in result

def test_tier4_returns_none():
    result = build_embedding_input("The Dark Knight", None, None, None)
    assert result is None

def test_review_truncation():
    long_review = " ".join(["word"] * 300)
    result = build_embedding_input("Title", "Genre", "Desc", long_review, max_review_words=10)
    review_part = result.split(" | ")[-1]
    assert len(review_part.split()) == 10

def test_separator_format():
    result = build_embedding_input("Title", "Genre", "Description", "Review")
    parts = result.split(" | ")
    assert len(parts) == 4
    assert parts[0] == "Title"
    assert parts[1] == "Genre"
