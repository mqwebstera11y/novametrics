"""
features.py — Pure functions for embedding input construction.

No Spark, no FAISS, no model loading. All functions are stateless
and fully unit-testable.

Four-tier fallback strategy (from EDA coverage analysis):
    Tier 1 — Full    : title + genres + description + review (~26% of items)
    Tier 2 — Good    : title + genres + description         (~20%)
    Tier 3 — Thin    : title + genres only                  (~11%)
    Tier 4 — Bridge  : title only or nothing                (~42%) → TMDB
"""

from __future__ import annotations


def _is_present(value: str | None) -> bool:
    """Return True if value is a non-empty, non-whitespace string."""
    return bool(value and value.strip())


def _truncate_to_words(text: str, max_words: int) -> str:
    """Return text truncated to at most max_words whitespace-split tokens."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def get_embedding_tier(
    title: str | None,
    genres_str: str | None,
    description_str: str | None,
    review_text: str | None,
) -> int:
    """
    Classify an item into one of four embedding tiers based on field availability.

    Tier definitions mirror the EDA coverage analysis:
        1 — Full    : title + genres + description + review
        2 — Good    : title + genres + description (no review)
        3 — Thin    : title + genres only
        4 — Bridge  : title only, or no fields at all → TMDB enrichment needed

    Parameters
    ----------
    title:
        Item title string, or None / empty string if missing.
    genres_str:
        Pipe- or comma-separated genre tags (e.g. "Action|Drama"), or None.
    description_str:
        Item description / synopsis, or None.
    review_text:
        Pre-selected most-helpful review text, or None.

    Returns
    -------
    int
        Tier number in [1, 2, 3, 4].
    """
    has_title = _is_present(title)
    has_genres = _is_present(genres_str)
    has_description = _is_present(description_str)
    has_review = _is_present(review_text)

    if has_title and has_genres and has_description and has_review:
        return 1
    if has_title and has_genres and has_description:
        return 2
    if has_title and has_genres:
        return 3
    return 4


def build_embedding_input(
    title: str | None,
    genres_str: str | None,
    description_str: str | None,
    review_text: str | None,
    max_review_words: int = 256,
) -> str | None:
    """
    Construct the text string that will be fed to the sentence-transformer.

    Implements the four-tier fallback strategy documented in the project
    blueprint.  Fields are concatenated in a fixed order separated by
    ``" | "`` so the model sees a consistent format regardless of tier.

    The most-helpful review is word-truncated to ``max_review_words`` before
    concatenation to stay within the model's 256-token context window.

    Parameters
    ----------
    title:
        Item title string, or None / empty string if missing.
    genres_str:
        Pipe- or comma-separated genre tags (e.g. "Action|Drama"), or None.
    description_str:
        Item description / synopsis, or None.
    review_text:
        Pre-selected most-helpful review text, or None.
    max_review_words:
        Maximum number of whitespace-delimited words to include from the
        review text.  Defaults to 256 (CONFIG PARAM — callers may override).

    Returns
    -------
    str or None
        Concatenated embedding input string for Tier 1–3 items.
        ``None`` for Tier 4 items that require TMDB enrichment before
        an embedding can be generated.
    """
    tier = get_embedding_tier(title, genres_str, description_str, review_text)

    if tier == 4:
        return None

    parts: list[str] = [title.strip()]  # type: ignore[union-attr]  # tier < 4 guarantees title

    if tier <= 3:  # Tier 1, 2, 3 all have genres
        parts.append(genres_str.strip())  # type: ignore[union-attr]

    if tier <= 2:  # Tier 1, 2 have description
        parts.append(description_str.strip())  # type: ignore[union-attr]

    if tier == 1:  # Only Tier 1 has a review
        truncated = _truncate_to_words(review_text.strip(), max_review_words)  # type: ignore[union-attr]
        parts.append(truncated)

    return " | ".join(parts)
