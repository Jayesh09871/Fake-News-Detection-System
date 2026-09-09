"""
Text Preprocessing Module
Provides deterministic cleaning routines shared identically across
Kaggle cloud training and local inference.
"""

from __future__ import annotations

import html
import re
from typing import Optional
import pandas as pd


# Precompiled regular expressions for high-throughput regex cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>", re.DOTALL)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """
    Clean and normalize raw text in a deterministic manner.

    Steps applied:
    1. Handle None / non-string gracefully.
    2. Decode HTML entities (e.g. &amp; -> &).
    3. Remove HTML tags.
    4. Remove URLs (http, https, www).
    5. Convert text to lowercase.
    6. Replace special characters / punctuation with spaces.
    7. Normalize and trim multiple whitespace characters.

    Args:
        text: Raw text string or None.

    Returns:
        Deterministic cleaned string.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # Decode HTML entities (e.g., &quot; -> ", &amp; -> &)
    cleaned = html.unescape(text)

    # Remove HTML tags
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # Remove URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # Lowercase
    cleaned = cleaned.lower()

    # Remove unnecessary special characters while keeping alphanumeric words
    cleaned = SPECIAL_CHAR_PATTERN.sub(" ", cleaned)

    # Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def combine_title_and_text(title: Optional[str] = "", text: Optional[str] = "") -> str:
    """
    Combine title and article text into a single cohesive string for vectorization.

    Args:
        title: Article title or headline.
        text: Article body content.

    Returns:
        Combined and cleaned string.
    """
    cleaned_title = clean_text(title)
    cleaned_body = clean_text(text)

    if cleaned_title and cleaned_body:
        return f"{cleaned_title} {cleaned_body}"
    elif cleaned_title:
        return cleaned_title
    elif cleaned_body:
        return cleaned_body
    return ""


def clean_dataframe(
    df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    label_col: Optional[str] = "label",
) -> pd.DataFrame:
    """
    Clean a pandas DataFrame containing news articles.
    Used during dataset preparation and Kaggle remote training.

    Operations:
    1. Fills missing title/text with empty strings.
    2. Combines title and text using `combine_title_and_text`.
    3. Drops duplicate articles based on the combined content.
    4. Drops empty records.
    5. Validates labels if label_col is provided.

    Args:
        df: Input DataFrame.
        title_col: Name of column containing title.
        text_col: Name of column containing text.
        label_col: Optional label column name.

    Returns:
        Cleaned, deduplicated DataFrame with 'combined_text' column.
    """
    df_clean = df.copy()

    # Ensure title and text columns exist
    if title_col not in df_clean.columns:
        df_clean[title_col] = ""
    if text_col not in df_clean.columns:
        df_clean[text_col] = ""

    # Handle null values
    df_clean[title_col] = df_clean[title_col].fillna("").astype(str)
    df_clean[text_col] = df_clean[text_col].fillna("").astype(str)

    # Deterministic combination
    df_clean["combined_text"] = [
        combine_title_and_text(t, b)
        for t, b in zip(df_clean[title_col], df_clean[text_col])
    ]

    # Remove empty articles (where combined_text is empty or purely whitespace)
    df_clean = df_clean[df_clean["combined_text"].str.strip() != ""].copy()

    # Remove duplicate articles to prevent data leakage across train/val/test splits
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["combined_text"]).copy()
    duplicates_removed = initial_count - len(df_clean)

    # Validate labels if present
    if label_col and label_col in df_clean.columns:
        # Drop records with missing or invalid labels
        df_clean = df_clean[df_clean[label_col].isin([0, 1])].copy()
        df_clean[label_col] = df_clean[label_col].astype(int)

    return df_clean
