"""Data loading, splitting, and subsampling utilities for the IMDB sentiment task."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

_HTML_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def load_imdb(path: str) -> pd.DataFrame:
    """Load the IMDB CSV, deduplicate, and clean HTML <br/> tags.

    The CSV is expected to have columns ``review`` and ``sentiment``.
    Returns a DataFrame with a fresh integer index.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    df["review"] = df["review"].astype(str).map(lambda t: _HTML_BR.sub(" ", t))
    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on ``sentiment``.

    Uses two sequential stratified splits to preserve class balance across
    all three partitions. The returned DataFrames keep their original
    row indices from ``df``, which is what ``select_few_shot_examples``
    and the leakage assertion rely on.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, (
        "Split sizes must sum to 1.0"
    )

    # First split: train vs. (val + test)
    train_df, rest_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df["sentiment"],
        random_state=seed,
    )

    # Second split: val vs. test, within the rest.
    val_fraction = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df,
        test_size=(1.0 - val_fraction),
        stratify=rest_df["sentiment"],
        random_state=seed,
    )

    return train_df, val_df, test_df


def subsample(
    df: pd.DataFrame,
    n_per_class: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified subsample with ``n_per_class`` rows per sentiment class."""
    parts = [
        df[df["sentiment"] == label].sample(n=n_per_class, random_state=seed)
        for label in sorted(df["sentiment"].unique())
    ]
    return pd.concat(parts).sort_index()


def select_few_shot_examples(
    df_train: pd.DataFrame,
    n_per_class: int = 1,
    max_words: int = 50,
) -> pd.DataFrame:
    """Deterministic few-shot example selection from the train split.

    Rule: sort each class by the DataFrame index (original row order),
    take the first ``n_per_class`` rows, and truncate the review text to
    ``max_words`` whitespace-separated tokens. Returns a DataFrame with
    columns ``review`` (truncated) and ``sentiment``, indexed by the
    original train-split index for logging.
    """
    parts = []
    for label in sorted(df_train["sentiment"].unique()):
        subset = df_train[df_train["sentiment"] == label].sort_index().head(n_per_class)
        truncated = subset["review"].map(
            lambda t: " ".join(str(t).split()[:max_words])
        )
        parts.append(
            pd.DataFrame({"review": truncated, "sentiment": subset["sentiment"]})
        )
    return pd.concat(parts)
