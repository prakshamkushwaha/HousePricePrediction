"""
src/data_split.py

Purpose
-------
This module provides a clean, reusable way to split features (X) and
target (y) into training and testing sets for the House Price
Prediction project.

This module does NOT preprocess, scale, impute, train, or evaluate
anything. It expects to receive data that has already been cleaned
and separated into X/y (e.g. by `preprocessing.preprocess_data()`),
and its only job is to split that data.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Fixed split configuration used across the project, so every split
# performed with this module is consistent and reproducible.
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.

    Uses `sklearn.model_selection.train_test_split` with a fixed
    `test_size=0.2` and `random_state=42`, so the same split is
    produced every time this function is called on the same data.

    Args:
        X (pd.DataFrame): Feature columns.
        y (pd.Series): Target values, aligned row-for-row with X.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test, in that order.

    Raises:
        TypeError: If `X` is not a pandas DataFrame, or `y` is not a
            pandas Series.
        ValueError: If `X` or `y` is empty, or if they do not have
            the same number of rows.
        RuntimeError: If the underlying split operation fails for
            any other reason.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("split_data() expects X to be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("split_data() expects y to be a pandas Series.")

    if X.empty:
        raise ValueError("split_data() cannot split an empty DataFrame X.")

    if y.empty:
        raise ValueError("split_data() cannot split an empty Series y.")

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows. "
            f"Got X: {len(X)} rows, y: {len(y)} rows."
        )

    logger.info(
        "split_data: splitting %s rows into train/test "
        "(test_size=%s, random_state=%s).",
        len(X), TEST_SIZE, RANDOM_STATE,
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    except Exception as error:
        logger.error("split_data: train_test_split failed: %s", error)
        raise RuntimeError(f"Failed to split data: {error}") from error

    logger.info(
        "split_data: done. X_train=%s, X_test=%s, y_train=%s, y_test=%s.",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Local imports so this file has no hard dependency on the other
    # modules unless it is actually run as a script.
    try:
        from src.data_loader import load_housing_data
        from src.preprocessing import preprocess_data
    except ImportError:
        from data_loader import load_housing_data
        from preprocessing import preprocess_data

    print("Step 1: Loading California Housing dataset...")
    housing_data = load_housing_data()

    print("Step 2: Cleaning data and separating X/y...")
    X, y = preprocess_data(housing_data)
    print("Total rows:", len(X))

    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Verify the split sizes are correct WITHOUT hard-coding the
    # expected row counts — they are computed from the actual total.
    total_rows = len(X)
    expected_train_rows = round(total_rows * (1 - TEST_SIZE))
    expected_test_rows = total_rows - expected_train_rows

    assert len(X_train) == len(y_train), "X_train and y_train row counts differ!"
    assert len(X_test) == len(y_test), "X_test and y_test row counts differ!"
    assert len(X_train) + len(X_test) == total_rows, "Split rows do not add up to the total!"
    assert len(X_train) == expected_train_rows, "Training set size is not as expected!"
    assert len(X_test) == expected_test_rows, "Testing set size is not as expected!"

    print("Verified: train/test split sizes are correct.")
