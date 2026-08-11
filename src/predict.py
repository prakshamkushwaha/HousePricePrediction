"""
src/predict.py

Purpose
-------
This module provides a reusable way to predict a house value from
user-provided California Housing feature values, using the ALREADY
saved and fitted model + preprocessing pipeline (see
`model_persistence.py`).

This module NEVER trains, fits, or refits anything. It only:
    - loads `models/best_model.pkl` and
      `models/preprocessing_pipeline.pkl`,
    - calls `pipeline.transform()` on new input data, and
    - calls `model.predict()` on the transformed input.

Units
-------
The California Housing target ("MedHouseVal") is measured in units
of $100,000. For example, a raw model prediction of `3.25` means
approximately $325,000. This module always converts predictions to
plain dollars before returning them, so callers never need to think
about the $100,000 scaling themselves.
"""

import logging
import math
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Reuse the existing save/load functions and default paths from
# model_persistence.py, so this module never re-implements loading
# logic or risks drifting from where artifacts are actually saved.
# This try/except supports running this module either as part of the
# `src` package or as a standalone script from inside the `src/`
# folder.
try:
    from src.model_persistence import MODEL_PATH, PIPELINE_PATH, load_saved_artifacts
except ImportError:
    from model_persistence import MODEL_PATH, PIPELINE_PATH, load_saved_artifacts

logger = logging.getLogger(__name__)

# The feature columns the model/pipeline were trained on, in the
# exact order used throughout the rest of this project (see
# data_validation.EXPECTED_FEATURE_COLUMNS). Any input DataFrame is
# reordered to match this before being passed to the pipeline, so a
# caller-supplied column order can never silently produce a wrong
# prediction.
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# The California Housing target is expressed in units of $100,000.
DOLLARS_PER_TARGET_UNIT: float = 100_000.0


def _validate_feature_value(name: str, value: Any) -> float:
    """
    Validate a single feature value and convert it to a plain float.

    Checks that the value is present, numeric, and finite. Does NOT
    impose any range restrictions (e.g. it will not reject an
    unusual latitude) — only type and finiteness are checked, since
    the task calls for validation, not arbitrary hard-coded limits.

    Args:
        name (str): The feature's name, used in error messages.
        value (Any): The raw value to validate.

    Returns:
        float: The validated value, as a plain Python float.

    Raises:
        ValueError: If `value` is None, NaN, or infinite.
        TypeError: If `value` is not a numeric type (or is a bool,
            which is technically a numeric subtype in Python but
            almost certainly not an intended feature value here).
    """
    if value is None:
        raise ValueError(f"'{name}' is required and cannot be None.")

    if isinstance(value, bool):
        raise TypeError(f"'{name}' must be numeric, got a boolean ({value}).")

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"'{name}' must be numeric, got {type(value).__name__}.")

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(f"'{name}' must be a finite number, got {value}.")

    return numeric_value


def _validate_input_dataframe(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a prediction input DataFrame and return a copy with
    columns reordered to match `FEATURE_COLUMNS`.

    Checks that:
        - `input_data` is a pandas DataFrame.
        - it is not empty.
        - it has exactly the expected feature columns (no missing,
          no unexpected extras — regardless of the order they were
          given in).
        - every value is numeric and finite (no missing/NaN values).

    Args:
        input_data (pd.DataFrame): The DataFrame to validate.

    Returns:
        pd.DataFrame: A new DataFrame containing the same data, with
        columns in the exact order the pipeline/model expect.

    Raises:
        TypeError: If `input_data` is not a pandas DataFrame, or
            contains non-numeric values.
        ValueError: If `input_data` is empty, has missing/unexpected
            columns, or contains NaN/infinite values.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "predict_house_price() expects a pandas DataFrame, "
            f"got {type(input_data).__name__}."
        )

    if input_data.empty:
        raise ValueError("predict_house_price() received an empty DataFrame.")

    actual_columns = set(input_data.columns)
    expected_columns = set(FEATURE_COLUMNS)

    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    if missing_columns or unexpected_columns:
        raise ValueError(
            f"input_data must contain exactly these columns: {FEATURE_COLUMNS}. "
            f"Missing: {missing_columns}. Unexpected: {unexpected_columns}."
        )

    ordered_input = input_data[FEATURE_COLUMNS].copy()

    try:
        numeric_array = ordered_input.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"input_data contains non-numeric value(s): {error}") from error

    if not np.isfinite(numeric_array).all():
        raise ValueError(
            "input_data contains missing (NaN) or infinite value(s). "
            "Every feature value must be a finite number."
        )

    return ordered_input


def load_prediction_artifacts() -> Tuple[Any, Pipeline]:
    """
    Load the saved best model and preprocessing pipeline used for
    making predictions.

    This is a thin, clearly-named wrapper around
    `model_persistence.load_saved_artifacts()`, so prediction code
    (including a future dashboard) has one obvious entry point here
    without needing to know about `model_persistence.py` directly.

    Returns:
        Tuple[Any, Pipeline]:
            - model: The loaded, already-fitted best model.
            - pipeline: The loaded, already-fitted preprocessing
              pipeline.

    Raises:
        FileNotFoundError: If either saved artifact file is missing.
        RuntimeError: If loading either file fails for any other
            reason.
    """
    logger.info("load_prediction_artifacts: loading saved model and pipeline...")

    try:
        model, pipeline = load_saved_artifacts()
    except FileNotFoundError:
        logger.error(
            "load_prediction_artifacts: saved artifacts not found at '%s' / "
            "'%s'. Run the training + model_persistence pipeline first.",
            MODEL_PATH, PIPELINE_PATH,
        )
        raise
    except Exception as error:
        logger.error("load_prediction_artifacts: failed to load artifacts: %s", error)
        raise RuntimeError(f"Failed to load prediction artifacts: {error}") from error

    logger.info(
        "load_prediction_artifacts: loaded %s and a fitted preprocessing pipeline.",
        type(model).__name__,
    )

    return model, pipeline


def create_input_dataframe(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float,
) -> pd.DataFrame:
    """
    Build a one-row pandas DataFrame from individual feature values,
    using the exact feature names and column order used during
    training (see `FEATURE_COLUMNS`).

    Args:
        MedInc (float): Median income in the block group (tens of
            thousands of dollars).
        HouseAge (float): Median house age in the block group
            (years).
        AveRooms (float): Average number of rooms per household.
        AveBedrms (float): Average number of bedrooms per household.
        Population (float): Block group population.
        AveOccup (float): Average number of household members.
        Latitude (float): Block group latitude.
        Longitude (float): Block group longitude.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns
        `FEATURE_COLUMNS`, in that order.

    Raises:
        TypeError: If any value is not numeric.
        ValueError: If any value is missing, NaN, or infinite.
    """
    raw_values: Dict[str, Any] = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }

    validated_values = {
        name: _validate_feature_value(name, value) for name, value in raw_values.items()
    }

    input_df = pd.DataFrame([validated_values], columns=FEATURE_COLUMNS)

    logger.info("create_input_dataframe: built input row %s", validated_values)

    return input_df


def predict_house_price(input_data: pd.DataFrame) -> float:
    """
    Predict a house price, in dollars, from a one-row feature
    DataFrame.

    This function NEVER fits, refits, or trains anything. It only:
        1. Loads the saved model + pipeline (`load_prediction_artifacts()`).
        2. Calls `pipeline.transform(input_data)` — never `.fit()` or
           `.fit_transform()`.
        3. Calls `model.predict(...)` — never `.fit()`.
        4. Converts the raw prediction (in $100,000 units) to plain
           dollars.

    Args:
        input_data (pd.DataFrame): A one-row DataFrame with exactly
            the columns in `FEATURE_COLUMNS` (any order — it is
            reordered internally), and numeric, finite values. Use
            `create_input_dataframe()` to build this easily.

    Returns:
        float: The predicted house price, in dollars.

    Raises:
        TypeError: If `input_data` is not a pandas DataFrame, or
            contains non-numeric values.
        ValueError: If `input_data` is empty, has missing/unexpected
            columns, or contains NaN/infinite values.
        FileNotFoundError: If the saved model/pipeline files are
            missing.
        RuntimeError: If transforming the input or generating the
            prediction fails.
    """
    validated_input = _validate_input_dataframe(input_data)

    model, pipeline = load_prediction_artifacts()

    try:
        transformed_input = pipeline.transform(validated_input)
    except Exception as error:
        logger.error("predict_house_price: pipeline.transform() failed: %s", error)
        raise RuntimeError(f"Failed to transform input data: {error}") from error

    try:
        raw_prediction = model.predict(transformed_input)
    except Exception as error:
        logger.error("predict_house_price: model.predict() failed: %s", error)
        raise RuntimeError(f"Failed to generate a prediction: {error}") from error

    prediction_in_target_units = float(raw_prediction[0])
    predicted_price_dollars = prediction_in_target_units * DOLLARS_PER_TARGET_UNIT

    logger.info(
        "predict_house_price: raw prediction=%.4f (x $100,000) -> $%.2f",
        prediction_in_target_units, predicted_price_dollars,
    )

    return predicted_price_dollars


def predict_from_values(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float,
) -> float:
    """
    Convenience function: build the input DataFrame from individual
    feature values, then predict a house price in dollars.

    Equivalent to:
        input_df = create_input_dataframe(...)
        predict_house_price(input_df)

    Args:
        MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
        Latitude, Longitude (float): See `create_input_dataframe()`.

    Returns:
        float: The predicted house price, in dollars.

    Raises:
        TypeError: If any value is not numeric.
        ValueError: If any value is missing, NaN, or infinite.
        FileNotFoundError: If the saved model/pipeline files are
            missing.
        RuntimeError: If transforming the input or generating the
            prediction fails.
    """
    input_df = create_input_dataframe(
        MedInc=MedInc,
        HouseAge=HouseAge,
        AveRooms=AveRooms,
        AveBedrms=AveBedrms,
        Population=Population,
        AveOccup=AveOccup,
        Latitude=Latitude,
        Longitude=Longitude,
    )

    return predict_house_price(input_df)



if __name__ == "__main__":
    print("Testing prediction module...")

    sample_input = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25,
    }

    print("\nInput values:")
    for feature, value in sample_input.items():
        print(f"{feature}: {value}")

    # Verify saved model and preprocessing pipeline can be loaded.
    model, pipeline = load_prediction_artifacts()

    # Generate prediction using the existing prediction function.
    predicted_price = predict_from_values(**sample_input)

    # Verify prediction.
    assert isinstance(predicted_price, (int, float))
    assert math.isfinite(predicted_price)
    assert predicted_price >= 0

    print(f"\nPredicted price: ${predicted_price:,.2f}")

    print("\nSaved model loaded successfully.")
    print("Preprocessing pipeline loaded successfully.")
    print("Prediction validation passed.")
    print("\nSUCCESS: predict.py test completed.")

