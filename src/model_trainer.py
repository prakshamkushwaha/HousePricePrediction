"""
src/model_trainer.py

Purpose
-------
This module is responsible ONLY for training regression models for
the House Price Prediction project:
    1. Linear Regression
    2. Ridge Regression
    3. Random Forest Regressor
    4. Gradient Boosting Regressor
    5. XGBoost Regressor

This module does NOT:
    - evaluate models (no MAE, MSE, RMSE, R², MAPE, etc.)
    - select a "best" model
    - perform hyperparameter tuning
    - save models to disk
    - create charts or the dashboard

Those responsibilities belong to separate modules.

A note on XGBoost
--------------------
`XGBRegressor` comes from the third-party `xgboost` package, which is
not part of scikit-learn and must be installed separately
(`pip install xgboost`). If it is missing, this module fails LOUDLY
with a clear `ImportError` as soon as it is imported, instead of
silently training only four models. See the import block below.
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except ImportError as import_error:
    logger.error(
        "model_trainer: could not import XGBRegressor from the "
        "'xgboost' package. Install it with: pip install xgboost"
    )
    raise ImportError(
        "The 'xgboost' package is required by src/model_trainer.py "
        "(for XGBRegressor) but is not installed in this environment. "
        "Install it with: pip install xgboost"
    ) from import_error

# Accepted input types for training data, used in type hints below.
FeatureData = Union[np.ndarray, pd.DataFrame]
TargetData = Union[np.ndarray, pd.Series]


def _ensure_valid_training_data(X_train: FeatureData, y_train: TargetData) -> None:
    """
    Validate X_train and y_train before they are used to train a model.

    Args:
        X_train: The training features.
        y_train: The training target values.

    Raises:
        TypeError: If `X_train` is not a NumPy array or pandas
            DataFrame, or `y_train` is not a NumPy array or pandas
            Series.
        ValueError: If `X_train` or `y_train` is empty, or if they
            do not have the same number of rows.
    """
    if not isinstance(X_train, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            "X_train must be a NumPy array or a pandas DataFrame, "
            f"got {type(X_train).__name__}."
        )

    if not isinstance(y_train, (np.ndarray, pd.Series)):
        raise TypeError(
            "y_train must be a NumPy array or a pandas Series, "
            f"got {type(y_train).__name__}."
        )

    num_feature_rows = X_train.shape[0]
    num_target_rows = y_train.shape[0]

    if num_feature_rows == 0:
        raise ValueError("X_train must not be empty.")

    if num_target_rows == 0:
        raise ValueError("y_train must not be empty.")

    if num_feature_rows != num_target_rows:
        raise ValueError(
            "X_train and y_train must have the same number of rows. "
            f"Got X_train: {num_feature_rows} rows, "
            f"y_train: {num_target_rows} rows."
        )


def create_models() -> Dict[str, Any]:
    """
    Create the untrained regression models used in this project.

    Returns:
        Dict[str, Any]: A dictionary mapping model name to an
        unfitted regressor:
            - "LinearRegression": LinearRegression()
            - "Ridge": Ridge(alpha=1.0)
            - "RandomForest": RandomForestRegressor(
                  n_estimators=100, random_state=42, n_jobs=-1
              )
            - "GradientBoosting": GradientBoostingRegressor(
                  n_estimators=100, learning_rate=0.1, max_depth=3,
                  random_state=42
              )
            - "XGBoost": XGBRegressor(
                  n_estimators=100, learning_rate=0.1, max_depth=3,
                  random_state=42, objective="reg:squarederror",
                  n_jobs=-1
              )
    """
    models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
    }

    logger.info(
        "create_models: created %s untrained model(s): %s.",
        len(models), list(models.keys()),
    )

    return models


def train_model(model: Any, X_train: FeatureData, y_train: TargetData) -> Any:
    """
    Fit a single, already-created model on the training data.

    Args:
        model: An unfitted regressor (must implement `.fit()`) —
            e.g. a scikit-learn estimator or an XGBRegressor.
        X_train: The training features.
        y_train: The training target values.

    Returns:
        The same model instance passed in, now fitted on
        X_train/y_train.

    Raises:
        TypeError: If `X_train`/`y_train` are not the expected types.
        ValueError: If `X_train`/`y_train` are empty or their row
            counts do not match.
        RuntimeError: If fitting the model fails for any other
            reason.
    """
    _ensure_valid_training_data(X_train, y_train)

    model_name = type(model).__name__

    try:
        logger.info(
            "train_model: fitting %s on %s row(s)...",
            model_name, X_train.shape[0],
        )
        model.fit(X_train, y_train)
    except Exception as error:
        logger.error("train_model: failed to fit %s: %s", model_name, error)
        raise RuntimeError(f"Failed to train {model_name}: {error}") from error

    logger.info("train_model: %s trained successfully.", model_name)

    return model


def train_all_models(X_train: FeatureData, y_train: TargetData) -> Dict[str, Any]:
    """
    Create and train every regression model defined in this project.

    Args:
        X_train: The training features.
        y_train: The training target values.

    Returns:
        Dict[str, Any]: A dictionary mapping model name to fitted
        model, using the same names as `create_models()`
        ("LinearRegression", "Ridge", "RandomForest",
        "GradientBoosting", "XGBoost").

    Raises:
        TypeError: If `X_train`/`y_train` are not the expected types.
        ValueError: If `X_train`/`y_train` are empty or their row
            counts do not match.
        RuntimeError: If training any model fails.
    """
    _ensure_valid_training_data(X_train, y_train)

    logger.info("train_all_models: starting training for all models...")

    untrained_models = create_models()
    fitted_models: Dict[str, Any] = {}

    for model_name, model in untrained_models.items():
        fitted_models[model_name] = train_model(model, X_train, y_train)

    logger.info(
        "train_all_models: finished training. Model(s): %s.",
        list(fitted_models.keys()),
    )

    return fitted_models


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from sklearn.utils.validation import check_is_fitted

    try:
        from src.data_loader import load_housing_data
        from src.preprocessing import preprocess_data, preprocess_training_data
        from src.data_split import split_data
    except ImportError:
        from data_loader import load_housing_data
        from preprocessing import preprocess_data, preprocess_training_data
        from data_split import split_data

    print("Step 1: Loading California Housing dataset...")
    housing_data = load_housing_data()

    print("Step 2: Cleaning data and separating X/y...")
    X, y = preprocess_data(housing_data)

    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Step 4: Fitting the preprocessing pipeline ONLY on X_train, "
          "and transforming X_train...")
    X_train_processed, fitted_pipeline = preprocess_training_data(X_train)
    print("Processed X_train shape:", X_train_processed.shape)

    print("Step 5: Training all five models...")
    trained_models = train_all_models(X_train_processed, y_train)

    print()
    print("Model names:", list(trained_models.keys()))
    print()

    print("Step 6: Verifying every model is fitted with check_is_fitted()...")
    for model_name, fitted_model in trained_models.items():
        try:
            check_is_fitted(fitted_model)
            print(f" - {model_name}: trained successfully.")
        except Exception as error:
            print(f" - {model_name}: NOT fitted correctly ({error}).")

    print()
    assert len(trained_models) == 5, f"Expected 5 models, got {len(trained_models)}!"
    print("All five models trained without errors.")
