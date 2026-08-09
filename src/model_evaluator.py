"""
src/model_evaluator.py

Purpose
-------
This module evaluates already-trained baseline regression models
(LinearRegression, Ridge, RandomForest) on the UNSEEN test dataset.

It calculates, for every model: MAE, MSE, RMSE, R2, and MAPE, and
helps select the best-performing model based on R2.

This module does NOT retrain or refit any model — every model is
used only via `.predict()`. It also does NOT perform hyperparameter
tuning, save anything to disk, or create charts/dashboards.

No data leakage
------------------
Models are evaluated ONLY on X_test_processed / y_test — features
that were transformed using a preprocessing pipeline fitted only on
the training data (see `preprocessing.preprocess_training_data()`
and `preprocessing.preprocess_test_data()`). Models are never
evaluated on X_train.

MAPE and zero-valued targets
------------------------------
MAPE (Mean Absolute Percentage Error) divides by the true target
value, which is mathematically undefined when a true value is 0.
This module's chosen approach: rows where `y_true == 0` are excluded
from the MAPE calculation only (all other metrics still use every
row), and the number of excluded rows is logged. See
`_calculate_safe_mape()` for details.
"""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# Accepted input types for evaluation data, used in type hints below.
FeatureData = Union[np.ndarray, pd.DataFrame]
TargetData = Union[np.ndarray, pd.Series]


def _ensure_valid_evaluation_data(X_test: FeatureData, y_test: TargetData) -> None:
    """
    Validate X_test and y_test before they are used for evaluation.

    Args:
        X_test: The test features.
        y_test: The true target values for the test set.

    Raises:
        TypeError: If `X_test` is not a NumPy array or pandas
            DataFrame, or `y_test` is not a NumPy array or pandas
            Series.
        ValueError: If `X_test` or `y_test` is empty, or if they do
            not have the same number of rows.
    """
    if not isinstance(X_test, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            "X_test must be a NumPy array or a pandas DataFrame, "
            f"got {type(X_test).__name__}."
        )

    if not isinstance(y_test, (np.ndarray, pd.Series)):
        raise TypeError(
            "y_test must be a NumPy array or a pandas Series, "
            f"got {type(y_test).__name__}."
        )

    num_feature_rows = X_test.shape[0]
    num_target_rows = y_test.shape[0]

    if num_feature_rows == 0:
        raise ValueError("X_test must not be empty.")

    if num_target_rows == 0:
        raise ValueError("y_test must not be empty.")

    if num_feature_rows != num_target_rows:
        raise ValueError(
            "X_test and y_test must have the same number of rows. "
            f"Got X_test: {num_feature_rows} rows, "
            f"y_test: {num_target_rows} rows."
        )


def _calculate_safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MAPE while safely handling zero values in y_true.

    Chosen approach
    -----------------
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100 is mathematically
    undefined when y_true == 0 (division by zero). Rather than adding
    a small epsilon (which would silently distort the metric for
    those rows), this function EXCLUDES rows where y_true == 0 from
    the MAPE calculation entirely, and logs how many rows were
    excluded. If every row has y_true == 0, MAPE cannot be computed
    and NaN is returned.

    Args:
        y_true (np.ndarray): Actual target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: MAPE expressed as a percentage, or NaN if no valid
        (non-zero) rows remain.
    """
    zero_target_mask = y_true == 0
    num_zero_targets = int(zero_target_mask.sum())

    if num_zero_targets > 0:
        logger.warning(
            "_calculate_safe_mape: %s row(s) have y_true == 0 and were "
            "excluded from the MAPE calculation to avoid division by zero.",
            num_zero_targets,
        )

    non_zero_mask = ~zero_target_mask
    if not non_zero_mask.any():
        logger.warning(
            "_calculate_safe_mape: no non-zero y_true values remain; "
            "MAPE is undefined and NaN will be returned."
        )
        return float("nan")

    mape_fraction = mean_absolute_percentage_error(
        y_true[non_zero_mask], y_pred[non_zero_mask]
    )

    return float(mape_fraction * 100)


def calculate_metrics(y_true: TargetData, y_pred: TargetData) -> Dict[str, float]:
    """
    Calculate standard regression metrics comparing predictions to
    true values.

    Args:
        y_true: The true target values.
        y_pred: The predicted target values, aligned row-for-row
            with `y_true`.

    Returns:
        Dict[str, float]: A dictionary with keys:
            - "MAE": Mean Absolute Error
            - "MSE": Mean Squared Error
            - "RMSE": Root Mean Squared Error
            - "R2": R-squared (coefficient of determination)
            - "MAPE": Mean Absolute Percentage Error (as a
              percentage), with zero-valued targets excluded — see
              `_calculate_safe_mape()`.

    Raises:
        ValueError: If `y_true` and `y_pred` have different lengths,
            or either is empty.
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.shape[0] == 0 or y_pred_array.shape[0] == 0:
        raise ValueError("calculate_metrics() cannot be called with empty inputs.")

    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError(
            "y_true and y_pred must have the same number of rows. "
            f"Got y_true: {y_true_array.shape[0]} rows, "
            f"y_pred: {y_pred_array.shape[0]} rows."
        )

    mean_absolute_error_value = float(mean_absolute_error(y_true_array, y_pred_array))
    mean_squared_error_value = float(mean_squared_error(y_true_array, y_pred_array))
    root_mean_squared_error_value = float(np.sqrt(mean_squared_error_value))
    r2_score_value = float(r2_score(y_true_array, y_pred_array))
    mape_value = _calculate_safe_mape(y_true_array, y_pred_array)

    return {
        "MAE": mean_absolute_error_value,
        "MSE": mean_squared_error_value,
        "RMSE": root_mean_squared_error_value,
        "R2": r2_score_value,
        "MAPE": mape_value,
    }


def evaluate_model(model: Any, X_test: FeatureData, y_test: TargetData) -> Dict[str, float]:
    """
    Evaluate a single, already-trained model on the test set.

    This function does NOT retrain or refit the model — it only
    calls `model.predict()`.

    Args:
        model: A fitted scikit-learn regressor (must implement
            `.predict()`).
        X_test: The (already preprocessed) test features
            (X_test_processed).
        y_test: The true target values for the test set.

    Returns:
        Dict[str, float]: The metrics dictionary from
        `calculate_metrics()` ("MAE", "MSE", "RMSE", "R2", "MAPE").

    Raises:
        TypeError: If `X_test`/`y_test` are not the expected types.
        ValueError: If `X_test`/`y_test` are empty or their row
            counts do not match.
        RuntimeError: If generating predictions fails.
    """
    _ensure_valid_evaluation_data(X_test, y_test)

    model_name = type(model).__name__

    try:
        y_pred = model.predict(X_test)
    except Exception as error:
        logger.error("evaluate_model: %s failed to generate predictions: %s", model_name, error)
        raise RuntimeError(f"Failed to generate predictions for {model_name}: {error}") from error

    metrics = calculate_metrics(y_test, y_pred)

    logger.info("evaluate_model: %s metrics -> %s", model_name, metrics)

    return metrics


def evaluate_all_models(
    models: Dict[str, Any], X_test: FeatureData, y_test: TargetData
) -> pd.DataFrame:
    """
    Evaluate every trained model on the test set and compile the
    results into a single comparison table.

    No model is retrained or refit here — each is used only via
    `.predict()` (through `evaluate_model()`). Evaluation always
    uses X_test / y_test, never X_train / y_train.

    Args:
        models (Dict[str, Any]): Mapping of model name -> fitted
            model (e.g. the output of
            `model_trainer.train_all_models()`).
        X_test: The (already preprocessed) test features
            (X_test_processed).
        y_test: The true target values for the test set.

    Returns:
        pd.DataFrame: One row per model, with columns "Model",
        "MAE", "MSE", "RMSE", "R2", "MAPE".

    Raises:
        ValueError: If `models` is empty, or `X_test`/`y_test` are
            invalid.
    """
    if not models:
        raise ValueError("evaluate_all_models() received an empty 'models' dictionary.")

    _ensure_valid_evaluation_data(X_test, y_test)

    logger.info("evaluate_all_models: evaluating %s model(s)...", len(models))

    result_rows = []
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        result_rows.append({"Model": model_name, **metrics})

    results_df = pd.DataFrame(
        result_rows, columns=["Model", "MAE", "MSE", "RMSE", "R2", "MAPE"]
    )

    logger.info("evaluate_all_models: evaluation complete for %s model(s).", len(results_df))

    return results_df


def get_best_model(results: pd.DataFrame) -> Tuple[str, float]:
    """
    Select the best-performing model from an evaluation results table.

    Why R2 is used for selection
    -------------------------------
    R2 (the coefficient of determination) measures the proportion of
    variance in the target variable that a model explains, on a
    scale that's easy to interpret: 1.0 means perfect predictions,
    0.0 means the model does no better than always predicting the
    average target value, and negative values mean it does worse
    than that simple baseline. Unlike MAE, MSE, or RMSE, R2 does not
    depend on the units or scale of the target, which makes it a
    fair, standard metric for comparing different regression models
    against each other. A higher R2 is better, so this function
    picks the model with the maximum R2 in the results table.

    Args:
        results (pd.DataFrame): The evaluation table returned by
            `evaluate_all_models()`, with at least "Model" and "R2"
            columns.

    Returns:
        Tuple[str, float]:
            - best_model_name: The name of the best-performing model.
            - best_r2_score: That model's R2 score.

    Raises:
        ValueError: If `results` is empty or missing required
            columns.
    """
    required_columns = {"Model", "R2"}
    if not required_columns.issubset(results.columns):
        raise ValueError(
            f"results must contain columns {sorted(required_columns)}, "
            f"got {list(results.columns)}."
        )

    if results.empty:
        raise ValueError("get_best_model() received an empty results table.")

    best_row = results.loc[results["R2"].idxmax()]
    best_model_name = str(best_row["Model"])
    best_r2_score = float(best_row["R2"])

    logger.info(
        "get_best_model: best model is '%s' with R2=%.4f.",
        best_model_name, best_r2_score,
    )

    return best_model_name, best_r2_score


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from src.data_loader import load_housing_data
        from src.preprocessing import (
            preprocess_data,
            preprocess_test_data,
            preprocess_training_data,
        )
        from src.data_split import split_data
        from src.model_trainer import train_all_models
    except ImportError:
        from data_loader import load_housing_data
        from preprocessing import (
            preprocess_data,
            preprocess_test_data,
            preprocess_training_data,
        )
        from data_split import split_data
        from model_trainer import train_all_models

    print("Step 1: Loading California Housing dataset...")
    housing_data = load_housing_data()

    print("Step 2: Cleaning data and separating X/y...")
    X, y = preprocess_data(housing_data)

    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Step 4: Fitting preprocessing ONLY on X_train...")
    X_train_processed, fitted_pipeline = preprocess_training_data(X_train)

    print("Step 5: Transforming X_test with the already-fitted pipeline...")
    X_test_processed = preprocess_test_data(X_test, fitted_pipeline)

    print("Step 6: Training all models...")
    trained_models = train_all_models(X_train_processed, y_train)

    print("Step 7-8: Generating predictions on X_test_processed and evaluating...")
    results = evaluate_all_models(trained_models, X_test_processed, y_test)

    print()
    print("Step 8 (table): Comparison table:")
    print(results.to_string(index=False))

    print()
    print("Step 9-10: Selecting the best model...")
    best_model_name, best_r2_score = get_best_model(results)
    print(f"Best Model: {best_model_name}")
    print(f"Best R2 Score: {best_r2_score:.4f}")

    # --- Verification checks ---
    assert len(results) == len(trained_models), "Not every model was evaluated!"

    metric_values = results[["MAE", "MSE", "RMSE", "R2", "MAPE"]].to_numpy(dtype=float)
    assert not np.isnan(metric_values).any(), "Found a NaN metric value!"
    assert not np.isinf(metric_values).any(), "Found an infinite metric value!"

    assert best_model_name == results.loc[results["R2"].idxmax(), "Model"], (
        "Best model was not selected correctly using R2!"
    )

    print()
    print(
        "Verified: all three models were evaluated on X_test_processed/y_test, "
        "no NaN or infinite metric values were produced, and the best model "
        "was correctly selected using R2."
    )
