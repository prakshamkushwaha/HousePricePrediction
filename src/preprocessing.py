"""
src/preprocessing.py

Purpose
-------
This module builds a reusable preprocessing pipeline for the
California Housing dataset. It cleans the raw DataFrame produced by
`data_loader.load_housing_data()` and turns it into model-ready
features (X) and target (y).

This module TRANSFORMS data (removes duplicates, fills missing
values, caps outliers, scales features) but it never trains a
machine learning model. Model training happens in a later module.

Data-leakage prevention
------------------------
The scikit-learn preprocessing pipeline (imputer + scaler) LEARNS
statistics from the data it is fit on (e.g. median values, mean,
standard deviation). If the pipeline were fit on the full dataset
before splitting into train/test sets, information from the test
set would leak into those statistics, making evaluation results
overly optimistic.

To prevent this, the pipeline is never fit on the full dataset in
this module. Instead:
    - `preprocess_data(df)` only cleans the data (duplicates,
      missing values, outliers) and separates X/y. It does NOT fit
      or use the scikit-learn pipeline at all.
    - `fit_preprocessor(X_train)` / `preprocess_training_data(X_train)`
      build and fit the pipeline, and must only ever be called on
      the TRAINING split.
    - `transform_features(X, fitted_pipeline)` /
      `preprocess_test_data(X_test, fitted_pipeline)` reuse an
      already-fitted pipeline and only ever call `.transform()`,
      never `.fit()` or `.fit_transform()`.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the target column name and the validation check already
# defined in the earlier modules, so all three modules always agree
# on column names and validation rules. This try/except supports
# running this module either as part of the `src` package (e.g.
# `from src.preprocessing import ...`) or as a standalone script
# from inside the `src/` folder.
try:
    from src.data_loader import TARGET_COLUMN
    from src.data_validation import validate_dataframe
except ImportError:
    from data_loader import TARGET_COLUMN
    from data_validation import validate_dataframe

logger = logging.getLogger(__name__)


def _ensure_is_dataframe(df: pd.DataFrame, function_name: str) -> None:
    """
    Raise a clear error if `df` is not a pandas DataFrame.

    Args:
        df (pd.DataFrame): The object to check.
        function_name (str): Name of the calling function, used to
            make the error message easier to trace.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{function_name}() expects a pandas DataFrame.")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicated rows from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to de-duplicate.

    Returns:
        pd.DataFrame: A new DataFrame with duplicate rows removed and
        the row index reset.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "remove_duplicates")

    rows_before = len(df)
    deduplicated_df = df.drop_duplicates().reset_index(drop=True)
    rows_after = len(deduplicated_df)
    num_removed = rows_before - rows_after

    logger.info(
        "remove_duplicates: removed %s duplicate row(s) (before=%s, after=%s).",
        num_removed, rows_before, rows_after,
    )

    return deduplicated_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in numerical columns using median imputation.

    Strategy
    --------
    For every numeric column that has missing values, this fills
    them with that column's median. The median is used instead of
    the mean because it is less sensitive to outliers, which keeps
    skewed columns (like "Population" or "AveOccup") from being
    filled with a distorted value.

    No rows or columns are ever dropped here, so the dataset never
    shrinks because of missing values.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: A new DataFrame with missing numeric values
        filled in.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "handle_missing_values")

    filled_df = df.copy()
    numeric_columns = filled_df.select_dtypes(include=[np.number]).columns

    for column in numeric_columns:
        missing_count = int(filled_df[column].isnull().sum())
        if missing_count > 0:
            median_value = filled_df[column].median()
            filled_df[column] = filled_df[column].fillna(median_value)
            logger.info(
                "handle_missing_values: filled %s missing value(s) in '%s' "
                "with the column median (%.4f).",
                missing_count, column, median_value,
            )

    remaining_missing = int(filled_df.isnull().sum().sum())
    if remaining_missing == 0:
        logger.info("handle_missing_values: no missing values remain.")
    else:
        logger.warning(
            "handle_missing_values: %s missing value(s) remain in "
            "non-numeric column(s) and were not touched.",
            remaining_missing,
        )

    return filled_df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in numerical feature columns using IQR-based
    capping (winsorization).

    Strategy
    --------
    For every numeric FEATURE column (the target column is
    intentionally skipped — see note below):
        1. Compute Q1 (25th percentile) and Q3 (75th percentile).
        2. Compute the interquartile range: IQR = Q3 - Q1.
        3. Define "normal" bounds as:
               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR
        4. Any value outside these bounds is CAPPED (clipped) to the
           nearest bound, instead of being removed.

    Capping (rather than deleting rows) is used because it keeps
    every row in the dataset while still reducing the influence of
    extreme values on the model.

    The target column ("MedHouseVal") is never modified by this
    function. Extreme house values are real, valid outcomes we want
    to predict, so this function does not cap or remove rows based
    on the target column containing extreme values.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: A new DataFrame with feature outliers capped.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "handle_outliers")

    capped_df = df.copy()
    feature_columns = [
        column
        for column in capped_df.select_dtypes(include=[np.number]).columns
        if column != TARGET_COLUMN
    ]

    affected_row_mask = pd.Series(False, index=capped_df.index)

    for column in feature_columns:
        first_quartile = capped_df[column].quantile(0.25)
        third_quartile = capped_df[column].quantile(0.75)
        interquartile_range = third_quartile - first_quartile

        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range

        is_outlier = (capped_df[column] < lower_bound) | (capped_df[column] > upper_bound)
        affected_row_mask = affected_row_mask | is_outlier

        capped_df[column] = capped_df[column].clip(lower=lower_bound, upper=upper_bound)

    num_affected_rows = int(affected_row_mask.sum())
    logger.info(
        "handle_outliers: %s row(s) had at least one feature value capped "
        "(target column '%s' was not touched).",
        num_affected_rows, TARGET_COLUMN,
    )

    return capped_df


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y).

    Args:
        df (pd.DataFrame): The DataFrame to split. Must contain the
            target column "MedHouseVal".

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: All columns except the target column.
            - y: The target column ("MedHouseVal") as a Series.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If the target column is missing from `df`.
    """
    _ensure_is_dataframe(df, "prepare_features_and_target")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Cannot separate features and target: column '{TARGET_COLUMN}' "
            f"was not found in the DataFrame."
        )

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    logger.info(
        "prepare_features_and_target: X shape=%s, y shape=%s.",
        features.shape, target.shape,
    )

    return features, target


def create_preprocessing_pipeline() -> Pipeline:
    """
    Build a reusable scikit-learn preprocessing pipeline.

    The California Housing dataset contains only numerical features,
    so this pipeline only needs two steps:
        1. Impute any remaining missing values with the median.
        2. Scale all features to zero mean / unit variance using
           StandardScaler.

    No categorical encoding is included, since this dataset has no
    categorical columns.

    Important: this function only BUILDS the pipeline — it does not
    fit it. Fitting should only ever happen on training data, using
    `fit_preprocessor()` or `preprocess_training_data()`, to avoid
    leaking information from validation, test, or future prediction
    data into the model.

    Returns:
        Pipeline: An unfitted scikit-learn Pipeline with an imputer
        step and a scaler step.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    step_names = [name for name, _ in pipeline.steps]
    logger.info("create_preprocessing_pipeline: built pipeline with steps %s.", step_names)

    return pipeline


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and clean a raw housing DataFrame, then split it into
    features (X) and target (y).

    This function does NOT build, fit, or apply the scikit-learn
    preprocessing pipeline (imputer + scaler). It only performs the
    row-level cleaning steps and the X/y split. This is intentional:
    an earlier version of this function used to fit the pipeline on
    the entire dataset here, before any train/test split existed,
    which leaked test-set information into the training statistics.
    Stopping before pipeline-fitting removes that risk entirely.

    Correct workflow using this function
    -------------------------------------
        1. X, y = preprocess_data(df)
        2. X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
        3. X_train_processed, fitted_pipeline = preprocess_training_data(X_train)
        4. X_test_processed = preprocess_test_data(X_test, fitted_pipeline)

    Steps performed here, in order:
        1. Validate the input DataFrame using
           `data_validation.validate_dataframe()`.
        2. Remove duplicate rows.
        3. Fill missing numeric values (median imputation).
        4. Cap outliers in feature columns (IQR-based).
        5. Separate features (X) from the target (y).

    Args:
        df (pd.DataFrame): The raw housing DataFrame to clean.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: Cleaned feature columns, still a plain DataFrame
              (not yet imputed/scaled by the pipeline).
            - y: The target Series ("MedHouseVal").

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `df` fails structural validation, or the
            target column is missing.
    """
    _ensure_is_dataframe(df, "preprocess_data")

    logger.info("preprocess_data: starting cleaning workflow (no pipeline fitting here).")

    validation_result = validate_dataframe(df)
    if not validation_result["is_valid"]:
        raise ValueError(
            f"Input DataFrame failed validation: {validation_result['issues']}"
        )

    cleaned_df = remove_duplicates(df)
    cleaned_df = handle_missing_values(cleaned_df)
    cleaned_df = handle_outliers(cleaned_df)

    features, target = prepare_features_and_target(cleaned_df)

    logger.info(
        "preprocess_data: cleaning complete. X shape=%s, y shape=%s. "
        "Split this with train_test_split() before fitting the "
        "preprocessing pipeline.",
        features.shape, target.shape,
    )

    return features, target


def fit_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Build a new preprocessing pipeline and fit it ONLY on X_train.

    Data-leakage prevention
    ------------------------
    Fitting a pipeline means learning statistics from the data
    (median values for imputation, mean/standard deviation for
    scaling). Those statistics must only ever be learned from
    training data. This function always builds a brand-new pipeline
    via `create_preprocessing_pipeline()` and fits it here, so there
    is no way to accidentally reuse a pipeline that was already fit
    on other data.

    Args:
        X_train (pd.DataFrame): The TRAINING features only. Never
            pass validation, test, or future prediction data here.

    Returns:
        Pipeline: The pipeline, now fitted on X_train.

    Raises:
        TypeError: If `X_train` is not a pandas DataFrame.
        RuntimeError: If fitting the pipeline fails.
    """
    _ensure_is_dataframe(X_train, "fit_preprocessor")

    pipeline = create_preprocessing_pipeline()

    try:
        pipeline.fit(X_train)
    except Exception as error:
        logger.error("fit_preprocessor: failed to fit pipeline on X_train: %s", error)
        raise RuntimeError(f"Failed to fit preprocessing pipeline: {error}") from error

    logger.info("fit_preprocessor: pipeline fitted on X_train with shape %s.", X_train.shape)

    return pipeline


def transform_features(X: pd.DataFrame, fitted_pipeline: Pipeline) -> np.ndarray:
    """
    Transform features using an already-fitted preprocessing pipeline.

    This function NEVER calls `.fit()` or `.fit_transform()` — only
    `.transform()`. That makes it safe to use on validation, test, or
    future prediction data without changing the pipeline's learned
    statistics.

    Args:
        X (pd.DataFrame): The features to transform. Can be
            training, validation, test, or future prediction data.
        fitted_pipeline (Pipeline): A pipeline that has already been
            fitted (e.g. via `fit_preprocessor()`).

    Returns:
        np.ndarray: The transformed feature array.

    Raises:
        TypeError: If `X` is not a pandas DataFrame.
        RuntimeError: If `fitted_pipeline` has not been fitted yet,
            or transforming fails for any other reason.
    """
    _ensure_is_dataframe(X, "transform_features")

    try:
        transformed = fitted_pipeline.transform(X)
    except NotFittedError as error:
        logger.error("transform_features: pipeline is not fitted yet: %s", error)
        raise RuntimeError(
            "The pipeline passed to transform_features() has not been "
            "fitted. Fit it first with fit_preprocessor() or "
            "preprocess_training_data()."
        ) from error
    except Exception as error:
        logger.error("transform_features: failed to transform data: %s", error)
        raise RuntimeError(f"Failed to transform features: {error}") from error

    logger.info("transform_features: transformed data with shape %s.", X.shape)

    return transformed


def preprocess_training_data(X_train: pd.DataFrame) -> Tuple[np.ndarray, Pipeline]:
    """
    Fit the preprocessing pipeline on X_train and transform X_train.

    This is a convenience wrapper around `fit_preprocessor()` +
    `transform_features()`. Call this ONCE, on the training split
    only — never on validation, test, or future prediction data.

    Args:
        X_train (pd.DataFrame): The TRAINING features only.

    Returns:
        Tuple[np.ndarray, Pipeline]:
            - X_train_processed: The transformed training features.
            - fitted_pipeline: The pipeline, fitted on X_train, to be
              reused (via `preprocess_test_data()` or
              `transform_features()`) on validation/test/future data.

    Raises:
        TypeError: If `X_train` is not a pandas DataFrame.
        RuntimeError: If fitting or transforming fails.
    """
    _ensure_is_dataframe(X_train, "preprocess_training_data")

    fitted_pipeline = fit_preprocessor(X_train)
    X_train_processed = transform_features(X_train, fitted_pipeline)

    logger.info(
        "preprocess_training_data: X_train_processed shape=%s.",
        X_train_processed.shape,
    )

    return X_train_processed, fitted_pipeline


def preprocess_test_data(X_test: pd.DataFrame, fitted_pipeline: Pipeline) -> np.ndarray:
    """
    Transform test (or validation/future) features using a pipeline
    already fitted on training data.

    Data-leakage prevention
    ------------------------
    This function does NOT fit or refit the pipeline — it only calls
    `.transform()` (via `transform_features()`). This guarantees the
    imputation values and scaling statistics applied to `X_test` are
    exactly the ones learned from the training data, never from
    `X_test` itself.

    Args:
        X_test (pd.DataFrame): The test (or validation/future)
            features to transform.
        fitted_pipeline (Pipeline): A pipeline already fitted on
            training data (e.g. via `fit_preprocessor()` or
            `preprocess_training_data()`).

    Returns:
        np.ndarray: The transformed test feature array.

    Raises:
        TypeError: If `X_test` is not a pandas DataFrame.
        RuntimeError: If `fitted_pipeline` has not been fitted yet,
            or transforming fails for any other reason.
    """
    logger.info("preprocess_test_data: transforming test data with the fitted pipeline.")
    return transform_features(X_test, fitted_pipeline)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from sklearn.model_selection import train_test_split

    try:
        from src.data_loader import load_housing_data
    except ImportError:
        from data_loader import load_housing_data

    # Step 1: Load the raw dataset.
    print("Step 1: Loading California Housing dataset...")
    housing_data = load_housing_data()

    # Step 2: Basic cleaning (duplicates, missing values, outliers)
    # and splitting into X/y. No pipeline fitting happens yet.
    print("Step 2: Cleaning data and separating X/y...")
    X, y = preprocess_data(housing_data)
    print("Original X shape:", X.shape)

    # Step 3: Split BEFORE any pipeline fitting.
    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Steps 4-5: Fit the pipeline ONLY on X_train, and transform X_train.
    print("Step 4-5: Fitting pipeline on X_train and transforming X_train...")
    X_train_processed, fitted_pipeline = preprocess_training_data(X_train)
    print("Processed X_train shape:", X_train_processed.shape)

    # Capture the fitted statistics BEFORE touching X_test, so we can
    # prove afterwards that transforming X_test did not change them.
    imputer_stats_before = fitted_pipeline.named_steps["imputer"].statistics_.copy()
    scaler_mean_before = fitted_pipeline.named_steps["scaler"].mean_.copy()

    # Step 6: Transform X_test using the SAME fitted pipeline.
    print("Step 6: Transforming X_test with the already-fitted pipeline...")
    X_test_processed = preprocess_test_data(X_test, fitted_pipeline)
    print("Processed X_test shape:", X_test_processed.shape)

    # Step 8: Verify the pipeline was NOT refit while transforming X_test.
    imputer_stats_after = fitted_pipeline.named_steps["imputer"].statistics_
    scaler_mean_after = fitted_pipeline.named_steps["scaler"].mean_
    assert np.array_equal(imputer_stats_before, imputer_stats_after), (
        "Imputer statistics changed — the pipeline was refit on X_test!"
    )
    assert np.array_equal(scaler_mean_before, scaler_mean_after), (
        "Scaler statistics changed — the pipeline was refit on X_test!"
    )
    print("Verified: pipeline statistics unchanged after transforming X_test.")
    print("No data leakage: X_test was only transformed, never used to fit.")
