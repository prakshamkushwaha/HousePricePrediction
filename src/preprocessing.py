"""
src/preprocessing.py

Purpose
-------
This module builds a reusable, LEAKAGE-SAFE preprocessing pipeline
for the California Housing dataset. It cleans the raw DataFrame
produced by `data_loader.load_housing_data()` and turns it into
model-ready features (X) and target (y).

This module TRANSFORMS data (removes duplicates, fills missing
values, caps outliers, scales features) but it never trains a
machine learning model. Model training happens in a later module.

Data-leakage audit (this version)
------------------------------------
An earlier version of this module called `handle_missing_values()`
and `handle_outliers()` on the FULL dataset inside `preprocess_data()`
— i.e. BEFORE any train/test split existed. Both of those functions
LEARN a statistic from whatever data they are given (a column
median, or IQR-based outlier bounds) and immediately apply it to
that same data. Doing this before splitting meant the median/IQR
values used to clean rows that ended up in the training set were
partly computed from rows that ended up in the test set — a real
(if subtle) data leak.

The fix: every operation that LEARNS a statistic from the data
(median imputation, outlier-bound calculation, and scaling) now
lives INSIDE the scikit-learn `Pipeline` built by
`create_preprocessing_pipeline()`, including a new custom
`OutlierCapper` transformer for outlier handling. The pipeline is
only ever fit on X_train (`fit_preprocessor()` /
`preprocess_training_data()`), and only ever applied — never
re-fit — to X_test or future prediction data
(`transform_features()` / `preprocess_test_data()`).

    - `preprocess_data(df)` now performs ONLY the cleaning steps that
      do NOT learn any statistic from the data: structural
      validation, duplicate removal, and separating X/y. It no
      longer touches missing values, outliers, or scaling.
    - `fit_preprocessor(X_train)` / `preprocess_training_data(X_train)`
      build the 3-step pipeline (outlier capping, imputation,
      scaling) and fit it — must only ever be called on the
      TRAINING split.
    - `transform_features(X, fitted_pipeline)` /
      `preprocess_test_data(X_test, fitted_pipeline)` reuse an
      already-fitted pipeline and only ever call `.transform()`,
      never `.fit()` or `.fit_transform()`.

The standalone `handle_missing_values()` and `handle_outliers()`
functions are KEPT (per project convention) but are no longer called
by `preprocess_data()`. Their docstrings now clearly warn that they
both LEARN and APPLY a statistic in a single call, so they are only
safe for one-off/exploratory use on a dataset that will not later be
split for training — never as part of the train/test workflow.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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

    This is "basic cleaning": it only checks whether rows are exact
    matches of each other. It does not learn or store any statistic
    that gets reused later, so it is safe to run BEFORE a train/test
    split without causing data leakage.

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
    [STANDALONE UTILITY — NOT used by the leakage-safe workflow]

    Fill missing values in numerical columns using median imputation.

    Learning vs. applying (read this before using this function)
    -----------------------------------------------------------------
    This function both LEARNS a statistic (each column's median) AND
    APPLIES it to fill that SAME DataFrame's missing values, in one
    call. That combination is only safe when `df` will not later be
    split into training/test sets — e.g. for a quick, one-off,
    exploratory cleanup. It must NOT be used on a dataset that is
    about to be split for model training, because the median it
    learns would be computed using rows that later end up in the
    test set, leaking test-set information into training data.

    For the actual leakage-safe train/test workflow, missing-value
    imputation is instead handled by the `SimpleImputer` step inside
    the pipeline from `create_preprocessing_pipeline()`, which is
    fit ONLY on X_train (see `fit_preprocessor()` /
    `preprocess_training_data()`) and then only ever applied — never
    re-fit — to X_test or future data (see `transform_features()` /
    `preprocess_test_data()`).

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
    [STANDALONE UTILITY — NOT used by the leakage-safe workflow]

    Handle outliers in numerical feature columns using IQR-based
    capping (winsorization).

    Learning vs. applying (read this before using this function)
    -----------------------------------------------------------------
    Just like `handle_missing_values()`, this function both LEARNS a
    statistic (each column's IQR-based lower/upper bound) AND
    APPLIES it to cap that SAME DataFrame's values, in one call. It
    is only safe for one-off/exploratory use on data that will not
    later be split for training. Using it before a train/test split
    would leak test-set information into the bounds used to clean
    training rows.

    For the actual leakage-safe train/test workflow, outlier capping
    is instead handled by the `OutlierCapper` step (defined below)
    inside the pipeline from `create_preprocessing_pipeline()`, which
    learns its bounds ONLY from X_train and then only ever applies
    them — never relearns them — to X_test or future data.

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

    This only selects columns — it does not learn or store any
    statistic — so it is safe to use before or after a train/test
    split.

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


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that caps (winsorizes) outliers
    in numeric feature columns using IQR-based bounds.

    Why a custom transformer
    ---------------------------
    scikit-learn does not ship a built-in outlier-capping
    transformer. Wrapping this logic as a proper fit/transform
    transformer — instead of a plain DataFrame function like the old
    `handle_outliers()` — turns the outlier bounds into "statistics
    learned from data", exactly like `SimpleImputer`'s median or
    `StandardScaler`'s mean/std: learned ONLY during `.fit()` (on
    X_train), and applied unchanged during every later
    `.transform()` call (on X_train itself, X_test, or future
    prediction data). This is what prevents outlier-threshold
    leakage from the test set into the training statistics.

    Learning vs. applying
    ------------------------
    - `fit(X)`: LEARNS the lower/upper bound for every numeric
      column in X, using Q1 - 1.5*IQR and Q3 + 1.5*IQR. This should
      only ever be called with X_train.
    - `transform(X)`: APPLIES the bounds learned during `fit()` to
      clip values in X. It NEVER recomputes bounds from X, so it is
      always safe to call on X_train, X_test, or brand-new future
      prediction data.

    Note: this transformer only ever receives FEATURE columns (never
    the target), because it is used inside the pipeline AFTER
    `prepare_features_and_target()` has already separated X from y.
    So unlike the standalone `handle_outliers()` function, it does
    not need any special-case logic to skip a target column.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "OutlierCapper":
        """
        Learn IQR-based outlier bounds from X.

        Args:
            X (pd.DataFrame): The data to learn bounds from. Pass
                X_train only.
            y: Ignored. Present only for scikit-learn API
                compatibility.

        Returns:
            OutlierCapper: self, now fitted.
        """
        X = self._as_dataframe(X)

        self.feature_names_in_ = list(X.columns)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for column in self.feature_names_in_:
            if not pd.api.types.is_numeric_dtype(X[column]):
                continue

            first_quartile = X[column].quantile(0.25)
            third_quartile = X[column].quantile(0.75)
            interquartile_range = third_quartile - first_quartile

            self.lower_bounds_[column] = first_quartile - 1.5 * interquartile_range
            self.upper_bounds_[column] = third_quartile + 1.5 * interquartile_range

        logger.info(
            "OutlierCapper.fit: learned outlier bounds for %s column(s) "
            "from %s row(s).",
            len(self.lower_bounds_), X.shape[0],
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the outlier bounds learned during `fit()` to X.

        Args:
            X (pd.DataFrame): The data to cap. Can be X_train,
                X_test, or future prediction data.

        Returns:
            pd.DataFrame: A new DataFrame with values capped to the
            bounds learned during `fit()`.

        Raises:
            NotFittedError: If `fit()` has not been called yet.
        """
        if not hasattr(self, "lower_bounds_"):
            raise NotFittedError(
                "This OutlierCapper instance is not fitted yet. Call "
                "'fit' with X_train before calling 'transform'."
            )

        X = self._as_dataframe(X).copy()
        affected_row_mask = pd.Series(False, index=X.index)

        for column, lower_bound in self.lower_bounds_.items():
            if column not in X.columns:
                continue
            upper_bound = self.upper_bounds_[column]

            is_outlier = (X[column] < lower_bound) | (X[column] > upper_bound)
            affected_row_mask = affected_row_mask | is_outlier

            X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)

        logger.info(
            "OutlierCapper.transform: %s row(s) had at least one value "
            "capped using bounds learned during fit().",
            int(affected_row_mask.sum()),
        )

        return X

    @staticmethod
    def _as_dataframe(X) -> pd.DataFrame:
        """Convert X to a DataFrame if it isn't already one."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def create_preprocessing_pipeline() -> Pipeline:
    """
    Build a reusable, leakage-safe scikit-learn preprocessing
    pipeline.

    The California Housing dataset contains only numerical features,
    so this pipeline has three steps, in order:
        1. "outlier_capper" — `OutlierCapper()`: learns per-column
           IQR outlier bounds and caps values to them.
        2. "imputer" — `SimpleImputer(strategy="median")`: learns
           per-column medians and fills missing values. This runs
           AFTER outlier capping so the learned medians are not
           skewed by extreme values.
        3. "scaler" — `StandardScaler()`: learns per-column mean and
           standard deviation and scales features to zero mean /
           unit variance.

    EVERY step here LEARNS its statistics during `.fit()` and only
    APPLIES them during `.transform()`. No categorical encoding is
    included, since this dataset has no categorical columns.

    Important: this function only BUILDS the pipeline — it does not
    fit it. Fitting should only ever happen on training data, using
    `fit_preprocessor()` or `preprocess_training_data()`, to avoid
    leaking information from validation, test, or future prediction
    data into the model.

    Returns:
        Pipeline: An unfitted, 3-step scikit-learn Pipeline.
    """
    pipeline = Pipeline(
        steps=[
            ("outlier_capper", OutlierCapper()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    step_names = [name for name, _ in pipeline.steps]
    logger.info("create_preprocessing_pipeline: built pipeline with steps %s.", step_names)

    return pipeline


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validate and perform LEAKAGE-SAFE basic cleaning on a raw housing
    DataFrame, then split it into features (X) and target (y).

    Leakage-safety note
    ----------------------
    This function ONLY performs steps that do not learn any
    statistic from the data:
        1. Validate the input DataFrame's structure.
        2. Remove duplicate rows.
        3. Separate features (X) from the target (y).

    It deliberately does NOT impute missing values, cap outliers, or
    scale features here — those operations learn statistics (median,
    IQR bounds, mean/standard deviation) from whatever data they are
    given, so performing them before a train/test split would leak
    information from what will become the test set into the training
    data. Those steps now live inside the pipeline built by
    `create_preprocessing_pipeline()`, which must be fit on X_train
    only.

    Correct workflow using this function
    -------------------------------------
        1. X, y = preprocess_data(df)
        2. X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
        3. X_train_processed, fitted_pipeline = preprocess_training_data(X_train)
        4. X_test_processed = preprocess_test_data(X_test, fitted_pipeline)

    Args:
        df (pd.DataFrame): The raw housing DataFrame to clean.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: Feature columns with duplicate rows removed. May
              still contain missing values and outliers — those are
              handled later by the fitted pipeline, not here.
            - y: The target Series ("MedHouseVal").

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `df` fails structural validation, or the
            target column is missing.
    """
    _ensure_is_dataframe(df, "preprocess_data")

    logger.info(
        "preprocess_data: starting leakage-safe cleaning (duplicate "
        "removal only — no statistics are learned from the data here)."
    )

    validation_result = validate_dataframe(df)
    if not validation_result["is_valid"]:
        raise ValueError(
            f"Input DataFrame failed validation: {validation_result['issues']}"
        )

    cleaned_df = remove_duplicates(df)
    features, target = prepare_features_and_target(cleaned_df)

    logger.info(
        "preprocess_data: cleaning complete. X shape=%s, y shape=%s. "
        "Split this with train_test_split(), then fit the "
        "preprocessing pipeline (outlier capping, imputation, "
        "scaling) on X_train only.",
        features.shape, target.shape,
    )

    return features, target


def fit_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Build a new preprocessing pipeline and fit it ONLY on X_train.

    Data-leakage prevention
    ------------------------
    Fitting a pipeline means learning statistics from the data:
    outlier bounds (IQR), median values (imputation), and mean /
    standard deviation (scaling). All of these must only ever be
    learned from training data. This function always builds a
    brand-new pipeline via `create_preprocessing_pipeline()` and
    fits it here, so there is no way to accidentally reuse a
    pipeline that was already fit on other data.

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
    statistics (outlier bounds, medians, or scaling parameters).

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
            - X_train_processed: The transformed training features
              (outliers capped, missing values imputed, scaled).
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
    outlier bounds, imputation values, and scaling statistics applied
    to `X_test` are exactly the ones learned from the training data,
    never from `X_test` itself.

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

    # Step 2: Basic, leakage-free cleaning (duplicates only) and
    # splitting into X/y. No statistics are learned yet.
    print("Step 2: Cleaning data (duplicates only) and separating X/y...")
    X, y = preprocess_data(housing_data)
    print("Original X shape:", X.shape)

    # Step 3: Split BEFORE any statistics are learned.
    print("Step 3: Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # Steps 4-5: Fit the pipeline (outlier bounds, median, scaling)
    # ONLY on X_train, and transform X_train.
    print("Step 4-5: Fitting pipeline on X_train and transforming X_train...")
    X_train_processed, fitted_pipeline = preprocess_training_data(X_train)
    print("Processed X_train shape:", X_train_processed.shape)

    # Capture every learned statistic BEFORE touching X_test, so we
    # can prove afterwards that transforming X_test changed none of
    # them (i.e. X_test never influences preprocessing statistics).
    outlier_lower_bounds_before = dict(fitted_pipeline.named_steps["outlier_capper"].lower_bounds_)
    outlier_upper_bounds_before = dict(fitted_pipeline.named_steps["outlier_capper"].upper_bounds_)
    imputer_stats_before = fitted_pipeline.named_steps["imputer"].statistics_.copy()
    scaler_mean_before = fitted_pipeline.named_steps["scaler"].mean_.copy()

    # Step 6: Transform X_test using the SAME fitted pipeline.
    print("Step 6: Transforming X_test with the already-fitted pipeline...")
    X_test_processed = preprocess_test_data(X_test, fitted_pipeline)
    print("Processed X_test shape:", X_test_processed.shape)

    # Verify NONE of the learned statistics changed while
    # transforming X_test — this is the core leakage-prevention check.
    outlier_lower_bounds_after = fitted_pipeline.named_steps["outlier_capper"].lower_bounds_
    outlier_upper_bounds_after = fitted_pipeline.named_steps["outlier_capper"].upper_bounds_
    imputer_stats_after = fitted_pipeline.named_steps["imputer"].statistics_
    scaler_mean_after = fitted_pipeline.named_steps["scaler"].mean_

    assert outlier_lower_bounds_before == outlier_lower_bounds_after, (
        "Outlier LOWER bounds changed — X_test influenced the pipeline!"
    )
    assert outlier_upper_bounds_before == outlier_upper_bounds_after, (
        "Outlier UPPER bounds changed — X_test influenced the pipeline!"
    )
    assert np.array_equal(imputer_stats_before, imputer_stats_after), (
        "Imputer statistics changed — the pipeline was refit on X_test!"
    )
    assert np.array_equal(scaler_mean_before, scaler_mean_after), (
        "Scaler statistics changed — the pipeline was refit on X_test!"
    )
    print("Verified: outlier bounds, imputer medians, and scaler stats")
    print("all unchanged after transforming X_test.")
    print("No data leakage: X_test was only transformed, never used to fit.")
