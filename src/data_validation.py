"""
src/data_validation.py

Purpose
-------
This module validates the raw California Housing DataFrame produced
by `src/data_loader.py`, before any preprocessing takes place.

It only *checks* and *reports* on the data — it never changes it.
Specifically, this module does NOT:
    - modify the data
    - remove rows (e.g. duplicates)
    - handle outliers
    - encode features
    - scale features
    - train any model

Its only job is to answer the question: "Is this DataFrame in the
shape we expect, and is it clean enough to move on to preprocessing?"
"""

import logging
from typing import Any, Dict, List

import pandas as pd

# Reuse the target column name already defined in data_loader.py so
# both modules always agree on what the target column is called.
# This try/except supports running the module either as part of the
# `src` package (e.g. `from src.data_validation import ...`) or as a
# standalone script from inside the `src/` folder.
try:
    from src.data_loader import TARGET_COLUMN
except ImportError:
    from data_loader import TARGET_COLUMN

logger = logging.getLogger(__name__)

# The feature columns that the California Housing dataset is expected
# to have, based on what `data_loader.load_housing_data()` returns.
EXPECTED_FEATURE_COLUMNS: List[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


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


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the overall shape and structure of the housing DataFrame.

    Checks performed:
        - `df` is a pandas DataFrame
        - `df` is not empty
        - the target column ("MedHouseVal") exists
        - all expected California Housing feature columns exist

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        Dict[str, Any]: A structure report with keys:
            - "is_empty": bool
            - "has_target_column": bool
            - "missing_feature_columns": list of missing column names
            - "is_valid": bool, True only if all checks pass
            - "issues": list of human-readable issue descriptions

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "validate_dataframe")

    issues: List[str] = []

    is_empty = df.empty
    if is_empty:
        issues.append("DataFrame is empty (it has 0 rows).")

    has_target_column = TARGET_COLUMN in df.columns
    if not has_target_column:
        issues.append(f"Missing expected target column: '{TARGET_COLUMN}'.")

    missing_feature_columns = [
        column for column in EXPECTED_FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_feature_columns:
        issues.append(f"Missing expected feature column(s): {missing_feature_columns}")

    is_valid = len(issues) == 0

    if is_valid:
        logger.info("Structure validation passed: DataFrame shape looks correct.")
    else:
        logger.warning("Structure validation found issues: %s", issues)

    return {
        "is_empty": is_empty,
        "has_target_column": has_target_column,
        "missing_feature_columns": missing_feature_columns,
        "is_valid": is_valid,
        "issues": issues,
    }


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Count missing (NaN) values for every column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        Dict[str, int]: A mapping of column name -> number of missing
        values in that column.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "check_missing_values")

    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    if total_missing == 0:
        logger.info("No missing values found in the DataFrame.")
    else:
        logger.warning("Found %s missing value(s) across the DataFrame.", total_missing)

    return missing_counts


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Count the number of fully duplicated rows in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        int: The number of duplicate rows found.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "check_duplicates")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        logger.info("No duplicate rows found in the DataFrame.")
    else:
        logger.warning("Found %s duplicate row(s) in the DataFrame.", duplicate_count)

    return duplicate_count


def validate_numeric_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify that the expected feature and target columns are numeric.

    Only checks columns that are both expected (features + target)
    and actually present in `df`, so this stays useful even if
    `validate_dataframe()` already reported missing columns.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        Dict[str, Any]: A report with keys:
            - "non_numeric_columns": list of column names that are
              expected to be numeric but are not
            - "is_valid": bool, True if all checked columns are numeric

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    _ensure_is_dataframe(df, "validate_numeric_columns")

    columns_to_check = [
        column
        for column in EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
        if column in df.columns
    ]

    non_numeric_columns = [
        column
        for column in columns_to_check
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    is_valid = len(non_numeric_columns) == 0

    if is_valid:
        logger.info("All expected columns are numeric.")
    else:
        logger.warning("Found non-numeric column(s): %s", non_numeric_columns)

    return {
        "non_numeric_columns": non_numeric_columns,
        "is_valid": is_valid,
    }


def run_data_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all validation checks on the housing DataFrame and compile
    the results into a single validation report.

    "overall_valid" reflects whether the DataFrame is structurally
    ready for preprocessing (correct type, not empty, expected
    columns present, expected columns numeric). Missing-value and
    duplicate-row counts are included in the report for visibility,
    but do not by themselves affect "overall_valid" — deciding how
    to handle them is left to the preprocessing step, since this
    module never modifies data.

    Args:
        df (pd.DataFrame): The raw housing DataFrame to validate.

    Returns:
        Dict[str, Any]: A validation report with keys:
            - "overall_valid": bool
            - "structure": report from validate_dataframe()
            - "missing_values": report from check_missing_values()
            - "total_missing_values": int
            - "duplicate_rows": int, from check_duplicates()
            - "numeric_columns": report from validate_numeric_columns()

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    logger.info("Starting data validation...")

    structure_report = validate_dataframe(df)
    missing_values_report = check_missing_values(df)
    duplicate_rows = check_duplicates(df)
    numeric_report = validate_numeric_columns(df)

    total_missing_values = sum(missing_values_report.values())

    overall_valid = structure_report["is_valid"] and numeric_report["is_valid"]

    report: Dict[str, Any] = {
        "overall_valid": overall_valid,
        "structure": structure_report,
        "missing_values": missing_values_report,
        "total_missing_values": total_missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_report,
    }

    if overall_valid:
        logger.info("Data validation PASSED. DataFrame is ready for preprocessing.")
    else:
        logger.warning("Data validation FOUND ISSUES. Review the report before proceeding.")

    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from src.data_loader import load_housing_data
    except ImportError:
        from data_loader import load_housing_data

    housing_data = load_housing_data()
    validation_report = run_data_validation(housing_data)
    print(validation_report)
