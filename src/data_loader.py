"""
src/data_loader.py

Purpose
-------
This module is responsible for loading the raw dataset used in the
House Price Prediction project. For this first version, it loads the
California Housing dataset that ships with scikit-learn.

This module intentionally does NOT perform any cleaning, outlier
removal, encoding, or scaling. Its only responsibilities are:
    1. Fetching the raw dataset and returning it as a pandas DataFrame.
    2. Providing a quick summary of that DataFrame for inspection.
"""

import logging
from typing import Any, Dict

import pandas as pd
from sklearn.datasets import fetch_california_housing

logger = logging.getLogger(__name__)

# Name of the target column expected in the returned DataFrame.
TARGET_COLUMN: str = "MedHouseVal"


def load_housing_data() -> pd.DataFrame:
    """
    Load the California Housing dataset as a pandas DataFrame.

    Uses `sklearn.datasets.fetch_california_housing` to fetch the
    dataset (downloading it on first use, then reading from a local
    cache afterwards) and returns it as a single DataFrame containing
    both the feature columns and the target column.

    Returns:
        pd.DataFrame: A DataFrame with the housing feature columns
        plus a target column named "MedHouseVal" (median house value
        for California districts, in units of $100,000).

    Raises:
        RuntimeError: If the dataset cannot be fetched or converted
        into a DataFrame for any reason (e.g. no internet access).
    """
    try:
        logger.info("Loading California Housing dataset...")
        housing_bunch = fetch_california_housing(as_frame=True)

        # `as_frame=True` returns a Bunch whose `.frame` attribute is
        # already a DataFrame with features + target combined.
        housing_df: pd.DataFrame = housing_bunch.frame.copy()

        # Guarantee the target column has the exact expected name.
        if TARGET_COLUMN not in housing_df.columns:
            original_target_name = housing_bunch.target_names[0]
            housing_df = housing_df.rename(
                columns={original_target_name: TARGET_COLUMN}
            )

        logger.info("Dataset loaded successfully. Shape: %s", housing_df.shape)
        return housing_df

    except Exception as error:
        logger.error("Failed to load housing dataset: %s", error)
        raise RuntimeError(f"Could not load housing dataset: {error}") from error


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Print and return a basic summary of a dataset.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.

    Returns:
        Dict[str, Any]: A summary dictionary with the following keys:
            - "num_rows": total number of rows
            - "num_columns": total number of columns
            - "column_names": list of all column names
            - "missing_values": missing value count per column
            - "data_types": data type of each column

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("get_dataset_info() expects a pandas DataFrame.")

    num_rows, num_columns = df.shape
    missing_values = df.isnull().sum()
    data_types = df.dtypes

    dataset_info: Dict[str, Any] = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "column_names": list(df.columns),
        "missing_values": missing_values.to_dict(),
        "data_types": data_types.astype(str).to_dict(),
    }

    print("Number of rows:", num_rows)
    print("Number of columns:", num_columns)
    print("Column names:", dataset_info["column_names"])
    print("\nMissing values per column:")
    print(missing_values)
    print("\nData types:")
    print(data_types)

    logger.info("Dataset info generated: %s rows, %s columns", num_rows, num_columns)
    return dataset_info


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    housing_data = load_housing_data()
    get_dataset_info(housing_data)
