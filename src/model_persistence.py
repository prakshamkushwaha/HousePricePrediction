"""
src/model_persistence.py

Purpose
-------
This module provides reusable functions for saving and loading:
    1. The selected/best trained machine learning model.
    2. The fitted preprocessing pipeline.

Both are saved as SEPARATE files using joblib, so a dashboard (or
any other future code) can load the model and the pipeline
independently:
    - models/best_model.pkl
    - models/preprocessing_pipeline.pkl

This module also identifies WHICH model is "best" from
already-generated evaluation results (see `save_best_model()`), but
it never trains or evaluates a model itself — it only reads the
results that `model_trainer.py` / `model_evaluator.py` already
produced. This module does NOT contain any Streamlit/dashboard code.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

# Reuse the exact same "which model is best" logic already defined in
# model_evaluator.py, so this module never reimplements or
# second-guesses that selection — it just acts on it. This try/except
# supports running this module either as part of the `src` package or
# as a standalone script from inside the `src/` folder.
try:
    from src.model_evaluator import get_best_model
except ImportError:
    from model_evaluator import get_best_model

logger = logging.getLogger(__name__)

# A path can be given as a string or a pathlib.Path.
PathLike = Union[str, Path]

# Default save locations, used by these functions and by the
# demo/test in the __main__ block below.
MODEL_PATH: Path = Path("models/best_model.pkl")
PIPELINE_PATH: Path = Path("models/preprocessing_pipeline.pkl")


def _ensure_parent_directory_exists(path: Path) -> None:
    """
    Create the parent directory of `path` if it does not exist yet.

    Args:
        path (Path): The file path whose parent directory should
            exist before writing to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, path: PathLike = MODEL_PATH) -> Path:
    """
    Save a trained scikit-learn model to disk using joblib.

    The model must already be fitted. This is checked before saving,
    so an accidentally-untrained model can never be written to disk
    by mistake.

    Args:
        model: A fitted scikit-learn model (any object with a
            `.predict()` method).
        path (PathLike): Where to save the model. Defaults to
            `MODEL_PATH` ("models/best_model.pkl"). Accepts a string
            or a `pathlib.Path`.

    Returns:
        Path: The path the model was saved to.

    Raises:
        TypeError: If `model` does not look like a scikit-learn
            model (has no `.predict()` method).
        RuntimeError: If the model is not fitted yet, or saving
            fails for any other reason.
    """
    save_path = Path(path)
    model_name = type(model).__name__

    if not hasattr(model, "predict"):
        raise TypeError(
            "save_model() expects a scikit-learn model with a "
            f".predict() method, got {model_name}."
        )

    try:
        check_is_fitted(model)
    except NotFittedError as error:
        logger.error("save_model: %s is not fitted yet: %s", model_name, error)
        raise RuntimeError(
            f"Cannot save {model_name}: it has not been fitted yet."
        ) from error

    _ensure_parent_directory_exists(save_path)

    try:
        joblib.dump(model, save_path)
    except Exception as error:
        logger.error("save_model: failed to save %s to '%s': %s", model_name, save_path, error)
        raise RuntimeError(f"Failed to save {model_name} to '{save_path}': {error}") from error

    logger.info("save_model: saved %s to '%s'.", model_name, save_path)

    return save_path


def load_model(path: PathLike = MODEL_PATH) -> Any:
    """
    Load a previously saved scikit-learn model from disk.

    Args:
        path (PathLike): Where to load the model from. Defaults to
            `MODEL_PATH` ("models/best_model.pkl").

    Returns:
        Any: The loaded model object.

    Raises:
        FileNotFoundError: If no file exists at `path`.
        RuntimeError: If loading the file fails for any other
            reason.
    """
    load_path = Path(path)

    if not load_path.exists():
        raise FileNotFoundError(f"No model file found at '{load_path}'.")

    try:
        model = joblib.load(load_path)
    except Exception as error:
        logger.error("load_model: failed to load model from '%s': %s", load_path, error)
        raise RuntimeError(f"Failed to load model from '{load_path}': {error}") from error

    logger.info("load_model: loaded %s from '%s'.", type(model).__name__, load_path)

    return model


def save_preprocessing_pipeline(pipeline: Pipeline, path: PathLike = PIPELINE_PATH) -> Path:
    """
    Save a fitted scikit-learn preprocessing pipeline to disk using
    joblib.

    The pipeline must already be fitted. This is checked before
    saving.

    Args:
        pipeline (Pipeline): A fitted scikit-learn Pipeline (e.g.
            from `preprocessing.fit_preprocessor()` or
            `preprocessing.preprocess_training_data()`).
        path (PathLike): Where to save the pipeline. Defaults to
            `PIPELINE_PATH` ("models/preprocessing_pipeline.pkl").

    Returns:
        Path: The path the pipeline was saved to.

    Raises:
        TypeError: If `pipeline` is not a scikit-learn Pipeline.
        RuntimeError: If the pipeline is not fitted yet, or saving
            fails for any other reason.
    """
    save_path = Path(path)

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "save_preprocessing_pipeline() expects a scikit-learn "
            f"Pipeline, got {type(pipeline).__name__}."
        )

    try:
        check_is_fitted(pipeline)
    except NotFittedError as error:
        logger.error("save_preprocessing_pipeline: pipeline is not fitted: %s", error)
        raise RuntimeError(
            "Cannot save the preprocessing pipeline: it has not been fitted yet."
        ) from error

    _ensure_parent_directory_exists(save_path)

    try:
        joblib.dump(pipeline, save_path)
    except Exception as error:
        logger.error(
            "save_preprocessing_pipeline: failed to save to '%s': %s", save_path, error
        )
        raise RuntimeError(
            f"Failed to save preprocessing pipeline to '{save_path}': {error}"
        ) from error

    logger.info("save_preprocessing_pipeline: saved pipeline to '%s'.", save_path)

    return save_path


def load_preprocessing_pipeline(path: PathLike = PIPELINE_PATH) -> Pipeline:
    """
    Load a previously saved scikit-learn preprocessing pipeline from
    disk.

    Args:
        path (PathLike): Where to load the pipeline from. Defaults
            to `PIPELINE_PATH` ("models/preprocessing_pipeline.pkl").

    Returns:
        Pipeline: The loaded preprocessing pipeline.

    Raises:
        FileNotFoundError: If no file exists at `path`.
        RuntimeError: If loading the file fails for any other
            reason.
    """
    load_path = Path(path)

    if not load_path.exists():
        raise FileNotFoundError(f"No preprocessing pipeline file found at '{load_path}'.")

    try:
        pipeline = joblib.load(load_path)
    except Exception as error:
        logger.error(
            "load_preprocessing_pipeline: failed to load from '%s': %s", load_path, error
        )
        raise RuntimeError(
            f"Failed to load preprocessing pipeline from '{load_path}': {error}"
        ) from error

    logger.info("load_preprocessing_pipeline: loaded pipeline from '%s'.", load_path)

    return pipeline


def save_best_model(
    trained_models: Dict[str, Any],
    evaluation_results: pd.DataFrame,
    fitted_pipeline: Pipeline,
) -> Tuple[str, float]:
    """
    Determine the best model from already-generated evaluation
    results, then save that model and the fitted preprocessing
    pipeline used to produce the features it was trained/evaluated
    on.

    This function does NOT train or evaluate any model itself. It
    only:
        1. Selects the best model name from `evaluation_results`
           (using `model_evaluator.get_best_model()`, which compares
           the already-computed R2 scores — no new evaluation runs).
        2. Looks that model up inside `trained_models` (which must
           already contain fitted models — no new training runs).
        3. Saves the model to `MODEL_PATH` via `save_model()`, and
           `fitted_pipeline` to `PIPELINE_PATH` via
           `save_preprocessing_pipeline()`.

    Args:
        trained_models (Dict[str, Any]): Mapping of model name ->
            fitted model (e.g. the output of
            `model_trainer.train_all_models()`).
        evaluation_results (pd.DataFrame): The evaluation table
            (e.g. from `model_evaluator.evaluate_all_models()`),
            with at least "Model" and "R2" columns.
        fitted_pipeline (Pipeline): The preprocessing pipeline that
            was fit on X_train and used to produce the features the
            models were trained/evaluated on.

    Returns:
        Tuple[str, float]:
            - best_model_name: The name of the best model, as it
              appears in both `trained_models` and
              `evaluation_results`.
            - best_r2_score: That model's R2 score.

    Raises:
        ValueError: If `trained_models` is empty, if
            `evaluation_results` is not a DataFrame containing
            "Model" and "R2" columns, or if the best model name from
            `evaluation_results` cannot be found in `trained_models`.
        RuntimeError: If saving the model or the pipeline fails.
    """
    if not trained_models:
        raise ValueError("save_best_model() received an empty 'trained_models' dictionary.")

    required_columns = {"Model", "R2"}
    if not isinstance(evaluation_results, pd.DataFrame) or not required_columns.issubset(
        evaluation_results.columns
    ):
        raise ValueError(
            "evaluation_results must be a pandas DataFrame containing "
            f"columns {sorted(required_columns)}, got "
            f"{list(getattr(evaluation_results, 'columns', []))}."
        )

    best_model_name, best_r2_score = get_best_model(evaluation_results)

    if best_model_name not in trained_models:
        raise ValueError(
            f"Best model '{best_model_name}' (selected from evaluation_results) "
            f"was not found in trained_models. Available models: "
            f"{list(trained_models.keys())}."
        )

    best_model = trained_models[best_model_name]

    logger.info(
        "save_best_model: best model is '%s' (R2=%.4f). Saving model and pipeline...",
        best_model_name, best_r2_score,
    )
