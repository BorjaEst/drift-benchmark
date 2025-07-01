"""
Data loading and preprocessing module for drift-benchmark.

This module provides utilities for loading datasets from various sources including:
- Synthetic data generation with drift patterns
- File-based datasets (CSV, Parquet, Excel, etc.)
- Built-in scikit-learn datasets
- Custom datasets with preprocessing pipelines

The module follows a standardized interface for consistent data handling across
the drift-benchmark library.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

from drift_benchmark.constants import (
    DatasetConfig,
    DatasetMetadata,
    DatasetType,
    FileDataConfig,
    SklearnDataConfig,
    SyntheticDataConfig,
)
from drift_benchmark.data.drift_generators import generate_synthetic_data
from drift_benchmark.data.preprocessing import apply_preprocessing_pipeline
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)

# Type alias for dataset return format
DatasetTuple = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], DatasetMetadata]


def load_dataset(config: Union[str, Dict[str, Any], DatasetConfig]) -> DatasetTuple:
    """
    Load a dataset based on the provided configuration.

    This is the main entry point for loading datasets in drift-benchmark.
    It supports multiple data sources and returns standardized outputs.

    Args:
        config: Dataset configuration. Can be:
            - String: dataset name for built-in datasets
            - Dictionary: configuration parameters
            - DatasetConfig: Pydantic model with full configuration

    Returns:
        Tuple containing:
            - X_ref: Reference data features (numpy array)
            - X_test: Test data features (numpy array)
            - y_ref: Reference data labels (numpy array or None)
            - y_test: Test data labels (numpy array or None)
            - metadata: Dataset metadata information

    Raises:
        ValueError: If the dataset cannot be loaded or configuration is invalid
        FileNotFoundError: If a specified file path doesn't exist
    """
    # Convert input to DatasetConfig
    if isinstance(config, str):
        # Simple string name - assume it's a sklearn dataset
        config = DatasetConfig(name=config, type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name=config))
    elif isinstance(config, dict):
        config = DatasetConfig(**config)
    elif not isinstance(config, DatasetConfig):
        raise ValueError(f"Invalid config type: {type(config)}")

    logger.info(f"Loading dataset: {config.name} (type: {config.type})")

    # Route to appropriate loader based on dataset type
    if config.type == "SYNTHETIC":
        return _load_synthetic_dataset(config)
    elif config.type == "FILE":
        return _load_file_dataset(config)
    elif config.type in ["SKLEARN", "BUILTIN"]:
        return _load_sklearn_dataset(config)
    else:
        raise ValueError(f"Unsupported dataset type: {config.type}")


def _load_synthetic_dataset(config: DatasetConfig) -> DatasetTuple:
    """
    Load a synthetic dataset with drift patterns.

    Args:
        config: Dataset configuration with synthetic_config

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, metadata)
    """
    if not config.synthetic_config:
        raise ValueError("Synthetic dataset configuration missing")

    synthetic_config = config.synthetic_config
    logger.debug(f"Generating synthetic data with generator: {synthetic_config.generator_name}")

    # Generate synthetic data with drift
    X_ref, X_test, y_ref, y_test, drift_info = generate_synthetic_data(
        generator_name=synthetic_config.generator_name,
        n_samples=synthetic_config.n_samples,
        n_features=synthetic_config.n_features,
        drift_pattern=synthetic_config.drift_pattern,
        drift_characteristic=synthetic_config.drift_characteristic,
        drift_magnitude=synthetic_config.drift_magnitude,
        drift_position=synthetic_config.drift_position,
        drift_duration=synthetic_config.drift_duration,
        drift_affected_features=synthetic_config.drift_affected_features,
        noise=synthetic_config.noise,
        categorical_features=synthetic_config.categorical_features,
        random_state=synthetic_config.random_state,
        **synthetic_config.generator_params,
    )

    # Apply preprocessing if specified
    if config.preprocessing:
        X_ref = apply_preprocessing_pipeline(X_ref, config.preprocessing, fit=True)
        X_test = apply_preprocessing_pipeline(X_test, config.preprocessing, fit=False)
        preprocessing_applied = [step.method for step in config.preprocessing]
    else:
        preprocessing_applied = []

    # Create metadata
    metadata = DatasetMetadata(
        name=config.name,
        n_samples=len(X_ref) + len(X_test),
        n_features=X_ref.shape[1],
        feature_names=[f"feature_{i}" for i in range(X_ref.shape[1])],
        target_name="target" if y_ref is not None else None,
        data_types=_infer_data_types(X_ref),
        has_drift=True,
        drift_points=[int(synthetic_config.drift_position * len(X_test))],
        drift_metadata=drift_info,
        source="synthetic",
        creation_time=datetime.now().isoformat(),
        preprocessing_applied=preprocessing_applied,
    )

    logger.info(f"Generated synthetic dataset: {metadata.n_samples} samples, {metadata.n_features} features")

    return X_ref, X_test, y_ref, y_test, metadata


def _load_file_dataset(config: DatasetConfig) -> DatasetTuple:
    """
    Load a dataset from file(s).

    Args:
        config: Dataset configuration with file_config

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, metadata)
    """
    if not config.file_config:
        raise ValueError("File dataset configuration missing")

    file_config = config.file_config

    # Resolve file path
    file_path = Path(file_config.file_path)
    if not file_path.is_absolute():
        file_path = Path(settings.datasets_dir) / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.debug(f"Loading file dataset from: {file_path}")

    # Load data based on file format
    data = _load_file_data(file_path, file_config)

    # Extract features and target
    X, y, feature_names = _extract_features_and_target(data, file_config)

    # Handle drift information
    drift_labels, drift_points = _extract_drift_information(data, file_config)

    # Split data into reference and test sets
    X_ref, X_test, y_ref, y_test = _split_data(X, y, file_config, drift_labels)

    # Apply preprocessing if specified
    preprocessing_applied = []
    if config.preprocessing:
        X_ref = apply_preprocessing_pipeline(X_ref, config.preprocessing, fit=True)
        X_test = apply_preprocessing_pipeline(X_test, config.preprocessing, fit=False)
        preprocessing_applied = [step.method for step in config.preprocessing]

    # Create metadata
    metadata = DatasetMetadata(
        name=config.name,
        n_samples=len(X_ref) + len(X_test),
        n_features=X_ref.shape[1],
        feature_names=feature_names,
        target_name=file_config.target_column,
        data_types=_infer_data_types(X_ref),
        has_drift=drift_points is not None and len(drift_points) > 0,
        drift_points=drift_points,
        drift_metadata={"source": "file", "drift_column": file_config.drift_column},
        source=str(file_path),
        creation_time=datetime.now().isoformat(),
        preprocessing_applied=preprocessing_applied,
    )

    logger.info(f"Loaded file dataset: {metadata.n_samples} samples, {metadata.n_features} features")

    return X_ref, X_test, y_ref, y_test, metadata


def _load_sklearn_dataset(config: DatasetConfig) -> DatasetTuple:
    """
    Load a scikit-learn built-in dataset.

    Args:
        config: Dataset configuration with sklearn_config

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, metadata)
    """
    if not config.sklearn_config:
        raise ValueError("Sklearn dataset configuration missing")

    sklearn_config = config.sklearn_config
    dataset_name = sklearn_config.dataset_name.lower()

    logger.debug(f"Loading sklearn dataset: {dataset_name}")

    # Load the appropriate sklearn dataset
    if dataset_name == "iris":
        dataset = load_iris(return_X_y=False, as_frame=sklearn_config.as_frame)
    elif dataset_name == "wine":
        dataset = load_wine(return_X_y=False, as_frame=sklearn_config.as_frame)
    elif dataset_name == "breast_cancer":
        dataset = load_breast_cancer(return_X_y=False, as_frame=sklearn_config.as_frame)
    elif dataset_name == "diabetes":
        dataset = load_diabetes(return_X_y=False, as_frame=sklearn_config.as_frame)
    elif dataset_name == "digits":
        dataset = load_digits(return_X_y=False, as_frame=sklearn_config.as_frame)
    else:
        # Try to fetch from OpenML
        try:
            dataset = fetch_openml(name=dataset_name, return_X_y=False, as_frame=sklearn_config.as_frame, parser="auto")
        except Exception as e:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}. Error: {e}")

    # Extract data
    X = dataset.data.values if hasattr(dataset.data, "values") else dataset.data
    y = dataset.target.values if hasattr(dataset.target, "values") else dataset.target
    feature_names = (
        dataset.feature_names.tolist()
        if hasattr(dataset, "feature_names")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    # Split data
    X_ref, X_test, y_ref, y_test = train_test_split(
        X,
        y,
        test_size=sklearn_config.test_split,
        random_state=sklearn_config.random_state,
        stratify=y if len(np.unique(y)) < len(y) // 10 else None,
    )

    # Apply preprocessing if specified
    preprocessing_applied = []
    if config.preprocessing:
        X_ref = apply_preprocessing_pipeline(X_ref, config.preprocessing, fit=True)
        X_test = apply_preprocessing_pipeline(X_test, config.preprocessing, fit=False)
        preprocessing_applied = [step.method for step in config.preprocessing]

    # Create metadata
    metadata = DatasetMetadata(
        name=config.name,
        n_samples=len(X_ref) + len(X_test),
        n_features=X_ref.shape[1],
        feature_names=feature_names,
        target_name="target",
        data_types=_infer_data_types(X_ref),
        has_drift=False,  # Built-in datasets typically don't have drift
        drift_points=None,
        drift_metadata={},
        source=f"sklearn.datasets.{dataset_name}",
        creation_time=datetime.now().isoformat(),
        preprocessing_applied=preprocessing_applied,
    )

    logger.info(f"Loaded sklearn dataset: {metadata.n_samples} samples, {metadata.n_features} features")


def _load_file_data(file_path: Path, config: FileDataConfig) -> pd.DataFrame:
    """Load data from file based on format."""
    file_format = config.file_format

    # Auto-detect format if not specified
    if file_format is None:
        if file_path.is_dir():
            file_format = "DIRECTORY"
        else:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                file_format = "CSV"
            elif suffix == ".parquet":
                file_format = "PARQUET"
            elif suffix in [".xls", ".xlsx"]:
                file_format = "EXCEL"
            elif suffix == ".json":
                file_format = "JSON"
            else:
                raise ValueError(f"Cannot determine file format for: {file_path}")

    # Load data based on format
    if file_format == "CSV":
        data = pd.read_csv(file_path, sep=config.separator, header=config.header, encoding=config.encoding)
    elif file_format == "PARQUET":
        data = pd.read_parquet(file_path)
    elif file_format == "EXCEL":
        data = pd.read_excel(file_path, header=config.header)
    elif file_format == "JSON":
        data = pd.read_json(file_path, encoding=config.encoding)
    elif file_format == "DIRECTORY":
        # Concatenate all CSV files in directory
        data_frames = []
        for csv_file in file_path.glob("*.csv"):
            df = pd.read_csv(csv_file, sep=config.separator, encoding=config.encoding)
            data_frames.append(df)
        if not data_frames:
            raise ValueError(f"No CSV files found in directory: {file_path}")
        data = pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return data


def _extract_features_and_target(
    data: pd.DataFrame, config: FileDataConfig
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Extract features and target from loaded data."""
    # Handle target column
    y = None
    if config.target_column and config.target_column in data.columns:
        y = data[config.target_column].values
        data = data.drop(columns=[config.target_column])

    # Handle feature selection
    if config.feature_columns:
        # Use only specified features
        missing_cols = [col for col in config.feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Feature columns not found: {missing_cols}")
        data = data[config.feature_columns]
        feature_names = config.feature_columns
    else:
        # Exclude datetime and drift columns from features
        exclude_cols = []
        if config.datetime_column and config.datetime_column in data.columns:
            exclude_cols.append(config.datetime_column)
        if config.drift_column and config.drift_column in data.columns:
            exclude_cols.append(config.drift_column)

        if exclude_cols:
            data = data.drop(columns=exclude_cols)
        feature_names = data.columns.tolist()

    X = data.values
    return X, y, feature_names


def _extract_drift_information(
    data: pd.DataFrame, config: FileDataConfig
) -> Tuple[Optional[List[bool]], Optional[List[int]]]:
    """Extract drift information from the dataset."""
    drift_labels = None
    drift_points = None

    # Use explicit drift points if provided
    if config.drift_points:
        drift_points = config.drift_points
        # Create drift labels based on drift points
        drift_labels = [False] * len(data)
        for point in drift_points:
            if 0 <= point < len(data):
                drift_labels[point] = True

    # Use explicit drift labels if provided
    elif config.drift_labels:
        if len(config.drift_labels) != len(data):
            raise ValueError("Drift labels length must match dataset length")
        drift_labels = config.drift_labels
        # Extract drift points from labels
        drift_points = [i for i, is_drift in enumerate(drift_labels) if is_drift]

    # Use drift column if specified
    elif config.drift_column and config.drift_column in data.columns:
        drift_col = data[config.drift_column]
        if drift_col.dtype == bool:
            drift_labels = drift_col.tolist()
            drift_points = [i for i, is_drift in enumerate(drift_labels) if is_drift]
        else:
            # Assume drift column contains period identifiers
            unique_periods = drift_col.unique()
            if len(unique_periods) > 1:
                # Find change points between periods
                drift_points = []
                prev_period = drift_col.iloc[0]
                for i, current_period in enumerate(drift_col):
                    if current_period != prev_period:
                        drift_points.append(i)
                        prev_period = current_period

    return drift_labels, drift_points


def _split_data(
    X: np.ndarray, y: Optional[np.ndarray], config: FileDataConfig, drift_labels: Optional[List[bool]]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split data into reference and test sets."""

    # Time-based split if datetime column and train_end_time specified
    if config.train_end_time and config.datetime_column:
        # This would require the original dataframe - simplified for now
        # In practice, you'd pass the datetime column through the pipeline
        raise NotImplementedError("Time-based splitting not yet implemented")

    # Use test_split ratio if specified
    if config.test_split:
        return train_test_split(
            X,
            y,
            test_size=config.test_split,
            random_state=42,  # Fixed for reproducibility
            stratify=y if y is not None and len(np.unique(y)) < len(y) // 10 else None,
        )

    # Default split: 70% reference, 30% test
    split_idx = int(0.7 * len(X))
    X_ref, X_test = X[:split_idx], X[split_idx:]

    if y is not None:
        y_ref, y_test = y[:split_idx], y[split_idx:]
    else:
        y_ref, y_test = None, None

    return X_ref, X_test, y_ref, y_test


def _infer_data_types(X: np.ndarray) -> Dict[str, str]:
    """Infer data types for features."""
    data_types = {}

    for i in range(X.shape[1]):
        column = X[:, i]

        # Check if numeric
        try:
            # Try to convert to float
            float_col = column.astype(float)
            # Check if integer
            if np.all(float_col == float_col.astype(int)):
                # Check if looks like categorical (few unique values)
                unique_vals = len(np.unique(float_col))
                if unique_vals <= 10 and unique_vals < len(float_col) * 0.1:
                    data_types[f"feature_{i}"] = "categorical"
                else:
                    data_types[f"feature_{i}"] = "integer"
            else:
                data_types[f"feature_{i}"] = "continuous"
        except (ValueError, TypeError):
            # Non-numeric data
            data_types[f"feature_{i}"] = "categorical"

    return data_types


# Convenience functions for backward compatibility
def load_iris(**kwargs) -> DatasetTuple:
    """Load the Iris dataset."""
    config = DatasetConfig(name="iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", **kwargs))
    return load_dataset(config)


def load_wine(**kwargs) -> DatasetTuple:
    """Load the Wine dataset."""
    config = DatasetConfig(name="wine", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="wine", **kwargs))
    return load_dataset(config)


def load_breast_cancer(**kwargs) -> DatasetTuple:
    """Load the Breast Cancer dataset."""
    config = DatasetConfig(
        name="breast_cancer", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="breast_cancer", **kwargs)
    )
    return load_dataset(config)


# Registry of available built-in datasets
BUILTIN_DATASETS = {
    "iris": "Iris flower classification dataset",
    "wine": "Wine recognition dataset",
    "breast_cancer": "Breast cancer Wisconsin dataset",
    "diabetes": "Diabetes regression dataset",
    "digits": "Optical recognition of handwritten digits",
}


def list_builtin_datasets() -> Dict[str, str]:
    """List all available built-in datasets."""
    return BUILTIN_DATASETS.copy()
