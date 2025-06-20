"""
Data loading and preprocessing module for drift-benchmark.

This module provides utilities for loading standard datasets, custom datasets,
and applying various preprocessing operations to prepare data for drift detection
benchmarking.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drift_benchmark.settings import settings


def load_dataset(
    dataset_params: Union[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]]:
    """
    Load a dataset based on provided parameters.

    This is the main entry point for loading datasets in drift-benchmark.
    It takes either a dataset name or a dictionary of parameters and returns
    the dataset split into reference and test sets with optional labels.

    Args:
        dataset_params: Either a dataset name as string or a dictionary with dataset parameters
                       including 'name', 'type', and other dataset-specific fields.

    Returns:
        Tuple containing:
            - X_ref: Reference data features
            - X_test: Test data features
            - y_ref: Reference data labels (None if no target)
            - y_test: Test data labels (None if no target)
            - drift_labels: Boolean indicators of drift points (None if unknown)

    Raises:
        ValueError: If the dataset cannot be loaded or parameters are invalid
    """
    if isinstance(dataset_params, str):
        dataset_params = {"name": dataset_params, "type": "builtin"}

    dataset_type = dataset_params.get("type", "builtin")

    # Extract common parameters
    test_size = dataset_params.get("test_size", 0.3)
    train_size = dataset_params.get("train_size")
    if train_size is not None:
        test_size = 1 - train_size

    random_state = dataset_params.get("random_state", 42)
    preprocess = dataset_params.get("preprocess")

    # Handle different dataset types
    if dataset_type == "synthetic":
        return _load_synthetic_dataset(dataset_params, test_size=test_size, random_state=random_state)
    elif dataset_type == "file":
        return _load_file_dataset(dataset_params, test_size=test_size, random_state=random_state)
    elif dataset_type == "builtin":
        return _load_builtin_dataset(dataset_params, test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _load_builtin_dataset(
    dataset_params: Dict[str, Any], test_size: float = 0.3, random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]]:
    """
    Load a built-in dataset.

    Args:
        dataset_params: Dictionary with dataset parameters
        test_size: Proportion of data to include in test split
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, drift_labels)
    """
    dataset_name = dataset_params["name"]

    # Define available builtin datasets
    dataset_loaders = {
        "iris": _load_iris,
        "wine": _load_wine,
        "breast_cancer": _load_breast_cancer,
        "creditcard": _load_creditcard,
        "california_housing": _load_california_housing,
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Unknown built-in dataset: {dataset_name}. " f"Available datasets: {list(dataset_loaders.keys())}"
        )

    # Load the dataset
    X, y, metadata = dataset_loaders[dataset_name](random_state=random_state)

    # Apply preprocessing if specified
    preprocess = dataset_params.get("preprocess")
    if preprocess:
        X = _preprocess_data(X, preprocess)

    # Split data
    X_ref, X_test, y_ref, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
    )

    # Most built-in datasets don't have drift, so drift labels are None
    drift_labels = None

    return X_ref, X_test, y_ref, y_test, drift_labels


def _load_file_dataset(
    dataset_params: Dict[str, Any], test_size: float = 0.3, random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]]:
    """
    Load a dataset from a file.

    Args:
        dataset_params: Dictionary with dataset parameters
        test_size: Proportion of data to include in test split
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, drift_labels)
    """
    path = dataset_params.get("path")
    if not path:
        raise ValueError("Path must be specified for file datasets")

    # If path is not absolute, look in datasets directory
    if not os.path.isabs(path):
        path = os.path.join(settings.datasets_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Get parameters
    target_column = dataset_params.get("target_column")
    drift_column = dataset_params.get("drift_column")
    time_column = dataset_params.get("time_column", drift_column)
    train_end_time = dataset_params.get("train_end_time")
    file_format = dataset_params.get("file_format")

    # Determine file format if not provided
    if file_format is None:
        if os.path.isdir(path):
            file_format = "directory"
        else:
            file_ext = os.path.splitext(path)[1].lower()
            if file_ext == ".csv":
                file_format = "csv"
            elif file_ext == ".parquet":
                file_format = "parquet"
            elif file_ext in [".xls", ".xlsx"]:
                file_format = "excel"
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

    # Load the data
    if file_format == "csv":
        data = pd.read_csv(path)
    elif file_format == "parquet":
        data = pd.read_parquet(path)
    elif file_format == "excel":
        data = pd.read_excel(path)
    elif file_format == "directory":
        # Assume directory contains CSV files to be merged
        data_frames = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if file.endswith(".csv"):
                df = pd.read_csv(file_path)
                data_frames.append(df)
        if not data_frames:
            raise ValueError(f"No CSV files found in directory: {path}")
        data = pd.concat(data_frames, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Extract target if specified
    y = None
    if target_column and target_column in data.columns:
        y = data[target_column].values
        data = data.drop(columns=[target_column])
    else:
        # Create dummy target if not specified
        y = np.zeros(len(data))

    # Extract drift column if specified
    drift_labels = None
    if drift_column and drift_column in data.columns:
        drift_values = data[drift_column].values
        data = data.drop(columns=[drift_column])

        # Convert drift values to boolean labels
        if dataset_params.get("drift_threshold") is not None:
            threshold = dataset_params["drift_threshold"]
            drift_labels = [val > threshold for val in drift_values]
        else:
            # Try to convert to boolean directly
            try:
                drift_labels = [bool(val) for val in drift_values]
            except:
                # Use changes in drift values to detect drift points
                drift_labels = [False]
                for i in range(1, len(drift_values)):
                    drift_labels.append(drift_values[i] != drift_values[i - 1])

    # Apply preprocessing if specified
    preprocess = dataset_params.get("preprocess")
    if preprocess:
        data = _preprocess_data(data, preprocess)

    # Time-based split or random split
    if time_column and time_column in data.columns and train_end_time is not None:
        # Convert column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])

        # Split by time
        ref_indices = data[time_column] <= train_end_time
        test_indices = data[time_column] > train_end_time

        # Extract reference and test data
        X_ref = data[ref_indices].drop(columns=[time_column]).values
        X_test = data[test_indices].drop(columns=[time_column]).values

        # Extract corresponding targets
        y_ref = y[ref_indices]
        y_test = y[test_indices]

        # Extract corresponding drift labels if available
        if drift_labels is not None:
            drift_labels = [drift_labels[i] for i, is_test in enumerate(test_indices) if is_test]
    else:
        # Drop time column if it exists
        if time_column and time_column in data.columns:
            data = data.drop(columns=[time_column])

        # Convert to numpy array if still DataFrame
        X = data.values if isinstance(data, pd.DataFrame) else data

        # Random split
        X_ref, X_test, y_ref, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
        )

        # Drift labels are not meaningful in random split
        drift_labels = None

    return X_ref, X_test, y_ref, y_test, drift_labels


def _load_synthetic_dataset(
    dataset_params: Dict[str, Any], test_size: float = 0.3, random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List[bool]]]:
    """
    Generate a synthetic dataset with specified drift properties.

    Args:
        dataset_params: Dictionary with dataset parameters
        test_size: Proportion of data to include in test split
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test, drift_labels)
    """
    from drift_benchmark.data.drift_generators import generate_drift

    # Extract parameters for synthetic data generation
    n_samples = dataset_params.get("n_samples", 1000)
    n_features = dataset_params.get("n_features", 10)
    drift_type = dataset_params.get("drift_type", "sudden")
    drift_magnitude = dataset_params.get("drift_magnitude", 1.0)
    drift_position = dataset_params.get("drift_position", 0.5)
    noise = dataset_params.get("noise", 0.05)

    # Generator selection
    generator_name = dataset_params.get("generator", "gaussian")

    # Generate data with drift
    X_ref, X_test, metadata = generate_drift(
        generator_name=generator_name,
        n_samples=n_samples,
        n_features=n_features,
        drift_type=drift_type,
        drift_magnitude=drift_magnitude,
        drift_position=drift_position,
        noise=noise,
        random_state=random_state,
        **{
            k: v
            for k, v in dataset_params.items()
            if k
            not in [
                "name",
                "type",
                "n_samples",
                "n_features",
                "drift_type",
                "drift_magnitude",
                "drift_position",
                "noise",
                "random_state",
            ]
        },
    )

    # For synthetic data, we can generate drift labels based on the position
    drift_start = int(drift_position * n_samples)
    drift_labels = [i >= drift_start for i in range(n_samples)]

    # Create dummy labels (no actual classification task for drift detection)
    y_ref = np.zeros(len(X_ref))
    y_test = np.zeros(len(X_test))

    return X_ref.values, X_test.values, y_ref, y_test, drift_labels


def _preprocess_data(X: Union[pd.DataFrame, np.ndarray], preprocess_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply preprocessing steps to data.

    Args:
        X: Input data (DataFrame or numpy array)
        preprocess_config: Dictionary with preprocessing options

    Returns:
        Preprocessed data as DataFrame
    """
    # Convert to DataFrame if it's not already
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    X_processed = X.copy()

    # Handle missing values
    if preprocess_config.get("handle_missing", False):
        strategy = preprocess_config.get("missing_strategy", "mean")
        for col in X_processed.columns:
            if X_processed[col].isna().any():
                if X_processed[col].dtype.kind in "ifc":  # numeric columns
                    imputer = SimpleImputer(strategy=strategy)
                    X_processed[col] = imputer.fit_transform(X_processed[col].values.reshape(-1, 1)).ravel()
                else:  # categorical columns
                    # Use most frequent for categorical
                    imputer = SimpleImputer(strategy="most_frequent")
                    X_processed[col] = imputer.fit_transform(X_processed[col].values.reshape(-1, 1)).ravel()

    # Handle scaling
    if preprocess_config.get("scaling", False):
        scaling_method = preprocess_config.get("scaling_method", "standard")
        columns_to_scale = preprocess_config.get(
            "scale_columns", X_processed.select_dtypes(include=np.number).columns.tolist()
        )

        # Only apply scaling to numeric columns that exist
        columns_to_scale = [
            col for col in columns_to_scale if col in X_processed.select_dtypes(include=np.number).columns
        ]

        if columns_to_scale:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")

            X_processed[columns_to_scale] = scaler.fit_transform(X_processed[columns_to_scale])

    # Handle categorical encoding
    if preprocess_config.get("encode_categorical", False):
        encoding_method = preprocess_config.get("encoding_method", "onehot")
        categorical_columns = X_processed.select_dtypes(include=["object", "category"]).columns

        if encoding_method == "onehot":
            X_processed = pd.get_dummies(X_processed, columns=categorical_columns, drop_first=False)
        elif encoding_method == "label":
            for col in categorical_columns:
                # Create mapping from categories to integers
                categories = X_processed[col].unique()
                mapping = {cat: i for i, cat in enumerate(categories)}

                # Apply mapping
                X_processed[col] = X_processed[col].map(mapping)

    return X_processed


# Dataset loader functions for built-in datasets


def _load_iris(random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load the Iris dataset."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    metadata = {
        "name": "Iris",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Classic iris dataset with 3 classes of flowers and 4 features.",
    }

    return X, y, metadata


def _load_wine(random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load the Wine dataset."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    metadata = {
        "name": "Wine",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Wine dataset with 13 features and 3 classes.",
    }

    return X, y, metadata


def _load_breast_cancer(random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load the Breast Cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    metadata = {
        "name": "Breast Cancer",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Breast cancer diagnostic dataset with 30 features.",
    }

    return X, y, metadata


def _load_creditcard(random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load the Credit Card Fraud Detection dataset from OpenML."""
    try:
        data = fetch_openml(name="credit-card", version=1, as_frame=True)
        X = data.data
        y = data.target.astype(int)

        metadata = {
            "name": "Credit Card Fraud",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "Credit card fraud detection dataset with anonymized features.",
        }

        return X, y, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load credit card dataset: {str(e)}")


def _load_california_housing(random_state: Optional[int] = 42) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load the California Housing dataset from OpenML."""
    try:
        data = fetch_openml(name="california_housing", version=1, as_frame=True)
        X = data.data
        y = data.target.astype(float)

        metadata = {
            "name": "California Housing",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "California housing dataset with 8 features.",
        }

        return X, y, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load California housing dataset: {str(e)}")
