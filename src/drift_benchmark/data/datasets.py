import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def load_dataset(
    name: str,
    test_size: float = 0.3,
    random_state: Optional[int] = 42,
    preprocess: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load a dataset by name and split into reference and test sets.

    Args:
        name: Name of the dataset to load
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        preprocess: Dictionary with preprocessing options
        **kwargs: Additional arguments for specific datasets

    Returns:
        Tuple containing:
            - Reference data (X_ref)
            - Test data (X_test)
            - Metadata dictionary with dataset information

    Raises:
        ValueError: If the dataset name is not recognized
    """
    dataset_loaders = {
        "iris": _load_iris,
        "wine": _load_wine,
        "breast_cancer": _load_breast_cancer,
        "creditcard": _load_creditcard,
        "california_housing": _load_california_housing,
    }

    # Check if this is a custom dataset path (file or directory)
    if os.path.exists(name):
        X_ref, X_test, metadata = _load_custom_dataset(
            path=name, test_size=test_size, random_state=random_state, **kwargs
        )
    elif name not in dataset_loaders:
        # Try to load from datasets directory
        datasets_dir = kwargs.get("datasets_dir", os.path.join(os.getcwd(), "datasets"))
        custom_path = os.path.join(datasets_dir, name)

        if os.path.exists(custom_path):
            X_ref, X_test, metadata = _load_custom_dataset(
                path=custom_path, test_size=test_size, random_state=random_state, **kwargs
            )
        else:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(dataset_loaders.keys())}")
    else:
        X_ref, X_test, metadata = dataset_loaders[name](test_size=test_size, random_state=random_state, **kwargs)

    # Apply preprocessing if specified
    if preprocess:
        X_ref, X_test = apply_preprocessing(X_ref, X_test, preprocess)
        metadata["preprocessing"] = preprocess

    return X_ref, X_test, metadata


def apply_preprocessing(
    X_ref: pd.DataFrame, X_test: pd.DataFrame, preprocess: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply preprocessing steps to reference and test data.

    Args:
        X_ref: Reference data
        X_test: Test data
        preprocess: Dictionary with preprocessing options

    Returns:
        Preprocessed reference and test data
    """
    X_ref_processed = X_ref.copy()
    X_test_processed = X_test.copy()

    # Handle missing values
    if preprocess.get("handle_missing"):
        strategy = preprocess.get("missing_strategy", "mean")
        for col in X_ref.columns:
            if X_ref[col].isna().any() or X_test[col].isna().any():
                if X_ref[col].dtype.kind in "ifc":  # numeric columns
                    imputer = SimpleImputer(strategy=strategy)
                    X_ref_processed[col] = imputer.fit_transform(X_ref[col].values.reshape(-1, 1)).ravel()
                    X_test_processed[col] = imputer.transform(X_test[col].values.reshape(-1, 1)).ravel()
                else:  # categorical columns
                    # Use most frequent for categorical
                    imputer = SimpleImputer(strategy="most_frequent")
                    X_ref_processed[col] = imputer.fit_transform(X_ref[col].values.reshape(-1, 1)).ravel()
                    X_test_processed[col] = imputer.transform(X_test[col].values.reshape(-1, 1)).ravel()

    # Handle scaling
    if preprocess.get("scaling"):
        scaling_method = preprocess.get("scaling_method", "standard")
        columns_to_scale = preprocess.get("scale_columns", X_ref.select_dtypes(include=np.number).columns.tolist())

        # Only apply scaling to numeric columns
        columns_to_scale = [col for col in columns_to_scale if col in X_ref.select_dtypes(include=np.number).columns]

        if columns_to_scale:
            if scaling_method == "standard":
                scaler = StandardScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            elif scaling_method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")

            X_ref_processed[columns_to_scale] = scaler.fit_transform(X_ref_processed[columns_to_scale])
            X_test_processed[columns_to_scale] = scaler.transform(X_test_processed[columns_to_scale])

    # Handle categorical encoding
    if preprocess.get("encode_categorical", False):
        encoding_method = preprocess.get("encoding_method", "onehot")
        categorical_columns = X_ref.select_dtypes(include=["object", "category"]).columns

        if encoding_method == "onehot":
            for col in categorical_columns:
                # Get all possible categories from both datasets
                categories = pd.concat([X_ref_processed[col], X_test_processed[col]]).unique()

                # One-hot encode
                for category in categories:
                    col_name = f"{col}_{category}"
                    X_ref_processed[col_name] = (X_ref_processed[col] == category).astype(int)
                    X_test_processed[col_name] = (X_test_processed[col] == category).astype(int)

                # Drop original column
                X_ref_processed = X_ref_processed.drop(columns=[col])
                X_test_processed = X_test_processed.drop(columns=[col])

        elif encoding_method == "label":
            for col in categorical_columns:
                # Create mapping from categories to integers
                categories = pd.concat([X_ref_processed[col], X_test_processed[col]]).unique()
                mapping = {cat: i for i, cat in enumerate(categories)}

                # Apply mapping
                X_ref_processed[col] = X_ref_processed[col].map(mapping)
                X_test_processed[col] = X_test_processed[col].map(mapping)

    return X_ref_processed, X_test_processed


def _load_custom_dataset(
    path: str,
    test_size: float = 0.3,
    random_state: Optional[int] = 42,
    file_format: str = None,
    target_column: str = None,
    time_column: str = None,
    drift_column: str = None,
    train_end_time: Optional[Any] = None,
    train_size: Optional[float] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load a custom dataset from a file or directory.

    Args:
        path: Path to the dataset file or directory
        test_size: Proportion of data to use for test set (ignored if train_size/train_end_time is provided)
        random_state: Random state for reproducibility
        file_format: Format of the file (csv, parquet, etc)
        target_column: Name of the target column (will be removed from X)
        time_column: Name of the time column (used for time-based splits)
        drift_column: Alias for time_column, used for time-based splits
        train_end_time: If provided, split by time with this as cutoff
        train_size: If provided, use this proportion for training set
        **kwargs: Additional arguments for reading file

    Returns:
        Tuple containing reference data, test data, and metadata
    """
    # Use drift_column as time_column if provided (for compatibility with TOML config)
    if drift_column and not time_column:
        time_column = drift_column

    # Use train_size instead of test_size if provided (for compatibility with TOML config)
    if train_size is not None:
        test_size = 1 - train_size

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

    # Load data based on format
    if file_format == "csv":
        data = pd.read_csv(path, **kwargs)
    elif file_format == "parquet":
        data = pd.read_parquet(path, **kwargs)
    elif file_format == "excel":
        data = pd.read_excel(path, **kwargs)
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

    # Handle target column if provided
    y = None
    if target_column and target_column in data.columns:
        y = data[target_column].copy()
        data = data.drop(columns=[target_column])

    # Time-based split or random split
    if time_column and time_column in data.columns and train_end_time is not None:
        # Convert column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])

        # Split by time
        X_ref = data[data[time_column] <= train_end_time].copy()
        X_test = data[data[time_column] > train_end_time].copy()

        # Check if split worked properly
        if len(X_ref) == 0 or len(X_test) == 0:
            raise ValueError(f"Invalid time split: train_end_time={train_end_time} resulted in empty dataset")
    else:
        # Random split
        X_ref, X_test = train_test_split(data, test_size=test_size, random_state=random_state)

    # Create metadata
    metadata = {
        "name": os.path.basename(path),
        "path": path,
        "n_features": data.shape[1] - (1 if time_column else 0),
        "n_samples": data.shape[0],
        "feature_names": data.columns.tolist(),
        "has_drift": kwargs.get("has_drift", False),
        "description": kwargs.get("description", f"Custom dataset loaded from {path}"),
        "target_column": target_column,
        "time_column": time_column,
    }

    return X_ref, X_test, metadata


def _load_iris(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the Iris dataset."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    metadata = {
        "name": "Iris",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Classic iris dataset with 3 classes of flowers and 4 features.",
    }

    return X_ref, X_test, metadata


def _load_wine(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the Wine dataset."""
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    metadata = {
        "name": "Wine",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Wine dataset with 13 features and 3 classes.",
    }

    return X_ref, X_test, metadata


def _load_breast_cancer(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the Breast Cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    metadata = {
        "name": "Breast Cancer",
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "has_drift": False,
        "description": "Breast cancer diagnostic dataset with 30 features.",
    }

    return X_ref, X_test, metadata


def _load_creditcard(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the Credit Card Fraud Detection dataset from OpenML."""
    try:
        data = fetch_openml(name="credit-card", version=1, as_frame=True)
        X = data.data
        y = data.target

        X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

        metadata = {
            "name": "Credit Card Fraud",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "Credit card fraud detection dataset with anonymized features.",
        }

        return X_ref, X_test, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load credit card dataset: {str(e)}")


def _load_california_housing(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the California Housing dataset from OpenML."""
    try:
        data = fetch_openml(name="california_housing", version=1, as_frame=True)
        X = data.data
        y = data.target

        X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

        metadata = {
            "name": "California Housing",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "California housing dataset with 8 features.",
        }

        return X_ref, X_test, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load California housing dataset: {str(e)}")
