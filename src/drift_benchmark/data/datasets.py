import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split


def load_dataset(
    name: str, test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load a dataset by name and split into reference and test sets.

    Args:
        name: Name of the dataset to load
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
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

    if name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(dataset_loaders.keys())}")

    return dataset_loaders[name](test_size=test_size, random_state=random_state, **kwargs)


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
        "has_drift": False,  # By default, no drift
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

        X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

        metadata = {
            "name": "Credit Card Fraud",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "Credit card fraud detection dataset from OpenML.",
        }

        return X_ref, X_test, metadata
    except Exception as e:
        raise ValueError(f"Error loading credit card dataset: {e}")


def _load_california_housing(
    test_size: float = 0.3, random_state: Optional[int] = 42, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load the California Housing dataset from OpenML."""
    try:
        data = fetch_openml(name="california_housing", version=1, as_frame=True)
        X = data.data

        X_ref, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

        metadata = {
            "name": "California Housing",
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": X.columns.tolist(),
            "has_drift": False,
            "description": "California housing dataset with house prices and demographics.",
        }

        return X_ref, X_test, metadata
    except Exception as e:
        raise ValueError(f"Error loading California housing dataset: {e}")
