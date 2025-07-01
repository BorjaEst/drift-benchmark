"""
Data preprocessing module for drift-benchmark.

This module provides comprehensive preprocessing utilities for preparing data
for drift detection benchmarks, including scaling, imputation, encoding,
and outlier removal.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from drift_benchmark.constants import (
    EncodingConfig,
    ImputationConfig,
    OutlierConfig,
    PreprocessingConfig,
    ScalingConfig,
)

logger = logging.getLogger(__name__)

# Global preprocessing state for consistent transform application
_preprocessing_state = {}


def apply_preprocessing_pipeline(
    X: np.ndarray, preprocessing_steps: List[PreprocessingConfig], fit: bool = True
) -> np.ndarray:
    """
    Apply a preprocessing pipeline to data.

    Args:
        X: Input data array
        preprocessing_steps: List of preprocessing configurations
        fit: Whether to fit preprocessors (True) or use existing fit (False)

    Returns:
        Preprocessed data array

    Raises:
        ValueError: If preprocessing configuration is invalid
    """
    X_processed = X.copy()

    for i, step in enumerate(preprocessing_steps):
        step_key = f"step_{i}_{step.method}"

        if step.method == "STANDARDIZE":
            X_processed = _apply_scaling(X_processed, step, step_key, fit)
        elif step.method == "NORMALIZE":
            X_processed = _apply_scaling(X_processed, step, step_key, fit)
        elif step.method == "ROBUST_SCALE":
            X_processed = _apply_scaling(X_processed, step, step_key, fit)
        elif step.method == "HANDLE_MISSING":
            X_processed = _apply_imputation(X_processed, step, step_key, fit)
        elif step.method == "ENCODE_CATEGORICAL":
            X_processed = _apply_encoding(X_processed, step, step_key, fit)
        elif step.method == "PCA":
            X_processed = _apply_pca(X_processed, step, step_key, fit)
        elif step.method == "REMOVE_OUTLIERS":
            X_processed = _apply_outlier_removal(X_processed, step, step_key, fit)
        else:
            raise ValueError(f"Unknown preprocessing method: {step.method}")

        logger.debug(f"Applied {step.method} preprocessing step")

    return X_processed


def _apply_scaling(X: np.ndarray, step: PreprocessingConfig, step_key: str, fit: bool) -> np.ndarray:
    """Apply scaling preprocessing."""
    # Convert to ScalingConfig if needed
    if isinstance(step, ScalingConfig):
        config = step
    else:
        # Create ScalingConfig from basic PreprocessingConfig
        method_map = {"STANDARDIZE": "STANDARD", "NORMALIZE": "MINMAX", "ROBUST_SCALE": "ROBUST"}
        scaling_method = method_map.get(step.method, "STANDARD")
        config = ScalingConfig(method=scaling_method, features=step.features, parameters=step.parameters)

    # Get feature indices
    feature_indices = _get_feature_indices(X, config.features)

    # Initialize scaler
    if config.method == "STANDARD":
        scaler = StandardScaler()
    elif config.method == "MINMAX":
        feature_range = config.feature_range or (0, 1)
        scaler = MinMaxScaler(feature_range=feature_range)
    elif config.method == "ROBUST":
        scaler = RobustScaler()
    elif config.method == "MAXABS":
        scaler = MaxAbsScaler()
    elif config.method == "QUANTILE":
        scaler = QuantileTransformer()
    else:
        raise ValueError(f"Unknown scaling method: {config.method}")

    # Apply scaling
    X_scaled = X.copy()

    if fit:
        X_scaled[:, feature_indices] = scaler.fit_transform(X[:, feature_indices])
        _preprocessing_state[step_key] = scaler
    else:
        if step_key not in _preprocessing_state:
            raise ValueError(f"Preprocessor not fitted: {step_key}")
        scaler = _preprocessing_state[step_key]
        X_scaled[:, feature_indices] = scaler.transform(X[:, feature_indices])

    return X_scaled


def _apply_imputation(X: np.ndarray, step: PreprocessingConfig, step_key: str, fit: bool) -> np.ndarray:
    """Apply missing value imputation."""
    # Convert to ImputationConfig if needed
    if isinstance(step, ImputationConfig):
        config = step
    else:
        strategy = step.parameters.get("strategy", "MEAN")
        config = ImputationConfig(
            strategy=strategy,
            features=step.features,
            parameters=step.parameters,
            fill_value=step.parameters.get("fill_value"),
        )

    # Get feature indices
    feature_indices = _get_feature_indices(X, config.features)

    # Map strategy names
    strategy_map = {"MEAN": "mean", "MEDIAN": "median", "MODE": "most_frequent", "CONSTANT": "constant"}

    sklearn_strategy = strategy_map.get(config.strategy, "mean")

    # Initialize imputer
    if config.strategy == "CONSTANT":
        imputer = SimpleImputer(strategy=sklearn_strategy, fill_value=config.fill_value)
    else:
        imputer = SimpleImputer(strategy=sklearn_strategy)

    # Apply imputation
    X_imputed = X.copy()

    if fit:
        X_imputed[:, feature_indices] = imputer.fit_transform(X[:, feature_indices])
        _preprocessing_state[step_key] = imputer
    else:
        if step_key not in _preprocessing_state:
            raise ValueError(f"Preprocessor not fitted: {step_key}")
        imputer = _preprocessing_state[step_key]
        X_imputed[:, feature_indices] = imputer.transform(X[:, feature_indices])

    return X_imputed


def _apply_encoding(X: np.ndarray, step: PreprocessingConfig, step_key: str, fit: bool) -> np.ndarray:
    """Apply categorical encoding."""
    # Note: This is simplified - in practice, you'd need to track which
    # features are categorical and handle mixed data types properly
    if isinstance(step, EncodingConfig):
        config = step
    else:
        method = step.parameters.get("method", "ONEHOT")
        config = EncodingConfig(method=method, features=step.features, parameters=step.parameters)

    # Get feature indices
    feature_indices = _get_feature_indices(X, config.features)

    if config.method == "ONEHOT":
        encoder = OneHotEncoder(
            drop="first" if config.drop_first else None,
            handle_unknown="ignore" if config.handle_unknown == "ignore" else "error",
            sparse_output=False,
        )
    elif config.method == "LABEL":
        encoder = LabelEncoder()
    elif config.method == "ORDINAL":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unknown encoding method: {config.method}")

    # Apply encoding (simplified - assumes categorical features are string-encoded)
    X_encoded = X.copy()

    if fit:
        if config.method == "ONEHOT":
            # OneHotEncoder returns different shape, need special handling
            encoded_features = encoder.fit_transform(X[:, feature_indices])
            # Replace original features with encoded ones
            other_features = np.delete(X, feature_indices, axis=1)
            X_encoded = np.concatenate([other_features, encoded_features], axis=1)
        else:
            X_encoded[:, feature_indices] = encoder.fit_transform(X[:, feature_indices])
        _preprocessing_state[step_key] = encoder
    else:
        if step_key not in _preprocessing_state:
            raise ValueError(f"Preprocessor not fitted: {step_key}")
        encoder = _preprocessing_state[step_key]
        if config.method == "ONEHOT":
            encoded_features = encoder.transform(X[:, feature_indices])
            other_features = np.delete(X, feature_indices, axis=1)
            X_encoded = np.concatenate([other_features, encoded_features], axis=1)
        else:
            X_encoded[:, feature_indices] = encoder.transform(X[:, feature_indices])

    return X_encoded


def _apply_pca(X: np.ndarray, step: PreprocessingConfig, step_key: str, fit: bool) -> np.ndarray:
    """Apply Principal Component Analysis."""
    n_components = step.parameters.get("n_components", 0.95)

    # Initialize PCA
    pca = PCA(n_components=n_components)

    if fit:
        X_pca = pca.fit_transform(X)
        _preprocessing_state[step_key] = pca
    else:
        if step_key not in _preprocessing_state:
            raise ValueError(f"Preprocessor not fitted: {step_key}")
        pca = _preprocessing_state[step_key]
        X_pca = pca.transform(X)

    return X_pca


def _apply_outlier_removal(X: np.ndarray, step: PreprocessingConfig, step_key: str, fit: bool) -> np.ndarray:
    """Apply outlier detection and removal."""
    if isinstance(step, OutlierConfig):
        config = step
    else:
        method = step.parameters.get("method", "ISOLATION_FOREST")
        config = OutlierConfig(
            method=method,
            features=step.features,
            parameters=step.parameters,
            contamination=step.parameters.get("contamination", 0.1),
        )

    # Get feature indices
    feature_indices = _get_feature_indices(X, config.features)

    # Initialize outlier detector
    if config.method == "ISOLATION_FOREST":
        detector = IsolationForest(contamination=config.contamination, random_state=42)
    elif config.method == "LOCAL_OUTLIER_FACTOR":
        detector = LocalOutlierFactor(contamination=config.contamination)
    elif config.method == "ZSCORE":
        # Z-score based outlier detection
        threshold = config.threshold or 3.0
        X_subset = X[:, feature_indices]
        z_scores = np.abs((X_subset - np.mean(X_subset, axis=0)) / np.std(X_subset, axis=0))
        outlier_mask = np.any(z_scores > threshold, axis=1)
        return X[~outlier_mask]
    elif config.method == "IQR":
        # Interquartile range outlier detection
        X_subset = X[:, feature_indices]
        Q1 = np.percentile(X_subset, 25, axis=0)
        Q3 = np.percentile(X_subset, 75, axis=0)
        IQR = Q3 - Q1
        threshold = config.threshold or 1.5
        outlier_mask = np.any((X_subset < Q1 - threshold * IQR) | (X_subset > Q3 + threshold * IQR), axis=1)
        return X[~outlier_mask]
    else:
        raise ValueError(f"Unknown outlier detection method: {config.method}")

    # Apply outlier detection
    if fit:
        if config.method == "LOCAL_OUTLIER_FACTOR":
            outlier_labels = detector.fit_predict(X[:, feature_indices])
        else:
            outlier_labels = detector.fit_predict(X[:, feature_indices])
        _preprocessing_state[step_key] = detector
    else:
        if step_key not in _preprocessing_state:
            raise ValueError(f"Preprocessor not fitted: {step_key}")
        detector = _preprocessing_state[step_key]
        if config.method == "LOCAL_OUTLIER_FACTOR":
            # LOF doesn't have a predict method, use fit_predict
            outlier_labels = detector.fit_predict(X[:, feature_indices])
        else:
            outlier_labels = detector.predict(X[:, feature_indices])

    # Remove outliers (keep only inliers, marked as 1)
    inlier_mask = outlier_labels == 1
    return X[inlier_mask]


def _get_feature_indices(X: np.ndarray, features: Union[str, List[str], List[int]]) -> List[int]:
    """Get feature indices from feature specification."""
    n_features = X.shape[1]

    if features == "all":
        return list(range(n_features))
    elif isinstance(features, list):
        if all(isinstance(f, int) for f in features):
            # List of indices
            return [f for f in features if 0 <= f < n_features]
        else:
            # List of feature names - convert to indices (simplified)
            # In practice, you'd need feature name mapping
            raise NotImplementedError("Feature name mapping not implemented")
    else:
        raise ValueError(f"Invalid features specification: {features}")


def reset_preprocessing_state():
    """Reset the global preprocessing state."""
    global _preprocessing_state
    _preprocessing_state = {}


def get_preprocessing_state() -> Dict[str, Any]:
    """Get the current preprocessing state."""
    return _preprocessing_state.copy()


def set_preprocessing_state(state: Dict[str, Any]):
    """Set the preprocessing state."""
    global _preprocessing_state
    _preprocessing_state = state.copy()
