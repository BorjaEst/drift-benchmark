from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def generate_drift(
    generator_name: str,
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate synthetic data with drift for benchmarking.

    Args:
        generator_name: Name of the drift generator to use
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift (specific meaning depends on drift type)
        drift_ratio: Proportion of features affected by drift
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        **kwargs: Additional arguments specific to particular generators

    Returns:
        Tuple containing:
            - Reference data (X_ref)
            - Test data with drift (X_test)
            - Metadata dictionary with drift information

    Raises:
        ValueError: If the generator name is not recognized
    """
    if random_state is not None:
        np.random.seed(random_state)

    generators = {
        "gaussian": _generate_gaussian_data,
        "mixed": _generate_mixed_data,
        "multimodal": _generate_multimodal_data,
        "time_series": _generate_time_series_data,
    }

    if generator_name not in generators:
        raise ValueError(f"Unknown generator: {generator_name}. Available generators: {list(generators.keys())}")

    return generators[generator_name](
        n_samples=n_samples,
        n_features=n_features,
        drift_type=drift_type,
        drift_magnitude=drift_magnitude,
        drift_ratio=drift_ratio,
        categorical_features=categorical_features,
        random_state=random_state,
        **kwargs,
    )


def _apply_drift(
    X_test: pd.DataFrame,
    drift_type: str,
    drift_magnitude: float,
    drift_ratio: float,
    categorical_features: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Apply specified drift to the test data.

    Args:
        X_test: Test data to modify
        drift_type: Type of drift to apply
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features to affect
        categorical_features: Indices of categorical features

    Returns:
        Modified test data with drift applied
    """
    X_test_drift = X_test.copy()
    features = X_test.columns
    n_features = len(features)

    # Select features to drift based on drift_ratio
    n_features_to_drift = max(1, int(n_features * drift_ratio))
    features_to_drift = np.random.choice(features, size=n_features_to_drift, replace=False)

    if categorical_features is None:
        categorical_features = []

    for feature in features_to_drift:
        is_categorical = feature in categorical_features if isinstance(feature, int) else False

        if drift_type == "mean_shift" and not is_categorical:
            # Add a constant shift
            X_test_drift[feature] += drift_magnitude

        elif drift_type == "variance_shift" and not is_categorical:
            # Increase or decrease variance
            mean_val = X_test[feature].mean()
            X_test_drift[feature] = mean_val + (X_test[feature] - mean_val) * drift_magnitude

        elif drift_type == "category_swap" and is_categorical:
            # Swap categorical values
            unique_values = X_test[feature].unique()
            if len(unique_values) >= 2:
                swap_prob = min(1.0, drift_magnitude * 0.5)  # Scale probability with magnitude
                mask = np.random.random(size=len(X_test)) < swap_prob

                # Randomly assign new categories to selected rows
                new_values = np.random.choice(unique_values, size=mask.sum())
                X_test_drift.loc[mask, feature] = new_values

        elif drift_type == "covariate_shift":
            # Create a covariate shift by resampling part of the data with bias
            if not is_categorical:
                # For numerical features: preferentially sample from one region
                threshold = X_test[feature].median()
                weight = np.where(X_test[feature] > threshold, drift_magnitude, 1.0)
                mask = np.random.random(size=len(X_test)) < (weight / np.max(weight))

                if mask.sum() > 0:
                    # Resample these points
                    biased_samples = np.random.normal(
                        loc=X_test.loc[mask, feature].mean(), scale=X_test.loc[mask, feature].std(), size=mask.sum()
                    )
                    X_test_drift.loc[mask, feature] = biased_samples

    return X_test_drift


def _generate_gaussian_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate Gaussian data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    # Generate reference data
    mean = kwargs.get("mean", 0)
    std = kwargs.get("std", 1)

    ref_data = np.random.normal(loc=mean, scale=std, size=(n_samples, n_features))
    test_data = np.random.normal(loc=mean, scale=std, size=(n_samples, n_features))

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Apply drift to test data
    X_test = _apply_drift(X_test, drift_type, drift_magnitude, drift_ratio, categorical_features)

    metadata = {
        "name": "Gaussian Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "description": f"Synthetic Gaussian data with {drift_type} drift",
    }

    return X_ref, X_test, metadata


def _generate_mixed_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    n_categories: int = 5,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate mixed numerical and categorical data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        categorical_features: Indices of categorical features (if None, 30% of features will be categorical)
        random_state: Random state for reproducibility
        n_categories: Number of categories for categorical features
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    if categorical_features is None:
        # By default, make ~30% of features categorical
        n_cat_features = max(1, int(n_features * 0.3))
        categorical_features = np.random.choice(range(n_features), size=n_cat_features, replace=False).tolist()

    # Generate reference data
    ref_data = np.zeros((n_samples, n_features))
    test_data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        if i in categorical_features:
            # Generate categorical data
            ref_data[:, i] = np.random.choice(range(n_categories), size=n_samples)
            test_data[:, i] = np.random.choice(range(n_categories), size=n_samples)
        else:
            # Generate numerical data
            ref_data[:, i] = np.random.normal(0, 1, size=n_samples)
            test_data[:, i] = np.random.normal(0, 1, size=n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Apply drift to test data
    X_test = _apply_drift(X_test, drift_type, drift_magnitude, drift_ratio, categorical_features)

    metadata = {
        "name": "Mixed Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "categorical_features": categorical_features,
        "description": f"Mixed numerical and categorical data with {drift_type} drift",
    }

    return X_ref, X_test, metadata


def _generate_multimodal_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    n_modes: int = 3,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate multimodal data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        n_modes: Number of modes/clusters in the data
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    # Generate multimodal data by sampling from multiple distributions
    ref_data = np.zeros((n_samples, n_features))
    test_data = np.zeros((n_samples, n_features))

    # Create cluster assignments
    ref_clusters = np.random.choice(range(n_modes), size=n_samples)
    test_clusters = np.random.choice(range(n_modes), size=n_samples)

    # Generate data for each cluster with different means
    for mode in range(n_modes):
        mode_mean = np.random.uniform(-5, 5, size=n_features)
        mode_std = np.random.uniform(0.5, 2, size=n_features)

        ref_mask = ref_clusters == mode
        test_mask = test_clusters == mode

        for i in range(n_features):
            ref_data[ref_mask, i] = np.random.normal(mode_mean[i], mode_std[i], size=ref_mask.sum())
            test_data[test_mask, i] = np.random.normal(mode_mean[i], mode_std[i], size=test_mask.sum())

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Apply drift to test data
    X_test = _apply_drift(X_test, drift_type, drift_magnitude, drift_ratio, categorical_features)

    metadata = {
        "name": "Multimodal Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "n_modes": n_modes,
        "description": f"Multimodal data with {n_modes} modes and {drift_type} drift",
    }

    return X_ref, X_test, metadata


def _generate_time_series_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    drift_point: float = 0.7,
    seasonality: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate time series data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        drift_point: Point in the series where drift begins (fraction of total length)
        seasonality: Whether to include seasonal patterns
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    # Generate time index
    time_index = np.linspace(0, 10, n_samples)

    # Initialize data
    data = np.zeros((n_samples, n_features))

    # Generate features
    for i in range(n_features):
        # Base trend
        data[:, i] = 0.1 * time_index + np.random.normal(0, 0.5, n_samples)

        # Add seasonality if specified
        if seasonality:
            period = np.random.uniform(0.5, 2.0)  # Random period length
            amplitude = np.random.uniform(0.5, 2.0)  # Random amplitude
            data[:, i] += amplitude * np.sin(period * time_index)

        # Add noise
        data[:, i] += np.random.normal(0, 0.2, n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    all_data = pd.DataFrame(data, columns=feature_names)

    # Split into reference and test data
    split_point = int(n_samples * 0.5)  # Use first half as reference
    X_ref = all_data.iloc[:split_point].copy()
    X_test = all_data.iloc[split_point:].copy()

    # Determine where in the test set the drift begins
    drift_start = int(len(X_test) * drift_point)

    # Select features to drift based on drift_ratio
    n_features_to_drift = max(1, int(n_features * drift_ratio))
    features_to_drift = np.random.choice(feature_names, size=n_features_to_drift, replace=False)

    # Apply drift to selected features starting from drift_point
    for feature in features_to_drift:
        if drift_type == "mean_shift":
            X_test.loc[X_test.index[drift_start:], feature] += drift_magnitude
        elif drift_type == "trend_change":
            # Add an accelerating trend
            additional_trend = np.linspace(0, drift_magnitude, len(X_test) - drift_start)
            X_test.loc[X_test.index[drift_start:], feature] += additional_trend
        elif drift_type == "variance_increase":
            # Increase the variance/noise level
            noise = np.random.normal(0, drift_magnitude, len(X_test) - drift_start)
            X_test.loc[X_test.index[drift_start:], feature] += noise

    metadata = {
        "name": "Time Series Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "drift_point": drift_point,
        "seasonality": seasonality,
        "description": f"Time series data with {drift_type} drift starting at {drift_point*100:.1f}% of the test data",
    }

    return X_ref, X_test, metadata
