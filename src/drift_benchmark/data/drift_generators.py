"""
Drift generation module for drift-benchmark.

This module provides functions to generate synthetic data with various types of drift,
including mean shift, variance shift, sudden drift, and gradual drift. It supports
multiple data distributions like Gaussian, mixed types, multimodal, and time series.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def generate_drift(
    generator_name: str,
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "sudden",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    drift_position: float = 0.5,
    noise: float = 0.05,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic data with drift for benchmarking.

    Args:
        generator_name: Name of the drift generator to use
            (gaussian, mixed, multimodal, time_series)
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
            (mean_shift, variance_shift, sudden, gradual, incremental, recurring)
        drift_magnitude: Magnitude of drift (specific meaning depends on drift type)
        drift_ratio: Proportion of features affected by drift
        drift_position: Position where the drift occurs (0.0-1.0),
            used for gradual and sudden drift
        noise: Amount of noise to add to the data
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
        raise ValueError(f"Unknown generator name: {generator_name}. Available generators: {list(generators.keys())}")

    return generators[generator_name](
        n_samples=n_samples,
        n_features=n_features,
        drift_type=drift_type,
        drift_magnitude=drift_magnitude,
        drift_ratio=drift_ratio,
        drift_position=drift_position,
        noise=noise,
        categorical_features=categorical_features,
        random_state=random_state,
        **kwargs,
    )


def _apply_drift(
    X_test: pd.DataFrame,
    drift_type: str,
    drift_magnitude: float,
    drift_ratio: float,
    drift_position: float,
    noise: float,
    categorical_features: Optional[List[int]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply specified drift to the test data.

    Args:
        X_test: Test data to modify
        drift_type: Type of drift to apply (mean_shift, variance_shift, sudden, gradual, etc.)
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features to affect
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
        categorical_features: Indices of categorical features
        **kwargs: Additional parameters for specific drift types

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

    # Convert categorical_features to column indices if column names were provided
    categorical_indices = []
    if isinstance(categorical_features, list) and categorical_features:
        if isinstance(categorical_features[0], str):
            categorical_indices = [X_test.columns.get_loc(col) for col in categorical_features if col in X_test.columns]
        else:
            categorical_indices = categorical_features

    # Calculate position in samples
    position_idx = int(drift_position * len(X_test))

    # If the drift type is recurring, define cycle parameters
    recurring_cycle = kwargs.get("recurring_cycle", len(X_test) // 5)

    for feature in features_to_drift:
        col_idx = X_test.columns.get_loc(feature) if isinstance(feature, str) else feature
        is_categorical = col_idx in categorical_indices

        # Check if the feature is time-based (datetime or timedelta)
        is_time_feature = pd.api.types.is_datetime64_any_dtype(X_test[feature]) or pd.api.types.is_timedelta64_dtype(
            X_test[feature]
        )

        # Handle categorical features
        if is_categorical:
            categories = X_test[feature].unique()
            n_categories = len(categories)

            if drift_type in ["sudden", "mean_shift"]:
                # For categorical features, shift categories after drift point
                for i in range(position_idx, len(X_test)):
                    current_cat = X_test_drift.loc[i, feature]
                    cat_idx = np.where(categories == current_cat)[0][0]
                    new_cat_idx = (cat_idx + int(drift_magnitude)) % n_categories
                    X_test_drift.loc[i, feature] = categories[new_cat_idx]

            elif drift_type == "gradual":
                # Gradually increase probability of category shift
                for i in range(position_idx, len(X_test)):
                    drift_factor = (i - position_idx) / (len(X_test) - position_idx)
                    if np.random.random() < drift_factor * drift_magnitude:
                        current_cat = X_test_drift.loc[i, feature]
                        cat_idx = np.where(categories == current_cat)[0][0]
                        new_cat_idx = (cat_idx + 1) % n_categories
                        X_test_drift.loc[i, feature] = categories[new_cat_idx]

            elif drift_type == "recurring":
                # Recurring pattern of category shifts
                for i in range(position_idx, len(X_test)):
                    cycle_position = (i - position_idx) % recurring_cycle
                    if cycle_position < recurring_cycle / 2:  # First half of cycle: apply drift
                        current_cat = X_test_drift.loc[i, feature]
                        cat_idx = np.where(categories == current_cat)[0][0]
                        new_cat_idx = (cat_idx + int(drift_magnitude)) % n_categories
                        X_test_drift.loc[i, feature] = categories[new_cat_idx]

        # Handle time-based features differently
        elif is_time_feature:
            # For datetime/timedelta features, add appropriate time-based drift
            if pd.api.types.is_datetime64_any_dtype(X_test[feature]):
                # For datetime type, add drift in days
                if drift_type in ["sudden", "mean_shift"]:
                    # Sudden drift: add offset after position_idx
                    X_test_drift.loc[position_idx:, feature] = X_test_drift.loc[position_idx:, feature] + pd.Timedelta(
                        days=drift_magnitude
                    )
                elif drift_type == "gradual":
                    # Gradual drift: incrementally increase offset
                    for i in range(position_idx, len(X_test)):
                        factor = (i - position_idx) / (len(X_test) - position_idx)
                        X_test_drift.loc[i, feature] = X_test_drift.loc[i, feature] + pd.Timedelta(
                            days=drift_magnitude * factor
                        )
                elif drift_type == "recurring":
                    # Recurring drift: oscillating pattern
                    for i in range(position_idx, len(X_test)):
                        cycle_position = (i - position_idx) % recurring_cycle
                        factor = np.sin(2 * np.pi * cycle_position / recurring_cycle)
                        X_test_drift.loc[i, feature] = X_test_drift.loc[i, feature] + pd.Timedelta(
                            days=drift_magnitude * factor
                        )
                else:
                    # Other drift types
                    X_test_drift.loc[position_idx:, feature] = X_test_drift.loc[position_idx:, feature] + pd.Timedelta(
                        days=drift_magnitude
                    )
            elif pd.api.types.is_timedelta64_dtype(X_test[feature]):
                # Handle timedelta similarly with appropriate scaling
                if drift_type in ["sudden", "mean_shift"]:
                    X_test_drift.loc[position_idx:, feature] = X_test_drift.loc[position_idx:, feature] * (
                        1.0 + drift_magnitude
                    )
                elif drift_type == "gradual":
                    for i in range(position_idx, len(X_test)):
                        factor = (i - position_idx) / (len(X_test) - position_idx)
                        X_test_drift.loc[i, feature] = X_test_drift.loc[i, feature] * (1.0 + drift_magnitude * factor)
                elif drift_type == "recurring":
                    for i in range(position_idx, len(X_test)):
                        cycle_position = (i - position_idx) % recurring_cycle
                        factor = np.sin(2 * np.pi * cycle_position / recurring_cycle)
                        X_test_drift.loc[i, feature] = X_test_drift.loc[i, feature] * (1.0 + drift_magnitude * factor)
                else:
                    X_test_drift.loc[position_idx:, feature] = X_test_drift.loc[position_idx:, feature] * (
                        1.0 + drift_magnitude
                    )

        # Handle numeric features (original implementation)
        else:
            feature_std = X_test[feature].std()
            if feature_std == 0:
                feature_std = 1.0  # Avoid division by zero

            if drift_type == "mean_shift":
                # Mean shift: add constant value after position_idx
                X_test_drift.loc[position_idx:, feature] += drift_magnitude * feature_std

            elif drift_type == "variance_shift":
                # Variance shift: multiply by factor after position_idx
                X_test_drift.loc[position_idx:, feature] *= 1.0 + drift_magnitude

            elif drift_type == "sudden":
                # Sudden drift: add constant value after position_idx
                X_test_drift.loc[position_idx:, feature] += drift_magnitude * feature_std

            elif drift_type == "gradual":
                # Gradual drift: incrementally increase offset
                for i in range(position_idx, len(X_test)):
                    factor = (i - position_idx) / (len(X_test) - position_idx)
                    X_test_drift.loc[i, feature] += drift_magnitude * feature_std * factor

            elif drift_type == "incremental":
                # Incremental drift: small steps over time
                step_size = drift_magnitude * feature_std / (len(X_test) - position_idx)
                for i in range(position_idx, len(X_test)):
                    steps = i - position_idx
                    X_test_drift.loc[i, feature] += step_size * steps

            elif drift_type == "recurring":
                # Recurring drift: oscillating pattern
                for i in range(position_idx, len(X_test)):
                    cycle_position = (i - position_idx) % recurring_cycle
                    factor = np.sin(2 * np.pi * cycle_position / recurring_cycle)
                    X_test_drift.loc[i, feature] += drift_magnitude * feature_std * factor

            # Add noise to the feature (only for numeric features)
            X_test_drift[feature] += np.random.normal(0, noise * feature_std, size=len(X_test))

    return X_test_drift


def _generate_gaussian_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    drift_position: float = 0.5,
    noise: float = 0.05,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate Gaussian data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
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
    X_test = _apply_drift(
        X_test, drift_type, drift_magnitude, drift_ratio, drift_position, noise, categorical_features, **kwargs
    )

    metadata = {
        "name": "Gaussian Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "drift_position": drift_position,
        "noise": noise,
        "description": f"Synthetic Gaussian data with {drift_type} drift",
    }

    return X_ref, X_test, metadata


def _generate_mixed_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    drift_position: float = 0.5,
    noise: float = 0.05,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    n_categories: int = 5,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate mixed numerical and categorical data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
        categorical_features: Indices of categorical features (if None, 30% of features will be categorical)
        random_state: Random state for reproducibility
        n_categories: Number of categories for categorical features
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    if categorical_features is None:
        n_cat_features = max(1, int(0.3 * n_features))
        categorical_features = np.random.choice(range(n_features), size=n_cat_features, replace=False).tolist()

    # Generate reference data
    ref_data = np.zeros((n_samples, n_features))
    test_data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        if i in categorical_features:
            # Generate categorical feature
            categories = np.random.randint(0, n_categories, size=n_samples)
            ref_data[:, i] = categories
            test_data[:, i] = np.random.randint(0, n_categories, size=n_samples)
        else:
            # Generate numerical feature
            ref_data[:, i] = np.random.normal(0, 1, size=n_samples)
            test_data[:, i] = np.random.normal(0, 1, size=n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Apply drift to test data
    X_test = _apply_drift(
        X_test, drift_type, drift_magnitude, drift_ratio, drift_position, noise, categorical_features, **kwargs
    )

    metadata = {
        "name": "Mixed Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "drift_position": drift_position,
        "noise": noise,
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
    drift_position: float = 0.5,
    noise: float = 0.05,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    n_modes: int = 3,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate multimodal data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        n_modes: Number of modes in the distribution
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    # Generate samples from multiple Gaussian distributions
    samples_per_mode = n_samples // n_modes
    remainder = n_samples % n_modes

    ref_data = []
    test_data = []
    mode_assignments = []

    for mode in range(n_modes):
        # Determine number of samples for this mode
        n_mode_samples = samples_per_mode + (1 if mode < remainder else 0)

        # Generate mode-specific mean
        mode_mean = np.random.uniform(-5, 5, size=n_features)

        # Generate data for this mode
        mode_ref_data = np.random.normal(loc=mode_mean, scale=1.0, size=(n_mode_samples, n_features))
        mode_test_data = np.random.normal(loc=mode_mean, scale=1.0, size=(n_mode_samples, n_features))

        ref_data.append(mode_ref_data)
        test_data.append(mode_test_data)
        mode_assignments.extend([mode] * n_mode_samples)

    # Concatenate all modes
    ref_data = np.vstack(ref_data)
    test_data = np.vstack(test_data)

    # Create DataFrames
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Add mode as an extra column
    X_ref["mode"] = mode_assignments
    X_test["mode"] = mode_assignments

    # Store mode column for later
    mode_column = X_test["mode"].copy()

    # Apply drift to test data (excluding the mode column)
    X_test_drifted = _apply_drift(
        X_test.drop(columns=["mode"]),
        drift_type,
        drift_magnitude,
        drift_ratio,
        drift_position,
        noise,
        categorical_features,
        **kwargs,
    )

    # Add mode back to the drifted data
    X_test_drifted["mode"] = mode_column

    metadata = {
        "name": "Multimodal Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_modes": n_modes,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "drift_position": drift_position,
        "noise": noise,
        "description": f"Synthetic multimodal data with {drift_type} drift",
    }

    return X_ref, X_test_drifted, metadata


def _generate_time_series_data(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "mean_shift",
    drift_magnitude: float = 1.0,
    drift_ratio: float = 0.5,
    drift_position: float = 0.7,
    noise: float = 0.05,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    seasonality: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate time series data with specified drift.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift to generate
        drift_magnitude: Magnitude of drift
        drift_ratio: Proportion of features affected by drift
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
        categorical_features: Indices of categorical features
        random_state: Random state for reproducibility
        seasonality: Whether to add seasonal patterns to the data
        **kwargs: Additional arguments

    Returns:
        Tuple containing reference data, test data with drift, and metadata
    """
    # Time index
    time_index = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")

    # Generate base time series with trend
    trend = np.linspace(0, 10, n_samples)

    # Add seasonality if requested
    seasonal_component = np.zeros(n_samples)
    if seasonality:
        # Weekly seasonality
        weekly_amplitude = kwargs.get("weekly_amplitude", 3.0)
        seasonal_component += weekly_amplitude * np.sin(2 * np.pi * np.arange(n_samples) / 7)

        # Monthly seasonality
        monthly_amplitude = kwargs.get("monthly_amplitude", 5.0)
        seasonal_component += monthly_amplitude * np.sin(2 * np.pi * np.arange(n_samples) / 30)

        # Add option for custom seasonality
        custom_period = kwargs.get("custom_period")
        custom_amplitude = kwargs.get("custom_amplitude", 4.0)
        if custom_period:
            seasonal_component += custom_amplitude * np.sin(2 * np.pi * np.arange(n_samples) / custom_period)

    # Generate features
    ref_data = np.zeros((n_samples, n_features))
    test_data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Base component is trend + seasonality + noise
        feature_base = trend + seasonal_component + np.random.normal(0, noise * np.std(trend), size=n_samples)

        # Add feature-specific variation
        feature_scale = np.random.uniform(0.5, 2.0)
        feature_offset = np.random.uniform(-10, 10)

        ref_data[:, i] = feature_scale * feature_base + feature_offset
        test_data[:, i] = feature_scale * feature_base + feature_offset

    # Create DataFrames
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref = pd.DataFrame(ref_data, columns=feature_names)
    X_test = pd.DataFrame(test_data, columns=feature_names)

    # Add time column
    X_ref["time"] = time_index
    X_test["time"] = time_index

    # Store time column for later
    time_column = X_test["time"].copy()

    # Apply drift to test data (excluding the time column)
    X_test_drifted = _apply_drift(
        X_test.drop(columns=["time"]),
        drift_type,
        drift_magnitude,
        drift_ratio,
        drift_position,
        noise,
        categorical_features,
        **kwargs,
    )

    # Add time column back to drifted data
    X_test_drifted["time"] = time_column

    metadata = {
        "name": "Time Series Data",
        "n_samples": n_samples,
        "n_features": n_features,
        "has_drift": True,
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_ratio": drift_ratio,
        "drift_position": drift_position,
        "noise": noise,
        "seasonality": seasonality,
        "description": f"Synthetic time series data with {drift_type} drift",
    }

    return X_ref, X_test_drifted, metadata


def generate_concept_drift(
    n_samples: int = 1000,
    n_features: int = 10,
    drift_type: str = "sudden",
    drift_position: float = 0.5,
    noise: float = 0.1,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[bool]]:
    """
    Generate data with concept drift where the relationship between X and y changes.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        drift_type: Type of drift (sudden, gradual, incremental, recurring)
        drift_position: Position where the drift occurs (0.0-1.0)
        noise: Amount of noise to add to the data
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for specific drift types

    Returns:
        Tuple containing:
            - Reference data features (X_ref)
            - Test data features (X_test)
            - Reference data labels (y_ref)
            - Test data labels (y_test)
            - Drift labels (True where drift occurs)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate feature data
    X = np.random.normal(0, 1, size=(n_samples, n_features))

    # Position in samples where drift occurs
    drift_idx = int(drift_position * n_samples)

    # Create labels with different concepts before and after drift
    y = np.zeros(n_samples)

    # Before drift: first feature determines the class
    before_weights = np.zeros(n_features)
    before_weights[0] = 1.0  # First feature is the only one that matters

    # After drift: different features determine the class
    after_weights = np.zeros(n_features)

    # Define recurring cycle for recurring drift type
    recurring_cycle = kwargs.get("recurring_cycle", n_samples // 5)

    if drift_type == "sudden":
        # Complete change in concept
        after_weights[1] = 1.0  # Second feature becomes important
    elif drift_type == "gradual":
        # Gradual transition between concepts
        after_weights[0] = 0.5  # First feature becomes less important
        after_weights[1] = 0.5  # Second feature becomes important
    elif drift_type == "incremental":
        # Small change in concept
        after_weights[0] = 0.8  # First feature remains important but less so
        after_weights[2] = 0.2  # Third feature becomes somewhat important
    else:  # Default or "recurring"
        # Oscillating concept
        after_weights[0] = -1.0  # First feature becomes negatively correlated

    # Generate labels based on concepts
    for i in range(n_samples):
        if i < drift_idx:
            # Before drift
            y[i] = np.dot(X[i], before_weights) > 0
        else:
            if drift_type == "recurring":
                # For recurring drift, alternate between concepts
                cycle_position = (i - drift_idx) % recurring_cycle
                if cycle_position < recurring_cycle / 2:
                    y[i] = np.dot(X[i], after_weights) > 0
                else:
                    y[i] = np.dot(X[i], before_weights) > 0
            else:
                # After drift for non-recurring types
                y[i] = np.dot(X[i], after_weights) > 0

    # Add noise to labels
    if noise > 0:
        noise_mask = np.random.random(n_samples) < noise
        y[noise_mask] = 1 - y[noise_mask]  # Flip labels for noise

    # Convert to binary labels
    y = y.astype(int)

    # Split data into reference (before drift) and test (all)
    X_ref = X[:drift_idx].copy()
    y_ref = y[:drift_idx].copy()
    X_test = X.copy()
    y_test = y.copy()

    # Create drift labels (True where drift occurs)
    drift_labels = [i >= drift_idx for i in range(n_samples)]

    # Convert to DataFrames
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_ref_df = pd.DataFrame(X_ref, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    return X_ref_df, X_test_df, y_ref, y_test, drift_labels


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Example 1: Gaussian data with sudden drift
    X_ref, X_test, metadata = generate_drift(
        generator_name="gaussian",
        n_samples=1000,
        n_features=5,
        drift_type="sudden",
        drift_magnitude=2.0,
        drift_position=0.5,
        random_state=42,
    )

    # Plot first feature to visualize drift
    plt.figure(figsize=(12, 5))
    plt.plot(X_test["feature_0"], label="Feature with drift")
    plt.axvline(
        x=metadata["drift_position"] * len(X_test),
        color="r",
        linestyle="--",
        label=f"Drift point ({metadata['drift_type']})",
    )
    plt.title(f"{metadata['name']} with {metadata['drift_type']} drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig("drift_example.png")

    print(f"Generated {metadata['name']} with {metadata['drift_type']} drift")
    print(f"Reference data shape: {X_ref.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Drift position at index: {int(metadata['drift_position'] * len(X_test))}")
