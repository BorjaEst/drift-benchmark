"""
Drift generation module for drift-benchmark.

This module provides functions to generate synthetic data with various types of drift,
including mean shift, variance shift, sudden drift, and gradual drift. It supports
multiple data distributions like Gaussian, mixed types, multimodal, and time series.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from drift_benchmark.constants import DataGenerator, DriftCharacteristic, DriftPattern
from drift_benchmark.constants.models import DatasetResult, DriftInfo, SyntheticDataConfig

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    generator_name: DataGenerator,
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic = "MEAN_SHIFT",
    drift_magnitude: float = 1.0,
    drift_position: float = 0.5,
    drift_duration: Optional[float] = None,
    drift_affected_features: Optional[List[int]] = None,
    noise: float = 0.0,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **generator_params,
) -> DatasetResult:
    """
    Generate synthetic data with specified drift patterns.

    Args:
        generator_name: Type of data generator to use
        n_samples: Total number of samples to generate
        n_features: Number of features
        drift_pattern: Pattern of drift (SUDDEN, GRADUAL, etc.)
        drift_characteristic: Type of drift characteristic (MEAN_SHIFT, etc.)
        drift_magnitude: Magnitude of the drift
        drift_position: Position where drift occurs (0.0-1.0)
        drift_duration: Duration of gradual drift (for GRADUAL pattern)
        drift_affected_features: Indices of features affected by drift
        noise: Amount of noise to add
        categorical_features: Indices of categorical features
        random_state: Random seed for reproducibility
        **generator_params: Additional generator-specific parameters

    Returns:
        DatasetResult containing:
            - X_ref: Reference data features as pandas DataFrame
            - X_test: Test data features (with drift) as pandas DataFrame
            - y_ref: Reference data labels as pandas Series (None for unsupervised)
            - y_test: Test data labels as pandas Series (None for unsupervised)
            - drift_info: DriftInfo object with drift metadata
            - metadata: Additional dataset metadata
    """
    if random_state is not None:
        np.random.seed(random_state)

    logger.info(f"Generating {generator_name} data with {drift_pattern} {drift_characteristic}")

    # Route to appropriate generator
    if generator_name == "GAUSSIAN":
        dataset_result = _generate_gaussian_data(
            n_samples,
            n_features,
            drift_pattern,
            drift_characteristic,
            drift_magnitude,
            drift_position,
            drift_duration,
            drift_affected_features,
            noise,
            **generator_params,
        )
    elif generator_name == "MIXED":
        dataset_result = _generate_mixed_data(
            n_samples,
            n_features,
            drift_pattern,
            drift_characteristic,
            drift_magnitude,
            drift_position,
            drift_duration,
            drift_affected_features,
            noise,
            categorical_features,
            **generator_params,
        )
    elif generator_name == "MULTIMODAL":
        dataset_result = _generate_multimodal_data(
            n_samples,
            n_features,
            drift_pattern,
            drift_characteristic,
            drift_magnitude,
            drift_position,
            drift_duration,
            drift_affected_features,
            noise,
            **generator_params,
        )
    elif generator_name == "TIME_SERIES":
        dataset_result = _generate_time_series_data(
            n_samples,
            n_features,
            drift_pattern,
            drift_characteristic,
            drift_magnitude,
            drift_position,
            drift_duration,
            drift_affected_features,
            noise,
            **generator_params,
        )
    else:
        raise ValueError(f"Unknown generator: {generator_name}")

    return dataset_result


def _create_dataset_result(
    X_ref: np.ndarray,
    X_test: np.ndarray,
    y_ref: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    drift_info_dict: Dict[str, Any],
    n_features: int,
) -> DatasetResult:
    """Convert raw data arrays to DatasetResult with proper models."""

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Convert to DataFrames
    X_ref_df = pd.DataFrame(X_ref, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Convert labels to Series if they exist
    y_ref_series = pd.Series(y_ref, name="target") if y_ref is not None else None
    y_test_series = pd.Series(y_test, name="target") if y_test is not None else None

    # Create DriftInfo object
    drift_info = DriftInfo(
        has_drift=True,  # We're generating synthetic data with drift
        drift_points=[int(len(X_ref))],  # Drift starts at the beginning of test data
        drift_pattern=drift_info_dict.get("drift_pattern"),
        drift_magnitude=drift_info_dict.get("drift_magnitude"),
        drift_characteristics=[drift_info_dict.get("drift_characteristic", "")],
        metadata=drift_info_dict,
    )

    # Create metadata
    metadata = {
        "generator": drift_info_dict.get("generator"),
        "n_samples_total": len(X_ref) + len(X_test),
        "n_ref_samples": len(X_ref),
        "n_test_samples": len(X_test),
        "n_features": n_features,
        "feature_names": feature_names,
        "has_labels": y_ref is not None or y_test is not None,
        "generation_parameters": drift_info_dict.get("parameters", {}),
    }

    return DatasetResult(
        X_ref=X_ref_df,
        X_test=X_test_df,
        y_ref=y_ref_series,
        y_test=y_test_series,
        drift_info=drift_info,
        metadata=metadata,
    )


def _generate_gaussian_data(
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic,
    drift_magnitude: float,
    drift_position: float,
    drift_duration: Optional[float],
    drift_affected_features: Optional[List[int]],
    noise: float,
    **params,
) -> DatasetResult:
    """Generate Gaussian distributed data with drift."""

    # Parameters with defaults
    mean_ref = params.get("mean", 0.0)
    std_ref = params.get("std", 1.0)

    # Determine affected features
    if drift_affected_features is None:
        drift_affected_features = list(range(n_features))

    # Split samples
    ref_samples = int(n_samples * drift_position)
    test_samples = n_samples - ref_samples

    # Generate reference data
    X_ref = np.random.normal(mean_ref, std_ref, (ref_samples, n_features))

    # Generate test data with drift
    X_test = _apply_drift_to_gaussian(
        ref_samples=ref_samples,
        test_samples=test_samples,
        n_features=n_features,
        drift_pattern=drift_pattern,
        drift_characteristic=drift_characteristic,
        drift_magnitude=drift_magnitude,
        drift_duration=drift_duration,
        drift_affected_features=drift_affected_features,
        mean_ref=mean_ref,
        std_ref=std_ref,
    )

    # Add noise if specified
    if noise > 0:
        X_ref += np.random.normal(0, noise, X_ref.shape)
        X_test += np.random.normal(0, noise, X_test.shape)

    # Create drift metadata
    drift_info_dict = {
        "generator": "GAUSSIAN",
        "drift_pattern": drift_pattern,
        "drift_characteristic": drift_characteristic,
        "drift_magnitude": drift_magnitude,
        "drift_position": drift_position,
        "drift_duration": drift_duration,
        "drift_affected_features": drift_affected_features,
        "noise": noise,
        "parameters": params,
    }

    return _create_dataset_result(X_ref, X_test, None, None, drift_info_dict, n_features)


def _apply_drift_to_gaussian(
    ref_samples: int,
    test_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic,
    drift_magnitude: float,
    drift_duration: Optional[float],
    drift_affected_features: List[int],
    mean_ref: float,
    std_ref: float,
) -> np.ndarray:
    """Apply drift to Gaussian data."""

    X_test = np.zeros((test_samples, n_features))

    for i in range(test_samples):
        # Calculate drift factor based on pattern
        if drift_pattern == "SUDDEN":
            drift_factor = 1.0  # Full drift immediately
        elif drift_pattern == "GRADUAL":
            if drift_duration is None:
                drift_duration = 1.0  # Default to full test period
            drift_end = int(drift_duration * test_samples)
            if i < drift_end:
                drift_factor = i / drift_end  # Linear increase
            else:
                drift_factor = 1.0
        elif drift_pattern == "INCREMENTAL":
            # Step-wise drift (multiple steps)
            n_steps = 5  # Default number of steps
            step_size = test_samples // n_steps
            step_number = min(i // step_size, n_steps - 1)
            drift_factor = step_number / (n_steps - 1)
        elif drift_pattern == "RECURRING":
            # Periodic drift
            period = int(0.2 * test_samples)  # 20% of test samples
            drift_factor = 0.5 * (1 + np.sin(2 * np.pi * i / period))
        elif drift_pattern == "SEASONAL":
            # Seasonal pattern (similar to recurring but smoother)
            period = int(0.25 * test_samples)
            drift_factor = 0.5 * (1 + np.cos(2 * np.pi * i / period))
        else:
            drift_factor = 1.0

        # Apply drift characteristic
        for j in range(n_features):
            if j in drift_affected_features:
                if drift_characteristic == "MEAN_SHIFT":
                    mean_drifted = mean_ref + drift_factor * drift_magnitude
                    X_test[i, j] = np.random.normal(mean_drifted, std_ref)
                elif drift_characteristic == "VARIANCE_SHIFT":
                    std_drifted = std_ref + drift_factor * drift_magnitude
                    X_test[i, j] = np.random.normal(mean_ref, std_drifted)
                elif drift_characteristic == "DISTRIBUTION_SHIFT":
                    # Shift to different distribution (e.g., uniform)
                    if drift_factor > 0.5:
                        # Switch to uniform distribution
                        X_test[i, j] = np.random.uniform(mean_ref - drift_magnitude, mean_ref + drift_magnitude)
                    else:
                        X_test[i, j] = np.random.normal(mean_ref, std_ref)
                else:
                    X_test[i, j] = np.random.normal(mean_ref, std_ref)
            else:
                # No drift for non-affected features
                X_test[i, j] = np.random.normal(mean_ref, std_ref)

    return X_test


def _generate_mixed_data(
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic,
    drift_magnitude: float,
    drift_position: float,
    drift_duration: Optional[float],
    drift_affected_features: Optional[List[int]],
    noise: float,
    categorical_features: Optional[List[int]],
    **params,
) -> DatasetResult:
    """Generate mixed continuous and categorical data with drift."""

    if categorical_features is None:
        # Default: half of features are categorical
        categorical_features = list(range(n_features // 2, n_features))

    # Split samples
    ref_samples = int(n_samples * drift_position)
    test_samples = n_samples - ref_samples

    # Generate base data
    X_ref = np.zeros((ref_samples, n_features))
    X_test = np.zeros((test_samples, n_features))

    # Generate continuous features
    continuous_features = [i for i in range(n_features) if i not in categorical_features]

    for i in continuous_features:
        X_ref[:, i] = np.random.normal(0, 1, ref_samples)

        # Apply drift for test data
        if drift_affected_features is None or i in drift_affected_features:
            if drift_characteristic == "MEAN_SHIFT":
                X_test[:, i] = np.random.normal(drift_magnitude, 1, test_samples)
            else:
                X_test[:, i] = np.random.normal(0, 1, test_samples)
        else:
            X_test[:, i] = np.random.normal(0, 1, test_samples)

    # Generate categorical features
    n_categories = params.get("n_categories", 3)

    for i in categorical_features:
        # Reference categories
        X_ref[:, i] = np.random.randint(0, n_categories, ref_samples)

        # Test categories with potential drift
        if drift_affected_features is None or i in drift_affected_features:
            if drift_characteristic == "DISTRIBUTION_SHIFT":
                # Shift category probabilities
                ref_probs = np.ones(n_categories) / n_categories
                drift_probs = ref_probs.copy()
                drift_probs[0] += drift_magnitude * 0.3  # Increase first category
                drift_probs[1:] -= drift_magnitude * 0.3 / (n_categories - 1)  # Decrease others
                drift_probs = np.clip(drift_probs, 0.01, 0.99)
                drift_probs /= drift_probs.sum()  # Normalize

                X_test[:, i] = np.random.choice(n_categories, test_samples, p=drift_probs)
            else:
                X_test[:, i] = np.random.randint(0, n_categories, test_samples)
        else:
            X_test[:, i] = np.random.randint(0, n_categories, test_samples)

    # Add noise to continuous features only
    if noise > 0:
        for i in continuous_features:
            X_ref[:, i] += np.random.normal(0, noise, ref_samples)
            X_test[:, i] += np.random.normal(0, noise, test_samples)

    drift_info_dict = {
        "generator": "MIXED",
        "drift_pattern": drift_pattern,
        "drift_characteristic": drift_characteristic,
        "drift_duration": drift_duration,
        "categorical_features": categorical_features,
        "continuous_features": continuous_features,
        "drift_magnitude": drift_magnitude,
        "parameters": params,
    }

    return _create_dataset_result(X_ref, X_test, None, None, drift_info_dict, n_features)


def _generate_multimodal_data(
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic,
    drift_magnitude: float,
    drift_position: float,
    drift_duration: Optional[float],
    drift_affected_features: Optional[List[int]],
    noise: float,
    **params,
) -> DatasetResult:
    """Generate multimodal data with drift."""

    n_modes = params.get("n_modes", 3)
    mode_separation = params.get("mode_separation", 3.0)

    # Split samples
    ref_samples = int(n_samples * drift_position)
    test_samples = n_samples - ref_samples

    # Generate reference data (multimodal)
    X_ref = _generate_multimodal_samples(ref_samples, n_features, n_modes, mode_separation)

    # Generate test data with drift
    if drift_characteristic == "DISTRIBUTION_SHIFT":
        # Change number of modes or their positions
        new_n_modes = n_modes + int(drift_magnitude)
        new_separation = mode_separation + drift_magnitude
        X_test = _generate_multimodal_samples(test_samples, n_features, new_n_modes, new_separation)
    else:
        # Apply standard drift to multimodal data
        X_test = _generate_multimodal_samples(test_samples, n_features, n_modes, mode_separation)

        # Apply mean shift to affected features
        if drift_affected_features is None:
            drift_affected_features = list(range(n_features))

        for i in drift_affected_features:
            if drift_characteristic == "MEAN_SHIFT":
                X_test[:, i] += drift_magnitude

    # Add noise
    if noise > 0:
        X_ref += np.random.normal(0, noise, X_ref.shape)
        X_test += np.random.normal(0, noise, X_test.shape)

    drift_info_dict = {
        "generator": "MULTIMODAL",
        "drift_pattern": drift_pattern,
        "drift_characteristic": drift_characteristic,
        "drift_duration": drift_duration,
        "n_modes": n_modes,
        "mode_separation": mode_separation,
        "drift_magnitude": drift_magnitude,
        "parameters": params,
    }

    return _create_dataset_result(X_ref, X_test, None, None, drift_info_dict, n_features)


def _generate_multimodal_samples(n_samples: int, n_features: int, n_modes: int, separation: float) -> np.ndarray:
    """Generate samples from a multimodal distribution."""
    X = np.zeros((n_samples, n_features))

    # Assign samples to modes
    mode_assignments = np.random.randint(0, n_modes, n_samples)

    # Generate mode centers
    mode_centers = np.random.uniform(-separation, separation, (n_modes, n_features))

    for i in range(n_samples):
        mode = mode_assignments[i]
        X[i] = np.random.multivariate_normal(mode_centers[mode], np.eye(n_features))

    return X


def _generate_time_series_data(
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic,
    drift_magnitude: float,
    drift_position: float,
    drift_duration: Optional[float],
    drift_affected_features: Optional[List[int]],
    noise: float,
    **params,
) -> DatasetResult:
    """Generate time series data with drift."""

    trend = params.get("trend", 0.0)
    seasonality_period = params.get("seasonality", 50)
    seasonality_amplitude = params.get("seasonality_amplitude", 0.5)

    # Split samples
    ref_samples = int(n_samples * drift_position)
    test_samples = n_samples - ref_samples

    # Generate time indices
    ref_time = np.arange(ref_samples)
    test_time = np.arange(ref_samples, ref_samples + test_samples)

    # Generate reference data
    X_ref = np.zeros((ref_samples, n_features))
    for i in range(n_features):
        trend_component = trend * ref_time
        seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * ref_time / seasonality_period)
        random_component = np.random.normal(0, 1, ref_samples)
        X_ref[:, i] = trend_component + seasonal_component + random_component

    # Generate test data with drift
    X_test = np.zeros((test_samples, n_features))
    for i in range(n_features):
        if drift_affected_features is None or i in drift_affected_features:
            # Apply drift to trend or seasonality
            if drift_characteristic == "MEAN_SHIFT":
                trend_drifted = trend + drift_magnitude
            else:
                trend_drifted = trend

            trend_component = trend_drifted * test_time
            seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * test_time / seasonality_period)
            random_component = np.random.normal(0, 1, test_samples)
            X_test[:, i] = trend_component + seasonal_component + random_component
        else:
            # No drift
            trend_component = trend * test_time
            seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * test_time / seasonality_period)
            random_component = np.random.normal(0, 1, test_samples)
            X_test[:, i] = trend_component + seasonal_component + random_component

    # Add noise
    if noise > 0:
        X_ref += np.random.normal(0, noise, X_ref.shape)
        X_test += np.random.normal(0, noise, X_test.shape)

    drift_info_dict = {
        "generator": "TIME_SERIES",
        "drift_pattern": drift_pattern,
        "drift_characteristic": drift_characteristic,
        "drift_duration": drift_duration,
        "trend": trend,
        "seasonality_period": seasonality_period,
        "seasonality_amplitude": seasonality_amplitude,
        "drift_magnitude": drift_magnitude,
        "parameters": params,
    }

    return _create_dataset_result(X_ref, X_test, None, None, drift_info_dict, n_features)


# Legacy function for backward compatibility
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
    Legacy function for backward compatibility.

    Maps old parameter names to new function.
    """
    # Map old names to new constants
    generator_map = {
        "gaussian": "GAUSSIAN",
        "mixed": "MIXED",
        "multimodal": "MULTIMODAL",
        "time_series": "TIME_SERIES",
    }

    pattern_map = {
        "sudden": "SUDDEN",
        "gradual": "GRADUAL",
        "incremental": "INCREMENTAL",
        "recurring": "RECURRING",
        "mean_shift": "SUDDEN",  # Assume sudden for mean_shift
        "variance_shift": "SUDDEN",
    }

    characteristic_map = {
        "mean_shift": "MEAN_SHIFT",
        "variance_shift": "VARIANCE_SHIFT",
        "sudden": "MEAN_SHIFT",  # Default to mean shift
        "gradual": "MEAN_SHIFT",
        "incremental": "MEAN_SHIFT",
        "recurring": "MEAN_SHIFT",
    }

    new_generator = generator_map.get(generator_name.lower(), "GAUSSIAN")
    new_pattern = pattern_map.get(drift_type.lower(), "SUDDEN")
    new_characteristic = characteristic_map.get(drift_type.lower(), "MEAN_SHIFT")

    # Calculate affected features from drift_ratio
    n_affected = max(1, int(drift_ratio * n_features))
    drift_affected_features = list(range(n_affected))

    dataset_result = generate_synthetic_data(
        generator_name=new_generator,
        n_samples=n_samples,
        n_features=n_features,
        drift_pattern=new_pattern,
        drift_characteristic=new_characteristic,
        drift_magnitude=drift_magnitude,
        drift_position=drift_position,
        drift_affected_features=drift_affected_features,
        noise=noise,
        categorical_features=categorical_features,
        random_state=random_state,
        **kwargs,
    )

    # Extract DataFrames from DatasetResult for backward compatibility
    X_ref_df = dataset_result.X_ref
    X_test_df = dataset_result.X_test

    # Legacy metadata format
    metadata = {
        "name": f"{generator_name}_drift",
        "drift_type": drift_type,
        "drift_magnitude": drift_magnitude,
        "drift_position": drift_position,
        "n_features": n_features,
        "n_samples": n_samples,
    }

    return X_ref_df, X_test_df, metadata


def generate_synthetic_data_from_config(config: SyntheticDataConfig) -> DatasetResult:
    """
    Generate synthetic data using a SyntheticDataConfig object.

    Args:
        config: SyntheticDataConfig object containing all generation parameters

    Returns:
        DatasetResult containing:
            - X_ref: Reference data features as pandas DataFrame
            - X_test: Test data features (with drift) as pandas DataFrame
            - y_ref: Reference data labels as pandas Series (None for unsupervised)
            - y_test: Test data labels as pandas Series (None for unsupervised)
            - drift_info: DriftInfo object with drift metadata
            - metadata: Additional dataset metadata
    """
    return generate_synthetic_data(
        generator_name=config.generator_name,
        n_samples=config.n_samples,
        n_features=config.n_features,
        drift_pattern=config.drift_pattern,
        drift_characteristic=config.drift_characteristic,
        drift_magnitude=config.drift_magnitude,
        drift_position=config.drift_position,
        drift_duration=config.drift_duration,
        drift_affected_features=config.drift_affected_features,
        noise=config.noise,
        categorical_features=config.categorical_features,
        random_state=config.random_state,
        **config.generator_params,
    )


def create_synthetic_data_config(
    generator_name: DataGenerator,
    n_samples: int,
    n_features: int,
    drift_pattern: DriftPattern,
    drift_characteristic: DriftCharacteristic = "MEAN_SHIFT",
    drift_magnitude: float = 1.0,
    drift_position: float = 0.5,
    drift_duration: Optional[float] = None,
    drift_affected_features: Optional[List[int]] = None,
    noise: float = 0.0,
    categorical_features: Optional[List[int]] = None,
    random_state: Optional[int] = None,
    **generator_params,
) -> SyntheticDataConfig:
    """
    Create a SyntheticDataConfig object from individual parameters.

    Args:
        generator_name: Type of data generator to use
        n_samples: Total number of samples to generate
        n_features: Number of features
        drift_pattern: Pattern of drift (SUDDEN, GRADUAL, etc.)
        drift_characteristic: Type of drift characteristic (MEAN_SHIFT, etc.)
        drift_magnitude: Magnitude of the drift
        drift_position: Position where drift occurs (0.0-1.0)
        drift_duration: Duration of gradual drift (for GRADUAL pattern)
        drift_affected_features: Indices of features affected by drift
        noise: Amount of noise to add
        categorical_features: Indices of categorical features
        random_state: Random seed for reproducibility
        **generator_params: Additional generator-specific parameters

    Returns:
        SyntheticDataConfig object with all parameters
    """
    return SyntheticDataConfig(
        generator_name=generator_name,
        n_samples=n_samples,
        n_features=n_features,
        drift_pattern=drift_pattern,
        drift_characteristic=drift_characteristic,
        drift_magnitude=drift_magnitude,
        drift_position=drift_position,
        drift_duration=drift_duration,
        drift_affected_features=drift_affected_features,
        noise=noise,
        categorical_features=categorical_features,
        random_state=random_state,
        generator_params=generator_params,
    )
