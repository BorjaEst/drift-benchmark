"""
Predefined drift scenarios module using scikit-learn datasets.

This module provides dataset-specific drift scenarios tailored to sklearn datasets.
Each scenario is designed to create meaningful drift patterns that match the
characteristics and domain of a specific sklearn dataset, preventing misuse
and ensuring realistic drift detection experiments.

KEY CAPABILITIES:
1. Dataset-specific drift scenarios (iris, wine, breast_cancer, diabetes, digits)
2. Multiple drift types: class-based, feature-based, and target-based
3. Scientifically meaningful drift patterns for each dataset
4. Type-safe scenario creation with proper validation

SCENARIO TYPES:
- Class-based: Different target classes for reference vs test (e.g., species drift)
- Feature-based: Feature value ranges for drift (e.g., size-based splits)
- Target-based: Target value ranges for regression datasets (e.g., progression levels)

The module ensures users cannot apply incompatible scenarios to datasets,
simplifying the library and preventing common misuse patterns.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer as sk_load_breast_cancer
from sklearn.datasets import load_diabetes as sk_load_diabetes
from sklearn.datasets import load_digits as sk_load_digits
from sklearn.datasets import load_iris as sk_load_iris
from sklearn.datasets import load_wine as sk_load_wine
from sklearn.model_selection import train_test_split

from drift_benchmark.constants.literals import DataType
from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, SklearnDataConfig
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS FOR PREDEFINED SCENARIOS
# =============================================================================

# Available sklearn datasets with descriptions
SKLEARN_DATASETS = {
    "iris": "Iris flower classification dataset (3 classes, 4 features)",
    "wine": "Wine recognition dataset (3 classes, 13 features)",
    "breast_cancer": "Breast cancer Wisconsin dataset (2 classes, 30 features)",
    "diabetes": "Diabetes regression dataset (442 samples, 10 features)",
    "digits": "Optical recognition of handwritten digits (10 classes, 64 features)",
}

# Predefined drift scenarios tied to specific sklearn datasets
SKLEARN_DRIFT_SCENARIOS = {
    "iris_species_drift": {
        "dataset": "iris",
        "description": "Iris species drift - Setosa vs Versicolor/Virginica classification",
        "ref_classes": [0],  # Setosa only
        "test_classes": [1, 2],  # Versicolor and Virginica
        "characteristics": ["COVARIATE", "CONCEPT"],
        "drift_type": "class_based",
    },
    "iris_feature_drift": {
        "dataset": "iris",
        "description": "Iris feature drift - samples with smaller vs larger measurements",
        "ref_condition": "sepal length (cm) <= 5.5",  # Smaller flowers
        "test_condition": "sepal length (cm) > 6.0",  # Larger flowers
        "characteristics": ["COVARIATE"],
        "drift_type": "feature_based",
    },
    "wine_quality_drift": {
        "dataset": "wine",
        "description": "Wine quality drift - class 0 vs classes 1&2",
        "ref_classes": [0],  # Class 0 wines
        "test_classes": [1, 2],  # Class 1 and 2 wines
        "characteristics": ["COVARIATE", "CONCEPT"],
        "drift_type": "class_based",
    },
    "wine_alcohol_drift": {
        "dataset": "wine",
        "description": "Wine alcohol content drift - low vs high alcohol wines",
        "ref_condition": "alcohol <= 12.5",  # Lower alcohol content
        "test_condition": "alcohol >= 13.5",  # Higher alcohol content
        "characteristics": ["COVARIATE"],
        "drift_type": "feature_based",
    },
    "breast_cancer_severity_drift": {
        "dataset": "breast_cancer",
        "description": "Breast cancer severity drift - benign vs malignant",
        "ref_classes": [0],  # Malignant
        "test_classes": [1],  # Benign
        "characteristics": ["COVARIATE", "CONCEPT"],
        "drift_type": "class_based",
    },
    "breast_cancer_size_drift": {
        "dataset": "breast_cancer",
        "description": "Breast cancer tumor size drift - smaller vs larger tumors",
        "ref_condition": "mean radius <= 13.0",  # Smaller tumors
        "test_condition": "mean radius >= 16.0",  # Larger tumors
        "characteristics": ["COVARIATE"],
        "drift_type": "feature_based",
    },
    "diabetes_progression_drift": {
        "dataset": "diabetes",
        "description": "Diabetes progression drift - low vs high progression scores",
        "ref_condition": "target <= 100",  # Lower progression
        "test_condition": "target >= 200",  # Higher progression
        "characteristics": ["CONCEPT"],
        "drift_type": "target_based",
    },
    "digits_complexity_drift": {
        "dataset": "digits",
        "description": "Digits complexity drift - simple vs complex digits",
        "ref_classes": [0, 1],  # Simple digits (0, 1)
        "test_classes": [8, 9],  # Complex digits (8, 9)
        "characteristics": ["COVARIATE", "CONCEPT"],
        "drift_type": "class_based",
    },
}


# =============================================================================
# MAIN SCENARIO API FUNCTIONS
# =============================================================================


def create_sklearn_drift_scenario(scenario_name: str, random_state: int = 42, name: Optional[str] = None) -> DatasetResult:
    """
    Create a predefined drift detection scenario using sklearn datasets.

    This function creates specific drift scenarios that are tailored to each
    sklearn dataset's characteristics, ensuring meaningful and realistic drift
    patterns that match the dataset's structure and domain.

    Args:
        scenario_name: Name of predefined scenario (must match SKLEARN_DRIFT_SCENARIOS)
        random_state: Random state for reproducible splits
        name: Custom name for the dataset result

    Returns:
        DatasetResult with reference and test sets exhibiting the specified drift

    Available Scenarios:
        - "iris_species_drift": Setosa vs Versicolor/Virginica
        - "iris_feature_drift": Small vs large flowers (sepal length)
        - "wine_quality_drift": Class 0 vs classes 1&2
        - "wine_alcohol_drift": Low vs high alcohol content
        - "breast_cancer_severity_drift": Malignant vs benign
        - "breast_cancer_size_drift": Small vs large tumors
        - "diabetes_progression_drift": Low vs high progression scores
        - "digits_complexity_drift": Simple (0,1) vs complex (8,9) digits

    Examples:
        # Create iris species drift scenario
        result = create_sklearn_drift_scenario("iris_species_drift")

        # Create wine alcohol content drift scenario
        result = create_sklearn_drift_scenario("wine_alcohol_drift", random_state=123)
    """
    if scenario_name not in SKLEARN_DRIFT_SCENARIOS:
        available = list(SKLEARN_DRIFT_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    scenario = SKLEARN_DRIFT_SCENARIOS[scenario_name]
    dataset_name = scenario["dataset"]

    if name is None:
        name = scenario_name

    logger.info(f"Creating sklearn drift scenario: {scenario_name}")
    logger.info(f"Dataset: {dataset_name}, Drift type: {scenario['drift_type']}")

    # Load the base sklearn dataset
    base_result = load_sklearn_dataset(dataset_name)

    # Combine reference and test data for filtering
    X_full = np.vstack([base_result.X_ref, base_result.X_test])
    y_full = np.hstack([base_result.y_ref, base_result.y_test])

    # Apply scenario-specific filtering
    if scenario["drift_type"] == "class_based":
        X_ref, y_ref = _filter_by_classes(X_full, y_full, scenario["ref_classes"])
        X_test, y_test = _filter_by_classes(X_full, y_full, scenario["test_classes"])
    elif scenario["drift_type"] == "feature_based":
        X_ref, y_ref = _filter_by_feature_condition(X_full, y_full, scenario["ref_condition"], base_result.metadata["feature_names"])
        X_test, y_test = _filter_by_feature_condition(X_full, y_full, scenario["test_condition"], base_result.metadata["feature_names"])
    elif scenario["drift_type"] == "target_based":
        X_ref, y_ref = _filter_by_target_condition(X_full, y_full, scenario["ref_condition"])
        X_test, y_test = _filter_by_target_condition(X_full, y_full, scenario["test_condition"])
    else:
        raise ValueError(f"Unknown drift type: {scenario['drift_type']}")

    # Ensure we have enough samples
    if len(X_ref) == 0 or len(X_test) == 0:
        raise ValueError(f"Scenario '{scenario_name}' resulted in empty reference or test set")

    # Create drift info
    drift_info = DriftInfo(
        has_drift=True,
        drift_points=[len(X_ref)],  # Drift occurs at boundary between ref and test
        drift_pattern="sudden",
        drift_magnitude=0.5,  # Moderate drift
        drift_characteristics=scenario["characteristics"],
        metadata={
            "scenario_name": scenario_name,
            "dataset": dataset_name,
            "drift_type": scenario["drift_type"],
            "description": scenario["description"],
        },
    )

    # Create updated metadata
    metadata = base_result.metadata.copy()
    metadata.update(
        {
            "name": name,
            "n_samples": len(X_ref) + len(X_test),
            "has_drift": True,
            "drift_scenario": scenario_name,
            "drift_points": [len(X_ref)],
            "source": f"sklearn.{dataset_name}.{scenario_name}",
            "creation_time": datetime.now().isoformat(),
        }
    )

    logger.info(f"Scenario created: {len(X_ref)} ref + {len(X_test)} test samples")

    return DatasetResult(
        X_ref=X_ref,
        X_test=X_test,
        y_ref=y_ref,
        y_test=y_test,
        drift_info=drift_info,
        metadata=metadata,
    )


def load_sklearn_dataset(config: Union[str, Dict, DatasetConfig]) -> DatasetResult:
    """
    Load a scikit-learn dataset with support for drift detection scenarios.

    Args:
        config: Dataset configuration. Can be:
            - String: sklearn dataset name (e.g., "iris", "wine")
            - Dictionary: configuration parameters
            - DatasetConfig: Pydantic model with full configuration

    Returns:
        DatasetResult: Complete dataset with X_ref, X_test, y_ref, y_test,
                      drift_info, and metadata

    Examples:
        # Load iris dataset by name
        result = load_sklearn_dataset("iris")

        # Load with custom configuration
        result = load_sklearn_dataset({
            "name": "custom_iris",
            "type": "SKLEARN",
            "sklearn_config": {
                "dataset_name": "iris",
                "test_split": 0.2,
                "random_state": 42
            }
        })
    """
    try:
        # Validate and parse configuration
        dataset_config = _validate_and_parse_sklearn_config(config)

        logger.info(f"Loading sklearn dataset: {dataset_config.name}")

        # Load the sklearn dataset
        return _load_sklearn_dataset(dataset_config)

    except ValidationError as e:
        # Handle Pydantic validation errors
        error_msg = str(e)
        if "test_split" in error_msg and ("less than" in error_msg or "greater than" in error_msg):
            raise ValueError("Invalid test split value")
        else:
            raise ValueError(f"Invalid sklearn dataset configuration: {e}")
    except Exception as e:
        # Re-raise ValueError as-is, wrap other exceptions
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Failed to load sklearn dataset: {e}")


# =============================================================================
# INTERNAL SKLEARN LOADING FUNCTIONS
# =============================================================================


def _load_sklearn_dataset(config: DatasetConfig) -> DatasetResult:
    """Load a built-in scikit-learn dataset using proper configuration."""
    try:
        if not config.sklearn_config:
            raise ValueError("Sklearn dataset configuration missing")

        sklearn_config = config.sklearn_config
        dataset_name = sklearn_config.dataset_name.lower()

        logger.debug(f"Loading sklearn dataset: {dataset_name}")

        # Load the appropriate sklearn dataset
        if dataset_name == "iris":
            dataset = sk_load_iris(return_X_y=False)
            description = "Iris flower classification dataset"
        elif dataset_name == "wine":
            dataset = sk_load_wine(return_X_y=False)
            description = "Wine recognition dataset"
        elif dataset_name == "breast_cancer":
            dataset = sk_load_breast_cancer(return_X_y=False)
            description = "Breast cancer Wisconsin dataset"
        elif dataset_name == "diabetes":
            dataset = sk_load_diabetes(return_X_y=False)
            description = "Diabetes regression dataset"
        elif dataset_name == "digits":
            dataset = sk_load_digits(return_X_y=False)
            description = "Optical recognition of handwritten digits"
        else:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        # Extract data
        X = dataset.data
        y = dataset.target
        feature_names = getattr(dataset, "feature_names", [f"feature_{i}" for i in range(X.shape[1])])

        # Convert feature names to list if it's an ndarray
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()

        # Split data
        X_ref, X_test, y_ref, y_test = train_test_split(
            X,
            y,
            test_size=sklearn_config.test_split,
            random_state=sklearn_config.random_state or 42,
            stratify=y if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) // 10 else None,
        )

        # Create dataset metadata
        metadata = DatasetMetadata(
            name=config.name,
            n_samples=len(X),
            n_features=X.shape[1],
            feature_names=feature_names,
            target_name="target",
            data_types=_infer_data_types_from_array(X, feature_names),
            has_drift=False,  # Built-in datasets typically don't have drift
            drift_points=None,
            drift_metadata={},
            source=f"sklearn.datasets.{dataset_name}",
            creation_time=datetime.now().isoformat(),
            preprocessing_applied=[],
        )

        # Create drift info
        drift_info = DriftInfo(
            has_drift=False,
            drift_points=None,
            drift_pattern=None,
            drift_magnitude=None,
            drift_characteristics=[],
            metadata={"description": description},
        )

        logger.info(f"Sklearn dataset loaded: {len(X_ref)} ref + {len(X_test)} test samples, {X.shape[1]} features")

        return DatasetResult(X_ref=X_ref, X_test=X_test, y_ref=y_ref, y_test=y_test, drift_info=drift_info, metadata=metadata.model_dump())

    except ValidationError as e:
        # Handle Pydantic validation errors
        error_msg = str(e)
        if "test_split" in error_msg and ("less than" in error_msg or "greater than" in error_msg):
            raise ValueError("Invalid test split value")
        else:
            raise ValueError(f"Invalid sklearn dataset configuration: {e}")
    except AttributeError as e:
        # Handle missing attributes that might occur with invalid configs
        if "sklearn_config" in str(e):
            raise ValueError("Sklearn dataset configuration missing")
        else:
            raise ValueError(f"Invalid sklearn dataset configuration: {e}")
    except Exception as e:
        # Re-raise ValueError as-is, wrap other exceptions
        if isinstance(e, ValueError):
            raise
        else:
            dataset_name_for_error = dataset_name if "dataset_name" in locals() else "unknown"
            raise ValueError(f"Failed to load sklearn dataset '{dataset_name_for_error}': {e}")


def _validate_and_parse_sklearn_config(config: Union[str, Dict, DatasetConfig]) -> DatasetConfig:
    """
    Validate and parse sklearn dataset configuration input.

    Args:
        config: Raw configuration input

    Returns:
        Validated DatasetConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        if isinstance(config, str):
            # String input - check for sklearn dataset
            if config.lower() in SKLEARN_DATASETS:
                return DatasetConfig(name=config, type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name=config))
            else:
                available = list(SKLEARN_DATASETS.keys())
                raise ValueError(f"Sklearn dataset '{config}' not found. Available: {available}")

        elif isinstance(config, dict):
            # Dictionary input - validate and convert to DatasetConfig
            return DatasetConfig(**config)

        elif isinstance(config, DatasetConfig):
            # Already a DatasetConfig - return as-is
            return config

        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected str, dict, or DatasetConfig")

    except ValidationError as e:
        # Handle Pydantic validation errors with more user-friendly messages
        error_msg = str(e)
        if "test_split" in error_msg and "less than" in error_msg:
            raise ValueError("Invalid test split value")
        else:
            raise ValueError(f"Invalid sklearn dataset configuration: {e}")


def _infer_data_types_from_array(X: np.ndarray, feature_names: List[str]) -> Dict[str, DataType]:
    """Infer data types for numpy array features."""
    data_types = {}

    for i, name in enumerate(feature_names):
        # All sklearn datasets have continuous features
        data_types[name] = "CONTINUOUS"

    return data_types


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SKLEARN DATASETS
# =============================================================================


def load_iris(test_size: float = 0.3, random_state: int = 42) -> DatasetResult:
    """Load the Iris dataset using proper configuration."""
    config = DatasetConfig(
        name="iris", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="iris", test_split=test_size, random_state=random_state)
    )
    return _load_sklearn_dataset(config)


def load_wine(test_size: float = 0.3, random_state: int = 42) -> DatasetResult:
    """Load the Wine dataset using proper configuration."""
    config = DatasetConfig(
        name="wine", type="SKLEARN", sklearn_config=SklearnDataConfig(dataset_name="wine", test_split=test_size, random_state=random_state)
    )
    return _load_sklearn_dataset(config)


def load_breast_cancer(test_size: float = 0.3, random_state: int = 42) -> DatasetResult:
    """Load the Breast Cancer dataset using proper configuration."""
    config = DatasetConfig(
        name="breast_cancer",
        type="SKLEARN",
        sklearn_config=SklearnDataConfig(dataset_name="breast_cancer", test_split=test_size, random_state=random_state),
    )
    return _load_sklearn_dataset(config)


def load_diabetes(test_size: float = 0.3, random_state: int = 42) -> DatasetResult:
    """Load the Diabetes dataset using proper configuration."""
    config = DatasetConfig(
        name="diabetes",
        type="SKLEARN",
        sklearn_config=SklearnDataConfig(dataset_name="diabetes", test_split=test_size, random_state=random_state),
    )
    return _load_sklearn_dataset(config)


def load_digits(test_size: float = 0.3, random_state: int = 42) -> DatasetResult:
    """Load the Digits dataset using proper configuration."""
    config = DatasetConfig(
        name="digits",
        type="SKLEARN",
        sklearn_config=SklearnDataConfig(dataset_name="digits", test_split=test_size, random_state=random_state),
    )
    return _load_sklearn_dataset(config)


# =============================================================================
# DISCOVERY AND LISTING FUNCTIONS
# =============================================================================


def list_available_scenarios() -> Dict[str, str]:
    """
    List all available predefined sklearn drift scenarios.

    Returns:
        Dictionary mapping scenario names to their descriptions
    """
    return {name: scenario["description"] for name, scenario in SKLEARN_DRIFT_SCENARIOS.items()}


def list_sklearn_datasets() -> List[str]:
    """List available sklearn datasets."""
    return list(SKLEARN_DATASETS.keys())


def describe_sklearn_datasets() -> Dict[str, str]:
    """Get descriptions of all available sklearn datasets."""
    return SKLEARN_DATASETS.copy()


def get_scenario_details(scenario_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific sklearn drift scenario.

    Args:
        scenario_name: Name of the scenario to describe

    Returns:
        Dictionary with scenario details including dataset, drift type, and characteristics

    Raises:
        ValueError: If scenario_name is not found
    """
    if scenario_name not in SKLEARN_DRIFT_SCENARIOS:
        available = list(SKLEARN_DRIFT_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    return SKLEARN_DRIFT_SCENARIOS[scenario_name].copy()


def suggest_scenarios_for_sklearn_dataset(dataset_name: str) -> List[str]:
    """
    Suggest appropriate drift scenarios for a given sklearn dataset.

    Args:
        dataset_name: Name of the sklearn dataset

    Returns:
        List of suggested scenario names

    Example:
        suggestions = suggest_scenarios_for_sklearn_dataset("iris")
        # Returns: ["iris_species_drift", "iris_feature_drift"]
    """
    if dataset_name not in SKLEARN_DATASETS:
        logger.warning(f"Unknown sklearn dataset: {dataset_name}")
        return []

    # Find all scenarios for the specified dataset
    suggested_scenarios = [
        scenario_name for scenario_name, scenario in SKLEARN_DRIFT_SCENARIOS.items() if scenario["dataset"] == dataset_name
    ]

    return suggested_scenarios


# =============================================================================
# HELPER FUNCTIONS FOR SKLEARN SCENARIO FILTERING
# =============================================================================


def _filter_by_classes(X: np.ndarray, y: np.ndarray, target_classes: List[int]) -> tuple[np.ndarray, np.ndarray]:
    """Filter dataset to include only specified target classes."""
    mask = np.isin(y, target_classes)
    return X[mask], y[mask]


def _filter_by_feature_condition(X: np.ndarray, y: np.ndarray, condition: str, feature_names: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """Filter dataset based on feature conditions (e.g., 'sepal_length <= 5.5')."""
    # Convert to DataFrame for easier condition evaluation
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Parse and evaluate condition
    # Replace feature names with column references
    eval_condition = condition
    for feature in feature_names:
        eval_condition = eval_condition.replace(feature, f"df['{feature}']")

    try:
        mask = eval(eval_condition)
        filtered_df = df[mask]
        return filtered_df[feature_names].values, filtered_df["target"].values
    except Exception as e:
        raise ValueError(f"Invalid condition '{condition}': {e}")


def _filter_by_target_condition(X: np.ndarray, y: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Filter dataset based on target conditions (e.g., 'target <= 100')."""
    # Replace 'target' with actual target values
    eval_condition = condition.replace("target", "y")

    try:
        mask = eval(eval_condition)
        return X[mask], y[mask]
    except Exception as e:
        raise ValueError(f"Invalid target condition '{condition}': {e}")
