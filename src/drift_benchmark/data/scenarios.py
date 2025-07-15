"""
Predefined drift scenarios module using scikit-learn datasets.

This module provides convenient access to scikit-learn datasets and predefined
drift detection scenarios. It builds upon the universal dataset loading
functionality to create specialized scenarios for testing drift detection
algorithms with well-known benchmark datasets.

KEY CAPABILITIES:
1. Direct access to all major scikit-learn datasets
2. Predefined drift scenarios based on common real-world patterns
3. Scenario-based testing for drift detection algorithms
4. Integration with the universal dataset loading system

The module is designed to complement the datasets.py module by providing
specialized functionality for research and benchmarking scenarios.
"""

import logging
from datetime import datetime
from pathlib import Path
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
from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, FileDataConfig, SklearnDataConfig
from drift_benchmark.settings import settings

# Import the universal dataset loading functionality
from .datasets import load_dataset_with_filters

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

# Typical drift scenarios for testing
COMMON_DRIFT_SCENARIOS = {
    "education_drift": {
        "description": "Educational level drift - test performance across education levels",
        "ref_filter": {"education": ["Bachelor", "Master"]},
        "test_filter": {"education": ["PhD", "Associate"]},
        "characteristics": ["COVARIATE", "CONCEPT"],
    },
    "geographic_drift": {
        "description": "Geographic drift - test regional differences",
        "ref_filter": {"region": ["North", "East"]},
        "test_filter": {"region": ["South", "West"]},
        "characteristics": ["COVARIATE"],
    },
    "generational_drift": {
        "description": "Age-based drift - test generational differences",
        "ref_filter": {"age": (25, 45)},
        "test_filter": {"age": (46, 65)},
        "characteristics": ["COVARIATE", "CONCEPT"],
    },
}


# =============================================================================
# MAIN SCENARIO API FUNCTIONS
# =============================================================================


def create_drift_scenario(
    file_path: str,
    scenario_name: str,
    target_column: Optional[str] = None,
    custom_ref_filter: Optional[Dict[str, Any]] = None,
    custom_test_filter: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> DatasetResult:
    """
    Create a predefined drift detection scenario from a dataset.

    This function simplifies drift detection experiments by providing common
    drift scenarios that can be applied to any dataset with appropriate features.
    It's designed for datasets with demographic or categorical features that
    naturally create different populations.

    Args:
        file_path: Path to the CSV file
        scenario_name: Name of predefined scenario or "custom"
        target_column: Name of the target column
        custom_ref_filter: Custom reference filter (for scenario_name="custom")
        custom_test_filter: Custom test filter (for scenario_name="custom")
        name: Custom name for the dataset

    Returns:
        DatasetResult with filtered datasets and drift metadata

    Available Scenarios:
        - "education_drift": Bachelor/Master vs PhD/Associate
        - "geographic_drift": North/East vs South/West
        - "generational_drift": Ages 25-45 vs 46-65
        - "custom": Use custom_ref_filter and custom_test_filter

    Examples:
        # Use predefined education drift scenario
        result = create_drift_scenario(
            "employee_data.csv",
            scenario_name="education_drift",
            target_column="salary"
        )

        # Create custom drift scenario
        result = create_drift_scenario(
            "customer_data.csv",
            scenario_name="custom",
            custom_ref_filter={"segment": ["Premium", "Gold"]},
            custom_test_filter={"segment": ["Basic", "Silver"]},
            target_column="purchase_amount"
        )
    """
    if scenario_name == "custom":
        if not custom_ref_filter or not custom_test_filter:
            raise ValueError("custom_ref_filter and custom_test_filter required for scenario_name='custom'")
        ref_filter = custom_ref_filter
        test_filter = custom_test_filter
        drift_characteristics = ["COVARIATE"]  # Default for custom scenarios
    elif scenario_name in COMMON_DRIFT_SCENARIOS:
        scenario = COMMON_DRIFT_SCENARIOS[scenario_name]
        ref_filter = scenario["ref_filter"]
        test_filter = scenario["test_filter"]
        drift_characteristics = scenario["characteristics"]
    else:
        available = list(COMMON_DRIFT_SCENARIOS.keys()) + ["custom"]
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    if name is None:
        name = f"{scenario_name}_{Path(file_path).stem}"

    logger.info(f"Creating {scenario_name} drift scenario from {file_path}")
    logger.info(f"Reference filter: {ref_filter}")
    logger.info(f"Test filter: {test_filter}")

    # Use the universal dataset loading with filters
    result = load_dataset_with_filters(
        file_path=file_path,
        ref_filter=ref_filter,
        test_filter=test_filter,
        target_column=target_column,
        filter_mode="include",
        name=name,
    )

    # Update drift info with scenario-specific characteristics
    if hasattr(result.drift_info, "drift_characteristics"):
        result.drift_info.drift_characteristics = drift_characteristics

    # Add scenario metadata
    if result.metadata:
        result.metadata["drift_scenario"] = scenario_name
        result.metadata["ref_filter"] = ref_filter
        result.metadata["test_filter"] = test_filter

    return result


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
    List all available predefined drift scenarios.

    Returns:
        Dictionary mapping scenario names to their descriptions
    """
    return {name: scenario["description"] for name, scenario in COMMON_DRIFT_SCENARIOS.items()}


def list_sklearn_datasets() -> List[str]:
    """List available sklearn datasets."""
    return list(SKLEARN_DATASETS.keys())


def describe_sklearn_datasets() -> Dict[str, str]:
    """Get descriptions of all available sklearn datasets."""
    return SKLEARN_DATASETS.copy()


def get_scenario_details(scenario_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific drift scenario.

    Args:
        scenario_name: Name of the scenario to describe

    Returns:
        Dictionary with scenario details including filters and characteristics

    Raises:
        ValueError: If scenario_name is not found
    """
    if scenario_name not in COMMON_DRIFT_SCENARIOS:
        available = list(COMMON_DRIFT_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")

    return COMMON_DRIFT_SCENARIOS[scenario_name].copy()


def suggest_scenarios_for_dataset(file_path: str) -> List[str]:
    """
    Suggest appropriate drift scenarios for a given CSV dataset.

    Analyzes the dataset's columns to suggest which predefined scenarios
    might be applicable based on common feature patterns.

    Args:
        file_path: Path to the CSV file to analyze

    Returns:
        List of suggested scenario names

    Example:
        suggestions = suggest_scenarios_for_dataset("employee_data.csv")
        # Might return: ["education_drift", "generational_drift"]
    """
    # Import here to avoid circular imports
    from .datasets import validate_dataset_for_drift_detection

    try:
        # Get basic validation from datasets module
        validation = validate_dataset_for_drift_detection(file_path)
        
        # Load data for scenario-specific analysis
        # Resolve path
        if not Path(file_path).is_absolute():
            file_path = str(Path(settings.datasets_dir) / file_path)
        
        data = pd.read_csv(file_path)
        
        suggested_scenarios = []
        
        # Analyze features for scenario suggestions
        for col in data.columns:
            if data[col].dtype == "object" or data[col].nunique() <= 10:
                # Check for common drift scenario features
                if col.lower() in ["education", "education_level"]:
                    suggested_scenarios.append("education_drift")
                elif col.lower() in ["region", "geography", "location"]:
                    suggested_scenarios.append("geographic_drift")
            else:
                # Check if age-like feature
                if col.lower() in ["age", "years", "experience"] and data[col].max() <= 150:
                    suggested_scenarios.append("generational_drift")
        
        # Remove duplicates and return
        return list(set(suggested_scenarios))
        
    except Exception as e:
        logger.warning(f"Could not analyze dataset {file_path}: {e}")
        return []
