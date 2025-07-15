"""
Universal dataset loading module for drift-benchmark.

This module provides core dataset loading utilities with support for CSV files
and basic data processing. It focuses on the fundamental functionality needed
to load datasets from various sources and apply filtering for drift detection.

KEY CAPABILITIES:
1. Load datasets from CSV files with flexible configuration
2. Support custom filtering for reference/test splits
3. Automatic data type inference and validation
4. Robust error handling and logging
5. Integration with Pydantic models for type safety

The module is designed to be the core dataset loading engine that other modules
can build upon for more specialized functionality.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import ValidationError

from drift_benchmark.constants.literals import DatasetType, DataType
from drift_benchmark.constants.models import DatasetConfig, DatasetMetadata, DatasetResult, DriftInfo, FileDataConfig
from drift_benchmark.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS FOR DATA PROCESSING
# =============================================================================

# Data type inference thresholds for categorical detection
CATEGORICAL_UNIQUE_THRESHOLD = 10  # Max unique values to consider categorical
CATEGORICAL_RATIO_THRESHOLD = 0.5  # Max ratio of unique/total values for categorical


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================


def load_dataset(config: Union[str, Dict, DatasetConfig]) -> DatasetResult:
    """
    Load a dataset with support for basic data filtering.

    This is the main entry point for loading datasets. Currently supports
    CSV files with flexible filtering options.

    Args:
        config: Dataset configuration. Can be:
            - String: CSV filename (looked up in datasets directory)
            - Dictionary: configuration parameters
            - DatasetConfig: Pydantic model with full configuration

    Returns:
        DatasetResult: Complete dataset with X_ref, X_test, y_ref, y_test,
                      drift_info, and metadata

    Raises:
        ValueError: If the dataset cannot be loaded or configuration is invalid
        FileNotFoundError: If a specified file doesn't exist

    Examples:
        # Load CSV dataset by name
        result = load_dataset("my_data.csv")

        # Load with filtering for custom data splits
        result = load_dataset({
            "name": "custom_split",
            "type": "FILE",
            "file_config": {
                "file_path": "data.csv",
                "target_column": "target",
                "ref_filter": {"category": ["A", "B"]},
                "test_filter": {"category": ["C"]}
            }
        })
    """
    try:
        # Validate and parse configuration
        dataset_config = _validate_and_parse_config(config)

        logger.info(f"Loading dataset: {dataset_config.name} (type: {dataset_config.type})")

        # Route to appropriate loader based on dataset type
        if dataset_config.type == "FILE":
            return _load_file_dataset(dataset_config)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config.type}. Use 'scenarios' module for sklearn datasets.")

    except ValidationError as e:
        # Handle Pydantic validation errors
        error_msg = str(e)
        if "type" in error_msg and "Input should be" in error_msg:
            raise ValueError("Unsupported dataset type")
        else:
            raise ValueError(f"Invalid dataset configuration: {e}")
    except Exception as e:
        # Re-raise ValueError as-is, wrap other exceptions
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Failed to load dataset: {e}")


def load_dataset_with_filters(
    file_path: str,
    ref_filter: Optional[Dict[str, Any]] = None,
    test_filter: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None,
    filter_mode: str = "include",
    name: Optional[str] = None,
) -> DatasetResult:
    """
    Load a dataset with filtering for custom data splits.

    This function makes it easy to create reference and test sets based on
    feature criteria rather than random splits, which is useful for creating
    controlled data splits based on specific conditions.

    Args:
        file_path: Path to the CSV file
        ref_filter: Filter criteria for reference/training data
        test_filter: Filter criteria for test/validation data
        target_column: Name of the target column
        filter_mode: "include" to keep matching rows, "exclude" to remove them
        name: Name for the dataset

    Returns:
        DatasetResult with filtered reference and test sets

    Example:
        # Load data, train on certain categories, test on others
        result = load_dataset_with_filters(
            "data.csv",
            ref_filter={"category": ["A", "B"]},
            test_filter={"category": ["C", "D"]},
            target_column="target"
        )

        # Load data filtering by value ranges
        result = load_dataset_with_filters(
            "data.csv",
            ref_filter={"score": (0, 50), "type": ["premium"]},
            test_filter={"score": (51, 100), "type": ["basic"]},
            target_column="outcome"
        )
    """
    if name is None:
        name = f"filtered_{Path(file_path).stem}"

    config = DatasetConfig(
        name=name,
        type="FILE",
        file_config=FileDataConfig(
            file_path=file_path, target_column=target_column, ref_filter=ref_filter, test_filter=test_filter, filter_mode=filter_mode
        ),
    )

    return load_dataset(config)


# =============================================================================
# INTERNAL DATASET LOADING FUNCTIONS
# =============================================================================


def _load_file_dataset(config: DatasetConfig) -> DatasetResult:
    """Load a dataset from CSV file using proper configuration."""
    if not config.file_config:
        raise ValueError("File dataset configuration missing")

    file_config = config.file_config
    file_path = Path(file_config.file_path)

    # Resolve relative paths
    if not file_path.is_absolute():
        file_path = Path(settings.datasets_dir) / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.debug(f"Loading CSV dataset from: {file_path}")

    try:
        # Load CSV data
        data = pd.read_csv(file_path, sep=file_config.separator, header=file_config.header, encoding=file_config.encoding)
        logger.info(f"Loaded CSV with shape: {data.shape}")

        # Extract features and target
        target_column = file_config.target_column
        if target_column is None:
            # Use last column as target if not specified
            target_column = data.columns[-1]

        if target_column in data.columns:
            X = data.drop(columns=[target_column]).values
            y = data[target_column].values
            feature_names = data.drop(columns=[target_column]).columns.tolist()
        else:
            # No target column found, treat as unsupervised
            X = data.values
            y = None
            feature_names = data.columns.tolist()
            target_column = None

        # Create reference and test sets
        if file_config.ref_filter or file_config.test_filter:
            # Use filtering for controlled data splits
            X_ref, X_test, y_ref, y_test = _create_filtered_datasets(data, file_config, target_column)
        else:
            # Use traditional random split
            test_split = file_config.test_split or 0.3
            X_ref, X_test, y_ref, y_test = _create_random_split_datasets(X, y, test_split)

        # Create dataset metadata
        has_filtering_drift = bool(file_config.ref_filter or file_config.test_filter)
        metadata = DatasetMetadata(
            name=config.name,
            n_samples=len(X),
            n_features=X.shape[1],
            feature_names=feature_names,
            target_name=target_column,
            data_types=_infer_data_types(data.drop(columns=[target_column]) if target_column in data.columns else data),
            has_drift=bool(file_config.drift_points or file_config.drift_labels or has_filtering_drift),
            drift_points=file_config.drift_points,
            drift_metadata={
                "source": "file",
                "drift_column": file_config.drift_column,
                "artificial_drift": has_filtering_drift,
                "ref_filter": file_config.ref_filter,
                "test_filter": file_config.test_filter,
                "filter_mode": file_config.filter_mode,
            },
            source=str(file_path),
            creation_time=datetime.now().isoformat(),
            preprocessing_applied=[],
        )

        # Create drift info
        drift_characteristics = []
        if has_filtering_drift:
            drift_characteristics.append("COVARIATE")  # Filtering creates covariate shift
            if file_config.ref_filter and file_config.test_filter:
                drift_characteristics.append("CONCEPT")  # Different populations may have concept drift too

        drift_info = DriftInfo(
            has_drift=metadata.has_drift,
            drift_points=metadata.drift_points,
            drift_pattern="SUDDEN" if has_filtering_drift else None,  # Filter-based drift is sudden
            drift_magnitude=1.0 if has_filtering_drift else None,  # Assume full magnitude for filter-based drift
            drift_characteristics=drift_characteristics,
            metadata=metadata.drift_metadata,
        )

        logger.info(f"CSV dataset loaded: {len(X_ref)} ref + {len(X_test)} test samples, {X.shape[1]} features")

        return DatasetResult(X_ref=X_ref, X_test=X_test, y_ref=y_ref, y_test=y_test, drift_info=drift_info, metadata=metadata.model_dump())

    except Exception as e:
        raise ValueError(f"Failed to load CSV dataset from {file_path}: {e}")


def _validate_and_parse_config(config: Union[str, Dict, DatasetConfig]) -> DatasetConfig:
    """
    Validate and parse dataset configuration input.

    Args:
        config: Raw configuration input

    Returns:
        Validated DatasetConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        if isinstance(config, str):
            # String input - check for CSV file
            csv_path = Path(settings.datasets_dir) / config
            if not csv_path.suffix:
                csv_path = csv_path.with_suffix(".csv")

            if csv_path.exists():
                return DatasetConfig(name=csv_path.stem, type="FILE", file_config=FileDataConfig(file_path=str(csv_path)))
            else:
                raise ValueError(f"CSV file '{csv_path}' not found in datasets directory")

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
        if "type" in error_msg and "Input should be" in error_msg:
            raise ValueError("Unsupported dataset type")
        elif "test_split" in error_msg and "less than" in error_msg:
            raise ValueError("Invalid test split value")
        else:
            raise ValueError(f"Invalid dataset configuration: {e}")


# =============================================================================
# DATA PROCESSING AND FILTERING FUNCTIONS
# =============================================================================


def _apply_data_filter(data: pd.DataFrame, filter_criteria: Dict[str, Any], filter_mode: str = "include") -> pd.DataFrame:
    """
    Apply filtering criteria to a DataFrame for custom data splits.

    This function supports multiple filter types:
    - Categorical: {"category": ["A", "B"]}
    - Numerical range: {"score": (25, 40)}
    - Single value: {"type": "premium"}
    - Mixed: {"category": ["A"], "score": (30, 50)}

    Args:
        data: Input DataFrame to filter
        filter_criteria: Dictionary mapping column names to filter values
        filter_mode: "include" to keep matching rows, "exclude" to remove them

    Returns:
        Filtered DataFrame

    Examples:
        # Keep only specific categories
        filtered = _apply_data_filter(data, {"category": ["A", "B"]})

        # Keep scores between 25 and 40
        filtered = _apply_data_filter(data, {"score": (25, 40)})

        # Complex filter
        filtered = _apply_data_filter(data, {
            "category": ["premium"],
            "region": ["north", "south"],
            "score": (5, 15)
        })
    """
    if not filter_criteria:
        return data

    mask = pd.Series(True, index=data.index)

    for column, criteria in filter_criteria.items():
        if column not in data.columns:
            logger.warning(f"Filter column '{column}' not found in data. Skipping.")
            continue

        if isinstance(criteria, (tuple, list)) and len(criteria) == 2 and all(isinstance(x, (int, float)) for x in criteria):
            # Numerical range filter (min, max)
            min_val, max_val = criteria
            column_mask = (data[column] >= min_val) & (data[column] <= max_val)
            logger.debug(f"Applied range filter {column}: {min_val}-{max_val}")
        elif isinstance(criteria, (list, tuple)):
            # Categorical value list filter
            column_mask = data[column].isin(criteria)
            logger.debug(f"Applied categorical filter {column}: {criteria}")
        else:
            # Single value filter
            column_mask = data[column] == criteria
            logger.debug(f"Applied single value filter {column}: {criteria}")

        mask = mask & column_mask

    if filter_mode == "exclude":
        mask = ~mask

    filtered_data = data[mask].copy()
    logger.info(f"Filtered data from {len(data)} to {len(filtered_data)} rows ({len(filtered_data)/len(data)*100:.1f}%)")

    return filtered_data


def _create_filtered_datasets(data: pd.DataFrame, file_config: FileDataConfig, target_column: Optional[str]) -> tuple:
    """
    Create reference and test datasets using filtering for custom data splits.

    Args:
        data: Full dataset DataFrame
        file_config: File configuration with filtering parameters
        target_column: Name of target column (if any)

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test)
    """
    logger.info("Creating filtered datasets for custom data splits")

    # Apply filters to create reference and test datasets
    if file_config.ref_filter:
        ref_data = _apply_data_filter(data, file_config.ref_filter, file_config.filter_mode)
        logger.info(f"Reference data: {len(ref_data)} samples with filter {file_config.ref_filter}")
    else:
        ref_data = data.copy()
        logger.info(f"Reference data: {len(ref_data)} samples (no filter)")

    if file_config.test_filter:
        test_data = _apply_data_filter(data, file_config.test_filter, file_config.filter_mode)
        logger.info(f"Test data: {len(test_data)} samples with filter {file_config.test_filter}")
    else:
        test_data = data.copy()
        logger.info(f"Test data: {len(test_data)} samples (no filter)")

    # Extract features and targets
    if target_column and target_column in data.columns:
        X_ref = ref_data.drop(columns=[target_column]).values
        y_ref = ref_data[target_column].values if len(ref_data) > 0 else np.array([])
        X_test = test_data.drop(columns=[target_column]).values
        y_test = test_data[target_column].values if len(test_data) > 0 else np.array([])
    else:
        X_ref = ref_data.values
        y_ref = None
        X_test = test_data.values
        y_test = None

    return X_ref, X_test, y_ref, y_test


def _create_random_split_datasets(X: np.ndarray, y: Optional[np.ndarray], test_split: float) -> tuple:
    """
    Create reference and test datasets using traditional random split.

    Args:
        X: Feature matrix
        y: Target vector (optional)
        test_split: Fraction of data to use for test set

    Returns:
        Tuple of (X_ref, X_test, y_ref, y_test)
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Creating random split datasets (test_split={test_split})")

    if y is not None:
        # Use stratification if target has reasonable number of classes
        stratify = y if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) // 10 else None
        X_ref, X_test, y_ref, y_test = train_test_split(X, y, test_size=test_split, random_state=42, stratify=stratify)
    else:
        X_ref, X_test = train_test_split(X, test_size=test_split, random_state=42)
        y_ref, y_test = None, None

    return X_ref, X_test, y_ref, y_test


def _infer_data_types(df: pd.DataFrame) -> Dict[str, DataType]:
    """Infer data types for DataFrame columns using proper literals."""
    data_types = {}

    for col in df.columns:
        if df[col].dtype in ["object", "category"]:
            data_types[col] = "CATEGORICAL"
        elif df[col].dtype in ["int64", "int32", "int16", "int8"]:
            # Check if it looks categorical (few unique values)
            unique_vals = df[col].nunique()
            if unique_vals <= CATEGORICAL_UNIQUE_THRESHOLD and unique_vals <= len(df) * CATEGORICAL_RATIO_THRESHOLD:
                data_types[col] = "CATEGORICAL"
            else:
                data_types[col] = "CONTINUOUS"
        elif df[col].dtype in ["float64", "float32"]:
            data_types[col] = "CONTINUOUS"
        else:
            data_types[col] = "CATEGORICAL"

    return data_types


# =============================================================================
# UTILITY AND DISCOVERY FUNCTIONS
# =============================================================================


def list_csv_datasets() -> List[str]:
    """List available CSV datasets in the datasets directory."""
    datasets_path = Path(settings.datasets_dir)
    if not datasets_path.exists():
        return []

    return [csv_file.stem for csv_file in datasets_path.glob("*.csv")]


def validate_dataset_for_drift_detection(
    file_path: str, required_features: Optional[List[str]] = None, min_samples_per_group: int = 50
) -> Dict[str, Any]:
    """
    Validate a dataset for filtering and data split suitability.

    Analyzes a dataset to determine if it's suitable for custom filtering
    and provides information about feature characteristics.

    Args:
        file_path: Path to the CSV file to analyze
        required_features: List of features that should be present
        min_samples_per_group: Minimum samples required per filter group

    Returns:
        Dictionary with validation results and feature analysis

    Example:
        validation = validate_dataset_for_drift_detection(
            "data.csv",
            required_features=["category", "type"],
            min_samples_per_group=100
        )

        if validation["suitable"]:
            print("Dataset is suitable for filtering")
            print("Categorical features:", validation["categorical_features"])
    """
    # Resolve path
    if not Path(file_path).is_absolute():
        file_path = str(Path(settings.datasets_dir) / file_path)

    # Load basic dataset info
    data = pd.read_csv(file_path)

    results = {
        "suitable": True,
        "n_samples": len(data),
        "n_features": len(data.columns),
        "feature_analysis": {},
        "warnings": [],
        "categorical_features": [],
    }

    # Analyze each feature for drift detection potential
    for col in data.columns:
        if data[col].dtype == "object" or data[col].nunique() <= CATEGORICAL_UNIQUE_THRESHOLD:
            # Categorical feature analysis
            unique_vals = data[col].unique()
            value_counts = data[col].value_counts()

            results["feature_analysis"][col] = {
                "type": "categorical",
                "unique_values": len(unique_vals),
                "values": list(unique_vals)[:10],  # Show first 10 values
                "min_count": value_counts.min(),
                "max_count": value_counts.max(),
                "suitable_for_filtering": value_counts.min() >= min_samples_per_group,
            }

            if value_counts.min() >= min_samples_per_group and len(unique_vals) >= 2:
                results["categorical_features"].append(col)

        else:
            # Numerical feature analysis
            results["feature_analysis"][col] = {
                "type": "numerical",
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
            }

    # Validation checks
    if required_features:
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            results["warnings"].append(f"Missing required features: {missing_features}")
            results["suitable"] = False

    if len(results["categorical_features"]) == 0:
        results["warnings"].append("No suitable categorical features found for filtering")
        results["suitable"] = False

    if results["n_samples"] < min_samples_per_group * 4:  # Need at least 4 groups worth
        results["warnings"].append(f"Dataset too small (need ≥{min_samples_per_group * 4} samples)")
        results["suitable"] = False

    logger.info(f"Dataset validation complete: {results['suitable']}")
    if results["warnings"]:
        for warning in results["warnings"]:
            logger.warning(warning)

    return results
