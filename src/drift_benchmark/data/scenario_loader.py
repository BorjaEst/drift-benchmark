"""
Scenario data loading for drift-benchmark - REQ-DAT-XXX

Provides scenario loading utilities that apply filtering to create ScenarioResult objects.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from ..exceptions import DataLoadingError, DataValidationError
from ..literals import DataDimension, DataType, ScenarioSourceType
from ..models.metadata import ScenarioDefinition
from ..models.results import ScenarioResult
from ..settings import settings

# Dataset categorization for filtering validation - REQ-DAT-009
SYNTHETIC_DATASETS = {"make_classification", "make_regression", "make_blobs"}
REAL_DATASETS = {"load_breast_cancer", "load_diabetes", "load_iris", "load_wine"}

# Modification parameters allowed only for synthetic datasets - REQ-DAT-010, REQ-DAT-016
MODIFICATION_PARAMETERS = {
    "noise_factor",
    "feature_scaling",
    "n_samples",
    "random_state",
    "n_features",
    "n_redundant",
    "n_informative",
    "n_clusters",
    "cluster_std",
    "centers",
    "noise",
}


def load_scenario(scenario_id: str) -> ScenarioResult:
    """
    Load scenario from definition file and apply filters to create ScenarioResult.

    REQ-DAT-001: Scenario loading interface for loading scenario definitions, fetching source data,
    applying filters, and returning a ScenarioResult object
    """
    # Load scenario definition
    scenario_def = _load_scenario_definition(scenario_id)

    # REQ-DAT-016: Validate modification parameters are only used with synthetic datasets
    _validate_modification_parameters(scenario_def)

    # Load source data based on source_type
    source_data = _load_source_data(scenario_def)

    # Apply filters to create ref_data and test_data
    ref_data = _apply_filter(source_data, scenario_def.ref_filter, scenario_def.source_name, "ref_filter")
    test_data = _apply_filter(source_data, scenario_def.test_filter, scenario_def.source_name, "test_filter")

    # REQ-DAT-017: Check for empty subsets
    if len(ref_data) == 0:
        raise DataValidationError(
            f"Reference filter resulted in empty dataset. "
            f"Filter criteria: {scenario_def.ref_filter}. "
            f"Please adjust filter conditions to include some samples."
        )

    if len(test_data) == 0:
        raise DataValidationError(
            f"Test filter resulted in empty dataset. "
            f"Filter criteria: {scenario_def.test_filter}. "
            f"Please adjust filter conditions to include some samples."
        )

    # Split features and labels
    X_ref, y_ref = _split_features_labels(ref_data, scenario_def.target_column)
    X_test, y_test = _split_features_labels(test_data, scenario_def.target_column)

    # Create metadata objects
    dataset_metadata, scenario_metadata = _create_metadata(scenario_def, X_ref, X_test, y_ref, y_test)

    # REQ-DAT-006: Return X_ref, X_test as pandas.DataFrame objects per REQ-MDL-004
    return ScenarioResult(
        name=scenario_id,
        X_ref=X_ref,
        X_test=X_test,
        y_ref=y_ref,
        y_test=y_test,
        dataset_metadata=dataset_metadata,
        scenario_metadata=scenario_metadata,
        definition=scenario_def,
    )


def _load_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    """Load scenario definition from TOML file."""
    scenarios_dir = Path(settings.scenarios_dir if hasattr(settings, "scenarios_dir") else "scenarios")
    scenario_file = scenarios_dir / f"{scenario_id}.toml"

    # REQ-DAT-004: Validate file exists and is readable
    if not scenario_file.exists():
        raise DataLoadingError(f"Scenario definition file not found: {scenario_file}")

    if not scenario_file.is_file():
        raise DataLoadingError(f"Scenario path is not a file: {scenario_file}")

    try:
        import toml

        with open(scenario_file, "r") as f:
            scenario_data = toml.load(f)

        return ScenarioDefinition(**scenario_data)
    except Exception as e:
        raise DataLoadingError(f"Failed to load scenario definition {scenario_file}: {e}")


def _load_source_data(scenario_def: ScenarioDefinition) -> pd.DataFrame:
    """Load source data based on scenario definition."""
    if scenario_def.source_type == "file":
        return _load_file_data(scenario_def.source_name)
    elif scenario_def.source_type == "sklearn":
        return _load_sklearn_data(scenario_def.source_name)
    else:
        raise DataLoadingError(f"Unsupported source type: {scenario_def.source_type}")


def _load_file_data(file_name: str) -> pd.DataFrame:
    """Load data from file."""
    file_path = Path(file_name)

    # If it's an absolute path, use it directly
    if file_path.is_absolute():
        if not file_path.exists():
            raise DataLoadingError(f"Source data file not found: {file_path}")
    else:
        # If it's a relative path, look in datasets_dir
        datasets_dir = Path(settings.datasets_dir if hasattr(settings, "datasets_dir") else "datasets")
        file_path = datasets_dir / file_name

        # REQ-DAT-004: Validate file exists and is readable
        if not file_path.exists():
            raise DataLoadingError(f"Source data file not found: {file_path}")

    try:
        # REQ-DAT-002: Support csv format using pandas.read_csv() with default parameters
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise DataLoadingError(f"Unsupported file format: {file_path.suffix}")

        # REQ-DAT-007: Handle missing values using pandas defaults
        # pandas automatically converts empty strings to NaN, no additional processing needed

        return df
    except pd.errors.EmptyDataError:
        raise DataLoadingError(f"Source data file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise DataLoadingError(f"Failed to parse csv file {file_path}: {e}")
    except Exception as e:
        raise DataLoadingError(f"Unexpected error loading source data {file_path}: {e}")


def _load_sklearn_data(dataset_name: str) -> pd.DataFrame:
    """Load data from sklearn datasets."""
    try:
        from sklearn.datasets import (
            load_breast_cancer,
            load_diabetes,
            load_iris,
            load_wine,
            make_blobs,
            make_classification,
            make_regression,
        )

        if dataset_name == "make_classification":
            X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=42)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df["target"] = y
            return df
        elif dataset_name == "make_regression":
            X, y = make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df["target"] = y
            return df
        elif dataset_name == "make_blobs":
            X, y = make_blobs(n_samples=600, n_features=4, centers=3, cluster_std=2.0, random_state=42)
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            df["target"] = y
            return df
        elif dataset_name == "load_iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df
        elif dataset_name == "load_breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df
        elif dataset_name == "load_wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df
        elif dataset_name == "load_diabetes":
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            return df
        else:
            raise DataLoadingError(f"Unsupported sklearn dataset: {dataset_name}")
    except ImportError:
        raise DataLoadingError("scikit-learn is required for sklearn data sources")


def _apply_filter(data: pd.DataFrame, filter_config: Dict, source_name: str, filter_name: str) -> pd.DataFrame:
    """
    Apply filter configuration to extract subset of data.

    REQ-DAT-012: Feature-based filtering support
    REQ-DAT-013: AND logic implementation
    REQ-DAT-014: Sample range filtering with inclusive endpoints
    """
    filtered_data = data.copy()

    # Apply sample_range filter if specified - REQ-DAT-014
    if "sample_range" in filter_config:
        start, end = filter_config["sample_range"]

        # Validate sample range bounds
        if start < 0:
            raise DataValidationError(
                f"Invalid sample range in {filter_name}: start index {start} cannot be negative. "
                f"Dataset '{source_name}' has {len(data)} samples (indices 0-{len(data)-1})."
            )

        if start >= len(data):
            raise DataValidationError(
                f"Invalid sample range in {filter_name}: start index {start} is beyond dataset size. "
                f"Dataset '{source_name}' has {len(data)} samples (indices 0-{len(data)-1})."
            )

        if end > len(data):
            raise DataValidationError(
                f"Invalid sample range in {filter_name}: end index {end} is beyond dataset size. "
                f"Dataset '{source_name}' has {len(data)} samples (indices 0-{len(data)-1})."
            )

        if start > end:
            raise DataValidationError(f"Invalid sample range in {filter_name}: start index {start} is greater than end index {end}.")

        # REQ-DAT-014: Apply inclusive endpoints: data[start:end+1]
        # This means [start, end] includes both start and end indices
        filtered_data = filtered_data.iloc[start : end + 1].copy()

    # REQ-DAT-012: Apply feature-based filtering
    if "feature_filters" in filter_config:
        feature_filters = filter_config["feature_filters"]

        for feature_filter in feature_filters:
            # Validate required fields
            if not all(key in feature_filter for key in ["column", "condition", "value"]):
                raise DataValidationError(
                    f"Invalid feature filter in {filter_name}: {feature_filter}. " f"Required fields: column, condition, value"
                )

            column = feature_filter["column"]
            condition = feature_filter["condition"]
            value = feature_filter["value"]

            # Validate column exists
            if column not in filtered_data.columns:
                available_columns = list(filtered_data.columns)
                raise DataValidationError(
                    f"Column '{column}' not found in dataset '{source_name}' during {filter_name} filtering. "
                    f"Available columns: {available_columns}"
                )

            # Apply condition - REQ-DAT-013: AND logic (all conditions must be true)
            if condition == "<=":
                mask = filtered_data[column] <= value
            elif condition == ">=":
                mask = filtered_data[column] >= value
            elif condition == ">":
                mask = filtered_data[column] > value
            elif condition == "<":
                mask = filtered_data[column] < value
            elif condition == "==":
                mask = filtered_data[column] == value
            elif condition == "!=":
                mask = filtered_data[column] != value
            else:
                raise DataValidationError(
                    f"Unsupported condition '{condition}' in {filter_name}. " f"Supported conditions: <=, >=, >, <, ==, !="
                )

            # Apply mask (AND logic with previous filters)
            filtered_data = filtered_data[mask].copy()

    return filtered_data


def _validate_modification_parameters(scenario_def: ScenarioDefinition):
    """
    Validate that modification parameters are only used with synthetic datasets.

    REQ-DAT-016: Validation of modifications for real datasets
    """
    source_name = scenario_def.source_name

    # Check if this is a real dataset
    if source_name in REAL_DATASETS:
        forbidden_params = []

        # Check both ref_filter and test_filter for modification parameters
        for filter_name, filter_config in [("ref_filter", scenario_def.ref_filter), ("test_filter", scenario_def.test_filter)]:
            for param in filter_config:
                if param in MODIFICATION_PARAMETERS:
                    forbidden_params.append(f"{filter_name}.{param}")

        if forbidden_params:
            raise DataValidationError(
                f"Real dataset '{source_name}' cannot use modification parameters: {forbidden_params}. "
                f"Real datasets only support filtering operations (feature_filters) "
                f"to preserve data authenticity. Remove these forbidden parameters: {forbidden_params}"
            )


def _create_scenario_metadata(scenario_def: ScenarioDefinition, ref_data: pd.DataFrame, test_data: pd.DataFrame) -> ScenarioDefinition:
    """Create enhanced scenario metadata with inferred properties."""
    # For now, return the scenario definition directly
    # In a more complete implementation, we could enhance it with inferred properties
    # REQ-DAT-005: Automatic data type inference would be applied here

    # Add inferred data type to scenario definition if needed
    if not hasattr(scenario_def, "data_type"):
        combined_data = pd.concat([ref_data, test_data])
        data_type = _infer_data_type(combined_data)
        # Note: ScenarioDefinition doesn't have data_type field,
        # so for now we just return the definition as-is

    return scenario_def


def _split_features_labels(data: pd.DataFrame, target_column: Union[str, None]) -> tuple[pd.DataFrame, Union[pd.Series, None]]:
    """Split data into features (X) and labels (y)."""
    if target_column is None or target_column not in data.columns:
        return data, None

    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def _create_metadata(
    scenario_def: ScenarioDefinition,
    X_ref: pd.DataFrame,
    X_test: pd.DataFrame,
    y_ref: Union[pd.Series, None],
    y_test: Union[pd.Series, None],
) -> tuple:
    """Create DatasetMetadata and ScenarioMetadata objects."""
    from ..models.metadata import DatasetMetadata, ScenarioMetadata

    # Infer data type from features - use only reference data if test data is empty
    if len(X_test) > 0:
        combined_data = pd.concat([X_ref, X_test])
    else:
        combined_data = X_ref
    data_type = _infer_data_type(combined_data)

    # Determine dimension
    n_features = X_ref.shape[1]
    dimension = "univariate" if n_features == 1 else "multivariate"

    # REQ-DAT-009: Dataset categorization
    if scenario_def.source_name in SYNTHETIC_DATASETS:
        dataset_category = "synthetic"
    elif scenario_def.source_name in REAL_DATASETS:
        dataset_category = "real"
    else:
        dataset_category = "unknown"

    # Handle edge case where filtering results in empty test set
    n_samples_test = len(X_test)
    if n_samples_test == 0:
        # For DatasetMetadata, we need at least 1 sample to satisfy validation
        # Use 1 as a placeholder since this represents the original dataset capability
        n_samples_test_for_metadata = 1
    else:
        n_samples_test_for_metadata = n_samples_test

    # Check for empty reference set - REQ-DAT-017
    if len(X_ref) == 0:
        raise DataValidationError(
            f"Filtering resulted in empty reference dataset. "
            f"Filter criteria: {scenario_def.ref_filter}. "
            f"Please adjust filter conditions to include some samples."
        )

    # Create DatasetMetadata
    dataset_metadata = DatasetMetadata(
        name=scenario_def.source_name,
        data_type=data_type,
        dimension=dimension,
        n_samples_ref=len(X_ref),
        n_samples_test=n_samples_test_for_metadata,
        n_features=n_features,
    )

    # Create ScenarioMetadata (allows empty test sets as it represents actual filtering results)
    scenario_metadata = ScenarioMetadata(
        total_samples=len(X_ref) + len(X_test),
        ref_samples=len(X_ref),
        test_samples=len(X_test),
        n_features=n_features,
        has_labels=y_ref is not None,
        data_type=data_type,
        dimension=dimension,
        dataset_category=dataset_category,  # REQ-DAT-009: Include dataset categorization
    )

    return dataset_metadata, scenario_metadata


# REQ-DAT-008: Data type inference algorithm (preserved for compatibility)
def _infer_data_type(df: pd.DataFrame) -> DataType:
    """
    Infer data type based on pandas dtypes.

    REQ-DAT-008: Data type inference algorithm
    continuous: numeric dtypes (int, float)
    categorical: object/string dtypes
    mixed: datasets with both numeric and object columns
    """
    import numpy as np

    # Include all possible numeric types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Include string and object types for categorical
    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    # Also check for boolean columns (should be categorical)
    bool_cols = df.select_dtypes(include=["bool"]).columns

    has_numeric = len(numeric_cols) > 0
    has_categorical = len(object_cols) > 0 or len(bool_cols) > 0

    if has_numeric and has_categorical:
        return "mixed"
    elif has_categorical:
        return "categorical"
    elif has_numeric:
        return "continuous"
    else:
        # Edge case: no recognized columns, default to continuous
        return "continuous"
