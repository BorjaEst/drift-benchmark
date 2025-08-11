"""
Scenario data loading for drift-benchmark - REQ-DAT-XXX

Provides scenario loading utilities that apply filtering to create ScenarioResult objects.
Enhanced with comprehensive real-world data integration following TDD principles.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import toml

from ..exceptions import DataLoadingError, DataValidationError
from ..literals import DataDimension, DataType
from ..models.metadata import DatasetMetadata, ScenarioDefinition, ScenarioMetadata
from ..models.results import ScenarioResult
from ..settings import settings


def load_scenario(scenario_id: str) -> ScenarioResult:
    """
    Load scenario data by scenario ID and return ScenarioResult.

    REQ-DAT-001: Scenario Loading Interface

    Args:
        scenario_id: Identifier for the scenario definition file

    Returns:
        ScenarioResult: Scenario with reference and test data

    Raises:
        DataLoadingError: If scenario file or data source cannot be loaded
        DataValidationError: If scenario data fails validation
    """
    # Load scenario definition from TOML file
    scenario_definition = _load_scenario_definition(scenario_id)

    # Load source data based on source_type
    source_data = _load_source_data(scenario_definition)

    # Apply filters to create reference and test datasets
    X_ref, y_ref = _apply_filter(source_data, scenario_definition.ref_filter, scenario_definition.target_column)
    X_test, y_test = _apply_filter(source_data, scenario_definition.test_filter, scenario_definition.target_column)

    # Create metadata objects
    dataset_metadata = _create_dataset_metadata(source_data, scenario_definition, X_ref, X_test)
    scenario_metadata = _create_scenario_metadata(scenario_definition, X_ref, X_test, y_ref, y_test)

    # Create and return ScenarioResult
    return ScenarioResult(
        name=scenario_id,
        X_ref=X_ref,
        X_test=X_test,
        y_ref=y_ref,
        y_test=y_test,
        dataset_metadata=dataset_metadata,
        scenario_metadata=scenario_metadata,
        definition=scenario_definition,
    )


def _load_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    """
    Load scenario definition from TOML file.

    Args:
        scenario_id: ID of the scenario definition file

    Returns:
        ScenarioDefinition: Parsed scenario definition

    Raises:
        DataLoadingError: If scenario file cannot be found or parsed
    """
    # REQ-DAT-003: Path Validation
    scenario_file = settings.scenarios_dir / f"{scenario_id}.toml"

    if not scenario_file.exists():
        raise DataLoadingError(message=f"Scenario definition file not found: {scenario_file}")

    try:
        with open(scenario_file, "r") as f:
            scenario_data = toml.load(f)

        # Convert to ScenarioDefinition model
        return ScenarioDefinition(**scenario_data)

    except Exception as e:
        raise DataLoadingError(message=f"Failed to load scenario definition from {scenario_file}: {str(e)}")


def _load_source_data(scenario_definition: ScenarioDefinition) -> pd.DataFrame:
    """
    Load source data based on scenario definition.

    Args:
        scenario_definition: Scenario definition with source information

    Returns:
        pd.DataFrame: Complete source dataset

    Raises:
        DataLoadingError: If data source cannot be loaded
    """
    source_type = scenario_definition.source_type
    source_name = scenario_definition.source_name

    if source_type == "file":
        return _load_csv_file(source_name)
    elif source_type == "synthetic":
        return _load_sklearn_data(source_name, scenario_definition.test_filter)
    elif source_type == "uci":
        return _load_uci_data(source_name)
    else:
        raise DataLoadingError(message=f"Unsupported source type: {source_type}")


def _load_csv_file(file_path: str) -> pd.DataFrame:
    """
    Load CSV file using pandas.

    REQ-DAT-002: CSV Format Support
    REQ-DAT-003: Path Validation

    Args:
        file_path: Path to CSV file

    Returns:
        pd.DataFrame: Loaded CSV data

    Raises:
        DataLoadingError: If CSV file cannot be loaded
    """
    # Convert to Path object for validation
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        raise DataLoadingError(message=f"CSV file not found: {file_path}")

    try:
        # REQ-DAT-002: Use pandas.read_csv() with default parameters
        return pd.read_csv(path)
    except Exception as e:
        raise DataLoadingError(message=f"Failed to load CSV file {file_path}: {str(e)}")


def _load_sklearn_data(dataset_name: str, test_filter: Dict) -> pd.DataFrame:
    """
    Load sklearn synthetic dataset.

    Args:
        dataset_name: Name of sklearn dataset function
        test_filter: Test filter parameters for generation

    Returns:
        pd.DataFrame: Generated sklearn dataset

    Raises:
        DataLoadingError: If sklearn dataset cannot be generated
    """
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

        # Get generation parameters from test_filter
        n_samples = test_filter.get("n_samples", 1000)
        random_state = test_filter.get("random_state", 42)

        if dataset_name in ["make_classification", "classification"]:
            X, y = make_classification(
                n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=random_state
            )
            feature_names = ["feature_1", "feature_2"]
        elif dataset_name in ["make_regression", "regression"]:
            X, y = make_regression(n_samples=n_samples, n_features=2, noise=0.1, random_state=random_state)
            feature_names = ["feature_1", "feature_2"]
        elif dataset_name == "make_blobs":
            X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=random_state)
            feature_names = ["feature_1", "feature_2"]
        elif dataset_name == "load_iris":
            iris = load_iris()
            X, y = iris.data, iris.target
            feature_names = iris.feature_names
        elif dataset_name == "load_breast_cancer":
            cancer = load_breast_cancer()
            X, y = cancer.data, cancer.target
            feature_names = cancer.feature_names
        elif dataset_name == "load_wine":
            wine = load_wine()
            X, y = wine.data, wine.target
            feature_names = wine.feature_names
        elif dataset_name == "load_diabetes":
            diabetes = load_diabetes()
            X, y = diabetes.data, diabetes.target
            feature_names = diabetes.feature_names
        else:
            raise DataLoadingError(message=f"Unsupported sklearn dataset: {dataset_name}")

        # Convert to DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        return df

    except ImportError:
        raise DataLoadingError(message="scikit-learn is required for synthetic datasets")
    except Exception as e:
        raise DataLoadingError(message=f"Failed to load sklearn dataset {dataset_name}: {str(e)}")


def _load_uci_data(dataset_name: str) -> pd.DataFrame:
    """
    Load UCI dataset.

    Args:
        dataset_name: UCI dataset identifier

    Returns:
        pd.DataFrame: UCI dataset

    Raises:
        DataLoadingError: If UCI dataset cannot be loaded
    """
    try:
        from ucimlrepo import fetch_ucirepo

        # Map common dataset names to UCI IDs
        uci_id_map = {"wine-quality-red": 186, "wine-quality-white": 186, "breast-cancer": 17, "iris": 53, "diabetes": 34}

        if dataset_name in uci_id_map:
            dataset_id = uci_id_map[dataset_name]
        else:
            # Try to parse as numeric ID
            try:
                dataset_id = int(dataset_name)
            except ValueError:
                raise DataLoadingError(message=f"Unknown UCI dataset: {dataset_name}")

        # Fetch dataset
        dataset = fetch_ucirepo(id=dataset_id)

        # Combine features and targets
        df = dataset.data.features.copy()
        if dataset.data.targets is not None:
            df["target"] = dataset.data.targets

        return df

    except ImportError:
        raise DataLoadingError(message="ucimlrepo is required for UCI datasets")
    except Exception as e:
        raise DataLoadingError(message=f"Failed to load UCI dataset {dataset_name}: {str(e)}")


def _apply_filter(data: pd.DataFrame, filter_config: Dict, target_column: Optional[str]) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Apply filter configuration to extract subset of data.

    Args:
        data: Source DataFrame
        filter_config: Filter configuration
        target_column: Name of target column (None if unsupervised)

    Returns:
        tuple: (features DataFrame, target Series or None)
    """
    # Start with full dataset
    filtered_data = data.copy()

    # Apply sample range filter if specified
    if "sample_range" in filter_config:
        start, end = filter_config["sample_range"]
        # Use inclusive slicing [start:end+1] to match endpoint behavior
        filtered_data = filtered_data.iloc[start : end + 1]

    # Apply feature filters if specified
    if "feature_filters" in filter_config:
        for feature_filter in filter_config["feature_filters"]:
            column = feature_filter["column"]
            condition = feature_filter["condition"]
            value = feature_filter["value"]

            if condition == "<=":
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif condition == ">=":
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif condition == ">":
                filtered_data = filtered_data[filtered_data[column] > value]
            elif condition == "<":
                filtered_data = filtered_data[filtered_data[column] < value]
            elif condition == "==":
                filtered_data = filtered_data[filtered_data[column] == value]
            elif condition == "!=":
                filtered_data = filtered_data[filtered_data[column] != value]

    # Apply noise factor for synthetic datasets (modify existing data)
    if "noise_factor" in filter_config:
        noise_factor = filter_config["noise_factor"]
        # Apply noise to numeric columns only
        for col in filtered_data.select_dtypes(include=[np.number]).columns:
            if col != target_column:  # Don't add noise to target
                std = filtered_data[col].std()
                noise = np.random.normal(0, std * noise_factor, len(filtered_data))
                filtered_data[col] += noise

    # Separate features and target
    if target_column and target_column in filtered_data.columns:
        y = filtered_data[target_column].copy()
        X = filtered_data.drop(columns=[target_column])
    else:
        y = None
        X = filtered_data.copy()

    return X, y


def _create_dataset_metadata(
    source_data: pd.DataFrame, scenario_definition: ScenarioDefinition, X_ref: pd.DataFrame, X_test: pd.DataFrame
) -> DatasetMetadata:
    """
    Create DatasetMetadata object.

    Args:
        source_data: Original source dataset
        scenario_definition: Scenario definition
        X_ref: Reference features
        X_test: Test features

    Returns:
        DatasetMetadata: Dataset metadata
    """
    # Infer data type and dimension
    data_type = _infer_data_type(X_ref)
    dimension = _infer_data_dimension(X_ref)

    return DatasetMetadata(
        name=scenario_definition.source_name,
        data_type=data_type,
        dimension=dimension,
        n_samples_ref=len(X_ref),
        n_samples_test=len(X_test),
        n_features=len(X_ref.columns),
        total_instances=len(source_data),
        feature_descriptions=_get_feature_descriptions(scenario_definition),
        missing_data_indicators=_get_missing_data_indicators(scenario_definition),
        data_quality_score=_get_data_quality_score(scenario_definition),
    )


def _create_scenario_metadata(
    scenario_definition: ScenarioDefinition,
    X_ref: pd.DataFrame,
    X_test: pd.DataFrame,
    y_ref: Optional[pd.Series],
    y_test: Optional[pd.Series],
) -> ScenarioMetadata:
    """
    Create ScenarioMetadata object.

    Args:
        scenario_definition: Scenario definition
        X_ref: Reference features
        X_test: Test features
        y_ref: Reference labels (None if unsupervised)
        y_test: Test labels (None if unsupervised)

    Returns:
        ScenarioMetadata: Scenario metadata
    """
    # Infer data type and dimension
    data_type = _infer_data_type(X_ref)
    dimension = _infer_data_dimension(X_ref)

    # Extract ground truth information
    ground_truth = scenario_definition.ground_truth or {}
    drift_periods = ground_truth.get("drift_periods", [])
    drift_intensity = ground_truth.get("drift_intensity")

    # Determine dataset category
    dataset_category = _determine_dataset_category(scenario_definition.source_type)

    return ScenarioMetadata(
        has_ground_truth=bool(ground_truth),
        drift_periods=drift_periods,
        drift_intensity=drift_intensity,
        total_samples=len(X_ref) + len(X_test),
        ref_samples=len(X_ref),
        test_samples=len(X_test),
        n_features=len(X_ref.columns),
        has_labels=(y_ref is not None and y_test is not None),
        data_type=data_type,
        dimension=dimension,
        dataset_category=dataset_category,
        # Enhanced metadata from scenario definition
        **_extract_enhanced_metadata(scenario_definition),
    )


def _infer_data_type(df: pd.DataFrame) -> DataType:
    """
    Infer data type from DataFrame columns.

    REQ-DAT-007: Data Type Algorithm

    Args:
        df: DataFrame to analyze

    Returns:
        DataType: Inferred data type
    """
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=["object", "category"]).columns)

    if numeric_cols > 0 and categorical_cols > 0:
        return "mixed"
    elif numeric_cols > 0:
        return "continuous"
    else:
        return "categorical"


def _infer_data_dimension(df: pd.DataFrame) -> DataDimension:
    """
    Infer data dimension from DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        DataDimension: Inferred dimension
    """
    return "multivariate" if len(df.columns) > 1 else "univariate"


def _determine_dataset_category(source_type: str) -> str:
    """
    Determine dataset category from source type.

    Args:
        source_type: Source type from scenario definition

    Returns:
        str: Dataset category
    """
    if source_type == "synthetic":
        return "synthetic"
    elif source_type == "uci":
        return "uci"
    elif source_type == "file":
        return "real"
    else:
        return "unknown"


def _get_feature_descriptions(scenario_definition: ScenarioDefinition) -> Optional[List[str]]:
    """Extract feature descriptions from scenario definition."""
    enhanced_metadata = getattr(scenario_definition, "enhanced_metadata", None)
    if enhanced_metadata:
        return enhanced_metadata.get("feature_descriptions")
    return None


def _get_missing_data_indicators(scenario_definition: ScenarioDefinition) -> Optional[List[str]]:
    """Extract missing data indicators from scenario definition."""
    enhanced_metadata = getattr(scenario_definition, "enhanced_metadata", None)
    if enhanced_metadata:
        return enhanced_metadata.get("missing_data_indicators")
    return None


def _get_data_quality_score(scenario_definition: ScenarioDefinition) -> Optional[float]:
    """Extract data quality score from scenario definition."""
    enhanced_metadata = getattr(scenario_definition, "enhanced_metadata", None)
    if enhanced_metadata:
        return enhanced_metadata.get("data_quality_score")
    return None


def _extract_enhanced_metadata(scenario_definition: ScenarioDefinition) -> Dict[str, Any]:
    """
    Extract enhanced metadata fields from scenario definition.

    Args:
        scenario_definition: Scenario definition

    Returns:
        Dict: Enhanced metadata fields
    """
    metadata = {}

    # Extract enhanced metadata
    enhanced_metadata = getattr(scenario_definition, "enhanced_metadata", None)
    if enhanced_metadata:
        metadata.update(
            {
                "total_instances": enhanced_metadata.get("total_instances"),
                "feature_descriptions": enhanced_metadata.get("feature_descriptions"),
                "missing_data_indicators": enhanced_metadata.get("missing_data_indicators"),
                "data_quality_score": enhanced_metadata.get("data_quality_score"),
            }
        )

    # Extract UCI metadata
    uci_metadata = getattr(scenario_definition, "uci_metadata", None)
    if uci_metadata:
        metadata.update(
            {
                "acquisition_date": uci_metadata.get("acquisition_date"),
                "data_source": uci_metadata.get("original_source"),
                "repository_reference": f"UCI ML Repository ID: {uci_metadata.get('dataset_id', 'unknown')}",
            }
        )

    return {k: v for k, v in metadata.items() if v is not None}
