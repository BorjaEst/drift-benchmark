"""
File data loading for drift-benchmark - REQ-DAT-XXX

Provides file loading utilities for different data formats.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from ..exceptions import DataLoadingError, DataValidationError
from ..literals import DataDimension, DataType
from ..models.configurations import DatasetConfig
from ..models.metadata import DatasetMetadata
from ..models.results import DatasetResult


def load_dataset(config: DatasetConfig) -> DatasetResult:
    """
    Load dataset from file according to configuration.

    REQ-DAT-001: Data module must provide load_dataset interface for loading datasets from files
    """
    # REQ-DAT-004: Validate file exists and is readable
    file_path = Path(config.path)
    if not file_path.exists():
        raise DataLoadingError(f"Dataset file not found: {file_path}")

    if not file_path.is_file():
        raise DataLoadingError(f"Dataset path is not a file: {file_path}")

    try:
        # REQ-DAT-002: Support CSV format using pandas.read_csv() with default parameters
        if config.format == "CSV":
            df = pd.read_csv(file_path)
        else:
            raise DataLoadingError(f"Unsupported file format: {config.format}")

        # REQ-DAT-007: Handle missing values using pandas defaults
        # pandas automatically converts empty strings to NaN, no additional processing needed

        # REQ-DAT-003: Support reference_split ratio for creating X_ref/X_test divisions
        n_total = len(df)
        n_ref = int(n_total * config.reference_split)

        if n_ref == 0 or n_ref == n_total:
            raise DataValidationError(
                f"Invalid split ratio {config.reference_split} for dataset with {n_total} samples. "
                f"Results in {n_ref} reference samples."
            )

        # REQ-DAT-006: Return X_ref and X_test as pandas.DataFrame objects
        X_ref = df.iloc[:n_ref].copy()
        X_test = df.iloc[n_ref:].copy()

        # REQ-DAT-005: Automatically infer data types and set appropriate DataType
        data_type = _infer_data_type(df)

        # Determine dimensionality
        dimension: DataDimension = "MULTIVARIATE" if len(df.columns) > 1 else "UNIVARIATE"

        # Create metadata
        metadata = DatasetMetadata(
            name=file_path.stem,  # Use filename without extension as name
            data_type=data_type,
            dimension=dimension,
            n_samples_ref=len(X_ref),
            n_samples_test=len(X_test),
        )

        return DatasetResult(X_ref=X_ref, X_test=X_test, metadata=metadata)

    except pd.errors.EmptyDataError:
        raise DataLoadingError(f"Dataset file is empty: {file_path}")
    except pd.errors.ParserError as e:
        raise DataLoadingError(f"Failed to parse CSV file {file_path}: {e}")
    except Exception as e:
        raise DataLoadingError(f"Unexpected error loading dataset {file_path}: {e}")


def _infer_data_type(df: pd.DataFrame) -> DataType:
    """
    Infer data type based on pandas dtypes.

    REQ-DAT-008: Data type inference algorithm
    CONTINUOUS: numeric dtypes (int, float)
    CATEGORICAL: object/string dtypes
    MIXED: datasets with both numeric and object columns
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    object_cols = df.select_dtypes(include=["object", "string"]).columns

    has_numeric = len(numeric_cols) > 0
    has_categorical = len(object_cols) > 0

    if has_numeric and has_categorical:
        return "MIXED"
    elif has_numeric:
        return "CONTINUOUS"
    elif has_categorical:
        return "CATEGORICAL"
    else:
        # Edge case: no recognized columns, default to continuous
        return "CONTINUOUS"
