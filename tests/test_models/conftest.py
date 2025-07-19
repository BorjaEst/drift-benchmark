"""
Model-specific test fixtures for drift-benchmark models testing.

This module provides fixtures specifically for testing the models module,
including sample data, configurations, and mock objects that support
comprehensive validation of model behavior and requirements.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.literals import (
    ClassificationMetric,
    DataDimension,
    DataType,
    DetectionMetric,
    DetectorFamily,
    DriftPattern,
    DriftType,
    ExecutionMode,
    FileFormat,
    LogLevel,
    PerformanceMetric,
)
from drift_benchmark.models import (
    BenchmarkConfig,
    BenchmarkMetadata,
    DatasetConfig,
    DatasetMetadata,
    DatasetResult,
    DetectorConfig,
    DetectorMetadata,
    DriftMetadata,
    EvaluationConfig,
    ScoreResult,
)


@pytest.fixture
def sample_drift_metadata() -> DriftMetadata:
    """Provide realistic drift metadata for testing REQ-MET-003."""
    return DriftMetadata(drift_type="COVARIATE", drift_position=0.6, drift_magnitude=2.5, drift_pattern="GRADUAL")


@pytest.fixture
def sample_dataset_metadata() -> DatasetMetadata:
    """Provide realistic dataset metadata for testing REQ-MET-002."""
    return DatasetMetadata(
        name="iris_drift_experiment",
        description="Iris dataset with introduced covariate drift",
        n_samples=1000,
        n_features=4,
        has_drift=True,
        data_types=["CONTINUOUS"],
        dimension="MULTIVARIATE",
        labeling="SUPERVISED",
    )


@pytest.fixture
def sample_detector_metadata() -> DetectorMetadata:
    """Provide realistic detector metadata for testing REQ-MET-004."""
    return DetectorMetadata(
        method_id="kolmogorov_smirnov",
        implementation_id="ks_batch",
        name="Kolmogorov-Smirnov Test (Batch)",
        description="Two-sample Kolmogorov-Smirnov test for batch processing",
        category="STATISTICAL_TEST",
        data_type="CONTINUOUS",
        streaming=False,
    )


@pytest.fixture
def sample_score_result() -> ScoreResult:
    """Provide realistic score result for testing REQ-RES-005."""
    return ScoreResult(
        drift_detected=True,
        drift_score=0.087,
        threshold=0.05,
        p_value=0.023,
        confidence_interval=(0.015, 0.045),
        metadata={
            "test_statistic": 0.087,
            "critical_value": 0.05,
            "sample_size_ref": 500,
            "sample_size_test": 500,
            "execution_time": 0.124,
        },
    )


@pytest.fixture
def sample_benchmark_metadata() -> BenchmarkMetadata:
    """Provide realistic benchmark metadata for testing REQ-CFG-002."""
    return BenchmarkMetadata(
        name="Comprehensive Drift Detection Benchmark",
        description="Multi-method evaluation across diverse drift scenarios",
        author="Drift Research Team",
        version="2.1.0",
    )


@pytest.fixture
def sample_dataset_config() -> DatasetConfig:
    """Provide realistic dataset configuration for testing REQ-CFG-003."""
    return DatasetConfig(
        datasets=[
            {"name": "iris_covariate_drift", "type": "scenario", "config": {"scenario_name": "iris_species_drift", "drift_magnitude": 2.0}},
            {
                "name": "wine_quality_shift",
                "type": "file",
                "config": {
                    "path": "/data/wine_quality.csv",
                    "format": "CSV",
                    "preprocessing": {"normalize": True, "remove_outliers": True},
                },
            },
            {
                "name": "synthetic_gradual_drift",
                "type": "synthetic",
                "config": {
                    "generator": "gaussian",
                    "n_samples": 2000,
                    "n_features": 5,
                    "drift_pattern": "gradual",
                    "drift_position": 0.5,
                    "drift_magnitude": 1.5,
                },
            },
        ]
    )


@pytest.fixture
def sample_detector_config() -> DetectorConfig:
    """Provide realistic detector configuration for testing REQ-CFG-004."""
    return DetectorConfig(
        algorithms=[
            {
                "adapter": "evidently_adapter",
                "method_id": "kolmogorov_smirnov",
                "implementation_id": "ks_batch",
                "parameters": {"threshold": 0.05, "alternative": "two-sided"},
            },
            {
                "adapter": "alibi_adapter",
                "method_id": "maximum_mean_discrepancy",
                "implementation_id": "mmd_streaming",
                "parameters": {"sigma": 1.0, "kernel": "rbf", "window_size": 100},
            },
            {
                "adapter": "frouros_adapter",
                "method_id": "page_hinkley",
                "implementation_id": "ph_streaming",
                "parameters": {"threshold": 10.0, "alpha": 0.01},
            },
        ]
    )


@pytest.fixture
def sample_evaluation_config() -> EvaluationConfig:
    """Provide realistic evaluation configuration for testing REQ-CFG-005."""
    return EvaluationConfig(
        classification_metrics=["accuracy", "precision", "recall", "f1_score"],
        detection_metrics=["detection_delay", "detection_rate", "auc_roc"],
        statistical_tests=["ttest", "mannwhitneyu", "wilcoxon"],
        performance_analysis=["rankings", "statistical_significance", "confidence_intervals"],
        runtime_analysis=["memory_usage", "cpu_time", "throughput"],
    )


@pytest.fixture
def sample_complete_benchmark_config(
    sample_benchmark_metadata: BenchmarkMetadata,
    sample_dataset_config: DatasetConfig,
    sample_detector_config: DetectorConfig,
    sample_evaluation_config: EvaluationConfig,
) -> BenchmarkConfig:
    """Provide complete benchmark configuration for testing REQ-CFG-001."""
    return BenchmarkConfig(
        metadata=sample_benchmark_metadata,
        data=sample_dataset_config,
        detectors=sample_detector_config,
        evaluation=sample_evaluation_config,
    )


@pytest.fixture
def sample_pandas_dataframes() -> Dict[str, pd.DataFrame]:
    """Provide realistic pandas DataFrames for testing REQ-RES-002."""
    np.random.seed(42)

    # Reference data - original distribution
    n_ref = 500
    X_ref = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_ref),
            "feature_2": np.random.normal(0, 1, n_ref),
            "feature_3": np.random.exponential(1, n_ref),
            "feature_4": np.random.choice(["A", "B", "C"], n_ref, p=[0.5, 0.3, 0.2]),
        }
    )

    # Test data - drifted distribution
    n_test = 500
    X_test = pd.DataFrame(
        {
            "feature_1": np.random.normal(1.5, 1.2, n_test),  # Mean and variance shift
            "feature_2": np.random.normal(-0.5, 0.8, n_test),  # Mean shift
            "feature_3": np.random.exponential(2, n_test),  # Scale change
            "feature_4": np.random.choice(["A", "B", "C"], n_test, p=[0.2, 0.4, 0.4]),  # Distribution change
        }
    )

    y_ref = pd.Series(np.random.choice([0, 1, 2], n_ref, p=[0.4, 0.4, 0.2]))
    y_test = pd.Series(np.random.choice([0, 1, 2], n_test, p=[0.2, 0.3, 0.5]))  # Label shift

    return {"X_ref": X_ref, "X_test": X_test, "y_ref": y_ref, "y_test": y_test}


@pytest.fixture
def sample_dataset_result(
    sample_pandas_dataframes: Dict[str, pd.DataFrame], sample_drift_metadata: DriftMetadata, sample_dataset_metadata: DatasetMetadata
) -> DatasetResult:
    """Provide complete dataset result for testing REQ-RES-002."""
    return DatasetResult(
        X_ref=sample_pandas_dataframes["X_ref"],
        X_test=sample_pandas_dataframes["X_test"],
        y_ref=sample_pandas_dataframes["y_ref"],
        y_test=sample_pandas_dataframes["y_test"],
        drift_info=sample_drift_metadata,
        metadata=sample_dataset_metadata,
    )


@pytest.fixture
def invalid_data_samples() -> Dict[str, Any]:
    """Provide invalid data samples for validation testing."""
    return {
        "negative_drift_position": -0.5,
        "drift_position_over_one": 1.5,
        "negative_n_samples": -100,
        "zero_n_features": 0,
        "invalid_drift_type": "INVALID_DRIFT",
        "invalid_data_dimension": "INVALID_DIMENSION",
        "invalid_execution_mode": "INVALID_MODE",
        "invalid_detector_family": "INVALID_FAMILY",
        "empty_string": "",
        "none_required_field": None,
    }


@pytest.fixture
def temporary_file_paths() -> Dict[str, Path]:
    """Provide temporary file paths for testing file operations."""
    temp_dir = Path(tempfile.mkdtemp(prefix="drift_benchmark_models_"))

    return {
        "temp_dir": temp_dir,
        "json_file": temp_dir / "test_config.json",
        "toml_file": temp_dir / "test_config.toml",
        "csv_file": temp_dir / "test_results.csv",
        "parquet_file": temp_dir / "test_results.parquet",
    }


@pytest.fixture
def mock_execution_metadata() -> Dict[str, Any]:
    """Provide mock execution metadata for testing REQ-MET-001."""
    return {
        "start_time": datetime(2024, 1, 15, 10, 30, 0),
        "end_time": datetime(2024, 1, 15, 11, 45, 30),
        "duration": 4530.0,  # seconds
        "status": "completed",
        "summary": {
            "total_detectors": 5,
            "successful_runs": 5,
            "failed_runs": 0,
            "datasets_processed": 3,
            "total_drift_detected": 8,
            "average_detection_time": 2.4,
        },
    }


@pytest.fixture
def literal_type_samples() -> Dict[str, Any]:
    """Provide valid literal type samples for testing type safety requirements."""
    return {
        "drift_types": list(DriftType.__args__),
        "data_types": list(DataType.__args__),
        "data_dimensions": list(DataDimension.__args__),
        "execution_modes": list(ExecutionMode.__args__),
        "detector_families": list(DetectorFamily.__args__),
        "drift_patterns": list(DriftPattern.__args__),
        "file_formats": list(FileFormat.__args__),
        "log_levels": list(LogLevel.__args__),
        "classification_metrics": list(ClassificationMetric.__args__),
        "detection_metrics": list(DetectionMetric.__args__),
        "performance_metrics": list(PerformanceMetric.__args__),
    }
