# Feature-specific fixtures for models module testing

from typing import Any, Dict

import pandas as pd
import pytest


@pytest.fixture
def sample_benchmark_config_data():
    """Provide sample data for BenchmarkConfig testing"""
    return {
        "scenarios": [{"id": "covariate_drift_example"}],
        "detectors": [
            {"method_id": "ks_test", "variant_id": "scipy", "library_id": "scipy"},
            {"method_id": "drift_detector", "variant_id": "custom", "library_id": "custom"},
        ],
    }


@pytest.fixture
def sample_dataset_config_data():
    """Provide sample data for DatasetConfig testing"""
    return {"path": "datasets/example.csv", "format": "csv", "reference_split": 0.7}


@pytest.fixture
def sample_detector_config_data():
    """Provide sample data for DetectorConfig testing"""
    return {"method_id": "ks_test", "variant_id": "scipy", "library_id": "scipy"}


@pytest.fixture
def sample_dataset_result_data():
    """Provide sample data for DatasetResult testing"""
    ref_data = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0], "feature_2": ["A", "B", "C"]})
    test_data = pd.DataFrame({"feature_1": [4.0, 5.0, 6.0], "feature_2": ["D", "E", "F"]})

    metadata = {"name": "test_dataset", "data_type": "mixed", "dimension": "multivariate", "n_samples_ref": 3, "n_samples_test": 3}

    return {"X_ref": ref_data, "X_test": test_data, "metadata": metadata}


@pytest.fixture
def sample_detector_result_data():
    """Provide sample data for DetectorResult testing"""
    return {
        "detector_id": "ks_test_scipy",
        "library_id": "scipy",
        "dataset_name": "covariate_drift_example",
        "drift_detected": True,
        "execution_time": 0.0123,
        "drift_score": 0.85,
    }


@pytest.fixture
def sample_dataset_metadata_data():
    """Provide sample data for DatasetMetadata testing"""
    return {
        "name": "sklearn_classification_source",
        "data_type": "continuous",
        "dimension": "multivariate",
        "n_samples_ref": 1000,
        "n_samples_test": 500,
    }


@pytest.fixture
def sample_detector_metadata_data():
    """Provide sample data for DetectorMetadata testing"""
    return {
        "method_id": "ks_test",
        "variant_id": "scipy",
        "library_id": "scipy",
        "name": "Kolmogorov-Smirnov Test",
        "family": "statistical-test",
    }


@pytest.fixture
def sample_benchmark_summary_data():
    """Provide sample data for BenchmarkSummary testing"""
    return {
        "total_detectors": 5,
        "successful_runs": 4,
        "failed_runs": 1,
        "avg_execution_time": 0.125,
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.9,
    }
