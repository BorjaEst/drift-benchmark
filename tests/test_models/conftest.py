# Feature-specific fixtures for models module testing
# Aligned with README TOML examples and REQUIREMENTS REQ-CFM-002 flat structure

from typing import Any, Dict

import pandas as pd
import pytest


@pytest.fixture
def sample_benchmark_config_data():
    """Provide sample data for BenchmarkConfig testing following README TOML structure"""
    return {
        "scenarios": [{"id": "covariate_drift_example"}, {"id": "concept_drift_example"}],
        "detectors": [
            # Library comparison examples from README
            {"method_id": "kolmogorov_smirnov", "variant_id": "batch", "library_id": "evidently"},
            {"method_id": "kolmogorov_smirnov", "variant_id": "batch", "library_id": "alibi-detect"},
            {"method_id": "cramer_von_mises", "variant_id": "batch", "library_id": "scipy"},
        ],
    }


@pytest.fixture
def sample_dataset_config_data():
    """Provide sample data for DatasetConfig testing"""
    return {"path": "datasets/example.csv", "format": "csv", "reference_split": 0.7}


@pytest.fixture
def sample_detector_config_data():
    """Provide sample data for DetectorConfig testing following README examples"""
    return {"method_id": "kolmogorov_smirnov", "variant_id": "batch", "library_id": "evidently"}


@pytest.fixture
def sample_dataset_result_data():
    """Provide sample data for DatasetResult testing"""
    ref_data = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0], "feature_2": ["A", "B", "C"]})
    test_data = pd.DataFrame({"feature_1": [4.0, 5.0, 6.0], "feature_2": ["D", "E", "F"]})

    metadata = {"name": "test_dataset", "data_type": "mixed", "dimension": "multivariate", "n_samples_ref": 3, "n_samples_test": 3}

    return {"X_ref": ref_data, "X_test": test_data, "metadata": metadata}


@pytest.fixture
def sample_detector_result_data():
    """Provide sample data for DetectorResult testing following REQ-MDL-002 structure"""
    return {
        "detector_id": "ks_test_scipy",
        "library_id": "scipy",
        "scenario_name": "covariate_drift_example",
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
        "n_features": 10,
    }


@pytest.fixture
def sample_detector_metadata_data():
    """Provide sample data for DetectorMetadata testing following REQ-MET-002"""
    return {
        "method_id": "kolmogorov_smirnov",
        "variant_id": "batch",
        "library_id": "evidently",
        "name": "Kolmogorov-Smirnov Test",
        "family": "statistical-test",
        "description": "Two-sample test for equality of continuous distributions",
    }


@pytest.fixture
def sample_benchmark_summary_data():
    """Provide sample data for BenchmarkSummary testing following REQ-MET-003 Phase 1 fields"""
    return {
        "total_detectors": 5,  # Test expects this specific value
        "successful_runs": 4,  # Test expects this specific value
        "failed_runs": 1,
        "avg_execution_time": 0.0196,  # Average of README examples: (0.0234 + 0.0156) / 2
        # Phase 1: Focus on performance metrics, Phase 2 will add ground truth evaluation
    }
