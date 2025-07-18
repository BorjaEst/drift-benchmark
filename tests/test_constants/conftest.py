"""
Fixtures and configuration for constants module tests.

This module provides constants-specific test fixtures including sample
literals, model validation data, and configuration examples for testing
the constants and models functionality.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_drift_info_data() -> Dict[str, Any]:
    """Provide sample drift information data for model testing."""
    return {"drift_type": "COVARIATE", "drift_position": 0.5, "drift_magnitude": 1.5, "drift_pattern": "SUDDEN"}


@pytest.fixture
def sample_dataset_metadata() -> Dict[str, Any]:
    """Provide sample dataset metadata for model testing."""
    return {
        "name": "test_dataset",
        "description": "Test dataset for validation",
        "n_samples": 1000,
        "n_features": 5,
        "has_drift": True,
        "data_types": ["CONTINUOUS", "CATEGORICAL"],
        "dimension": "MULTIVARIATE",
        "labeling": "SUPERVISED",
    }


@pytest.fixture
def sample_score_result_data() -> Dict[str, Any]:
    """Provide sample score result data for model testing."""
    return {
        "drift_detected": True,
        "drift_score": 0.85,
        "threshold": 0.5,
        "p_value": 0.02,
        "confidence_interval": (0.75, 0.95),
        "metadata": {"method": "test", "timestamp": "2024-01-01T00:00:00"},
    }


@pytest.fixture
def sample_benchmark_config_data() -> Dict[str, Any]:
    """Provide comprehensive benchmark configuration data for model testing."""
    return {
        "metadata": {
            "name": "Test Benchmark Config",
            "description": "Configuration for testing",
            "author": "Test Author",
            "version": "1.0.0",
        },
        "data": {
            "datasets": [
                {"name": "test_scenario", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}},
                {
                    "name": "test_synthetic",
                    "type": "synthetic",
                    "config": {"generator_name": "gaussian", "n_samples": 1000, "n_features": 4, "drift_pattern": "gradual"},
                },
            ]
        },
        "detectors": {
            "algorithms": [
                {
                    "adapter": "evidently_adapter",
                    "method_id": "kolmogorov_smirnov",
                    "implementation_id": "ks_batch",
                    "parameters": {"threshold": 0.05},
                },
                {
                    "adapter": "alibi_adapter",
                    "method_id": "maximum_mean_discrepancy",
                    "implementation_id": "mmd_batch",
                    "parameters": {"kernel": "rbf", "sigma": 1.0},
                },
            ]
        },
        "evaluation": {
            "classification_metrics": ["accuracy", "precision", "recall", "f1"],
            "detection_metrics": ["detection_delay", "auc_score", "false_alarm_rate"],
            "statistical_tests": ["ttest", "mannwhitneyu", "ks_test"],
            "performance_analysis": ["rankings", "statistical_significance", "robustness"],
            "runtime_analysis": ["memory_usage", "cpu_time", "training_time"],
        },
    }


@pytest.fixture
def invalid_config_samples() -> List[Dict[str, Any]]:
    """Provide various invalid configuration samples for validation testing."""
    return [
        # Missing required fields
        {
            "metadata": {"name": "Test"},
            # Missing data, detectors, evaluation
        },
        # Invalid literal values
        {
            "metadata": {"name": "Test", "description": "", "author": "", "version": ""},
            "data": {"datasets": [{"name": "test", "type": "invalid_type", "config": {}}]},
            "detectors": {"algorithms": []},
            "evaluation": {"classification_metrics": ["invalid_metric"]},
        },
        # Invalid value ranges
        {
            "metadata": {"name": "Test", "description": "", "author": "", "version": ""},
            "data": {
                "datasets": [
                    {
                        "name": "test",
                        "type": "synthetic",
                        "config": {
                            "n_samples": -100,  # Negative samples
                            "drift_position": 1.5,  # Position > 1
                            "drift_magnitude": -0.5,  # Negative magnitude
                        },
                    }
                ]
            },
            "detectors": {"algorithms": []},
            "evaluation": {"classification_metrics": []},
        },
    ]


@pytest.fixture
def literal_values_samples() -> Dict[str, List[str]]:
    """Provide sample literal values for constants testing."""
    return {
        "drift_types": ["COVARIATE", "CONCEPT", "PRIOR", "LABEL"],
        "drift_patterns": ["SUDDEN", "GRADUAL", "INCREMENTAL", "RECURRING"],
        "data_types": ["CONTINUOUS", "CATEGORICAL", "MIXED"],
        "dimensions": ["UNIVARIATE", "MULTIVARIATE"],
        "labeling_types": ["SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"],
        "dataset_types": ["scenario", "synthetic", "file"],
        "execution_modes": ["BATCH", "STREAMING"],
        "detector_families": [
            "STATISTICAL_TEST",
            "DISTANCE_BASED",
            "STATISTICAL_PROCESS_CONTROL",
            "CHANGE_DETECTION",
            "WINDOW_BASED",
            "ENSEMBLE",
            "MACHINE_LEARNING",
        ],
    }


@pytest.fixture
def model_edge_cases() -> Dict[str, Any]:
    """Provide edge cases for model validation testing."""
    return {
        "empty_strings": {"name": "", "description": "", "author": ""},
        "zero_values": {"n_samples": 0, "n_features": 0, "drift_magnitude": 0.0},
        "boundary_values": {
            "drift_position": [0.0, 1.0],  # Boundary positions
            "drift_magnitude": [0.001, 999.999],  # Extreme magnitudes
            "threshold": [0.0, 1.0],  # Boundary thresholds
        },
        "large_values": {"n_samples": 1000000, "n_features": 10000, "confidence_interval": (0.0, 1.0)},
    }
