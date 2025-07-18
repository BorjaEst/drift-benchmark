"""
Fixtures and configuration for benchmark module tests.

This module provides benchmark-specific test fixtures including mock runners,
configurations, and test environments for validating benchmarking workflows
and performance measurement capabilities.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_benchmark_runner():
    """Mock BenchmarkRunner for testing benchmark execution workflows."""
    runner = Mock()
    runner.run.return_value = {
        "benchmark_id": "test_benchmark",
        "total_detectors": 2,
        "total_datasets": 1,
        "execution_time": 45.2,
        "results": [
            {
                "detector": "ks_test_evidently",
                "dataset": "iris_drift",
                "drift_detected": True,
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
            }
        ],
    }
    return runner


@pytest.fixture
def benchmark_environment(test_workspace):
    """Provide isolated benchmark environment for testing."""
    env = {
        "workspace": test_workspace,
        "components_dir": test_workspace / "components",
        "configurations_dir": test_workspace / "configurations",
        "results_dir": test_workspace / "results",
        "logs_dir": test_workspace / "logs",
    }

    # Create required directories
    for dir_path in env.values():
        if hasattr(dir_path, "mkdir"):
            dir_path.mkdir(exist_ok=True)

    return env


@pytest.fixture
def multi_detector_config() -> Dict[str, Any]:
    """Provide configuration for testing multiple detector benchmarking."""
    return {
        "metadata": {
            "name": "Multi-Detector Benchmark",
            "description": "Test multiple drift detection methods",
            "author": "Test Suite",
            "version": "1.0.0",
        },
        "data": {
            "datasets": [
                {"name": "iris_drift_test", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}},
                {"name": "wine_drift_test", "type": "scenario", "config": {"scenario_name": "wine_quality_drift"}},
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
                    "parameters": {"kernel": "rbf"},
                },
            ]
        },
        "evaluation": {
            "classification_metrics": ["accuracy", "precision", "recall", "f1"],
            "detection_metrics": ["detection_delay", "auc_score"],
            "statistical_tests": ["ttest", "mannwhitneyu"],
            "performance_analysis": ["rankings"],
            "runtime_analysis": ["memory_usage", "cpu_time"],
        },
    }


@pytest.fixture
def mock_execution_strategy():
    """Mock execution strategy for testing benchmark strategies."""
    strategy = Mock()
    strategy.execute.return_value = [
        {
            "detector_id": "ks_evidently",
            "dataset": "test_dataset",
            "execution_time": 2.5,
            "memory_usage": 125.6,
            "drift_detected": True,
            "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
        }
    ]
    return strategy


@pytest.fixture
def benchmark_results_sample() -> Dict[str, Any]:
    """Provide sample benchmark results for testing result processing."""
    return {
        "benchmark_metadata": {
            "name": "Test Benchmark",
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:05:00",
            "total_duration": 300.0,
        },
        "detector_results": [
            {
                "detector_id": "ks_evidently_batch",
                "adapter": "evidently_adapter",
                "method_id": "kolmogorov_smirnov",
                "dataset": "iris_species_drift",
                "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85, "detection_delay": 5.2, "auc_score": 0.89},
                "runtime": {"fit_time": 0.15, "detect_time": 0.08, "memory_usage": 45.2},
            },
            {
                "detector_id": "mmd_alibi_batch",
                "adapter": "alibi_adapter",
                "method_id": "maximum_mean_discrepancy",
                "dataset": "iris_species_drift",
                "metrics": {"accuracy": 0.78, "precision": 0.75, "recall": 0.82, "f1": 0.78, "detection_delay": 8.1, "auc_score": 0.83},
                "runtime": {"fit_time": 0.32, "detect_time": 0.18, "memory_usage": 68.5},
            },
        ],
    }
