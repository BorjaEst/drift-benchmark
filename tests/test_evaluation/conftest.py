"""
Fixtures and configuration for evaluation module tests.

This module provides evaluation-specific test fixtures including mock results,
performance metrics, and statistical test data for comprehensive testing
of drift detection evaluation capabilities.
"""

from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_results() -> Dict[str, Any]:
    """Provide classification results for testing metrics calculation."""
    np.random.seed(42)

    # Simulate detection results
    n_samples = 1000
    true_drift = np.concatenate([np.zeros(500), np.ones(500)])  # No drift in first half  # Drift in second half

    # Perfect detector results
    perfect_predictions = true_drift.copy()

    # Noisy detector results (85% accuracy)
    noisy_predictions = true_drift.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    noisy_predictions[noise_indices] = 1 - noisy_predictions[noise_indices]

    # Conservative detector (high precision, low recall)
    conservative_predictions = np.zeros(n_samples)
    high_confidence_drift = np.where(true_drift == 1)[0][:200]  # Only 40% of drift detected
    conservative_predictions[high_confidence_drift] = 1

    return {
        "true_labels": true_drift,
        "perfect_detector": perfect_predictions,
        "noisy_detector": noisy_predictions,
        "conservative_detector": conservative_predictions,
        "n_samples": n_samples,
    }


@pytest.fixture
def detection_timing_data() -> Dict[str, Any]:
    """Provide detection timing data for delay analysis."""
    return {
        "true_drift_points": [250, 600, 850],  # Actual drift occurrences
        "detected_drift_points": {
            "fast_detector": [255, 610, 860],  # 5-10 sample delay
            "slow_detector": [280, 640, 900],  # 30-50 sample delay
            "early_detector": [245, 595, 845],  # Detects before actual (false early)
            "missed_detector": [260, None, 880],  # Misses middle drift
        },
        "false_alarms": {"fast_detector": [], "slow_detector": [150], "early_detector": [100, 400], "missed_detector": [120, 450, 780]},
    }


@pytest.fixture
def score_distributions() -> Dict[str, np.ndarray]:
    """Provide score distributions for ROC analysis."""
    np.random.seed(42)

    return {
        # High-performance detector: clear separation
        "high_performance": {
            "no_drift_scores": np.random.beta(2, 8, 500),  # Low scores for no drift
            "drift_scores": np.random.beta(8, 2, 500),  # High scores for drift
        },
        # Medium-performance detector: some overlap
        "medium_performance": {"no_drift_scores": np.random.beta(3, 5, 500), "drift_scores": np.random.beta(5, 3, 500)},
        # Poor-performance detector: high overlap
        "poor_performance": {"no_drift_scores": np.random.beta(4, 4, 500), "drift_scores": np.random.beta(4.5, 4, 500)},
    }


@pytest.fixture
def statistical_test_data() -> Dict[str, Any]:
    """Provide data for statistical tests."""
    np.random.seed(42)

    # Performance scores for different detectors
    detector_scores = {
        "detector_a": np.random.normal(0.85, 0.05, 30),  # High performance
        "detector_b": np.random.normal(0.75, 0.08, 30),  # Medium performance
        "detector_c": np.random.normal(0.65, 0.06, 30),  # Lower performance
        "detector_d": np.random.normal(0.82, 0.07, 30),  # High performance
    }

    # Paired comparison data
    paired_results = {"method_1": np.random.normal(0.8, 0.05, 50), "method_2": np.random.normal(0.78, 0.06, 50)}  # Slightly lower

    return {"detector_scores": detector_scores, "paired_results": paired_results, "significance_level": 0.05}


@pytest.fixture
def runtime_measurements() -> Dict[str, Any]:
    """Provide runtime and resource usage measurements."""
    return {
        "detectors": {
            "fast_detector": {
                "fit_times": [0.1, 0.12, 0.09, 0.11, 0.1],
                "detect_times": [0.05, 0.04, 0.06, 0.05, 0.04],
                "memory_usage": [45.2, 47.1, 44.8, 46.3, 45.9],  # MB
                "cpu_usage": [15.2, 16.8, 14.9, 15.7, 15.4],  # %
            },
            "slow_detector": {
                "fit_times": [2.5, 2.8, 2.3, 2.6, 2.7],
                "detect_times": [0.8, 0.9, 0.7, 0.8, 0.85],
                "memory_usage": [125.6, 132.1, 128.9, 130.4, 127.8],
                "cpu_usage": [65.2, 68.9, 63.7, 66.1, 67.3],
            },
            "memory_intensive": {
                "fit_times": [1.2, 1.4, 1.1, 1.3, 1.25],
                "detect_times": [0.3, 0.35, 0.28, 0.32, 0.31],
                "memory_usage": [256.7, 264.2, 251.3, 259.8, 255.4],
                "cpu_usage": [35.1, 37.8, 34.2, 36.5, 35.9],
            },
        }
    }


@pytest.fixture
def benchmark_results_comprehensive() -> Dict[str, Any]:
    """Provide comprehensive benchmark results for analysis."""
    return {
        "benchmark_metadata": {
            "name": "Comprehensive Evaluation Test",
            "datasets": ["iris_drift", "wine_drift", "synthetic_1"],
            "detectors": ["ks_test", "mmd_test", "chi2_test"],
            "total_executions": 9,
            "start_time": "2024-01-01T10:00:00",
            "end_time": "2024-01-01T10:30:00",
        },
        "detector_results": [
            {
                "detector_id": "ks_test_evidently",
                "dataset": "iris_drift",
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.88,
                    "f1": 0.85,
                    "specificity": 0.82,
                    "balanced_accuracy": 0.85,
                    "detection_delay": 5.2,
                    "auc_score": 0.89,
                    "false_alarm_rate": 0.15,
                },
                "runtime": {
                    "fit_time": 0.15,
                    "detect_time": 0.08,
                    "training_time": 0.15,
                    "inference_time": 0.08,
                    "memory_usage": 45.2,
                    "cpu_time": 0.23,
                },
            },
            {
                "detector_id": "mmd_test_alibi",
                "dataset": "iris_drift",
                "metrics": {
                    "accuracy": 0.78,
                    "precision": 0.75,
                    "recall": 0.82,
                    "f1": 0.78,
                    "specificity": 0.75,
                    "balanced_accuracy": 0.785,
                    "detection_delay": 8.1,
                    "auc_score": 0.83,
                    "false_alarm_rate": 0.22,
                },
                "runtime": {
                    "fit_time": 0.32,
                    "detect_time": 0.18,
                    "training_time": 0.32,
                    "inference_time": 0.18,
                    "memory_usage": 68.5,
                    "cpu_time": 0.50,
                },
            },
            {
                "detector_id": "chi2_test_custom",
                "dataset": "wine_drift",
                "metrics": {
                    "accuracy": 0.72,
                    "precision": 0.70,
                    "recall": 0.75,
                    "f1": 0.72,
                    "specificity": 0.70,
                    "balanced_accuracy": 0.725,
                    "detection_delay": 12.3,
                    "auc_score": 0.76,
                    "false_alarm_rate": 0.28,
                },
                "runtime": {
                    "fit_time": 0.08,
                    "detect_time": 0.03,
                    "training_time": 0.08,
                    "inference_time": 0.03,
                    "memory_usage": 28.7,
                    "cpu_time": 0.11,
                },
            },
        ],
    }


@pytest.fixture
def evaluation_configurations() -> List[Dict[str, Any]]:
    """Provide various evaluation configurations for testing."""
    return [
        # Basic evaluation
        {
            "classification_metrics": ["accuracy", "precision", "recall"],
            "detection_metrics": ["detection_delay", "auc_score"],
            "statistical_tests": ["ttest"],
            "performance_analysis": ["rankings"],
            "runtime_analysis": ["memory_usage"],
        },
        # Comprehensive evaluation
        {
            "classification_metrics": ["accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy"],
            "detection_metrics": ["detection_delay", "auc_score", "false_alarm_rate", "detection_power"],
            "statistical_tests": ["ttest", "mannwhitneyu", "ks_test", "wilcoxon", "friedman"],
            "performance_analysis": ["rankings", "robustness", "critical_difference", "statistical_significance"],
            "runtime_analysis": ["memory_usage", "cpu_time", "training_time", "inference_time"],
        },
        # Minimal evaluation
        {
            "classification_metrics": ["accuracy"],
            "detection_metrics": ["auc_score"],
            "statistical_tests": [],
            "performance_analysis": [],
            "runtime_analysis": [],
        },
    ]


@pytest.fixture
def expected_metric_ranges() -> Dict[str, Tuple[float, float]]:
    """Provide expected ranges for metric validation."""
    return {
        "accuracy": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
        "f1": (0.0, 1.0),
        "specificity": (0.0, 1.0),
        "balanced_accuracy": (0.0, 1.0),
        "auc_score": (0.0, 1.0),
        "detection_delay": (0.0, float("inf")),
        "false_alarm_rate": (0.0, 1.0),
        "detection_power": (0.0, 1.0),
        "memory_usage": (0.0, float("inf")),
        "cpu_time": (0.0, float("inf")),
        "fit_time": (0.0, float("inf")),
        "detect_time": (0.0, float("inf")),
    }
