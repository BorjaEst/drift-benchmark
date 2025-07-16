"""
Tests for metrics module.

This module contains comprehensive tests for the metrics.py module,
which provides metric calculation, evaluation, and analysis functionality
for drift detection benchmarks.

Test Classes Organization:
- TestMetricCalculation: Core metric calculation functions
- TestClassificationMetrics: Classification-specific metrics (accuracy, precision, recall, etc.)
- TestRateMetrics: Rate-based metrics (TPR, FPR, etc.)
- TestROCMetrics: ROC and AUC metrics
- TestDetectionMetrics: Drift detection-specific metrics (delay, rate, etc.)
- TestPerformanceMetrics: Performance and efficiency metrics
- TestScoreMetrics: Score and statistical metrics
- TestComparativeMetrics: Comparative and ranking metrics
- TestMetricAggregation: Aggregation and combination of multiple metrics
- TestMetricConfiguration: Configuration and setup of metrics
- TestMetricValidation: Validation and error handling
- TestMetricSummaryStatistics: Summary statistics and reporting
- TestMetricIntegration: Integration tests with real detector results
- TestMetricUtilities: Utility functions and helpers
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.constants.literals import (
    ClassificationMetric,
    ComparativeMetric,
    DetectionMetric,
    DetectionResult,
    Metric,
    PerformanceMetric,
    RateMetric,
    ROCMetric,
    ScoreMetric,
)
from drift_benchmark.constants.models import (
    BenchmarkResult,
    DetectorPrediction,
    DriftEvaluationResult,
    MetricConfiguration,
    MetricResult,
    MetricSummary,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_predictions():
    """Sample detector predictions for testing."""
    return [
        DetectorPrediction(
            dataset_name="test_data",
            window_id=1,
            has_true_drift=True,
            detected_drift=True,
            detection_time=0.001,
            scores={"confidence": 0.85, "drift_score": 0.75, "p_value": 0.02},
        ),
        DetectorPrediction(
            dataset_name="test_data",
            window_id=2,
            has_true_drift=False,
            detected_drift=False,
            detection_time=0.002,
            scores={"confidence": 0.92, "drift_score": 0.15, "p_value": 0.85},
        ),
        DetectorPrediction(
            dataset_name="test_data",
            window_id=3,
            has_true_drift=False,
            detected_drift=True,
            detection_time=0.001,
            scores={"confidence": 0.68, "drift_score": 0.45, "p_value": 0.12},
        ),
        DetectorPrediction(
            dataset_name="test_data",
            window_id=4,
            has_true_drift=True,
            detected_drift=False,
            detection_time=0.003,
            scores={"confidence": 0.78, "drift_score": 0.25, "p_value": 0.45},
        ),
    ]


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth labels for testing."""
    return np.array([True, False, False, True, True, False, True, False])


@pytest.fixture
def sample_predictions_binary():
    """Sample binary predictions for testing."""
    return np.array([True, False, True, True, False, False, True, False])


@pytest.fixture
def sample_scores():
    """Sample drift scores for testing."""
    return np.array([0.85, 0.12, 0.67, 0.89, 0.34, 0.08, 0.72, 0.19])


@pytest.fixture
def sample_probabilities():
    """Sample prediction probabilities for testing."""
    return np.array([0.85, 0.12, 0.67, 0.89, 0.34, 0.08, 0.72, 0.19])


@pytest.fixture
def sample_detection_times():
    """Sample detection times for testing."""
    return np.array([0.001, 0.002, 0.0015, 0.003, 0.0012, 0.0018, 0.0025, 0.0013])


@pytest.fixture
def sample_drift_points():
    """Sample drift points for testing detection delay."""
    return [100, 300, 500]


@pytest.fixture
def sample_detected_points():
    """Sample detected drift points for testing detection delay."""
    return [105, 310, 520]


@pytest.fixture
def sample_metric_configurations():
    """Sample metric configurations for testing."""
    return [
        MetricConfiguration(name="ACCURACY", enabled=True, weight=1.0),
        MetricConfiguration(name="PRECISION", enabled=True, weight=1.2),
        MetricConfiguration(name="RECALL", enabled=True, weight=0.8),
        MetricConfiguration(name="F1_SCORE", enabled=True, weight=1.0, threshold=0.5),
        MetricConfiguration(name="AUC_ROC", enabled=False, weight=1.5),
    ]


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results for testing aggregation."""
    predictions = [
        DetectorPrediction(
            dataset_name="test_data",
            window_id=1,
            has_true_drift=True,
            detected_drift=True,
            detection_time=0.001,
            scores={"confidence": 0.85, "drift_score": 0.75},
        ),
        DetectorPrediction(
            dataset_name="test_data",
            window_id=2,
            has_true_drift=False,
            detected_drift=False,
            detection_time=0.002,
            scores={"confidence": 0.92, "drift_score": 0.15},
        ),
    ]

    return [
        BenchmarkResult(
            detector_name="ks_batch",
            dataset_name="test_data_1",
            predictions=predictions,
            metrics={"fit_time": 0.01, "detect_time": 0.002},
        ),
        BenchmarkResult(
            detector_name="mmd_online",
            dataset_name="test_data_1",
            predictions=predictions,
            metrics={"fit_time": 0.02, "detect_time": 0.003},
        ),
    ]


@pytest.fixture
def confusion_matrix_data():
    """Sample confusion matrix data for testing."""
    return {
        "true_positives": 15,
        "true_negatives": 20,
        "false_positives": 5,
        "false_negatives": 10,
    }


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestMetricCalculation:
    """Test core metric calculation functionality."""

    def test_calculate_metric_with_valid_inputs(self, sample_ground_truth, sample_predictions_binary):
        """Test basic metric calculation with valid inputs."""
        # This will test the main calculate_metric function
        from drift_benchmark.metrics import calculate_metric

        result = calculate_metric(
            metric_name="ACCURACY",
            ground_truth=sample_ground_truth,
            predictions=sample_predictions_binary,
        )

        assert isinstance(result, MetricResult)
        assert result.name == "ACCURACY"
        assert 0.0 <= result.value <= 1.0
        assert result.metadata is not None

    def test_calculate_metric_with_scores(self, sample_ground_truth, sample_scores):
        """Test metric calculation with continuous scores."""
        from drift_benchmark.metrics import calculate_metric

        result = calculate_metric(
            metric_name="AUC_ROC",
            ground_truth=sample_ground_truth,
            scores=sample_scores,
        )

        assert isinstance(result, MetricResult)
        assert result.name == "AUC_ROC"
        assert 0.0 <= result.value <= 1.0

    def test_calculate_metric_with_probabilities(self, sample_ground_truth, sample_probabilities):
        """Test metric calculation with prediction probabilities."""
        from drift_benchmark.metrics import calculate_metric

        result = calculate_metric(
            metric_name="AUC_PR",
            ground_truth=sample_ground_truth,
            probabilities=sample_probabilities,
        )

        assert isinstance(result, MetricResult)
        assert result.name == "AUC_PR"
        assert 0.0 <= result.value <= 1.0

    def test_calculate_multiple_metrics(self, sample_ground_truth, sample_predictions_binary):
        """Test calculation of multiple metrics at once."""
        from drift_benchmark.metrics import calculate_multiple_metrics

        metric_names = ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]
        results = calculate_multiple_metrics(
            metric_names=metric_names,
            ground_truth=sample_ground_truth,
            predictions=sample_predictions_binary,
        )

        assert isinstance(results, list)
        assert len(results) == len(metric_names)
        for result in results:
            assert isinstance(result, MetricResult)
            assert result.name in metric_names

    def test_calculate_metric_with_confidence_interval(self, sample_ground_truth, sample_predictions_binary):
        """Test metric calculation with confidence intervals."""
        from drift_benchmark.metrics import calculate_metric

        result = calculate_metric(
            metric_name="ACCURACY",
            ground_truth=sample_ground_truth,
            predictions=sample_predictions_binary,
            confidence_level=0.95,
            bootstrap_samples=100,
        )

        assert isinstance(result, MetricResult)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.value <= result.confidence_interval[1]

    def test_calculate_metric_invalid_input_types(self):
        """Test metric calculation with invalid input types."""
        from drift_benchmark.metrics import calculate_metric

        with pytest.raises((ValueError, TypeError)):
            calculate_metric(
                metric_name="ACCURACY",
                ground_truth="invalid",
                predictions=[1, 2, 3],
            )

    def test_calculate_metric_mismatched_lengths(self):
        """Test metric calculation with mismatched input lengths."""
        from drift_benchmark.metrics import calculate_metric

        with pytest.raises(ValueError, match="length|size|shape"):
            calculate_metric(
                metric_name="ACCURACY",
                ground_truth=[True, False, True],
                predictions=[True, False],
            )

    def test_calculate_metric_empty_inputs(self):
        """Test metric calculation with empty inputs."""
        from drift_benchmark.metrics import calculate_metric

        with pytest.raises(ValueError, match="empty|length"):
            calculate_metric(
                metric_name="ACCURACY",
                ground_truth=[],
                predictions=[],
            )


class TestClassificationMetrics:
    """Test classification-specific metrics."""

    def test_accuracy_calculation(self, confusion_matrix_data):
        """Test accuracy metric calculation."""
        from drift_benchmark.metrics import calculate_accuracy

        accuracy = calculate_accuracy(**confusion_matrix_data)
        expected = (15 + 20) / (15 + 20 + 5 + 10)  # (TP + TN) / Total

        assert isinstance(accuracy, float)
        assert abs(accuracy - expected) < 1e-10

    def test_precision_calculation(self, confusion_matrix_data):
        """Test precision metric calculation."""
        from drift_benchmark.metrics import calculate_precision

        precision = calculate_precision(**confusion_matrix_data)
        expected = 15 / (15 + 5)  # TP / (TP + FP)

        assert isinstance(precision, float)
        assert abs(precision - expected) < 1e-10

    def test_recall_calculation(self, confusion_matrix_data):
        """Test recall metric calculation."""
        from drift_benchmark.metrics import calculate_recall

        recall = calculate_recall(**confusion_matrix_data)
        expected = 15 / (15 + 10)  # TP / (TP + FN)

        assert isinstance(recall, float)
        assert abs(recall - expected) < 1e-10

    def test_f1_score_calculation(self, confusion_matrix_data):
        """Test F1 score metric calculation."""
        from drift_benchmark.metrics import calculate_f1_score

        f1 = calculate_f1_score(**confusion_matrix_data)
        precision = 15 / (15 + 5)
        recall = 15 / (15 + 10)
        expected = 2 * (precision * recall) / (precision + recall)

        assert isinstance(f1, float)
        assert abs(f1 - expected) < 1e-10

    def test_specificity_calculation(self, confusion_matrix_data):
        """Test specificity metric calculation."""
        from drift_benchmark.metrics import calculate_specificity

        specificity = calculate_specificity(**confusion_matrix_data)
        expected = 20 / (20 + 5)  # TN / (TN + FP)

        assert isinstance(specificity, float)
        assert abs(specificity - expected) < 1e-10

    def test_sensitivity_calculation(self, confusion_matrix_data):
        """Test sensitivity metric calculation (same as recall)."""
        from drift_benchmark.metrics import calculate_sensitivity

        sensitivity = calculate_sensitivity(**confusion_matrix_data)
        expected = 15 / (15 + 10)  # TP / (TP + FN)

        assert isinstance(sensitivity, float)
        assert abs(sensitivity - expected) < 1e-10

    def test_confusion_matrix_from_predictions(self, sample_ground_truth, sample_predictions_binary):
        """Test confusion matrix generation from predictions."""
        from drift_benchmark.metrics import calculate_confusion_matrix

        cm = calculate_confusion_matrix(sample_ground_truth, sample_predictions_binary)

        assert isinstance(cm, dict)
        required_keys = ["true_positives", "true_negatives", "false_positives", "false_negatives"]
        for key in required_keys:
            assert key in cm
            assert isinstance(cm[key], (int, np.integer))
            assert cm[key] >= 0

    def test_zero_division_handling(self):
        """Test handling of zero division in classification metrics."""
        from drift_benchmark.metrics import calculate_precision, calculate_recall

        # Case with no true positives and no false positives (precision)
        precision = calculate_precision(
            true_positives=0,
            true_negatives=10,
            false_positives=0,
            false_negatives=5,
        )
        assert precision == 0.0 or np.isnan(precision)

        # Case with no true positives and no false negatives (recall)
        recall = calculate_recall(
            true_positives=0,
            true_negatives=10,
            false_positives=5,
            false_negatives=0,
        )
        assert recall == 0.0 or np.isnan(recall)


class TestRateMetrics:
    """Test rate-based metrics."""

    def test_true_positive_rate_calculation(self, confusion_matrix_data):
        """Test true positive rate calculation."""
        from drift_benchmark.metrics import calculate_true_positive_rate

        tpr = calculate_true_positive_rate(**confusion_matrix_data)
        expected = 15 / (15 + 10)  # TP / (TP + FN)

        assert isinstance(tpr, float)
        assert abs(tpr - expected) < 1e-10

    def test_true_negative_rate_calculation(self, confusion_matrix_data):
        """Test true negative rate calculation."""
        from drift_benchmark.metrics import calculate_true_negative_rate

        tnr = calculate_true_negative_rate(**confusion_matrix_data)
        expected = 20 / (20 + 5)  # TN / (TN + FP)

        assert isinstance(tnr, float)
        assert abs(tnr - expected) < 1e-10

    def test_false_positive_rate_calculation(self, confusion_matrix_data):
        """Test false positive rate calculation."""
        from drift_benchmark.metrics import calculate_false_positive_rate

        fpr = calculate_false_positive_rate(**confusion_matrix_data)
        expected = 5 / (5 + 20)  # FP / (FP + TN)

        assert isinstance(fpr, float)
        assert abs(fpr - expected) < 1e-10

    def test_false_negative_rate_calculation(self, confusion_matrix_data):
        """Test false negative rate calculation."""
        from drift_benchmark.metrics import calculate_false_negative_rate

        fnr = calculate_false_negative_rate(**confusion_matrix_data)
        expected = 10 / (10 + 15)  # FN / (FN + TP)

        assert isinstance(fnr, float)
        assert abs(fnr - expected) < 1e-10

    def test_rate_metrics_sum_to_one(self, confusion_matrix_data):
        """Test that complementary rate metrics sum to 1."""
        from drift_benchmark.metrics import (
            calculate_false_negative_rate,
            calculate_false_positive_rate,
            calculate_true_negative_rate,
            calculate_true_positive_rate,
        )

        tpr = calculate_true_positive_rate(**confusion_matrix_data)
        fnr = calculate_false_negative_rate(**confusion_matrix_data)
        tnr = calculate_true_negative_rate(**confusion_matrix_data)
        fpr = calculate_false_positive_rate(**confusion_matrix_data)

        assert abs(tpr + fnr - 1.0) < 1e-10
        assert abs(tnr + fpr - 1.0) < 1e-10


class TestROCMetrics:
    """Test ROC and AUC metrics."""

    def test_auc_roc_calculation(self, sample_ground_truth, sample_scores):
        """Test AUC-ROC calculation."""
        from drift_benchmark.metrics import calculate_auc_roc

        auc = calculate_auc_roc(sample_ground_truth, sample_scores)

        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_auc_pr_calculation(self, sample_ground_truth, sample_scores):
        """Test AUC-PR calculation."""
        from drift_benchmark.metrics import calculate_auc_pr

        auc_pr = calculate_auc_pr(sample_ground_truth, sample_scores)

        assert isinstance(auc_pr, float)
        assert 0.0 <= auc_pr <= 1.0

    def test_roc_curve_generation(self, sample_ground_truth, sample_scores):
        """Test ROC curve generation."""
        from drift_benchmark.metrics import generate_roc_curve

        fpr, tpr, thresholds = generate_roc_curve(sample_ground_truth, sample_scores)

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(thresholds, np.ndarray)
        assert len(fpr) == len(tpr) == len(thresholds)
        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)

    def test_pr_curve_generation(self, sample_ground_truth, sample_scores):
        """Test Precision-Recall curve generation."""
        from drift_benchmark.metrics import generate_pr_curve

        precision, recall, thresholds = generate_pr_curve(sample_ground_truth, sample_scores)

        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert isinstance(thresholds, np.ndarray)
        assert len(precision) == len(recall)
        assert len(thresholds) == len(precision) - 1  # sklearn convention
        assert np.all(precision >= 0) and np.all(precision <= 1)
        assert np.all(recall >= 0) and np.all(recall <= 1)

    def test_perfect_classifier_auc(self):
        """Test AUC for perfect classifier."""
        from drift_benchmark.metrics import calculate_auc_roc

        ground_truth = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])  # Perfect separation

        auc = calculate_auc_roc(ground_truth, scores)
        assert abs(auc - 1.0) < 1e-10

    def test_random_classifier_auc(self):
        """Test AUC for random classifier."""
        from drift_benchmark.metrics import calculate_auc_roc

        np.random.seed(42)
        ground_truth = np.random.choice([0, 1], size=100)
        scores = np.random.random(100)  # Random scores

        auc = calculate_auc_roc(ground_truth, scores)
        assert 0.3 <= auc <= 0.7  # Should be around 0.5 for random


class TestDetectionMetrics:
    """Test drift detection-specific metrics."""

    def test_detection_delay_calculation(self, sample_drift_points, sample_detected_points):
        """Test detection delay calculation."""
        from drift_benchmark.metrics import calculate_detection_delay

        delay = calculate_detection_delay(sample_drift_points, sample_detected_points)
        expected = np.mean([5, 10, 20])  # Delays: 105-100, 310-300, 520-500

        assert isinstance(delay, float)
        assert abs(delay - expected) < 1e-10

    def test_detection_rate_calculation(self, sample_drift_points, sample_detected_points):
        """Test detection rate calculation."""
        from drift_benchmark.metrics import calculate_detection_rate

        rate = calculate_detection_rate(sample_drift_points, sample_detected_points, tolerance=25)

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
        assert rate == 1.0  # All drifts detected within tolerance

    def test_missed_detection_rate_calculation(self, sample_drift_points):
        """Test missed detection rate calculation."""
        from drift_benchmark.metrics import calculate_missed_detection_rate

        # Only detect first drift point
        detected_points = [105]
        rate = calculate_missed_detection_rate(sample_drift_points, detected_points, tolerance=25)

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
        assert abs(rate - 2 / 3) < 1e-10  # 2 out of 3 drifts missed

    def test_detection_delay_with_missed_drifts(self):
        """Test detection delay calculation with missed drifts."""
        from drift_benchmark.metrics import calculate_detection_delay

        drift_points = [100, 300, 500]
        detected_points = [105, 520]  # Missing detection at 300

        delay = calculate_detection_delay(drift_points, detected_points, tolerance=25)
        expected = np.mean([5, 20])  # Only count detected drifts

        assert isinstance(delay, float)
        assert abs(delay - expected) < 1e-10

    def test_detection_delay_with_no_detections(self):
        """Test detection delay with no detections."""
        from drift_benchmark.metrics import calculate_detection_delay

        drift_points = [100, 300, 500]
        detected_points = []

        delay = calculate_detection_delay(drift_points, detected_points)
        assert np.isnan(delay) or delay == float("inf")

    def test_false_alarm_rate_calculation(self):
        """Test false alarm rate calculation."""
        from drift_benchmark.metrics import calculate_false_alarm_rate

        total_windows = 1000
        false_alarms = 50

        rate = calculate_false_alarm_rate(false_alarms, total_windows)
        expected = 50 / 1000

        assert isinstance(rate, float)
        assert abs(rate - expected) < 1e-10

    def test_detection_metrics_with_tolerance(self):
        """Test detection metrics with different tolerance values."""
        from drift_benchmark.metrics import calculate_detection_rate

        drift_points = [100, 300]
        detected_points = [110, 325]  # Delays of 10 and 25

        # With tolerance 15, only first detection is valid
        rate_15 = calculate_detection_rate(drift_points, detected_points, tolerance=15)
        assert abs(rate_15 - 0.5) < 1e-10

        # With tolerance 30, both detections are valid
        rate_30 = calculate_detection_rate(drift_points, detected_points, tolerance=30)
        assert abs(rate_30 - 1.0) < 1e-10


class TestPerformanceMetrics:
    """Test performance and efficiency metrics."""

    def test_computation_time_calculation(self, sample_detection_times):
        """Test computation time metric calculation."""
        from drift_benchmark.metrics import calculate_computation_time

        avg_time = calculate_computation_time(sample_detection_times)
        expected = np.mean(sample_detection_times)

        assert isinstance(avg_time, float)
        assert abs(avg_time - expected) < 1e-10

    def test_memory_usage_calculation(self):
        """Test memory usage calculation."""
        from drift_benchmark.metrics import calculate_memory_usage

        memory_samples = [512, 768, 1024, 640, 896]  # MB
        avg_memory = calculate_memory_usage(memory_samples)
        expected = np.mean(memory_samples)

        assert isinstance(avg_memory, float)
        assert abs(avg_memory - expected) < 1e-10

    def test_throughput_calculation(self):
        """Test throughput calculation."""
        from drift_benchmark.metrics import calculate_throughput

        n_samples = 10000
        total_time = 2.5  # seconds

        throughput = calculate_throughput(n_samples, total_time)
        expected = n_samples / total_time

        assert isinstance(throughput, float)
        assert abs(throughput - expected) < 1e-10

    def test_performance_metrics_aggregation(self, sample_benchmark_results):
        """Test aggregation of performance metrics across results."""
        from drift_benchmark.metrics import aggregate_performance_metrics

        aggregated = aggregate_performance_metrics(sample_benchmark_results)

        assert isinstance(aggregated, dict)
        assert "fit_time" in aggregated
        assert "detect_time" in aggregated
        for metric_name, stats in aggregated.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

    def test_efficiency_ratio_calculation(self):
        """Test efficiency ratio calculation."""
        from drift_benchmark.metrics import calculate_efficiency_ratio

        baseline_time = 0.1
        detector_time = 0.05

        ratio = calculate_efficiency_ratio(detector_time, baseline_time)
        expected = baseline_time / detector_time

        assert isinstance(ratio, float)
        assert abs(ratio - expected) < 1e-10

    def test_scalability_metrics(self):
        """Test scalability metrics calculation."""
        from drift_benchmark.metrics import calculate_scalability_metrics

        sample_sizes = [1000, 2000, 4000, 8000]
        computation_times = [0.01, 0.025, 0.055, 0.12]

        metrics = calculate_scalability_metrics(sample_sizes, computation_times)

        assert isinstance(metrics, dict)
        assert "complexity_order" in metrics
        assert "scaling_factor" in metrics


class TestScoreMetrics:
    """Test score and statistical metrics."""

    def test_drift_score_calculation(self, sample_predictions):
        """Test drift score metric calculation."""
        from drift_benchmark.metrics import calculate_drift_score

        scores = [pred.scores.get("drift_score", 0.5) for pred in sample_predictions]
        avg_score = calculate_drift_score(scores)
        expected = np.mean(scores)

        assert isinstance(avg_score, float)
        assert abs(avg_score - expected) < 1e-10

    def test_p_value_calculation(self, sample_predictions):
        """Test p-value extraction and analysis."""
        from drift_benchmark.metrics import calculate_p_value_metrics

        p_values = [pred.scores.get("p_value", 0.5) for pred in sample_predictions]
        metrics = calculate_p_value_metrics(p_values)

        assert isinstance(metrics, dict)
        assert "mean_p_value" in metrics
        assert "significant_count" in metrics  # p < 0.05
        assert "median_p_value" in metrics

    def test_confidence_score_calculation(self, sample_predictions):
        """Test confidence score calculation."""
        from drift_benchmark.metrics import calculate_confidence_score

        confidences = [pred.scores.get("confidence", 0.5) for pred in sample_predictions]
        avg_confidence = calculate_confidence_score(confidences)
        expected = np.mean(confidences)

        assert isinstance(avg_confidence, float)
        assert abs(avg_confidence - expected) < 1e-10

    def test_statistical_significance_testing(self, sample_ground_truth, sample_scores):
        """Test statistical significance testing."""
        from drift_benchmark.metrics import test_statistical_significance

        result = test_statistical_significance(sample_ground_truth, sample_scores)

        assert isinstance(result, dict)
        assert "p_value" in result
        assert "is_significant" in result
        assert "test_statistic" in result
        assert "test_name" in result

    def test_effect_size_calculation(self, sample_ground_truth, sample_scores):
        """Test effect size calculation."""
        from drift_benchmark.metrics import calculate_effect_size

        effect_size = calculate_effect_size(sample_ground_truth, sample_scores)

        assert isinstance(effect_size, float)
        assert effect_size >= 0  # Cohen's d can be positive or negative, but we take absolute


class TestComparativeMetrics:
    """Test comparative and ranking metrics."""

    def test_relative_accuracy_calculation(self):
        """Test relative accuracy calculation."""
        from drift_benchmark.metrics import calculate_relative_accuracy

        detector_accuracy = 0.85
        baseline_accuracy = 0.70

        relative_acc = calculate_relative_accuracy(detector_accuracy, baseline_accuracy)
        expected = detector_accuracy / baseline_accuracy

        assert isinstance(relative_acc, float)
        assert abs(relative_acc - expected) < 1e-10

    def test_improvement_ratio_calculation(self):
        """Test improvement ratio calculation."""
        from drift_benchmark.metrics import calculate_improvement_ratio

        detector_metric = 0.85
        baseline_metric = 0.70

        improvement = calculate_improvement_ratio(detector_metric, baseline_metric)
        expected = (detector_metric - baseline_metric) / baseline_metric

        assert isinstance(improvement, float)
        assert abs(improvement - expected) < 1e-10

    def test_ranking_score_calculation(self, sample_benchmark_results):
        """Test ranking score calculation."""
        from drift_benchmark.metrics import calculate_ranking_score

        metrics_data = {
            "accuracy": [0.85, 0.78],
            "precision": [0.82, 0.75],
            "recall": [0.88, 0.80],
        }

        ranking_scores = calculate_ranking_score(metrics_data)

        assert isinstance(ranking_scores, list)
        assert len(ranking_scores) == 2
        for score in ranking_scores:
            assert isinstance(score, float)
            assert score >= 0

    def test_dominance_analysis(self):
        """Test dominance analysis between detectors."""
        from drift_benchmark.metrics import perform_dominance_analysis

        detector_a_metrics = {"accuracy": 0.85, "precision": 0.80, "recall": 0.90}
        detector_b_metrics = {"accuracy": 0.83, "precision": 0.85, "recall": 0.88}

        dominance = perform_dominance_analysis(detector_a_metrics, detector_b_metrics)

        assert isinstance(dominance, dict)
        assert "dominates" in dominance
        assert "dominated_by" in dominance
        assert "pareto_efficient" in dominance

    def test_pareto_frontier_analysis(self):
        """Test Pareto frontier analysis for multi-objective optimization."""
        from drift_benchmark.metrics import analyze_pareto_frontier

        detectors_metrics = [
            {"accuracy": 0.85, "speed": 0.90},
            {"accuracy": 0.90, "speed": 0.70},
            {"accuracy": 0.80, "speed": 0.95},
            {"accuracy": 0.82, "speed": 0.85},  # Dominated
        ]

        pareto_efficient = analyze_pareto_frontier(detectors_metrics)

        assert isinstance(pareto_efficient, list)
        assert len(pareto_efficient) <= len(detectors_metrics)
        # Should exclude the dominated detector
        assert len(pareto_efficient) == 3


class TestMetricAggregation:
    """Test aggregation and combination of multiple metrics."""

    def test_aggregate_metrics_across_runs(self):
        """Test aggregating metrics across multiple runs."""
        from drift_benchmark.metrics import aggregate_metrics_across_runs

        runs_metrics = [
            [MetricResult(name="ACCURACY", value=0.85, metadata={})],
            [MetricResult(name="ACCURACY", value=0.82, metadata={})],
            [MetricResult(name="ACCURACY", value=0.88, metadata={})],
        ]

        aggregated = aggregate_metrics_across_runs(runs_metrics)

        assert isinstance(aggregated, list)
        assert len(aggregated) == 1  # One unique metric
        assert isinstance(aggregated[0], MetricSummary)
        assert aggregated[0].name == "ACCURACY"
        assert abs(aggregated[0].mean - 0.85) < 1e-10

    def test_weighted_metric_aggregation(self, sample_metric_configurations):
        """Test weighted aggregation of metrics."""
        from drift_benchmark.metrics import calculate_weighted_metric_score

        metric_values = {"ACCURACY": 0.85, "PRECISION": 0.80, "RECALL": 0.90, "F1_SCORE": 0.85}

        weighted_score = calculate_weighted_metric_score(
            metric_values,
            sample_metric_configurations,
        )

        assert isinstance(weighted_score, float)
        assert 0.0 <= weighted_score <= 1.0

    def test_metric_normalization(self):
        """Test metric normalization for comparison."""
        from drift_benchmark.metrics import normalize_metrics

        raw_metrics = {
            "accuracy": [0.70, 0.85, 0.90],
            "detection_delay": [15, 8, 5],  # Lower is better
            "computation_time": [0.05, 0.02, 0.01],  # Lower is better
        }

        normalized = normalize_metrics(raw_metrics, higher_is_better={"accuracy": True})

        assert isinstance(normalized, dict)
        for metric_name, values in normalized.items():
            assert len(values) == 3
            for value in values:
                assert 0.0 <= value <= 1.0

    def test_composite_score_calculation(self):
        """Test composite score calculation from multiple metrics."""
        from drift_benchmark.metrics import calculate_composite_score

        metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.90,
            "speed": 0.75,
        }
        weights = {"accuracy": 0.4, "precision": 0.2, "recall": 0.2, "speed": 0.2}

        composite = calculate_composite_score(metrics, weights)

        assert isinstance(composite, float)
        assert 0.0 <= composite <= 1.0

    def test_metric_correlation_analysis(self):
        """Test correlation analysis between metrics."""
        from drift_benchmark.metrics import analyze_metric_correlations

        metrics_data = {
            "accuracy": [0.70, 0.75, 0.80, 0.85, 0.90],
            "precision": [0.68, 0.73, 0.78, 0.83, 0.88],
            "recall": [0.72, 0.77, 0.82, 0.87, 0.92],
            "speed": [0.90, 0.85, 0.80, 0.75, 0.70],  # Negatively correlated
        }

        correlations = analyze_metric_correlations(metrics_data)

        assert isinstance(correlations, dict)
        for metric_pair, correlation in correlations.items():
            assert isinstance(correlation, float)
            assert -1.0 <= correlation <= 1.0


class TestMetricConfiguration:
    """Test metric configuration and setup."""

    def test_metric_configuration_creation(self):
        """Test creating metric configurations."""
        config = MetricConfiguration(
            name="ACCURACY",
            enabled=True,
            weight=1.5,
            threshold=0.8,
            parameters={"bootstrap_samples": 1000},
        )

        assert config.name == "ACCURACY"
        assert config.enabled is True
        assert config.weight == 1.5
        assert config.threshold == 0.8
        assert config.parameters["bootstrap_samples"] == 1000

    def test_metric_configuration_validation(self):
        """Test metric configuration validation."""
        # Test invalid weight
        with pytest.raises(ValueError):
            MetricConfiguration(name="ACCURACY", weight=-1.0)

        # Test invalid metric name
        with pytest.raises(ValueError):
            MetricConfiguration(name="INVALID_METRIC")

    def test_filter_enabled_metrics(self, sample_metric_configurations):
        """Test filtering enabled metrics."""
        from drift_benchmark.metrics import filter_enabled_metrics

        enabled = filter_enabled_metrics(sample_metric_configurations)

        assert isinstance(enabled, list)
        assert len(enabled) == 4  # AUC_ROC is disabled
        for config in enabled:
            assert config.enabled is True

    def test_get_metric_weights(self, sample_metric_configurations):
        """Test extracting metric weights."""
        from drift_benchmark.metrics import get_metric_weights

        weights = get_metric_weights(sample_metric_configurations)

        assert isinstance(weights, dict)
        assert "ACCURACY" in weights
        assert weights["ACCURACY"] == 1.0
        assert weights["PRECISION"] == 1.2

    def test_validate_metric_configurations(self, sample_metric_configurations):
        """Test validation of metric configurations."""
        from drift_benchmark.metrics import validate_metric_configurations

        # Should pass validation
        is_valid, errors = validate_metric_configurations(sample_metric_configurations)
        assert is_valid is True
        assert len(errors) == 0

    def test_default_metric_configurations(self):
        """Test creation of default metric configurations."""
        from drift_benchmark.metrics import get_default_metric_configurations

        defaults = get_default_metric_configurations()

        assert isinstance(defaults, list)
        assert len(defaults) > 0
        for config in defaults:
            assert isinstance(config, MetricConfiguration)
            assert config.enabled is True

    def test_metric_configuration_serialization(self, sample_metric_configurations):
        """Test serialization of metric configurations."""
        config = sample_metric_configurations[0]

        # Test to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "name" in config_dict
        assert "enabled" in config_dict

        # Test from dict
        recreated = MetricConfiguration.model_validate(config_dict)
        assert recreated.name == config.name
        assert recreated.enabled == config.enabled


class TestMetricValidation:
    """Test validation and error handling."""

    def test_validate_metric_inputs(self):
        """Test validation of metric calculation inputs."""
        from drift_benchmark.metrics import validate_metric_inputs

        # Valid inputs
        ground_truth = np.array([True, False, True, False])
        predictions = np.array([True, True, False, False])

        is_valid, error_msg = validate_metric_inputs(ground_truth, predictions)
        assert is_valid is True
        assert error_msg is None

    def test_validate_metric_inputs_mismatched_length(self):
        """Test validation with mismatched input lengths."""
        from drift_benchmark.metrics import validate_metric_inputs

        ground_truth = np.array([True, False, True])
        predictions = np.array([True, False])

        is_valid, error_msg = validate_metric_inputs(ground_truth, predictions)
        assert is_valid is False
        assert "length" in error_msg.lower()

    def test_validate_metric_inputs_empty(self):
        """Test validation with empty inputs."""
        from drift_benchmark.metrics import validate_metric_inputs

        ground_truth = np.array([])
        predictions = np.array([])

        is_valid, error_msg = validate_metric_inputs(ground_truth, predictions)
        assert is_valid is False
        assert "empty" in error_msg.lower()

    def test_handle_nan_values(self):
        """Test handling of NaN values in metrics."""
        from drift_benchmark.metrics import handle_nan_values

        values_with_nan = [0.85, np.nan, 0.90, 0.78]
        cleaned = handle_nan_values(values_with_nan, strategy="remove")

        assert len(cleaned) == 3
        assert not any(np.isnan(v) for v in cleaned)

    def test_handle_infinite_values(self):
        """Test handling of infinite values in metrics."""
        from drift_benchmark.metrics import handle_infinite_values

        values_with_inf = [0.85, float("inf"), 0.90, float("-inf")]
        cleaned = handle_infinite_values(values_with_inf, strategy="clip", bounds=(0, 1))

        assert len(cleaned) == 4
        assert all(0 <= v <= 1 for v in cleaned)

    def test_validate_probability_scores(self):
        """Test validation of probability scores."""
        from drift_benchmark.metrics import validate_probability_scores

        # Valid probabilities
        valid_probs = [0.1, 0.5, 0.9, 0.0, 1.0]
        assert validate_probability_scores(valid_probs) is True

        # Invalid probabilities
        invalid_probs = [0.1, 1.5, -0.2, 0.9]
        assert validate_probability_scores(invalid_probs) is False

    def test_validate_metric_bounds(self):
        """Test validation of metric value bounds."""
        from drift_benchmark.metrics import validate_metric_bounds

        # Test accuracy bounds
        assert validate_metric_bounds("ACCURACY", 0.85) is True
        assert validate_metric_bounds("ACCURACY", 1.5) is False
        assert validate_metric_bounds("ACCURACY", -0.1) is False

        # Test AUC bounds
        assert validate_metric_bounds("AUC_ROC", 0.75) is True
        assert validate_metric_bounds("AUC_ROC", 1.2) is False


class TestMetricSummaryStatistics:
    """Test summary statistics and reporting."""

    def test_generate_metric_summary(self):
        """Test generation of metric summary statistics."""
        from drift_benchmark.metrics import generate_metric_summary

        values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.78, 0.82, 0.88]
        summary = generate_metric_summary("ACCURACY", values)

        assert isinstance(summary, MetricSummary)
        assert summary.name == "ACCURACY"
        assert abs(summary.mean - np.mean(values)) < 1e-10
        assert abs(summary.std - np.std(values, ddof=1)) < 1e-10
        assert summary.min == min(values)
        assert summary.max == max(values)
        assert summary.count == len(values)

    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        from drift_benchmark.metrics import calculate_percentiles

        values = [0.70, 0.75, 0.80, 0.85, 0.90]
        percentiles = calculate_percentiles(values, [25, 50, 75, 95])

        assert isinstance(percentiles, dict)
        assert "25" in percentiles
        assert "50" in percentiles
        assert "75" in percentiles
        assert "95" in percentiles
        assert percentiles["50"] == 0.80  # Median

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        from drift_benchmark.metrics import calculate_confidence_interval

        values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.78, 0.82, 0.88]
        ci = calculate_confidence_interval(values, confidence_level=0.95)

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= np.mean(values) <= ci[1]

    def test_bootstrap_statistics(self):
        """Test bootstrap statistical analysis."""
        from drift_benchmark.metrics import bootstrap_metric_statistics

        values = [0.70, 0.75, 0.80, 0.85, 0.90]
        stats = bootstrap_metric_statistics(values, n_bootstrap=100, random_state=42)

        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "confidence_interval" in stats
        assert "bootstrap_samples" in stats

    def test_significance_testing(self):
        """Test statistical significance testing between groups."""
        from drift_benchmark.metrics import test_metric_significance

        group1 = [0.70, 0.75, 0.80, 0.85, 0.90]
        group2 = [0.65, 0.70, 0.75, 0.80, 0.85]

        result = test_metric_significance(group1, group2)

        assert isinstance(result, dict)
        assert "p_value" in result
        assert "is_significant" in result
        assert "test_statistic" in result
        assert "effect_size" in result

    def test_metric_distribution_analysis(self):
        """Test analysis of metric value distributions."""
        from drift_benchmark.metrics import analyze_metric_distribution

        values = np.random.normal(0.8, 0.1, 100)
        analysis = analyze_metric_distribution(values)

        assert isinstance(analysis, dict)
        assert "distribution_type" in analysis
        assert "normality_test" in analysis
        assert "skewness" in analysis
        assert "kurtosis" in analysis


class TestMetricIntegration:
    """Integration tests with real detector results."""

    def test_end_to_end_metric_calculation(self, sample_predictions):
        """Test complete metric calculation workflow."""
        from drift_benchmark.metrics import calculate_all_metrics

        ground_truth = [pred.has_true_drift for pred in sample_predictions]
        predictions = [pred.detected_drift for pred in sample_predictions]
        scores = [pred.scores.get("drift_score", 0.5) for pred in sample_predictions]

        all_metrics = calculate_all_metrics(
            ground_truth=ground_truth,
            predictions=predictions,
            scores=scores,
            include_performance=True,
        )

        assert isinstance(all_metrics, list)
        assert len(all_metrics) > 0
        for metric in all_metrics:
            assert isinstance(metric, MetricResult)

    def test_benchmark_result_metric_extraction(self, sample_benchmark_results):
        """Test metric extraction from benchmark results."""
        from drift_benchmark.metrics import extract_metrics_from_results

        metrics = extract_metrics_from_results(sample_benchmark_results)

        assert isinstance(metrics, dict)
        for detector_id, detector_metrics in metrics.items():
            assert isinstance(detector_metrics, list)
            for metric in detector_metrics:
                assert isinstance(metric, MetricResult)

    def test_cross_validation_metrics(self, sample_predictions):
        """Test cross-validation metric calculation."""
        from drift_benchmark.metrics import calculate_cv_metrics

        # Simulate CV folds
        cv_predictions = [sample_predictions[:2], sample_predictions[2:]]
        cv_metrics = calculate_cv_metrics(cv_predictions, metric_names=["ACCURACY", "F1_SCORE"])

        assert isinstance(cv_metrics, dict)
        for metric_name, stats in cv_metrics.items():
            assert "mean" in stats
            assert "std" in stats
            assert "cv_scores" in stats

    def test_temporal_metric_analysis(self):
        """Test temporal analysis of metrics over time."""
        from drift_benchmark.metrics import analyze_temporal_metrics

        timestamps = pd.date_range("2024-01-01", periods=10, freq="D")
        metric_values = [0.70, 0.72, 0.75, 0.73, 0.78, 0.80, 0.82, 0.79, 0.85, 0.87]

        analysis = analyze_temporal_metrics(timestamps, metric_values, metric_name="ACCURACY")

        assert isinstance(analysis, dict)
        assert "trend" in analysis
        assert "seasonality" in analysis
        assert "volatility" in analysis

    def test_comparative_detector_analysis(self, sample_benchmark_results):
        """Test comparative analysis between detectors."""
        from drift_benchmark.metrics import compare_detectors

        comparison = compare_detectors(sample_benchmark_results)

        assert isinstance(comparison, dict)
        assert "best_detector" in comparison
        assert "rankings" in comparison
        assert "statistical_tests" in comparison


class TestMetricUtilities:
    """Test utility functions and helpers."""

    def test_metric_name_normalization(self):
        """Test metric name normalization."""
        from drift_benchmark.metrics import normalize_metric_name

        assert normalize_metric_name("accuracy") == "ACCURACY"
        assert normalize_metric_name("f1_score") == "F1_SCORE"
        assert normalize_metric_name("AUC_ROC") == "AUC_ROC"

    def test_available_metrics_listing(self):
        """Test listing of available metrics."""
        from drift_benchmark.metrics import list_available_metrics

        metrics = list_available_metrics()

        assert isinstance(metrics, list)
        assert "ACCURACY" in metrics
        assert "PRECISION" in metrics
        assert "RECALL" in metrics
        assert "F1_SCORE" in metrics

    def test_metric_category_filtering(self):
        """Test filtering metrics by category."""
        from drift_benchmark.metrics import filter_metrics_by_category

        classification_metrics = filter_metrics_by_category("classification")
        performance_metrics = filter_metrics_by_category("performance")

        assert isinstance(classification_metrics, list)
        assert isinstance(performance_metrics, list)
        assert "ACCURACY" in classification_metrics
        assert "COMPUTATION_TIME" in performance_metrics

    def test_metric_dependencies_checking(self):
        """Test checking metric calculation dependencies."""
        from drift_benchmark.metrics import check_metric_dependencies

        # Accuracy needs predictions and ground truth
        deps = check_metric_dependencies("ACCURACY")
        assert "predictions" in deps
        assert "ground_truth" in deps

        # AUC needs scores/probabilities and ground truth
        deps = check_metric_dependencies("AUC_ROC")
        assert "ground_truth" in deps
        assert ("scores" in deps) or ("probabilities" in deps)

    def test_metric_result_serialization(self):
        """Test serialization of metric results."""
        from drift_benchmark.metrics import serialize_metric_results

        results = [
            MetricResult(name="ACCURACY", value=0.85, metadata={"method": "test"}),
            MetricResult(name="PRECISION", value=0.80, metadata={"method": "test"}),
        ]

        serialized = serialize_metric_results(results, format="json")
        assert isinstance(serialized, str)

        # Test deserialization
        from drift_benchmark.metrics import deserialize_metric_results

        deserialized = deserialize_metric_results(serialized, format="json")
        assert len(deserialized) == 2
        assert deserialized[0].name == "ACCURACY"

    def test_metric_export_formats(self):
        """Test exporting metrics in different formats."""
        from drift_benchmark.metrics import export_metrics

        results = [
            MetricResult(name="ACCURACY", value=0.85, metadata={}),
            MetricResult(name="PRECISION", value=0.80, metadata={}),
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            export_metrics(results, f.name, format="csv")

            # Verify file exists and has content
            exported_df = pd.read_csv(f.name)
            assert len(exported_df) == 2
            assert "name" in exported_df.columns
            assert "value" in exported_df.columns

    def test_metric_report_generation(self, sample_benchmark_results):
        """Test generation of comprehensive metric reports."""
        from drift_benchmark.metrics import generate_metric_report

        report = generate_metric_report(
            sample_benchmark_results,
            include_plots=False,
            include_statistical_tests=True,
        )

        assert isinstance(report, dict)
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "comparisons" in report

    @pytest.mark.parametrize("metric_name", ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE"])
    def test_individual_metric_calculation(self, metric_name, sample_ground_truth, sample_predictions_binary):
        """Test individual metric calculations."""
        from drift_benchmark.metrics import calculate_metric

        result = calculate_metric(
            metric_name=metric_name,
            ground_truth=sample_ground_truth,
            predictions=sample_predictions_binary,
        )

        assert isinstance(result, MetricResult)
        assert result.name == metric_name
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
    def test_confidence_intervals_different_levels(self, confidence_level):
        """Test confidence intervals at different confidence levels."""
        from drift_benchmark.metrics import calculate_confidence_interval

        values = np.random.normal(0.8, 0.1, 100)
        ci = calculate_confidence_interval(values, confidence_level=confidence_level)

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]

        # Higher confidence should give wider intervals
        if confidence_level > 0.90:
            ci_90 = calculate_confidence_interval(values, confidence_level=0.90)
            assert (ci[1] - ci[0]) >= (ci_90[1] - ci_90[0])


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.parametrize(
    "metric_name,expected_range",
    [
        ("ACCURACY", (0.0, 1.0)),
        ("PRECISION", (0.0, 1.0)),
        ("RECALL", (0.0, 1.0)),
        ("F1_SCORE", (0.0, 1.0)),
        ("AUC_ROC", (0.0, 1.0)),
        ("SPECIFICITY", (0.0, 1.0)),
    ],
)
def test_metric_value_ranges(metric_name, expected_range, sample_ground_truth, sample_predictions_binary, sample_scores):
    """Test that metrics return values within expected ranges."""
    from drift_benchmark.metrics import calculate_metric

    if metric_name in ["AUC_ROC", "AUC_PR"]:
        result = calculate_metric(
            metric_name=metric_name,
            ground_truth=sample_ground_truth,
            scores=sample_scores,
        )
    else:
        result = calculate_metric(
            metric_name=metric_name,
            ground_truth=sample_ground_truth,
            predictions=sample_predictions_binary,
        )

    assert expected_range[0] <= result.value <= expected_range[1]


@pytest.mark.parametrize(
    "aggregation_method",
    ["mean", "median", "weighted_mean", "geometric_mean", "harmonic_mean"],
)
def test_metric_aggregation_methods(aggregation_method):
    """Test different metric aggregation methods."""
    from drift_benchmark.metrics import aggregate_metric_values

    values = [0.70, 0.75, 0.80, 0.85, 0.90]
    weights = [1.0, 1.2, 0.8, 1.0, 1.5] if aggregation_method == "weighted_mean" else None

    result = aggregate_metric_values(values, method=aggregation_method, weights=weights)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize(
    "bootstrap_samples",
    [10, 50, 100, 500],
)
def test_bootstrap_sample_sizes(bootstrap_samples):
    """Test bootstrap confidence intervals with different sample sizes."""
    from drift_benchmark.metrics import bootstrap_confidence_interval

    values = [0.70, 0.75, 0.80, 0.85, 0.90]
    ci = bootstrap_confidence_interval(values, n_bootstrap=bootstrap_samples, random_state=42)

    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert ci[0] <= np.mean(values) <= ci[1]
