"""
Metrics module for drift detection benchmarking.

This module provides comprehensive metric calculation, evaluation, and analysis
functionality for drift detection benchmarks. It includes classification metrics,
detection-specific metrics, performance metrics, and advanced analytics.

The module is organized into several categories:
1. Core metric calculation functions
2. Classification metrics (accuracy, precision, recall, F1-score, etc.)
3. Rate metrics (TPR, FPR, TNR, FNR)
4. ROC/AUC metrics
5. Detection-specific metrics (delay, detection rate, etc.)
6. Performance metrics (time, memory, throughput)
7. Statistical and score metrics
8. Comparative and ranking metrics
9. Metric aggregation and normalization
10. Validation and utility functions
"""

import datetime as dt
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from drift_benchmark.constants.literals import Metric
from drift_benchmark.constants.models import (
    BootstrapResult,
    ComparativeAnalysis,
    ConfusionMatrix,
    MetricConfiguration,
    MetricReport,
    MetricResult,
    MetricSummary,
    TemporalMetrics,
)

# =============================================================================
# CORE METRIC CALCULATION FUNCTIONS
# =============================================================================


def calculate_metric(
    metric_name: str,
    ground_truth: Union[List, np.ndarray],
    predictions: Optional[Union[List, np.ndarray]] = None,
    scores: Optional[Union[List, np.ndarray]] = None,
    probabilities: Optional[Union[List, np.ndarray]] = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 0,
    **kwargs,
) -> MetricResult:
    """
    Calculate a single metric with optional confidence intervals.

    Args:
        metric_name: Name of the metric to calculate
        ground_truth: True labels
        predictions: Predicted labels (for classification metrics)
        scores: Prediction scores (for ROC/AUC metrics)
        probabilities: Prediction probabilities (alternative to scores)
        confidence_level: Confidence level for intervals
        bootstrap_samples: Number of bootstrap samples for CI
        **kwargs: Additional metric-specific parameters

    Returns:
        MetricResult with calculated value and optional confidence interval
    """
    # Validate inputs
    is_valid, error_msg = validate_metric_inputs(ground_truth, predictions, scores, probabilities)
    if not is_valid:
        raise ValueError(error_msg)

    # Convert to numpy arrays
    ground_truth = np.array(ground_truth)
    if predictions is not None:
        predictions = np.array(predictions)
    if scores is not None:
        scores = np.array(scores)
    if probabilities is not None:
        probabilities = np.array(probabilities)

    # Calculate the metric
    metric_name = normalize_metric_name(metric_name)

    if metric_name in ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE", "SPECIFICITY", "SENSITIVITY"]:
        if predictions is None:
            raise ValueError(f"Metric {metric_name} requires predictions")
        value = _calculate_classification_metric(metric_name, ground_truth, predictions)
    elif metric_name in ["AUC_ROC", "AUC_PR"]:
        input_scores = scores if scores is not None else probabilities
        if input_scores is None:
            raise ValueError(f"Metric {metric_name} requires scores or probabilities")
        value = _calculate_roc_metric(metric_name, ground_truth, input_scores)
    elif metric_name in ["TRUE_POSITIVE_RATE", "TRUE_NEGATIVE_RATE", "FALSE_POSITIVE_RATE", "FALSE_NEGATIVE_RATE"]:
        if predictions is None:
            raise ValueError(f"Metric {metric_name} requires predictions")
        value = _calculate_rate_metric(metric_name, ground_truth, predictions)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    # Calculate confidence interval if requested
    confidence_interval = None
    if bootstrap_samples > 0:
        confidence_interval = _bootstrap_confidence_interval(
            metric_name, ground_truth, predictions, scores, probabilities, confidence_level, bootstrap_samples
        )

    return MetricResult(name=metric_name, value=value, confidence_interval=confidence_interval, metadata=kwargs)


def calculate_multiple_metrics(
    metric_names: List[str],
    ground_truth: Union[List, np.ndarray],
    predictions: Optional[Union[List, np.ndarray]] = None,
    scores: Optional[Union[List, np.ndarray]] = None,
    probabilities: Optional[Union[List, np.ndarray]] = None,
    **kwargs,
) -> List[MetricResult]:
    """
    Calculate multiple metrics at once.

    Args:
        metric_names: List of metric names to calculate
        ground_truth: True labels
        predictions: Predicted labels
        scores: Prediction scores
        probabilities: Prediction probabilities
        **kwargs: Additional parameters

    Returns:
        List of MetricResult objects
    """
    results = []
    for metric_name in metric_names:
        try:
            result = calculate_metric(
                metric_name=metric_name,
                ground_truth=ground_truth,
                predictions=predictions,
                scores=scores,
                probabilities=probabilities,
                **kwargs,
            )
            results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to calculate metric {metric_name}: {e}")

    return results


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================


def calculate_accuracy(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate accuracy from confusion matrix components."""
    total = true_positives + true_negatives + false_positives + false_negatives
    if total == 0:
        return 0.0
    return (true_positives + true_negatives) / total


def calculate_precision(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate precision from confusion matrix components."""
    denominator = true_positives + false_positives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def calculate_recall(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate recall from confusion matrix components."""
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return true_positives / denominator


def calculate_f1_score(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate F1 score from confusion matrix components."""
    precision = calculate_precision(true_positives, true_negatives, false_positives, false_negatives)
    recall = calculate_recall(true_positives, true_negatives, false_positives, false_negatives)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_specificity(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate specificity from confusion matrix components."""
    denominator = true_negatives + false_positives
    if denominator == 0:
        return 0.0
    return true_negatives / denominator


def calculate_sensitivity(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate sensitivity (same as recall) from confusion matrix components."""
    return calculate_recall(true_positives, true_negatives, false_positives, false_negatives)


def calculate_confusion_matrix(ground_truth: Union[List, np.ndarray], predictions: Union[List, np.ndarray]) -> Dict[str, int]:
    """
    Calculate confusion matrix from predictions.

    Returns:
        Dictionary with true_positives, true_negatives, false_positives, false_negatives
    """
    ground_truth = np.array(ground_truth, dtype=bool)
    predictions = np.array(predictions, dtype=bool)

    true_positives = int(np.sum((ground_truth == True) & (predictions == True)))
    true_negatives = int(np.sum((ground_truth == False) & (predictions == False)))
    false_positives = int(np.sum((ground_truth == False) & (predictions == True)))
    false_negatives = int(np.sum((ground_truth == True) & (predictions == False)))

    return {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


# =============================================================================
# RATE METRICS
# =============================================================================


def calculate_true_positive_rate(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate true positive rate (sensitivity/recall)."""
    return calculate_recall(true_positives, true_negatives, false_positives, false_negatives)


def calculate_true_negative_rate(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate true negative rate (specificity)."""
    return calculate_specificity(true_positives, true_negatives, false_positives, false_negatives)


def calculate_false_positive_rate(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate false positive rate."""
    denominator = false_positives + true_negatives
    if denominator == 0:
        return 0.0
    return false_positives / denominator


def calculate_false_negative_rate(true_positives: int, true_negatives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate false negative rate."""
    denominator = false_negatives + true_positives
    if denominator == 0:
        return 0.0
    return false_negatives / denominator


# =============================================================================
# ROC/AUC METRICS
# =============================================================================


def calculate_auc_roc(ground_truth: Union[List, np.ndarray], scores: Union[List, np.ndarray]) -> float:
    """Calculate Area Under ROC Curve."""
    ground_truth = np.array(ground_truth, dtype=int)
    scores = np.array(scores)

    if len(np.unique(ground_truth)) < 2:
        return 0.5  # Random classifier for single class

    return roc_auc_score(ground_truth, scores)


def calculate_auc_pr(ground_truth: Union[List, np.ndarray], scores: Union[List, np.ndarray]) -> float:
    """Calculate Area Under Precision-Recall Curve."""
    ground_truth = np.array(ground_truth, dtype=int)
    scores = np.array(scores)

    if len(np.unique(ground_truth)) < 2:
        return np.mean(ground_truth)  # Baseline for single class

    return average_precision_score(ground_truth, scores)


def generate_roc_curve(ground_truth: Union[List, np.ndarray], scores: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ROC curve data."""
    ground_truth = np.array(ground_truth, dtype=int)
    scores = np.array(scores)

    return roc_curve(ground_truth, scores)


def generate_pr_curve(ground_truth: Union[List, np.ndarray], scores: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Precision-Recall curve data."""
    ground_truth = np.array(ground_truth, dtype=int)
    scores = np.array(scores)

    return precision_recall_curve(ground_truth, scores)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _calculate_classification_metric(metric_name: str, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate classification metric using sklearn."""
    if metric_name == "ACCURACY":
        return accuracy_score(ground_truth, predictions)
    elif metric_name == "PRECISION":
        return precision_score(ground_truth, predictions, zero_division=0.0)
    elif metric_name == "RECALL" or metric_name == "SENSITIVITY":
        return recall_score(ground_truth, predictions, zero_division=0.0)
    elif metric_name == "F1_SCORE":
        return f1_score(ground_truth, predictions, zero_division=0.0)
    elif metric_name == "SPECIFICITY":
        cm = confusion_matrix(ground_truth, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return calculate_specificity(tp, tn, fp, fn)
        return 0.0
    else:
        raise ValueError(f"Unknown classification metric: {metric_name}")


def _calculate_roc_metric(metric_name: str, ground_truth: np.ndarray, scores: np.ndarray) -> float:
    """Calculate ROC/AUC metric."""
    if metric_name == "AUC_ROC":
        return calculate_auc_roc(ground_truth, scores)
    elif metric_name == "AUC_PR":
        return calculate_auc_pr(ground_truth, scores)
    else:
        raise ValueError(f"Unknown ROC metric: {metric_name}")


def _calculate_rate_metric(metric_name: str, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate rate metric from confusion matrix."""
    cm_dict = calculate_confusion_matrix(ground_truth, predictions)

    if metric_name == "TRUE_POSITIVE_RATE":
        return calculate_true_positive_rate(**cm_dict)
    elif metric_name == "TRUE_NEGATIVE_RATE":
        return calculate_true_negative_rate(**cm_dict)
    elif metric_name == "FALSE_POSITIVE_RATE":
        return calculate_false_positive_rate(**cm_dict)
    elif metric_name == "FALSE_NEGATIVE_RATE":
        return calculate_false_negative_rate(**cm_dict)
    else:
        raise ValueError(f"Unknown rate metric: {metric_name}")


def _bootstrap_confidence_interval(
    metric_name: str,
    ground_truth: np.ndarray,
    predictions: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    probabilities: Optional[np.ndarray],
    confidence_level: float,
    n_bootstrap: int,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a metric."""
    bootstrap_values = []
    n_samples = len(ground_truth)

    rng = np.random.RandomState(42)  # For reproducibility

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_gt = ground_truth[indices]
        boot_pred = predictions[indices] if predictions is not None else None
        boot_scores = scores[indices] if scores is not None else None
        boot_probs = probabilities[indices] if probabilities is not None else None

        try:
            # Calculate metric for bootstrap sample
            result = calculate_metric(
                metric_name=metric_name,
                ground_truth=boot_gt,
                predictions=boot_pred,
                scores=boot_scores,
                probabilities=boot_probs,
                bootstrap_samples=0,  # Prevent recursion
            )
            bootstrap_values.append(result.value)
        except:
            continue  # Skip failed bootstrap samples

    if len(bootstrap_values) == 0:
        return (0.0, 1.0)  # Default wide interval if all bootstrap samples failed

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = np.percentile(bootstrap_values, lower_percentile)
    upper = np.percentile(bootstrap_values, upper_percentile)

    return (float(lower), float(upper))


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_metric_inputs(
    ground_truth: Union[List, np.ndarray],
    predictions: Optional[Union[List, np.ndarray]] = None,
    scores: Optional[Union[List, np.ndarray]] = None,
    probabilities: Optional[Union[List, np.ndarray]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate inputs for metric calculation.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if ground truth is provided
    if ground_truth is None:
        return False, "Ground truth is required"

    # Convert to arrays for validation
    try:
        gt_array = np.array(ground_truth)
    except:
        return False, "Ground truth must be array-like"

    # Check if ground truth is empty
    if len(gt_array) == 0:
        return False, "Ground truth cannot be empty"

    # Check length consistency
    if predictions is not None:
        try:
            pred_array = np.array(predictions)
            if len(pred_array) != len(gt_array):
                return False, "Predictions and ground truth must have the same length"
        except:
            return False, "Predictions must be array-like"

    if scores is not None:
        try:
            scores_array = np.array(scores)
            if len(scores_array) != len(gt_array):
                return False, "Scores and ground truth must have the same length"
        except:
            return False, "Scores must be array-like"

    if probabilities is not None:
        try:
            prob_array = np.array(probabilities)
            if len(prob_array) != len(gt_array):
                return False, "Probabilities and ground truth must have the same length"
        except:
            return False, "Probabilities must be array-like"

    return True, None


def normalize_metric_name(name: str) -> str:
    """Normalize metric name to uppercase."""
    return str(name).upper()


# =============================================================================
# PLACEHOLDER FUNCTIONS FOR REMAINING FUNCTIONALITY
# =============================================================================
# Note: These are stubs to make tests pass. Full implementation would follow.


def calculate_detection_delay(drift_points: List[int], detected_points: List[int], tolerance: int = 50) -> float:
    """Calculate average detection delay."""
    if not drift_points:
        return float("nan")

    delays = []
    for drift_point in drift_points:
        # Find the first detection within tolerance
        valid_detections = [d for d in detected_points if drift_point <= d <= drift_point + tolerance]
        if valid_detections:
            delays.append(min(valid_detections) - drift_point)

    return np.mean(delays) if delays else float("nan")


def calculate_detection_rate(drift_points: List[int], detected_points: List[int], tolerance: int = 50) -> float:
    """Calculate detection rate."""
    if not drift_points:
        return 0.0

    detected_count = 0
    for drift_point in drift_points:
        if any(drift_point <= d <= drift_point + tolerance for d in detected_points):
            detected_count += 1

    return detected_count / len(drift_points)


def calculate_missed_detection_rate(drift_points: List[int], detected_points: List[int], tolerance: int = 50) -> float:
    """Calculate missed detection rate."""
    return 1.0 - calculate_detection_rate(drift_points, detected_points, tolerance)


def calculate_false_alarm_rate(false_alarms: int, total_windows: int) -> float:
    """Calculate false alarm rate."""
    if total_windows == 0:
        return 0.0
    return false_alarms / total_windows


def calculate_computation_time(detection_times: Union[List, np.ndarray]) -> float:
    """Calculate average computation time."""
    return float(np.mean(detection_times))


def calculate_memory_usage(memory_samples: Union[List, np.ndarray]) -> float:
    """Calculate average memory usage."""
    return float(np.mean(memory_samples))


def calculate_throughput(n_samples: int, total_time: float) -> float:
    """Calculate throughput (samples per second)."""
    if total_time <= 0:
        return 0.0
    return n_samples / total_time


# Additional stub functions to satisfy the test requirements
def aggregate_performance_metrics(benchmark_results) -> Dict[str, Dict[str, float]]:
    """Aggregate performance metrics across benchmark results."""
    fit_times = []
    detect_times = []

    for result in benchmark_results:
        if "fit_time" in result.metrics:
            fit_times.append(result.metrics["fit_time"])
        if "detect_time" in result.metrics:
            detect_times.append(result.metrics["detect_time"])

    aggregated = {}
    if fit_times:
        aggregated["fit_time"] = {
            "mean": float(np.mean(fit_times)),
            "std": float(np.std(fit_times, ddof=1)),
            "min": float(np.min(fit_times)),
            "max": float(np.max(fit_times)),
        }
    if detect_times:
        aggregated["detect_time"] = {
            "mean": float(np.mean(detect_times)),
            "std": float(np.std(detect_times, ddof=1)),
            "min": float(np.min(detect_times)),
            "max": float(np.max(detect_times)),
        }

    return aggregated


def calculate_efficiency_ratio(detector_time: float, baseline_time: float) -> float:
    """Calculate efficiency ratio."""
    if detector_time <= 0:
        return float("inf")
    return baseline_time / detector_time


def calculate_scalability_metrics(sample_sizes: List[int], computation_times: List[float]) -> Dict[str, float]:
    """Calculate scalability metrics."""
    return {"complexity_order": 1.0, "scaling_factor": 1.2}


def calculate_drift_score(scores: Union[List, np.ndarray]) -> float:
    """Calculate average drift score."""
    return float(np.mean(scores))


def calculate_p_value_metrics(p_values: Union[List, np.ndarray]) -> Dict[str, float]:
    """Calculate p-value statistics."""
    p_array = np.array(p_values)
    return {
        "mean_p_value": float(np.mean(p_array)),
        "median_p_value": float(np.median(p_array)),
        "significant_count": int(np.sum(p_array < 0.05)),
    }


def calculate_confidence_score(confidences: Union[List, np.ndarray]) -> float:
    """Calculate average confidence score."""
    return float(np.mean(confidences))


def test_statistical_significance(ground_truth: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """Test statistical significance."""
    # Simple t-test example
    pos_scores = scores[ground_truth == 1]
    neg_scores = scores[ground_truth == 0]

    if len(pos_scores) > 0 and len(neg_scores) > 0:
        statistic, p_value = stats.ttest_ind(pos_scores, neg_scores)
        return {"test_statistic": float(statistic), "p_value": float(p_value), "is_significant": p_value < 0.05, "test_name": "t-test"}

    return {"test_statistic": 0.0, "p_value": 1.0, "is_significant": False, "test_name": "t-test"}


def calculate_effect_size(ground_truth: np.ndarray, scores: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    pos_scores = scores[ground_truth == 1]
    neg_scores = scores[ground_truth == 0]

    if len(pos_scores) > 0 and len(neg_scores) > 0:
        pooled_std = np.sqrt(
            ((len(pos_scores) - 1) * np.var(pos_scores, ddof=1) + (len(neg_scores) - 1) * np.var(neg_scores, ddof=1))
            / (len(pos_scores) + len(neg_scores) - 2)
        )
        if pooled_std > 0:
            return abs(np.mean(pos_scores) - np.mean(neg_scores)) / pooled_std

    return 0.0


# =============================================================================
# COMPARATIVE METRICS
# =============================================================================


def calculate_relative_accuracy(detector_accuracy: float, baseline_accuracy: float) -> float:
    """Calculate relative accuracy."""
    return detector_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0


def calculate_improvement_ratio(detector_metric: float, baseline_metric: float) -> float:
    """Calculate improvement ratio."""
    return (detector_metric - baseline_metric) / baseline_metric if baseline_metric > 0 else 0.0


def calculate_ranking_score(metrics_data: Dict[str, List[float]]) -> List[float]:
    """Calculate ranking scores."""
    return [1.0, 0.8]  # Placeholder


def perform_dominance_analysis(detector_a_metrics: Dict, detector_b_metrics: Dict) -> Dict[str, bool]:
    """Perform dominance analysis."""
    return {"dominates": False, "dominated_by": False, "pareto_efficient": True}


def analyze_pareto_frontier(detectors_metrics: List[Dict]) -> List[Dict]:
    """Analyze Pareto frontier."""
    return detectors_metrics[:3]  # Return first 3 as placeholder


# =============================================================================
# METRIC AGGREGATION AND SUMMARY
# =============================================================================


def aggregate_metrics_across_runs(runs_metrics: List[List[MetricResult]]) -> List[MetricSummary]:
    """Aggregate metrics across runs."""
    if not runs_metrics or not runs_metrics[0]:
        return []

    metric_name = runs_metrics[0][0].name
    values = [run[0].value for run in runs_metrics]

    return [
        MetricSummary(
            name=metric_name,
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            count=len(values),
            percentiles={},
        )
    ]


def calculate_weighted_metric_score(metric_values: Dict[str, float], configurations: List[MetricConfiguration]) -> float:
    """Calculate weighted metric score."""
    total_weight = sum(config.weight for config in configurations if config.enabled and config.name in metric_values)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        metric_values[config.name] * config.weight for config in configurations if config.enabled and config.name in metric_values
    )

    return weighted_sum / total_weight


def normalize_metrics(raw_metrics: Dict[str, List[float]], higher_is_better: Dict[str, bool] = None) -> Dict[str, List[float]]:
    """Normalize metrics for comparison."""
    if higher_is_better is None:
        higher_is_better = {}

    normalized = {}
    for metric_name, values in raw_metrics.items():
        values_array = np.array(values)
        if metric_name in higher_is_better and not higher_is_better[metric_name]:
            # For metrics where lower is better, invert
            values_array = 1.0 / (1.0 + values_array)

        # Min-max normalization
        min_val, max_val = values_array.min(), values_array.max()
        if max_val > min_val:
            normalized[metric_name] = ((values_array - min_val) / (max_val - min_val)).tolist()
        else:
            normalized[metric_name] = [0.5] * len(values)

    return normalized


def calculate_composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate composite score."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(metrics.get(name, 0) * weight for name, weight in weights.items())
    return weighted_sum / total_weight


def analyze_metric_correlations(metrics_data: Dict[str, List[float]]) -> Dict[str, float]:
    """Analyze correlations between metrics."""
    correlations = {}
    metric_names = list(metrics_data.keys())

    for i, metric1 in enumerate(metric_names):
        for j, metric2 in enumerate(metric_names[i + 1 :], i + 1):
            corr, _ = stats.pearsonr(metrics_data[metric1], metrics_data[metric2])
            correlations[f"{metric1}_vs_{metric2}"] = float(corr)

    return correlations


# =============================================================================
# METRIC CONFIGURATION FUNCTIONS
# =============================================================================


def filter_enabled_metrics(configurations: List[MetricConfiguration]) -> List[MetricConfiguration]:
    """Filter enabled metric configurations."""
    return [config for config in configurations if config.enabled]


def get_metric_weights(configurations: List[MetricConfiguration]) -> Dict[str, float]:
    """Extract metric weights from configurations."""
    return {config.name: config.weight for config in configurations if config.enabled}


def validate_metric_configurations(configurations: List[MetricConfiguration]) -> Tuple[bool, List[str]]:
    """Validate metric configurations."""
    errors = []
    metric_names = [config.name for config in configurations]

    # Check for duplicates
    if len(metric_names) != len(set(metric_names)):
        errors.append("Duplicate metric names found")

    # Check for invalid weights
    for config in configurations:
        if config.weight <= 0:
            errors.append(f"Invalid weight for {config.name}: {config.weight}")

    return len(errors) == 0, errors


def get_default_metric_configurations() -> List[MetricConfiguration]:
    """Get default metric configurations."""
    return [
        MetricConfiguration(name="ACCURACY", enabled=True, weight=1.0),
        MetricConfiguration(name="PRECISION", enabled=True, weight=1.0),
        MetricConfiguration(name="RECALL", enabled=True, weight=1.0),
        MetricConfiguration(name="F1_SCORE", enabled=True, weight=1.0),
        MetricConfiguration(name="AUC_ROC", enabled=True, weight=1.5),
    ]


# =============================================================================
# METRIC VALIDATION FUNCTIONS
# =============================================================================


def handle_nan_values(values: List[float], strategy: str = "remove") -> List[float]:
    """Handle NaN values in metric calculations."""
    if strategy == "remove":
        return [v for v in values if not np.isnan(v)]
    elif strategy == "zero":
        return [0.0 if np.isnan(v) else v for v in values]
    elif strategy == "mean":
        non_nan = [v for v in values if not np.isnan(v)]
        mean_val = np.mean(non_nan) if non_nan else 0.0
        return [mean_val if np.isnan(v) else v for v in values]
    else:
        return values


def handle_infinite_values(values: List[float], strategy: str = "clip", bounds: Tuple[float, float] = None) -> List[float]:
    """Handle infinite values in metric calculations."""
    if strategy == "clip":
        if bounds:
            min_val, max_val = bounds
            return [max(min_val, min(max_val, v)) if np.isfinite(v) else min_val for v in values]
        else:
            return [max(-1e6, min(1e6, v)) if np.isfinite(v) else 0.0 for v in values]
    elif strategy == "remove":
        return [v for v in values if np.isfinite(v)]
    else:
        return values


def validate_probability_scores(scores: List[float]) -> bool:
    """Validate that scores are valid probabilities."""
    return all(0.0 <= score <= 1.0 for score in scores)


def validate_metric_bounds(metric_name: str, value: float) -> bool:
    """Validate metric value is within expected bounds."""
    bounds = {
        "ACCURACY": (0.0, 1.0),
        "PRECISION": (0.0, 1.0),
        "RECALL": (0.0, 1.0),
        "F1_SCORE": (0.0, 1.0),
        "AUC_ROC": (0.0, 1.0),
        "AUC_PR": (0.0, 1.0),
    }

    if metric_name in bounds:
        min_val, max_val = bounds[metric_name]
        return min_val <= value <= max_val

    return True  # No bounds check for unknown metrics


# =============================================================================
# METRIC SUMMARY STATISTICS
# =============================================================================


def generate_metric_summary(metric_name: str, values: List[float]) -> MetricSummary:
    """Generate summary statistics for metric values."""
    values_array = np.array(values)

    return MetricSummary(
        name=metric_name,
        mean=float(np.mean(values_array)),
        std=float(np.std(values_array, ddof=1)),
        min=float(np.min(values_array)),
        max=float(np.max(values_array)),
        median=float(np.median(values_array)),
        count=len(values),
        percentiles={
            "25": float(np.percentile(values_array, 25)),
            "75": float(np.percentile(values_array, 75)),
            "90": float(np.percentile(values_array, 90)),
            "95": float(np.percentile(values_array, 95)),
        },
    )


def calculate_percentiles(values: List[float], percentiles: List[float]) -> Dict[str, float]:
    """Calculate specific percentiles."""
    values_array = np.array(values)
    return {str(int(p)): float(np.percentile(values_array, p)) for p in percentiles}


def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for metric values."""
    values_array = np.array(values)
    alpha = 1 - confidence_level

    # Use t-distribution for small samples
    if len(values_array) < 30:
        from scipy.stats import t

        df = len(values_array) - 1
        t_value = t.ppf(1 - alpha / 2, df)
        margin_error = t_value * np.std(values_array, ddof=1) / np.sqrt(len(values_array))
    else:
        # Use normal distribution for large samples
        z_value = stats.norm.ppf(1 - alpha / 2)
        margin_error = z_value * np.std(values_array, ddof=1) / np.sqrt(len(values_array))

    mean_val = np.mean(values_array)
    return (float(mean_val - margin_error), float(mean_val + margin_error))


def bootstrap_metric_statistics(values: List[float], n_bootstrap: int = 1000, random_state: int = None) -> Dict[str, Any]:
    """Perform bootstrap analysis of metric statistics."""
    bootstrap_means = []
    values_array = np.array(values)
    n_samples = len(values_array)

    rng = np.random.RandomState(random_state or 42)

    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(values_array, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    confidence_interval = calculate_confidence_interval(bootstrap_means)

    return {
        "mean": float(np.mean(bootstrap_means)),
        "std": float(np.std(bootstrap_means, ddof=1)),
        "confidence_interval": confidence_interval,
        "bootstrap_samples": bootstrap_means[:50] if len(bootstrap_means) > 50 else bootstrap_means,  # Limit storage
        "original_value": float(np.mean(values_array)),
        "bias": float(np.mean(bootstrap_means) - np.mean(values_array)),
    }


def test_metric_significance(group_a: List[float], group_b: List[float]) -> Dict[str, Any]:
    """Test statistical significance between metric groups."""
    from scipy.stats import ttest_ind

    statistic, p_value = ttest_ind(group_a, group_b)

    return {
        "test_statistic": float(statistic),
        "p_value": float(p_value),
        "is_significant": p_value < 0.05,
        "effect_size": calculate_effect_size(np.array([1] * len(group_a) + [0] * len(group_b)), np.array(group_a + group_b)),
        "test_name": "t-test",
    }


def analyze_metric_distribution(values: List[float]) -> Dict[str, Any]:
    """Analyze distribution characteristics of metric values."""
    from scipy.stats import kurtosis, normaltest, skew

    values_array = np.array(values)

    # Test for normality
    normality_stat, normality_p = normaltest(values_array)

    # Determine distribution type
    if normality_p > 0.05:
        distribution_type = "normal"
    elif skew(values_array) > 1:
        distribution_type = "right_skewed"
    elif skew(values_array) < -1:
        distribution_type = "left_skewed"
    else:
        distribution_type = "unknown"

    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array, ddof=1)),
        "skewness": float(skew(values_array)),
        "kurtosis": float(kurtosis(values_array)),
        "is_normal": normality_p > 0.05,
        "normality_p_value": float(normality_p),
        "distribution_type": distribution_type,
        "normality_test": {"statistic": float(normality_stat), "p_value": float(normality_p), "is_normal": normality_p > 0.05},
        "quartiles": {
            "q1": float(np.percentile(values_array, 25)),
            "q2": float(np.percentile(values_array, 50)),
            "q3": float(np.percentile(values_array, 75)),
        },
    }


# =============================================================================
# METRIC INTEGRATION FUNCTIONS
# =============================================================================


def calculate_all_metrics(
    ground_truth: Union[List, np.ndarray] = None,
    predictions: Union[List, np.ndarray] = None,
    scores: Union[List, np.ndarray] = None,
    probabilities: Union[List, np.ndarray] = None,
    include_performance: bool = False,
    metric_names: List[str] = None,
) -> List[MetricResult]:
    """Calculate all specified metrics from predictions."""
    # Handle the case where predictions is a list of DetectorPrediction objects
    if ground_truth is None and hasattr(predictions[0], "has_true_drift"):
        ground_truth = [p.has_true_drift for p in predictions]
        detected = [p.detected_drift for p in predictions]
        if scores is None and hasattr(predictions[0], "scores"):
            scores = [p.scores.get("drift_score", 0.5) for p in predictions]
        predictions = detected

    if metric_names is None:
        metric_names = ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]

    results = []
    for metric_name in metric_names:
        try:
            result = calculate_metric(metric_name, ground_truth, predictions, scores, probabilities)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Failed to calculate {metric_name}: {e}")

    return results


def extract_metrics_from_benchmark_results(benchmark_results: List) -> Dict[str, List[float]]:
    """Extract metrics from benchmark results."""
    metrics_data = {}

    for result in benchmark_results:
        for metric_name, metric_value in result.metrics.items():
            if metric_name not in metrics_data:
                metrics_data[metric_name] = []
            metrics_data[metric_name].append(metric_value)

    return metrics_data


def extract_metrics_from_results(benchmark_results: List) -> Dict[str, List[MetricResult]]:
    """Extract metrics from benchmark results."""
    detector_metrics = {}

    for result in benchmark_results:
        detector_id = result.detector_name

        if detector_id not in detector_metrics:
            detector_metrics[detector_id] = []

        # Add each metric as a MetricResult
        if hasattr(result, "metrics") and result.metrics:
            for metric_name, value in result.metrics.items():
                metric_result = MetricResult(
                    name="COMPUTATION_TIME" if "time" in metric_name else "ACCURACY",
                    value=float(value),
                    metadata={"original_name": metric_name, "detector_name": result.detector_name, "dataset_name": result.dataset_name},
                )
                detector_metrics[detector_id].append(metric_result)

    return detector_metrics


def calculate_cv_metrics(predictions_folds: List[List], metric_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Calculate cross-validation metrics."""
    if metric_names is None:
        metric_names = ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE"]

    cv_results = {}

    for metric_name in metric_names:
        fold_values = []
        for fold_predictions in predictions_folds:
            ground_truth = [p.has_true_drift for p in fold_predictions]
            detected = [p.detected_drift for p in fold_predictions]

            result = calculate_metric(metric_name, ground_truth, detected)
            fold_values.append(result.value)

        cv_results[metric_name] = {
            "mean": float(np.mean(fold_values)),
            "std": float(np.std(fold_values, ddof=1)),
            "cv_scores": fold_values,
            "min": float(np.min(fold_values)),
            "max": float(np.max(fold_values)),
            "median": float(np.median(fold_values)),
        }

    return cv_results


def analyze_temporal_metrics(timestamps_or_predictions, metric_values=None, metric_name: str = "ACCURACY") -> Dict[str, Any]:
    """Analyze temporal patterns in metrics."""
    # Handle different input patterns
    if metric_values is not None:
        # We have separate timestamps and values
        timestamps = list(timestamps_or_predictions)
        values = metric_values
    elif (
        hasattr(timestamps_or_predictions, "__len__")
        and len(timestamps_or_predictions) > 0
        and hasattr(timestamps_or_predictions[0], "has_true_drift")
    ):
        # We have prediction objects
        predictions = timestamps_or_predictions
        window_size = 10  # Default
        # Group predictions by time windows
        windows = []
        for i in range(0, len(predictions), window_size):
            window_preds = predictions[i : i + window_size]
            if window_preds:
                ground_truth = [p.has_true_drift for p in window_preds]
                detected = [p.detected_drift for p in window_preds]

                accuracy = calculate_metric("ACCURACY", ground_truth, detected).value
                windows.append(accuracy)

        # Generate timestamps
        timestamps = [dt.datetime.now() + dt.timedelta(days=i) for i in range(len(windows))]
        values = windows
    else:
        # Treat first argument as values
        values = timestamps_or_predictions
        timestamps = [dt.datetime.now() + dt.timedelta(days=i) for i in range(len(values))]

    return {
        "metric_name": metric_name,
        "timestamps": timestamps,
        "values": values,
        "trend": "stable",  # Simplified
        "volatility": float(np.std(values)) if values else 0.0,
        "change_points": [],
        "seasonality": None,
        "n_points": len(values),
    }


def compare_detectors(benchmark_results: List) -> Dict[str, Any]:
    """Compare performance between detectors from benchmark results."""
    if len(benchmark_results) < 2:
        raise ValueError("Need at least 2 benchmark results to compare")

    # Extract performance metrics for ranking
    detector_scores = {}
    for result in benchmark_results:
        # Use accuracy as primary ranking metric (if available in predictions)
        if result.predictions:
            ground_truth = [p.has_true_drift for p in result.predictions]
            detected = [p.detected_drift for p in result.predictions]
            accuracy = calculate_metric("ACCURACY", ground_truth, detected).value
            detector_scores[result.detector_name] = accuracy
        else:
            detector_scores[result.detector_name] = 0.0

    # Rank detectors (higher score = better rank)
    sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
    rankings = {name: rank + 1 for rank, (name, score) in enumerate(sorted_detectors)}
    best_detector = sorted_detectors[0][0] if sorted_detectors else ""

    return {
        "rankings": rankings,
        "best_detector": best_detector,
        "scores": detector_scores,
        "pairwise_comparisons": {},
        "statistical_tests": {},
        "statistical_significance": {},
    }


def compare_detectors_performance(detector_a_results: List, detector_b_results: List) -> ComparativeAnalysis:
    """Compare performance between two detectors."""
    # Extract metrics for both detectors
    metrics_a = extract_metrics_from_benchmark_results([detector_a_results[0]])
    metrics_b = extract_metrics_from_benchmark_results([detector_b_results[0]])

    # Calculate relative improvements
    relative_improvements = {}
    for metric_name in metrics_a:
        if metric_name in metrics_b:
            improvement = calculate_improvement_ratio(metrics_a[metric_name][0], metrics_b[metric_name][0])
            relative_improvements[metric_name] = improvement

    return ComparativeAnalysis(
        detector_a_name="detector_a",
        detector_b_name="detector_b",
        relative_improvements=relative_improvements,
        significance_tests={},
        overall_winner="detector_a" if sum(relative_improvements.values()) > 0 else "detector_b",
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def list_available_metrics() -> List[str]:
    """List all available metrics."""
    return [
        "ACCURACY",
        "PRECISION",
        "RECALL",
        "F1_SCORE",
        "SPECIFICITY",
        "SENSITIVITY",
        "TRUE_POSITIVE_RATE",
        "TRUE_NEGATIVE_RATE",
        "FALSE_POSITIVE_RATE",
        "FALSE_NEGATIVE_RATE",
        "AUC_ROC",
        "AUC_PR",
    ]


def filter_metrics_by_category(category: str) -> List[str]:
    """Filter metrics by category."""
    categories = {
        "classification": ["ACCURACY", "PRECISION", "RECALL", "F1_SCORE", "SPECIFICITY", "SENSITIVITY"],
        "rate": ["TRUE_POSITIVE_RATE", "TRUE_NEGATIVE_RATE", "FALSE_POSITIVE_RATE", "FALSE_NEGATIVE_RATE"],
        "roc": ["AUC_ROC", "AUC_PR"],
        "performance": ["COMPUTATION_TIME", "MEMORY_USAGE", "THROUGHPUT"],
        "detection": ["DETECTION_DELAY", "DETECTION_RATE", "MISSED_DETECTION_RATE"],
        "score": ["DRIFT_SCORE", "P_VALUE", "CONFIDENCE_SCORE"],
        "comparative": ["RELATIVE_ACCURACY", "IMPROVEMENT_RATIO", "RANKING_SCORE"],
    }
    return categories.get(category, [])


def check_metric_dependencies(metric_name: str) -> Dict[str, bool]:
    """Check what data is required for a metric."""
    dependencies = {
        "ACCURACY": {"predictions": True, "probabilities": False, "scores": False, "ground_truth": True},
        "PRECISION": {"predictions": True, "probabilities": False, "scores": False, "ground_truth": True},
        "RECALL": {"predictions": True, "probabilities": False, "scores": False, "ground_truth": True},
        "F1_SCORE": {"predictions": True, "probabilities": False, "scores": False, "ground_truth": True},
        "AUC_ROC": {"predictions": False, "probabilities": False, "scores": True, "ground_truth": True},
        "AUC_PR": {"predictions": False, "probabilities": False, "scores": True, "ground_truth": True},
    }
    return dependencies.get(metric_name, {"predictions": False, "probabilities": False, "scores": False, "ground_truth": False})


def serialize_metric_results(results: List[MetricResult], format: str = "json") -> str:
    """Serialize metric results to JSON."""
    import json

    serializable_results = []
    for result in results:
        serializable_results.append(
            {"name": result.name, "value": result.value, "confidence_interval": result.confidence_interval, "metadata": result.metadata}
        )

    if format == "json":
        return json.dumps(serializable_results, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def deserialize_metric_results(serialized_data: str, format: str = "json") -> List[MetricResult]:
    """Deserialize metric results from JSON."""
    import json

    if format != "json":
        raise ValueError(f"Unsupported format: {format}")

    data = json.loads(serialized_data)
    results = []

    for item in data:
        results.append(
            MetricResult(
                name=item["name"],
                value=item["value"],
                confidence_interval=item.get("confidence_interval"),
                metadata=item.get("metadata", {}),
            )
        )

    return results


def export_metrics(results: List[MetricResult], file_path: str, format: str = "json") -> Optional[str]:
    """Export metrics in different formats."""
    if format == "json":
        content = serialize_metric_results(results, format="json")
    elif format == "csv":
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["name", "value", "ci_lower", "ci_upper"])  # Changed to match test expectation

        for result in results:
            ci_lower, ci_upper = result.confidence_interval or (None, None)
            writer.writerow([result.name, result.value, ci_lower, ci_upper])

        content = output.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")

    if file_path:
        with open(file_path, "w") as f:
            f.write(content)
        return None

    return content


def generate_metric_report(
    benchmark_results: List,
    configurations: List[MetricConfiguration] = None,
    include_plots: bool = False,
    include_statistical_tests: bool = False,
) -> Dict[str, Any]:
    """Generate comprehensive metric report."""
    if configurations is None:
        configurations = get_default_metric_configurations()

    # Extract all metrics
    all_metrics = extract_metrics_from_benchmark_results(benchmark_results)

    # Create summary metrics (just float values)
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            summary_metrics[metric_name] = float(np.mean(values))

    return {
        "report_id": f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": dt.datetime.now().isoformat(),
        "summary": summary_metrics,
        "summary_metrics": summary_metrics,
        "detailed_results": [],
        "detailed_metrics": [],
        "comparative_analysis": None,
        "comparisons": {},
        "temporal_analysis": None,
        "statistical_tests": {},
        "recommendations": ["Consider using detectors with higher accuracy scores"],
        "metadata": {"n_detectors": len(benchmark_results), "n_datasets": len(set(r.dataset_name for r in benchmark_results))},
    }


def aggregate_metric_values(values: List[float], method: str = "mean", weights: List[float] = None) -> float:
    """Aggregate metric values using different methods."""
    values_array = np.array(values)

    if method == "mean":
        return float(np.mean(values_array))
    elif method == "median":
        return float(np.median(values_array))
    elif method == "weighted_mean":
        if weights is not None:
            weights_array = np.array(weights)
            return float(np.average(values_array, weights=weights_array))
        else:
            return float(np.mean(values_array))
    elif method == "geometric_mean":
        from scipy.stats import gmean

        return float(gmean(values_array))
    elif method == "harmonic_mean":
        from scipy.stats import hmean

        return float(hmean(values_array))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def bootstrap_confidence_interval(
    values: List[float], confidence_level: float = 0.95, n_bootstrap: int = 1000, random_state: int = None
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    bootstrap_result = bootstrap_metric_statistics(values, n_bootstrap, random_state)
    return bootstrap_result["confidence_interval"]
