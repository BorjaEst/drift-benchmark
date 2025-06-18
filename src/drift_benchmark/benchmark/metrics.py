"""
Evaluation metrics for drift detection benchmarks.

This module provides functions and data structures for computing and analyzing
the performance of drift detection algorithms. It includes metrics such as
detection delay, false positive/negative rates, F1-score, and computation time.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn import metrics as skmetrics


class DetectionResult(Enum):
    """Possible outcomes of drift detection."""

    TRUE_POSITIVE = "true_positive"  # Correctly detected drift
    TRUE_NEGATIVE = "true_negative"  # Correctly identified no drift
    FALSE_POSITIVE = "false_positive"  # Incorrectly detected drift (Type I error)
    FALSE_NEGATIVE = "false_negative"  # Missed drift (Type II error)


@dataclass
class DetectorPrediction:
    """Single prediction by a drift detector."""

    # Input data identifiers
    dataset_name: str
    window_id: int

    # True drift status
    has_true_drift: bool

    # Detector prediction
    detected_drift: bool

    # Timing information
    detection_time: float = 0.0  # in seconds

    # Additional metrics
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def result(self) -> DetectionResult:
        """Get the classification of this detection result."""
        if self.has_true_drift and self.detected_drift:
            return DetectionResult.TRUE_POSITIVE
        elif not self.has_true_drift and not self.detected_drift:
            return DetectionResult.TRUE_NEGATIVE
        elif not self.has_true_drift and self.detected_drift:
            return DetectionResult.FALSE_POSITIVE
        else:  # has_true_drift and not detected_drift
            return DetectionResult.FALSE_NEGATIVE


@dataclass
class BenchmarkResult:
    """Results for a single detector on a single dataset."""

    detector_name: str
    detector_params: Dict[str, Any]
    dataset_name: str
    dataset_params: Dict[str, Any]

    # Collection of all predictions
    predictions: List[DetectorPrediction] = field(default_factory=list)

    # Aggregated metrics computed from predictions
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_prediction(self, prediction: DetectorPrediction) -> None:
        """Add a new prediction to the results."""
        self.predictions.append(prediction)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics based on stored predictions."""
        if not self.predictions:
            return {}

        # Extract binary classification values
        y_true = np.array([p.has_true_drift for p in self.predictions])
        y_pred = np.array([p.detected_drift for p in self.predictions])

        # Basic counts
        tp = sum(1 for p in self.predictions if p.result == DetectionResult.TRUE_POSITIVE)
        tn = sum(1 for p in self.predictions if p.result == DetectionResult.TRUE_NEGATIVE)
        fp = sum(1 for p in self.predictions if p.result == DetectionResult.FALSE_POSITIVE)
        fn = sum(1 for p in self.predictions if p.result == DetectionResult.FALSE_NEGATIVE)

        # Avoid division by zero
        total_positives = tp + fn
        total_negatives = tn + fp
        predicted_positives = tp + fp

        # Calculate metrics
        self.metrics = {
            "accuracy": float((tp + tn) / len(self.predictions)) if self.predictions else 0.0,
            "precision": float(tp / predicted_positives) if predicted_positives > 0 else 0.0,
            "recall": float(tp / total_positives) if total_positives > 0 else 0.0,
            "f1_score": float(calculate_f1_score(tp, fp, fn)),
            "false_positive_rate": float(fp / total_negatives) if total_negatives > 0 else 0.0,
            "false_negative_rate": float(fn / total_positives) if total_positives > 0 else 0.0,
            "true_positive_rate": float(tp / total_positives) if total_positives > 0 else 0.0,
            "true_negative_rate": float(tn / total_negatives) if total_negatives > 0 else 0.0,
            "computation_time": float(np.mean([p.detection_time for p in self.predictions])),
            "detection_delay": float(calculate_detection_delay(self.predictions)),
        }

        # If there are enough samples and varied predictions, calculate AUC
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            try:
                self.metrics["auc_roc"] = float(skmetrics.roc_auc_score(y_true, y_pred))

                # Store ROC curve points for visualization
                fpr, tpr, thresholds = skmetrics.roc_curve(y_true, y_pred)
                self._roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
            except ValueError:
                # Not enough distinct values or other issue
                self.metrics["auc_roc"] = 0.0

        return self.metrics

    def get_roc_curve_data(self) -> Dict[str, List[float]]:
        """Return ROC curve data points if available."""
        if hasattr(self, "_roc_data"):
            return self._roc_data
        return {"fpr": [], "tpr": [], "thresholds": []}


@dataclass
class DriftEvaluationResult:
    """Overall benchmark results for multiple detectors and datasets."""

    # Individual benchmark results
    results: List[BenchmarkResult] = field(default_factory=list)

    # Settings used for the benchmark
    settings: Dict[str, Any] = field(default_factory=dict)

    # Overall ranking summary
    rankings: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add an individual benchmark result."""
        self.results.append(result)

    def get_results_for_detector(self, detector_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific detector."""
        return [res for res in self.results if res.detector_name == detector_name]

    def get_results_for_dataset(self, dataset_name: str) -> List[BenchmarkResult]:
        """Get all results for a specific dataset."""
        return [res for res in self.results if res.dataset_name == dataset_name]

    def compute_detector_rankings(self, metrics: List[str] = None) -> Dict[str, Dict[str, int]]:
        """Compute rankings of detectors across specified metrics."""
        if not metrics:
            metrics = ["f1_score", "detection_delay", "false_positive_rate", "computation_time"]

        all_detectors = {r.detector_name for r in self.results}
        datasets = {r.dataset_name for r in self.results}

        rankings = {}

        # Rank by metric for each dataset
        for metric in metrics:
            metric_rankings = {}

            for dataset in datasets:
                dataset_results = self.get_results_for_dataset(dataset)

                # Skip if no results have this metric
                if not any(metric in r.metrics for r in dataset_results):
                    continue

                # Sort by metric (higher is better for most metrics, lower for some)
                reverse = metric not in [
                    "detection_delay",
                    "false_positive_rate",
                    "false_negative_rate",
                    "computation_time",
                ]

                sorted_results = sorted(
                    [r for r in dataset_results if metric in r.metrics],
                    key=lambda r: r.metrics[metric],
                    reverse=reverse,
                )

                # Assign ranks (1 = best)
                for i, result in enumerate(sorted_results):
                    detector = result.detector_name
                    if detector not in metric_rankings:
                        metric_rankings[detector] = []
                    metric_rankings[detector].append(i + 1)

            # Calculate average rank across datasets
            avg_rankings = {}
            for detector, ranks in metric_rankings.items():
                avg_rankings[detector] = sum(ranks) / len(ranks)

            rankings[metric] = avg_rankings

        self.rankings = rankings
        return rankings

    def get_best_detector(self, metric: str = "f1_score") -> str:
        """Identify the best detector for a given metric based on average ranking."""
        if not self.rankings:
            self.compute_detector_rankings()

        if metric not in self.rankings:
            raise ValueError(f"Metric '{metric}' not found in rankings")

        metric_rankings = self.rankings[metric]
        return min(metric_rankings.items(), key=lambda x: x[1])[0]

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of benchmark results."""
        if not self.rankings:
            self.compute_detector_rankings()

        detectors = {r.detector_name for r in self.results}
        datasets = {r.dataset_name for r in self.results}

        # Aggregate metrics across all tests
        summary = {
            "datasets": list(datasets),
            "detectors": list(detectors),
            "metrics_available": list(set().union(*(r.metrics.keys() for r in self.results))),
            "rankings": self.rankings,
            "best_overall": {metric: self.get_best_detector(metric) for metric in self.rankings.keys()},
        }

        # Add dataset-specific summaries
        summary["dataset_summaries"] = {}
        for dataset in datasets:
            dataset_results = self.get_results_for_dataset(dataset)

            # Get best detector for each metric on this dataset
            best_by_metric = {}
            for metric in summary["metrics_available"]:
                valid_results = [r for r in dataset_results if metric in r.metrics]
                if not valid_results:
                    continue

                # Higher is better for most metrics except a few
                reverse = metric not in [
                    "detection_delay",
                    "false_positive_rate",
                    "false_negative_rate",
                    "computation_time",
                ]

                best = (
                    max(valid_results, key=lambda r: r.metrics.get(metric, float("-inf")), default=None)
                    if reverse
                    else min(valid_results, key=lambda r: r.metrics.get(metric, float("inf")), default=None)
                )

                if best:
                    best_by_metric[metric] = {"detector": best.detector_name, "value": best.metrics[metric]}

            summary["dataset_summaries"][dataset] = best_by_metric

        return summary


def calculate_f1_score(tp: int, fp: int, fn: int) -> float:
    """
    Calculate F1 score.

    Args:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives

    Returns:
        F1 score (harmonic mean of precision and recall)
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision == 0 and recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_detection_delay(predictions: List[DetectorPrediction]) -> float:
    """
    Calculate the average delay in detecting true drift.

    Args:
        predictions: List of detector predictions

    Returns:
        Average detection delay in number of windows
    """
    # Group predictions by dataset
    dataset_predictions = {}
    for pred in predictions:
        if pred.dataset_name not in dataset_predictions:
            dataset_predictions[pred.dataset_name] = []
        dataset_predictions[pred.dataset_name].append(pred)

    delays = []

    for dataset_name, preds in dataset_predictions.items():
        # Sort predictions by window ID
        sorted_preds = sorted(preds, key=lambda p: p.window_id)

        # Find first window with true drift
        first_drift = next((i for i, p in enumerate(sorted_preds) if p.has_true_drift), None)
        if first_drift is None:
            continue

        # Find first detection after the drift
        first_detection = next((i for i, p in enumerate(sorted_preds[first_drift:]) if p.detected_drift), None)

        if first_detection is not None:
            delays.append(first_detection)  # Number of windows until detection

    return np.mean(delays) if delays else 0.0


def time_execution(func, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure execution time of a function.

    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (function result, execution time in seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return result, end_time - start_time


def generate_binary_drift_vector(length: int, drift_points: List[int], drift_width: int = 1) -> np.ndarray:
    """
    Generate a binary vector indicating where drift occurs.

    Args:
        length: Total length of the vector
        drift_points: Indices where drift begins
        drift_width: Width of each drift region

    Returns:
        Binary numpy array with 1 at drift positions
    """
    drift_vector = np.zeros(length, dtype=int)

    for point in drift_points:
        if 0 <= point < length:
            for i in range(drift_width):
                idx = point + i
                if idx < length:
                    drift_vector[idx] = 1

    return drift_vector


def compute_confusion_matrix(predictions: List[DetectorPrediction]) -> Dict[str, int]:
    """
    Compute confusion matrix from detector predictions.

    Args:
        predictions: List of detector predictions

    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    tp = sum(1 for p in predictions if p.result == DetectionResult.TRUE_POSITIVE)
    tn = sum(1 for p in predictions if p.result == DetectionResult.TRUE_NEGATIVE)
    fp = sum(1 for p in predictions if p.result == DetectionResult.FALSE_POSITIVE)
    fn = sum(1 for p in predictions if p.result == DetectionResult.FALSE_NEGATIVE)

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total": len(predictions),
    }


# Example usage when running this module directly
if __name__ == "__main__":
    import random

    # Create some sample predictions
    predictions = []
    dataset = "synthetic_test"

    # Generate some random predictions
    for i in range(50):
        has_drift = i >= 25  # True drift in second half
        detected = random.random() > 0.2 if has_drift else random.random() < 0.1

        pred = DetectorPrediction(
            dataset_name=dataset,
            window_id=i,
            has_true_drift=has_drift,
            detected_drift=detected,
            detection_time=random.uniform(0.01, 0.2),
            scores={"p_value": random.random()},
        )
        predictions.append(pred)

    # Create a benchmark result
    result = BenchmarkResult(
        detector_name="TestDetector",
        detector_params={"threshold": 0.05},
        dataset_name=dataset,
        dataset_params={"drift_type": "sudden"},
    )

    # Add predictions and compute metrics
    for pred in predictions:
        result.add_prediction(pred)

    metrics = result.compute_metrics()
    print("Computed metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Create overall evaluation result
    eval_result = DriftEvaluationResult()
    eval_result.add_result(result)

    # Add another detector for comparison
    result2 = BenchmarkResult(
        detector_name="AnotherDetector",
        detector_params={"threshold": 0.01},
        dataset_name=dataset,
        dataset_params={"drift_type": "sudden"},
    )

    # Add predictions with different performance
    for i in range(50):
        has_drift = i >= 25  # True drift in second half
        detected = random.random() > 0.4 if has_drift else random.random() < 0.2

        pred = DetectorPrediction(
            dataset_name=dataset,
            window_id=i,
            has_true_drift=has_drift,
            detected_drift=detected,
            detection_time=random.uniform(0.05, 0.3),
            scores={"p_value": random.random()},
        )
        result2.add_prediction(pred)

    result2.compute_metrics()
    eval_result.add_result(result2)

    # Compute rankings
    rankings = eval_result.compute_detector_rankings()
    print("\nDetector Rankings:")
    for metric, detectors in rankings.items():
        print(f"  {metric}:")
        for detector, rank in sorted(detectors.items(), key=lambda x: x[1]):
            print(f"    {detector}: {rank:.2f}")

    # Get overall summary
    summary = eval_result.summary()
    print("\nBest detector by metric:")
    for metric, detector in summary["best_overall"].items():
        print(f"  {metric}: {detector}")
