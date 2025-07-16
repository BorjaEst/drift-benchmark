"""
Tests for results module.

This module contains comprehensive tests for the results.py module,
which provides result formatting, serialization, and parsing functionality
for drift detection benchmark results.

Test Classes Organization:
- TestResultFormatting: Core result formatting and serialization
- TestResultParsing: Result loading and deserialization
- TestFileOperations: File I/O operations for results
- TestResultExporting: Export functionality to various formats
- TestResultImporting: Import functionality from various formats
- TestResultValidation: Result validation and error handling
- TestResultAggregation: Result aggregation and summarization
- TestResultFiltering: Result filtering and querying
- TestResultComparison: Result comparison and analysis
- TestResultReporting: Report generation functionality
- TestResultSerialization: Serialization format handling
- TestResultCompression: Compression and optimization
- TestResultVersioning: Version compatibility handling
- TestResultIntegration: Integration tests with real data
- TestResultUtilities: Utility functions and helpers
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.constants.literals import DetectionResult, ExportFormat, FileFormat, Metric
from drift_benchmark.constants.models import (
    BenchmarkResult,
    ComparativeAnalysis,
    ConfusionMatrix,
    DetectorPrediction,
    DriftEvaluationResult,
    MetricReport,
    MetricResult,
    MetricSummary,
    TemporalMetrics,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_detector_predictions():
    """Create sample detector predictions for testing."""
    return [
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=0,
            has_true_drift=False,
            detected_drift=False,
            detection_time=0.001,
            scores={"confidence": 0.2, "statistic": 0.15},
        ),
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=1,
            has_true_drift=False,
            detected_drift=True,
            detection_time=0.002,
            scores={"confidence": 0.8, "statistic": 0.75},
        ),
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=2,
            has_true_drift=True,
            detected_drift=True,
            detection_time=0.0015,
            scores={"confidence": 0.9, "statistic": 0.85},
        ),
        DetectorPrediction(
            dataset_name="test_dataset",
            window_id=3,
            has_true_drift=True,
            detected_drift=False,
            detection_time=0.0012,
            scores={"confidence": 0.3, "statistic": 0.25},
        ),
    ]


@pytest.fixture
def sample_benchmark_result(sample_detector_predictions):
    """Create a sample benchmark result for testing."""
    result = BenchmarkResult(
        detector_name="test_detector",
        dataset_name="test_dataset",
        detector_params={"threshold": 0.5, "window_size": 100},
        dataset_params={"drift_type": "COVARIATE", "drift_magnitude": 1.0},
        predictions=sample_detector_predictions,
        metrics={
            "accuracy": 0.75,
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
            "specificity": 0.5,
        },
        roc_data={"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.5, 1.0], "thresholds": [1.0, 0.5, 0.0]},
    )
    return result


@pytest.fixture
def sample_drift_evaluation_result(sample_benchmark_result):
    """Create a sample drift evaluation result for testing."""
    return DriftEvaluationResult(
        results=[sample_benchmark_result],
        settings={"n_runs": 5, "random_seed": 42},
        rankings={"accuracy": {"test_detector": 1}},
        statistical_summaries={"test_detector": {"mean_accuracy": 0.75, "std_accuracy": 0.05}},
        best_performers={"accuracy": "test_detector"},
    )


@pytest.fixture
def sample_metric_results():
    """Create sample metric results for testing."""
    return [
        MetricResult(
            name="ACCURACY",
            value=0.85,
            confidence_interval=(0.80, 0.90),
            metadata={"n_samples": 1000, "bootstrap_samples": 100},
        ),
        MetricResult(
            name="PRECISION",
            value=0.78,
            confidence_interval=(0.72, 0.84),
            metadata={"n_samples": 1000, "bootstrap_samples": 100},
        ),
        MetricResult(
            name="RECALL",
            value=0.82,
            confidence_interval=(0.77, 0.87),
            metadata={"n_samples": 1000, "bootstrap_samples": 100},
        ),
    ]


@pytest.fixture
def sample_metric_summaries():
    """Create sample metric summaries for testing."""
    return [
        MetricSummary(
            name="ACCURACY",
            mean=0.85,
            std=0.05,
            min=0.75,
            max=0.95,
            median=0.84,
            count=10,
            percentiles={"25": 0.82, "75": 0.88, "90": 0.92, "95": 0.94},
        ),
        MetricSummary(
            name="F1_SCORE",
            mean=0.78,
            std=0.06,
            min=0.68,
            max=0.88,
            median=0.77,
            count=10,
            percentiles={"25": 0.74, "75": 0.82, "90": 0.85, "95": 0.87},
        ),
    ]


@pytest.fixture
def sample_confusion_matrix():
    """Create a sample confusion matrix for testing."""
    return ConfusionMatrix(
        true_positives=50,
        true_negatives=40,
        false_positives=5,
        false_negatives=5,
    )


@pytest.fixture
def sample_metric_report(sample_metric_results):
    """Create a sample metric report for testing."""
    return MetricReport(
        report_id="test_report_001",
        summary_metrics={"overall_accuracy": 0.85, "mean_f1": 0.78},
        detailed_results=sample_metric_results,
        recommendations=["Consider using ensemble methods", "Increase training data"],
        metadata={"benchmark_version": "1.0", "execution_time": 3600},
    )


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_multiple_benchmark_results(sample_detector_predictions):
    """Create multiple benchmark results for testing."""
    results = []
    detectors = ["detector_a", "detector_b", "detector_c"]
    datasets = ["dataset_1", "dataset_2"]

    for detector in detectors:
        for dataset in datasets:
            result = BenchmarkResult(
                detector_name=detector,
                dataset_name=dataset,
                detector_params={"threshold": 0.5},
                dataset_params={"drift_type": "COVARIATE"},
                predictions=sample_detector_predictions.copy(),
                metrics={
                    "accuracy": np.random.uniform(0.7, 0.9),
                    "precision": np.random.uniform(0.6, 0.8),
                    "recall": np.random.uniform(0.6, 0.8),
                },
            )
            results.append(result)
    return results


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestResultFormatting:
    """Test core result formatting and serialization functionality."""

    def test_format_detector_prediction_to_dict(self, sample_detector_predictions):
        """Test formatting detector prediction to dictionary."""
        from drift_benchmark.results import format_prediction_to_dict

        prediction = sample_detector_predictions[0]
        formatted = format_prediction_to_dict(prediction)

        assert isinstance(formatted, dict)
        assert formatted["dataset_name"] == "test_dataset"
        assert formatted["window_id"] == 0
        assert formatted["has_true_drift"] is False
        assert formatted["detected_drift"] is False
        assert formatted["detection_time"] == 0.001
        assert "scores" in formatted
        assert formatted["result"] == "TRUE_NEGATIVE"

    def test_format_benchmark_result_to_dict(self, sample_benchmark_result):
        """Test formatting benchmark result to dictionary."""
        from drift_benchmark.results import format_benchmark_result_to_dict

        formatted = format_benchmark_result_to_dict(sample_benchmark_result)

        assert isinstance(formatted, dict)
        assert formatted["detector_name"] == "test_detector"
        assert formatted["dataset_name"] == "test_dataset"
        assert "detector_params" in formatted
        assert "dataset_params" in formatted
        assert "predictions" in formatted
        assert "metrics" in formatted
        assert len(formatted["predictions"]) == 4

    def test_format_drift_evaluation_result_to_dict(self, sample_drift_evaluation_result):
        """Test formatting drift evaluation result to dictionary."""
        from drift_benchmark.results import format_evaluation_result_to_dict

        formatted = format_evaluation_result_to_dict(sample_drift_evaluation_result)

        assert isinstance(formatted, dict)
        assert "results" in formatted
        assert "settings" in formatted
        assert "rankings" in formatted
        assert "statistical_summaries" in formatted
        assert "best_performers" in formatted

    def test_format_metric_result_to_dict(self, sample_metric_results):
        """Test formatting metric result to dictionary."""
        from drift_benchmark.results import format_metric_result_to_dict

        metric_result = sample_metric_results[0]
        formatted = format_metric_result_to_dict(metric_result)

        assert isinstance(formatted, dict)
        assert formatted["name"] == "ACCURACY"
        assert formatted["value"] == 0.85
        assert "confidence_interval" in formatted
        assert "metadata" in formatted

    def test_format_with_custom_serializer(self, sample_benchmark_result):
        """Test formatting with custom serialization options."""
        from drift_benchmark.results import format_benchmark_result_to_dict

        formatted = format_benchmark_result_to_dict(
            sample_benchmark_result,
            include_raw_predictions=False,
            decimal_places=3,
        )

        assert "predictions" not in formatted
        # Check that metrics are rounded to 3 decimal places
        for metric_value in formatted["metrics"].values():
            if isinstance(metric_value, float):
                assert len(str(metric_value).split(".")[-1]) <= 3

    def test_format_hierarchical_structure(self, sample_drift_evaluation_result):
        """Test formatting results with hierarchical structure preservation."""
        from drift_benchmark.results import format_evaluation_result_to_dict

        formatted = format_evaluation_result_to_dict(
            sample_drift_evaluation_result,
            preserve_hierarchy=True,
        )

        assert "results" in formatted
        assert isinstance(formatted["results"], list)
        assert len(formatted["results"]) == 1
        assert "predictions" in formatted["results"][0]


class TestResultParsing:
    """Test result loading and deserialization functionality."""

    def test_parse_detector_prediction_from_dict(self):
        """Test parsing detector prediction from dictionary."""
        from drift_benchmark.results import parse_prediction_from_dict

        prediction_dict = {
            "dataset_name": "test_dataset",
            "window_id": 0,
            "has_true_drift": False,
            "detected_drift": False,
            "detection_time": 0.001,
            "scores": {"confidence": 0.2},
        }

        prediction = parse_prediction_from_dict(prediction_dict)

        assert isinstance(prediction, DetectorPrediction)
        assert prediction.dataset_name == "test_dataset"
        assert prediction.window_id == 0
        assert prediction.has_true_drift is False
        assert prediction.detected_drift is False
        assert prediction.detection_time == 0.001

    def test_parse_benchmark_result_from_dict(self):
        """Test parsing benchmark result from dictionary."""
        from drift_benchmark.results import parse_benchmark_result_from_dict

        result_dict = {
            "detector_name": "test_detector",
            "dataset_name": "test_dataset",
            "detector_params": {"threshold": 0.5},
            "dataset_params": {"drift_type": "COVARIATE"},
            "predictions": [
                {
                    "dataset_name": "test_dataset",
                    "window_id": 0,
                    "has_true_drift": False,
                    "detected_drift": False,
                    "detection_time": 0.001,
                    "scores": {},
                }
            ],
            "metrics": {"accuracy": 0.75},
        }

        result = parse_benchmark_result_from_dict(result_dict)

        assert isinstance(result, BenchmarkResult)
        assert result.detector_name == "test_detector"
        assert result.dataset_name == "test_dataset"
        assert len(result.predictions) == 1

    def test_parse_drift_evaluation_result_from_dict(self):
        """Test parsing drift evaluation result from dictionary."""
        from drift_benchmark.results import parse_evaluation_result_from_dict

        evaluation_dict = {
            "results": [
                {
                    "detector_name": "test_detector",
                    "dataset_name": "test_dataset",
                    "detector_params": {},
                    "dataset_params": {},
                    "predictions": [],
                    "metrics": {},
                }
            ],
            "settings": {"n_runs": 5},
            "rankings": {},
            "statistical_summaries": {},
            "best_performers": {},
        }

        evaluation = parse_evaluation_result_from_dict(evaluation_dict)

        assert isinstance(evaluation, DriftEvaluationResult)
        assert len(evaluation.results) == 1

    def test_parse_with_validation_errors(self):
        """Test parsing with validation errors."""
        from drift_benchmark.results import parse_prediction_from_dict

        invalid_dict = {
            "dataset_name": "test_dataset",
            # Missing required fields
            "detection_time": "invalid_type",  # Wrong type
        }

        with pytest.raises(ValueError):
            parse_prediction_from_dict(invalid_dict)

    def test_parse_with_missing_optional_fields(self):
        """Test parsing with missing optional fields uses defaults."""
        from drift_benchmark.results import parse_prediction_from_dict

        minimal_dict = {
            "dataset_name": "test_dataset",
            "window_id": 0,
            "has_true_drift": False,
            "detected_drift": False,
        }

        prediction = parse_prediction_from_dict(minimal_dict)

        assert prediction.detection_time == 0.0  # Default value
        assert prediction.scores == {}  # Default empty dict


class TestFileOperations:
    """Test file I/O operations for results."""

    def test_save_benchmark_result_to_file(self, sample_benchmark_result, temp_results_dir):
        """Test saving benchmark result to file."""
        from drift_benchmark.results import save_benchmark_result

        file_path = temp_results_dir / "benchmark_result.json"
        save_benchmark_result(sample_benchmark_result, file_path)

        assert file_path.exists()
        assert file_path.stat().st_size > 0

        # Verify content can be loaded back
        with open(file_path, "r") as f:
            data = json.load(f)
            assert data["detector_name"] == "test_detector"

    def test_load_benchmark_result_from_file(self, sample_benchmark_result, temp_results_dir):
        """Test loading benchmark result from file."""
        from drift_benchmark.results import load_benchmark_result, save_benchmark_result

        file_path = temp_results_dir / "benchmark_result.json"
        save_benchmark_result(sample_benchmark_result, file_path)

        loaded_result = load_benchmark_result(file_path)

        assert isinstance(loaded_result, BenchmarkResult)
        assert loaded_result.detector_name == sample_benchmark_result.detector_name
        assert loaded_result.dataset_name == sample_benchmark_result.dataset_name
        assert len(loaded_result.predictions) == len(sample_benchmark_result.predictions)

    def test_save_drift_evaluation_result_to_file(self, sample_drift_evaluation_result, temp_results_dir):
        """Test saving drift evaluation result to file."""
        from drift_benchmark.results import save_evaluation_result

        file_path = temp_results_dir / "evaluation_result.json"
        save_evaluation_result(sample_drift_evaluation_result, file_path)

        assert file_path.exists()
        with open(file_path, "r") as f:
            data = json.load(f)
            assert "results" in data
            assert "settings" in data

    def test_load_drift_evaluation_result_from_file(self, sample_drift_evaluation_result, temp_results_dir):
        """Test loading drift evaluation result from file."""
        from drift_benchmark.results import load_evaluation_result, save_evaluation_result

        file_path = temp_results_dir / "evaluation_result.json"
        save_evaluation_result(sample_drift_evaluation_result, file_path)

        loaded_result = load_evaluation_result(file_path)

        assert isinstance(loaded_result, DriftEvaluationResult)
        assert len(loaded_result.results) == len(sample_drift_evaluation_result.results)

    def test_file_operations_with_compression(self, sample_benchmark_result, temp_results_dir):
        """Test file operations with compression."""
        from drift_benchmark.results import load_benchmark_result, save_benchmark_result

        file_path = temp_results_dir / "compressed_result.json.gz"
        save_benchmark_result(sample_benchmark_result, file_path, compress=True)

        assert file_path.exists()
        loaded_result = load_benchmark_result(file_path)
        assert loaded_result.detector_name == sample_benchmark_result.detector_name

    def test_file_operations_error_handling(self, temp_results_dir):
        """Test file operations error handling."""
        from drift_benchmark.results import load_benchmark_result

        non_existent_path = temp_results_dir / "non_existent.json"

        with pytest.raises(FileNotFoundError):
            load_benchmark_result(non_existent_path)

    def test_file_operations_with_invalid_content(self, temp_results_dir):
        """Test file operations with invalid content."""
        from drift_benchmark.results import load_benchmark_result

        invalid_file = temp_results_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_benchmark_result(invalid_file)


class TestResultExporting:
    """Test export functionality to various formats."""

    @pytest.mark.parametrize("export_format", ["JSON", "CSV", "PICKLE", "EXCEL"])
    def test_export_benchmark_result_to_format(self, sample_benchmark_result, temp_results_dir, export_format):
        """Test exporting benchmark result to different formats."""
        from drift_benchmark.results import export_benchmark_result

        file_path = temp_results_dir / f"result.{export_format.lower()}"
        export_benchmark_result(sample_benchmark_result, file_path, format=export_format)

        assert file_path.exists()

    @pytest.mark.parametrize("export_format", ["JSON", "CSV", "PICKLE"])
    def test_export_drift_evaluation_result_to_format(self, sample_drift_evaluation_result, temp_results_dir, export_format):
        """Test exporting drift evaluation result to different formats."""
        from drift_benchmark.results import export_evaluation_result

        file_path = temp_results_dir / f"evaluation.{export_format.lower()}"
        export_evaluation_result(sample_drift_evaluation_result, file_path, format=export_format)

        assert file_path.exists()

    def test_export_predictions_to_csv(self, sample_detector_predictions, temp_results_dir):
        """Test exporting predictions to CSV format."""
        from drift_benchmark.results import export_predictions_to_csv

        file_path = temp_results_dir / "predictions.csv"
        export_predictions_to_csv(sample_detector_predictions, file_path)

        assert file_path.exists()

        # Verify CSV content
        df = pd.read_csv(file_path)
        assert len(df) == len(sample_detector_predictions)
        assert "dataset_name" in df.columns
        assert "detected_drift" in df.columns

    def test_export_metrics_to_csv(self, sample_multiple_benchmark_results, temp_results_dir):
        """Test exporting metrics to CSV format."""
        from drift_benchmark.results import export_metrics_to_csv

        file_path = temp_results_dir / "metrics.csv"
        export_metrics_to_csv(sample_multiple_benchmark_results, file_path)

        assert file_path.exists()

        # Verify CSV content
        df = pd.read_csv(file_path)
        assert "detector_name" in df.columns
        assert "dataset_name" in df.columns
        assert "accuracy" in df.columns

    def test_export_summary_report(self, sample_drift_evaluation_result, temp_results_dir):
        """Test exporting summary report."""
        from drift_benchmark.results import export_summary_report

        file_path = temp_results_dir / "summary_report.json"
        export_summary_report(sample_drift_evaluation_result, file_path)

        assert file_path.exists()

        with open(file_path, "r") as f:
            report = json.load(f)
            assert "summary" in report
            assert "best_performers" in report
            assert "statistical_summaries" in report

    def test_export_with_custom_metadata(self, sample_benchmark_result, temp_results_dir):
        """Test exporting with custom metadata."""
        from drift_benchmark.results import export_benchmark_result

        metadata = {
            "export_timestamp": "2023-07-16T10:00:00Z",
            "exported_by": "test_user",
            "version": "1.0",
        }

        file_path = temp_results_dir / "result_with_metadata.json"
        export_benchmark_result(
            sample_benchmark_result,
            file_path,
            format="JSON",
            metadata=metadata,
        )

        with open(file_path, "r") as f:
            data = json.load(f)
            assert "metadata" in data
            assert data["metadata"]["exported_by"] == "test_user"


class TestResultImporting:
    """Test import functionality from various formats."""

    def test_import_benchmark_result_from_json(self, sample_benchmark_result, temp_results_dir):
        """Test importing benchmark result from JSON."""
        from drift_benchmark.results import export_benchmark_result, import_benchmark_result

        file_path = temp_results_dir / "result.json"
        export_benchmark_result(sample_benchmark_result, file_path, format="JSON")

        imported_result = import_benchmark_result(file_path)

        assert isinstance(imported_result, BenchmarkResult)
        assert imported_result.detector_name == sample_benchmark_result.detector_name

    def test_import_predictions_from_csv(self, sample_detector_predictions, temp_results_dir):
        """Test importing predictions from CSV."""
        from drift_benchmark.results import export_predictions_to_csv, import_predictions_from_csv

        file_path = temp_results_dir / "predictions.csv"
        export_predictions_to_csv(sample_detector_predictions, file_path)

        imported_predictions = import_predictions_from_csv(file_path)

        assert len(imported_predictions) == len(sample_detector_predictions)
        assert all(isinstance(p, DetectorPrediction) for p in imported_predictions)

    def test_import_with_format_auto_detection(self, sample_benchmark_result, temp_results_dir):
        """Test importing with automatic format detection."""
        from drift_benchmark.results import export_benchmark_result, import_benchmark_result

        # Test JSON
        json_path = temp_results_dir / "result.json"
        export_benchmark_result(sample_benchmark_result, json_path, format="JSON")

        imported_json = import_benchmark_result(json_path)  # Should auto-detect JSON
        assert isinstance(imported_json, BenchmarkResult)

    def test_import_with_validation(self, temp_results_dir):
        """Test importing with validation."""
        from drift_benchmark.results import import_benchmark_result

        # Create invalid JSON file
        invalid_file = temp_results_dir / "invalid_result.json"
        with open(invalid_file, "w") as f:
            json.dump({"invalid": "structure"}, f)

        with pytest.raises(ValueError):
            import_benchmark_result(invalid_file, validate=True)

    def test_import_legacy_format(self, temp_results_dir):
        """Test importing legacy format results."""
        from drift_benchmark.results import import_benchmark_result

        # Create legacy format file
        legacy_data = {
            "detector": "old_detector",  # Legacy field name
            "dataset": "old_dataset",  # Legacy field name
            "results": [],
        }

        legacy_file = temp_results_dir / "legacy_result.json"
        with open(legacy_file, "w") as f:
            json.dump(legacy_data, f)

        imported_result = import_benchmark_result(legacy_file, legacy_support=True)
        assert isinstance(imported_result, BenchmarkResult)


class TestResultValidation:
    """Test result validation and error handling."""

    def test_validate_benchmark_result_structure(self, sample_benchmark_result):
        """Test validating benchmark result structure."""
        from drift_benchmark.results import validate_benchmark_result

        # Valid result should pass
        is_valid, errors = validate_benchmark_result(sample_benchmark_result)
        assert is_valid
        assert len(errors) == 0

    def test_validate_benchmark_result_with_errors(self):
        """Test validating benchmark result with errors."""
        from drift_benchmark.results import validate_benchmark_result

        # Create invalid result
        invalid_result = BenchmarkResult(
            detector_name="",  # Empty name should be invalid
            dataset_name="test_dataset",
            predictions=[],  # Empty predictions might be flagged
        )

        is_valid, errors = validate_benchmark_result(invalid_result)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_prediction_consistency(self, sample_detector_predictions):
        """Test validating prediction consistency."""
        from drift_benchmark.results import validate_prediction_consistency

        # Valid predictions should pass
        is_consistent, issues = validate_prediction_consistency(sample_detector_predictions)
        assert is_consistent

        # Create inconsistent predictions
        inconsistent_predictions = sample_detector_predictions.copy()
        inconsistent_predictions[0].dataset_name = "different_dataset"

        is_consistent, issues = validate_prediction_consistency(inconsistent_predictions)
        assert not is_consistent
        assert len(issues) > 0

    def test_validate_metric_values(self, sample_benchmark_result):
        """Test validating metric values."""
        from drift_benchmark.results import validate_metric_values

        # Valid metrics should pass
        is_valid, errors = validate_metric_values(sample_benchmark_result.metrics)
        assert is_valid

        # Create invalid metrics
        invalid_metrics = {"accuracy": 1.5, "precision": -0.1}  # Out of valid range

        is_valid, errors = validate_metric_values(invalid_metrics)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_temporal_consistency(self, sample_detector_predictions):
        """Test validating temporal consistency of predictions."""
        from drift_benchmark.results import validate_temporal_consistency

        # Reorder predictions to create temporal inconsistency
        shuffled_predictions = [
            sample_detector_predictions[2],  # window_id=2
            sample_detector_predictions[0],  # window_id=0
            sample_detector_predictions[1],  # window_id=1
        ]

        is_consistent, issues = validate_temporal_consistency(shuffled_predictions)
        # This might be valid depending on implementation - temporal order might not be required
        # or might be flagged as a warning rather than error


class TestResultAggregation:
    """Test result aggregation and summarization."""

    def test_aggregate_metrics_across_datasets(self, sample_multiple_benchmark_results):
        """Test aggregating metrics across multiple datasets."""
        from drift_benchmark.results import aggregate_metrics_by_detector

        aggregated = aggregate_metrics_by_detector(sample_multiple_benchmark_results)

        assert isinstance(aggregated, dict)
        # Should have one entry per detector
        expected_detectors = {"detector_a", "detector_b", "detector_c"}
        assert set(aggregated.keys()) == expected_detectors

        # Each detector should have aggregated metrics
        for detector_stats in aggregated.values():
            assert "mean_accuracy" in detector_stats
            assert "std_accuracy" in detector_stats
            assert "count" in detector_stats

    def test_aggregate_metrics_across_detectors(self, sample_multiple_benchmark_results):
        """Test aggregating metrics across multiple detectors."""
        from drift_benchmark.results import aggregate_metrics_by_dataset

        aggregated = aggregate_metrics_by_dataset(sample_multiple_benchmark_results)

        assert isinstance(aggregated, dict)
        expected_datasets = {"dataset_1", "dataset_2"}
        assert set(aggregated.keys()) == expected_datasets

    def test_compute_summary_statistics(self, sample_multiple_benchmark_results):
        """Test computing summary statistics."""
        from drift_benchmark.results import compute_summary_statistics

        summary = compute_summary_statistics(sample_multiple_benchmark_results)

        assert isinstance(summary, dict)
        assert "total_evaluations" in summary
        assert "unique_detectors" in summary
        assert "unique_datasets" in summary
        assert "overall_metrics" in summary

    def test_create_ranking_table(self, sample_multiple_benchmark_results):
        """Test creating ranking table from results."""
        from drift_benchmark.results import create_ranking_table

        rankings = create_ranking_table(sample_multiple_benchmark_results, metric="accuracy")

        assert isinstance(rankings, list)
        assert len(rankings) == 3  # Three detectors

        # Check ranking structure
        for rank_entry in rankings:
            assert "detector_name" in rank_entry
            assert "rank" in rank_entry
            assert "score" in rank_entry

        # Check rankings are sorted
        scores = [entry["score"] for entry in rankings]
        assert scores == sorted(scores, reverse=True)

    def test_compute_pairwise_comparisons(self, sample_multiple_benchmark_results):
        """Test computing pairwise comparisons between detectors."""
        from drift_benchmark.results import compute_pairwise_comparisons

        comparisons = compute_pairwise_comparisons(sample_multiple_benchmark_results)

        assert isinstance(comparisons, dict)
        # Should have comparisons for each detector pair
        detector_names = {"detector_a", "detector_b", "detector_c"}
        for detector1 in detector_names:
            assert detector1 in comparisons
            for detector2 in detector_names:
                if detector1 != detector2:
                    assert detector2 in comparisons[detector1]


class TestResultFiltering:
    """Test result filtering and querying."""

    def test_filter_results_by_detector(self, sample_multiple_benchmark_results):
        """Test filtering results by detector name."""
        from drift_benchmark.results import filter_results_by_detector

        filtered = filter_results_by_detector(sample_multiple_benchmark_results, "detector_a")

        assert len(filtered) == 2  # detector_a on 2 datasets
        assert all(result.detector_name == "detector_a" for result in filtered)

    def test_filter_results_by_dataset(self, sample_multiple_benchmark_results):
        """Test filtering results by dataset name."""
        from drift_benchmark.results import filter_results_by_dataset

        filtered = filter_results_by_dataset(sample_multiple_benchmark_results, "dataset_1")

        assert len(filtered) == 3  # 3 detectors on dataset_1
        assert all(result.dataset_name == "dataset_1" for result in filtered)

    def test_filter_results_by_metric_threshold(self, sample_multiple_benchmark_results):
        """Test filtering results by metric threshold."""
        from drift_benchmark.results import filter_results_by_metric

        filtered = filter_results_by_metric(
            sample_multiple_benchmark_results,
            metric="accuracy",
            min_value=0.75,
        )

        assert all(result.metrics.get("accuracy", 0) >= 0.75 for result in filtered)

    def test_filter_results_by_multiple_criteria(self, sample_multiple_benchmark_results):
        """Test filtering results by multiple criteria."""
        from drift_benchmark.results import filter_results

        criteria = {
            "detector_names": ["detector_a", "detector_b"],
            "dataset_names": ["dataset_1"],
            "min_accuracy": 0.7,
        }

        filtered = filter_results(sample_multiple_benchmark_results, **criteria)

        assert len(filtered) <= 2  # At most 2 results (detector_a and detector_b on dataset_1)
        assert all(result.detector_name in criteria["detector_names"] for result in filtered)
        assert all(result.dataset_name in criteria["dataset_names"] for result in filtered)

    def test_query_results_with_complex_conditions(self, sample_multiple_benchmark_results):
        """Test querying results with complex conditions."""
        from drift_benchmark.results import query_results

        # Query for high-performing detectors
        query = "accuracy > 0.8 AND precision > 0.7"
        filtered = query_results(sample_multiple_benchmark_results, query)

        # Results should match the query conditions
        for result in filtered:
            assert result.metrics.get("accuracy", 0) > 0.8
            assert result.metrics.get("precision", 0) > 0.7


class TestResultComparison:
    """Test result comparison and analysis."""

    def test_compare_detector_performance(self, sample_multiple_benchmark_results):
        """Test comparing detector performance."""
        from drift_benchmark.results import compare_detector_performance

        comparison = compare_detector_performance(
            sample_multiple_benchmark_results,
            detectors=["detector_a", "detector_b"],
            metric="accuracy",
        )

        assert isinstance(comparison, dict)
        assert "detector_a" in comparison
        assert "detector_b" in comparison
        assert "statistical_test" in comparison
        assert "effect_size" in comparison

    def test_statistical_significance_testing(self, sample_multiple_benchmark_results):
        """Test statistical significance testing between detectors."""
        from drift_benchmark.results import test_statistical_significance

        test_result = test_statistical_significance(
            sample_multiple_benchmark_results,
            detector1="detector_a",
            detector2="detector_b",
            metric="accuracy",
        )

        assert isinstance(test_result, dict)
        assert "p_value" in test_result
        assert "statistic" in test_result
        assert "test_name" in test_result
        assert "significant" in test_result

    def test_compute_effect_sizes(self, sample_multiple_benchmark_results):
        """Test computing effect sizes between detectors."""
        from drift_benchmark.results import compute_effect_sizes

        effect_sizes = compute_effect_sizes(
            sample_multiple_benchmark_results,
            detector1="detector_a",
            detector2="detector_b",
            metric="accuracy",
        )

        assert isinstance(effect_sizes, dict)
        assert "cohens_d" in effect_sizes
        assert "hedges_g" in effect_sizes
        assert "effect_magnitude" in effect_sizes

    def test_create_comparison_matrix(self, sample_multiple_benchmark_results):
        """Test creating comparison matrix for all detector pairs."""
        from drift_benchmark.results import create_comparison_matrix

        matrix = create_comparison_matrix(sample_multiple_benchmark_results, metric="accuracy")

        assert isinstance(matrix, pd.DataFrame)
        detector_names = {"detector_a", "detector_b", "detector_c"}
        assert set(matrix.index) == detector_names
        assert set(matrix.columns) == detector_names


class TestResultReporting:
    """Test report generation functionality."""

    def test_generate_basic_report(self, sample_drift_evaluation_result, temp_results_dir):
        """Test generating basic benchmark report."""
        from drift_benchmark.results import generate_benchmark_report

        report_path = temp_results_dir / "benchmark_report.html"
        generate_benchmark_report(sample_drift_evaluation_result, report_path)

        assert report_path.exists()
        # Verify it's a valid HTML file
        content = report_path.read_text()
        assert "<html>" in content
        assert "</html>" in content

    def test_generate_detailed_report(self, sample_drift_evaluation_result, temp_results_dir):
        """Test generating detailed benchmark report."""
        from drift_benchmark.results import generate_detailed_report

        report_path = temp_results_dir / "detailed_report.html"
        generate_detailed_report(
            sample_drift_evaluation_result,
            report_path,
            include_plots=True,
            include_statistics=True,
        )

        assert report_path.exists()

    def test_generate_summary_report(self, sample_drift_evaluation_result, temp_results_dir):
        """Test generating summary report."""
        from drift_benchmark.results import generate_summary_report

        report_path = temp_results_dir / "summary_report.pdf"
        generate_summary_report(sample_drift_evaluation_result, report_path)

        assert report_path.exists()

    def test_generate_comparison_report(self, sample_multiple_benchmark_results, temp_results_dir):
        """Test generating comparison report."""
        from drift_benchmark.results import generate_comparison_report

        report_path = temp_results_dir / "comparison_report.html"
        generate_comparison_report(
            sample_multiple_benchmark_results,
            report_path,
            detectors=["detector_a", "detector_b"],
        )

        assert report_path.exists()

    def test_generate_custom_report_template(self, sample_drift_evaluation_result, temp_results_dir):
        """Test generating report with custom template."""
        from drift_benchmark.results import generate_custom_report

        template_content = """
        <html>
            <head><title>Custom Report</title></head>
            <body>
                <h1>{{ title }}</h1>
                <p>Total Results: {{ total_results }}</p>
            </body>
        </html>
        """

        template_path = temp_results_dir / "custom_template.html"
        template_path.write_text(template_content)

        report_path = temp_results_dir / "custom_report.html"
        generate_custom_report(
            sample_drift_evaluation_result,
            report_path,
            template_path=template_path,
            context={"title": "My Custom Report"},
        )

        assert report_path.exists()


class TestResultSerialization:
    """Test serialization format handling."""

    def test_serialize_to_json_with_custom_encoder(self, sample_benchmark_result):
        """Test JSON serialization with custom encoder."""
        from drift_benchmark.results import serialize_to_json

        json_str = serialize_to_json(sample_benchmark_result, use_custom_encoder=True)

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "detector_name" in parsed

    def test_serialize_to_pickle(self, sample_benchmark_result):
        """Test pickle serialization."""
        from drift_benchmark.results import serialize_to_pickle

        pickle_bytes = serialize_to_pickle(sample_benchmark_result)

        assert isinstance(pickle_bytes, bytes)

        # Test deserialization
        import pickle

        deserialized = pickle.loads(pickle_bytes)
        assert isinstance(deserialized, BenchmarkResult)

    def test_serialize_with_compression(self, sample_drift_evaluation_result):
        """Test serialization with compression."""
        from drift_benchmark.results import serialize_with_compression

        compressed_data = serialize_with_compression(sample_drift_evaluation_result)

        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0

    def test_deserialize_from_compressed(self, sample_drift_evaluation_result):
        """Test deserialization from compressed data."""
        from drift_benchmark.results import deserialize_from_compressed, serialize_with_compression

        compressed_data = serialize_with_compression(sample_drift_evaluation_result)
        deserialized = deserialize_from_compressed(compressed_data, DriftEvaluationResult)

        assert isinstance(deserialized, DriftEvaluationResult)
        assert len(deserialized.results) == len(sample_drift_evaluation_result.results)

    def test_handle_large_results_streaming(self, temp_results_dir):
        """Test handling large results with streaming serialization."""
        from drift_benchmark.results import stream_serialize_large_result

        # Create a large result set
        large_predictions = []
        for i in range(10000):
            prediction = DetectorPrediction(
                dataset_name="large_dataset",
                window_id=i,
                has_true_drift=i % 100 == 0,
                detected_drift=i % 90 == 0,
                detection_time=0.001,
            )
            large_predictions.append(prediction)

        large_result = BenchmarkResult(
            detector_name="test_detector",
            dataset_name="large_dataset",
            predictions=large_predictions,
        )

        output_path = temp_results_dir / "large_result.jsonl"
        stream_serialize_large_result(large_result, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestResultCompression:
    """Test compression and optimization."""

    def test_compress_benchmark_results(self, sample_multiple_benchmark_results, temp_results_dir):
        """Test compressing multiple benchmark results."""
        from drift_benchmark.results import compress_benchmark_results

        archive_path = temp_results_dir / "results_archive.zip"
        compress_benchmark_results(sample_multiple_benchmark_results, archive_path)

        assert archive_path.exists()
        assert archive_path.stat().st_size > 0

    def test_decompress_benchmark_results(self, sample_multiple_benchmark_results, temp_results_dir):
        """Test decompressing benchmark results."""
        from drift_benchmark.results import compress_benchmark_results, decompress_benchmark_results

        archive_path = temp_results_dir / "results_archive.zip"
        compress_benchmark_results(sample_multiple_benchmark_results, archive_path)

        decompressed_results = decompress_benchmark_results(archive_path)

        assert len(decompressed_results) == len(sample_multiple_benchmark_results)
        assert all(isinstance(result, BenchmarkResult) for result in decompressed_results)

    def test_optimize_result_storage(self, sample_drift_evaluation_result):
        """Test optimizing result storage."""
        from drift_benchmark.results import optimize_result_storage

        optimized = optimize_result_storage(sample_drift_evaluation_result)

        assert isinstance(optimized, DriftEvaluationResult)
        # Should maintain essential information while reducing size


class TestResultVersioning:
    """Test version compatibility handling."""

    def test_convert_legacy_result_format(self, temp_results_dir):
        """Test converting legacy result format."""
        from drift_benchmark.results import convert_legacy_result

        # Create legacy format file
        legacy_data = {
            "version": "0.1.0",
            "detector": "legacy_detector",
            "dataset": "legacy_dataset",
            "results": {"accuracy": 0.8},
        }

        legacy_file = temp_results_dir / "legacy_result.json"
        with open(legacy_file, "w") as f:
            json.dump(legacy_data, f)

        converted_result = convert_legacy_result(legacy_file)

        assert isinstance(converted_result, BenchmarkResult)
        assert converted_result.detector_name == "legacy_detector"

    def test_version_compatibility_check(self, sample_benchmark_result):
        """Test version compatibility checking."""
        from drift_benchmark.results import check_version_compatibility

        # Add version metadata
        sample_benchmark_result.metrics["_version"] = "1.0.0"

        is_compatible, warnings = check_version_compatibility(sample_benchmark_result)

        assert isinstance(is_compatible, bool)
        assert isinstance(warnings, list)

    def test_upgrade_result_format(self, temp_results_dir):
        """Test upgrading result format to latest version."""
        from drift_benchmark.results import upgrade_result_format

        # Create old format result
        old_format_data = {
            "version": "0.5.0",
            "detector_name": "old_detector",
            "dataset_name": "old_dataset",
            "predictions": [],
            "metrics": {"accuracy": 0.75},
        }

        old_file = temp_results_dir / "old_format.json"
        with open(old_file, "w") as f:
            json.dump(old_format_data, f)

        upgraded_file = temp_results_dir / "upgraded_format.json"
        upgrade_result_format(old_file, upgraded_file)

        assert upgraded_file.exists()


class TestResultIntegration:
    """Integration tests with real data."""

    def test_end_to_end_result_workflow(self, sample_multiple_benchmark_results, temp_results_dir):
        """Test complete end-to-end result workflow."""
        from drift_benchmark.results import (
            aggregate_metrics_by_detector,
            create_ranking_table,
            export_evaluation_result,
            generate_benchmark_report,
        )

        # Create evaluation result
        evaluation_result = DriftEvaluationResult(results=sample_multiple_benchmark_results)

        # Export results
        export_path = temp_results_dir / "complete_results.json"
        export_evaluation_result(evaluation_result, export_path)

        # Generate aggregations
        aggregated = aggregate_metrics_by_detector(sample_multiple_benchmark_results)
        rankings = create_ranking_table(sample_multiple_benchmark_results, metric="accuracy")

        # Generate report
        report_path = temp_results_dir / "final_report.html"
        generate_benchmark_report(evaluation_result, report_path)

        # Verify all outputs exist
        assert export_path.exists()
        assert report_path.exists()
        assert len(aggregated) > 0
        assert len(rankings) > 0

    def test_large_scale_result_processing(self, temp_results_dir):
        """Test processing large-scale results."""
        from drift_benchmark.results import process_large_result_batch

        # Generate large batch of results
        large_batch = []
        for i in range(100):
            result = BenchmarkResult(
                detector_name=f"detector_{i % 10}",
                dataset_name=f"dataset_{i % 20}",
                metrics={"accuracy": np.random.uniform(0.5, 0.95)},
            )
            large_batch.append(result)

        # Process in batches
        batch_size = 10
        processed_results = process_large_result_batch(large_batch, batch_size=batch_size)

        assert len(processed_results) == len(large_batch)

    def test_concurrent_result_processing(self, sample_multiple_benchmark_results):
        """Test concurrent processing of results."""
        from drift_benchmark.results import process_results_concurrently

        # Process results concurrently
        processed_results = process_results_concurrently(
            sample_multiple_benchmark_results,
            max_workers=2,
        )

        assert len(processed_results) == len(sample_multiple_benchmark_results)


class TestResultUtilities:
    """Test utility functions and helpers."""

    def test_calculate_result_checksums(self, sample_benchmark_result):
        """Test calculating checksums for result integrity."""
        from drift_benchmark.results import calculate_result_checksum

        checksum = calculate_result_checksum(sample_benchmark_result)

        assert isinstance(checksum, str)
        assert len(checksum) > 0

        # Same result should produce same checksum
        checksum2 = calculate_result_checksum(sample_benchmark_result)
        assert checksum == checksum2

    def test_merge_benchmark_results(self, sample_multiple_benchmark_results):
        """Test merging multiple benchmark results."""
        from drift_benchmark.results import merge_benchmark_results

        # Split results into two groups
        group1 = sample_multiple_benchmark_results[:3]
        group2 = sample_multiple_benchmark_results[3:]

        merged = merge_benchmark_results([group1, group2])

        assert len(merged) == len(sample_multiple_benchmark_results)

    def test_split_results_by_criteria(self, sample_multiple_benchmark_results):
        """Test splitting results by criteria."""
        from drift_benchmark.results import split_results_by_criteria

        groups = split_results_by_criteria(
            sample_multiple_benchmark_results,
            criteria_func=lambda r: r.detector_name,
        )

        assert isinstance(groups, dict)
        assert len(groups) == 3  # Three different detectors

    def test_validate_result_integrity(self, sample_benchmark_result):
        """Test validating result integrity."""
        from drift_benchmark.results import validate_result_integrity

        is_valid, report = validate_result_integrity(sample_benchmark_result)

        assert isinstance(is_valid, bool)
        assert isinstance(report, dict)

    def test_anonymize_results(self, sample_benchmark_result):
        """Test anonymizing results for sharing."""
        from drift_benchmark.results import anonymize_results

        anonymized = anonymize_results(sample_benchmark_result)

        assert isinstance(anonymized, BenchmarkResult)
        assert anonymized.detector_name != sample_benchmark_result.detector_name
        assert anonymized.dataset_name != sample_benchmark_result.dataset_name

    def test_result_memory_usage_analysis(self, sample_multiple_benchmark_results):
        """Test analyzing memory usage of results."""
        from drift_benchmark.results import analyze_result_memory_usage

        memory_report = analyze_result_memory_usage(sample_multiple_benchmark_results)

        assert isinstance(memory_report, dict)
        assert "total_size_bytes" in memory_report
        assert "size_by_component" in memory_report


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


@pytest.mark.parametrize(
    "export_format,expected_extension",
    [
        ("JSON", ".json"),
        ("CSV", ".csv"),
        ("PICKLE", ".pkl"),
        ("EXCEL", ".xlsx"),
    ],
)
def test_export_format_extensions(export_format, expected_extension, sample_benchmark_result, temp_results_dir):
    """Test that export formats use correct file extensions."""
    from drift_benchmark.results import export_benchmark_result

    file_path = temp_results_dir / f"test_result{expected_extension}"
    export_benchmark_result(sample_benchmark_result, file_path, format=export_format)

    assert file_path.exists()


@pytest.mark.parametrize(
    "metric_name,expected_range",
    [
        ("accuracy", (0.0, 1.0)),
        ("precision", (0.0, 1.0)),
        ("recall", (0.0, 1.0)),
        ("f1_score", (0.0, 1.0)),
    ],
)
def test_metric_validation_ranges(metric_name, expected_range, sample_benchmark_result):
    """Test that metrics are validated within expected ranges."""
    from drift_benchmark.results import validate_metric_range

    # Test valid value
    valid_value = (expected_range[0] + expected_range[1]) / 2
    assert validate_metric_range(metric_name, valid_value)

    # Test invalid values
    assert not validate_metric_range(metric_name, expected_range[0] - 0.1)
    assert not validate_metric_range(metric_name, expected_range[1] + 0.1)


@pytest.mark.parametrize(
    "compression_level",
    [1, 5, 9],
)
def test_compression_levels(compression_level, sample_benchmark_result, temp_results_dir):
    """Test different compression levels."""
    from drift_benchmark.results import save_benchmark_result

    file_path = temp_results_dir / f"result_compressed_{compression_level}.json.gz"
    save_benchmark_result(
        sample_benchmark_result,
        file_path,
        compress=True,
        compression_level=compression_level,
    )

    assert file_path.exists()
    assert file_path.stat().st_size > 0


@pytest.mark.parametrize(
    "batch_size",
    [1, 10, 50, 100],
)
def test_batch_processing_sizes(batch_size, sample_multiple_benchmark_results):
    """Test different batch sizes for processing."""
    from drift_benchmark.results import process_results_in_batches

    processed = process_results_in_batches(sample_multiple_benchmark_results, batch_size=batch_size)

    assert len(processed) == len(sample_multiple_benchmark_results)
