"""
Test suite for models.results module - REQ-MDL-XXX

This module tests the result data models using Pydantic v2 for storing
and managing benchmark execution results.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import pytest


# REQ-MDL-002: DetectorResult Model Tests
class TestDetectorResultModel:
    """Test REQ-MDL-002: DetectorResult model with execution results fields"""

    def test_should_define_detector_result_model_when_imported(self):
        """Test that DetectorResult model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.results import DetectorResult
        except ImportError:
            pytest.skip("DetectorResult not implemented yet")

        # Assert
        assert DetectorResult is not None
        from pydantic import BaseModel

        assert issubclass(DetectorResult, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test DetectorResult accepts all required fields"""
        # Arrange
        try:
            from drift_benchmark.models.results import DetectorResult
        except ImportError:
            pytest.skip("DetectorResult not implemented yet")

        # Act
        result = DetectorResult(
            detector_id="ks_test_scipy_evidently",
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="evidently",
            scenario_name="covariate_drift_example",
            drift_detected=True,
            execution_time=0.0234,
            drift_score=0.75,
        )

        # Assert
        assert result.detector_id == "ks_test_scipy_evidently"
        assert result.method_id == "kolmogorov_smirnov"
        assert result.variant_id == "batch"
        assert result.library_id == "evidently"
        assert result.scenario_name == "covariate_drift_example"
        assert result.drift_detected is True
        assert result.execution_time == 0.0234
        assert result.drift_score == 0.75

    def test_should_support_optional_execution_time_for_failures_when_created(self):
        """Test DetectorResult supports None execution_time for failed detectors"""
        # Arrange
        try:
            from drift_benchmark.models.results import DetectorResult
        except ImportError:
            pytest.skip("DetectorResult not implemented yet")

        # Act - failed detector (execution_time=None)
        failed_result = DetectorResult(
            detector_id="failed_detector",
            method_id="ks_test",
            variant_id="scipy",
            library_id="scipy",
            scenario_name="test_scenario",
            drift_detected=False,
            execution_time=None,  # Indicates failure
            drift_score=None,
        )

        # Assert
        assert failed_result.execution_time is None
        assert failed_result.drift_score is None

    def test_should_support_optional_drift_score_when_not_available(self):
        """Test DetectorResult supports None drift_score when detector can't provide score"""
        # Arrange
        try:
            from drift_benchmark.models.results import DetectorResult
        except ImportError:
            pytest.skip("DetectorResult not implemented yet")

        # Act
        result = DetectorResult(
            detector_id="basic_detector",
            method_id="simple_test",
            variant_id="basic",
            library_id="custom",
            scenario_name="test_scenario",
            drift_detected=True,
            execution_time=0.1,
            drift_score=None,  # Some detectors may not provide scores
        )

        # Assert
        assert result.drift_score is None
        assert result.drift_detected is True
        assert result.execution_time is not None

    def test_should_validate_field_types_when_created(self):
        """Test DetectorResult validates field types"""
        # Arrange
        try:
            from drift_benchmark.models.results import DetectorResult
        except ImportError:
            pytest.skip("DetectorResult not implemented yet")

        base_data = {
            "detector_id": "test_detector",
            "method_id": "test_method",
            "variant_id": "test_variant",
            "library_id": "test_library",
            "scenario_name": "test_scenario",
            "drift_detected": True,
            "execution_time": 0.1,
            "drift_score": 0.5,
        }

        # Act & Assert - invalid types should raise validation errors
        with pytest.raises(ValueError):
            invalid_data = base_data.copy()
            invalid_data["drift_detected"] = "not_boolean"
            DetectorResult(**invalid_data)

        with pytest.raises(ValueError):
            invalid_data = base_data.copy()
            invalid_data["execution_time"] = "not_number"
            DetectorResult(**invalid_data)


# REQ-MDL-003: BenchmarkResult Model Tests
class TestBenchmarkResultModel:
    """Test REQ-MDL-003: BenchmarkResult model with complete benchmark results"""

    def test_should_define_benchmark_result_model_when_imported(self):
        """Test that BenchmarkResult model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.results import BenchmarkResult
        except ImportError:
            pytest.skip("BenchmarkResult not implemented yet")

        # Assert
        assert BenchmarkResult is not None
        from pydantic import BaseModel

        assert issubclass(BenchmarkResult, BaseModel)

    def test_should_accept_all_required_fields_when_created(self, mock_benchmark_config):
        """Test BenchmarkResult accepts required fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
            from drift_benchmark.models.results import BenchmarkResult, DetectorResult
        except ImportError:
            pytest.skip("Result models not implemented yet")

        detector_results = [
            DetectorResult(
                detector_id="test_detector",
                method_id="ks_test",
                variant_id="scipy",
                library_id="scipy",
                scenario_name="test_scenario",
                drift_detected=True,
                execution_time=0.1,
                drift_score=0.8,
            )
        ]

        summary = BenchmarkSummary(total_detectors=1, successful_runs=1, failed_runs=0, avg_execution_time=0.1, total_scenarios=1)

        timestamp = datetime.now()
        output_dir = "/results/20250720_143022"

        # Act
        result = BenchmarkResult(
            config=mock_benchmark_config,
            detector_results=detector_results,
            summary=summary,
            timestamp=timestamp,
            output_directory=output_dir,
        )

        # Assert
        assert result.config == mock_benchmark_config
        assert len(result.detector_results) == 1
        assert result.summary == summary
        assert result.timestamp == timestamp
        assert result.output_directory == output_dir

    def test_should_validate_detector_results_list_when_created(self, mock_benchmark_config):
        """Test BenchmarkResult validates detector_results is proper list"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
            from drift_benchmark.models.results import BenchmarkResult
        except ImportError:
            pytest.skip("Result models not implemented yet")

        summary = BenchmarkSummary(total_detectors=0, successful_runs=0, failed_runs=0, avg_execution_time=0.0, total_scenarios=0)

        # Act & Assert
        with pytest.raises(ValueError):
            BenchmarkResult(
                config=mock_benchmark_config,
                detector_results="not_a_list",
                summary=summary,
                timestamp=datetime.now(),
                output_directory="/results/test",
            )


# REQ-MDL-004: ScenarioResult Model Tests
class TestScenarioResultModel:
    """Test REQ-MDL-004: ScenarioResult model with scenario data and metadata"""

    def test_should_define_scenario_result_model_when_imported(self):
        """Test that ScenarioResult model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.results import ScenarioResult
        except ImportError:
            pytest.skip("ScenarioResult not implemented yet")

        # Assert
        assert ScenarioResult is not None
        from pydantic import BaseModel

        assert issubclass(ScenarioResult, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test ScenarioResult accepts all required fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata, ScenarioDefinition, ScenarioMetadata
            from drift_benchmark.models.results import ScenarioResult
        except ImportError:
            pytest.skip("ScenarioResult and metadata models not implemented yet")

        import numpy as np

        # Create test data
        X_ref = pd.DataFrame({"feature_1": np.random.normal(0, 1, 100), "feature_2": np.random.normal(0, 1, 100)})
        X_test = pd.DataFrame({"feature_1": np.random.normal(0.5, 1, 50), "feature_2": np.random.normal(0, 1.2, 50)})
        y_ref = pd.Series(np.random.choice([0, 1], 100))
        y_test = pd.Series(np.random.choice([0, 1], 50))

        dataset_metadata = DatasetMetadata(
            name="test_dataset", data_type="continuous", dimension="multivariate", n_samples_ref=100, n_samples_test=50, n_features=2
        )

        scenario_metadata = ScenarioMetadata(
            total_samples=150,
            ref_samples=100,
            test_samples=50,
            n_features=2,
            has_labels=True,
            data_type="continuous",
            dimension="multivariate",
        )

        definition = ScenarioDefinition(
            description="Test scenario",
            source_type="synthetic",
            source_name="make_classification",
            target_column="target",
            ref_filter={"sample_range": [0, 100]},
            test_filter={"sample_range": [100, 150]},
        )

        # Act
        result = ScenarioResult(
            name="test_scenario",
            X_ref=X_ref,
            X_test=X_test,
            y_ref=y_ref,
            y_test=y_test,
            dataset_metadata=dataset_metadata,
            scenario_metadata=scenario_metadata,
            definition=definition,
        )

        # Assert
        assert result.name == "test_scenario"
        assert result.X_ref.shape == (100, 2)
        assert result.X_test.shape == (50, 2)
        assert len(result.y_ref) == 100
        assert len(result.y_test) == 50
        assert result.dataset_metadata.name == "test_dataset"
        assert result.scenario_metadata.total_samples == 150
        assert result.definition.description == "Test scenario"

    def test_should_support_optional_labels_when_not_available(self):
        """Test ScenarioResult supports None for y_ref and y_test when labels not available"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata, ScenarioDefinition, ScenarioMetadata
            from drift_benchmark.models.results import ScenarioResult
        except ImportError:
            pytest.skip("ScenarioResult and metadata models not implemented yet")

        import numpy as np

        X_ref = pd.DataFrame({"feature": np.random.normal(0, 1, 10)})
        X_test = pd.DataFrame({"feature": np.random.normal(0.5, 1, 5)})

        dataset_metadata = DatasetMetadata(
            name="unlabeled_dataset", data_type="continuous", dimension="univariate", n_samples_ref=10, n_samples_test=5, n_features=1
        )

        scenario_metadata = ScenarioMetadata(
            total_samples=15, ref_samples=10, test_samples=5, n_features=1, has_labels=False, data_type="continuous", dimension="univariate"
        )

        definition = ScenarioDefinition(
            description="Unlabeled scenario",
            source_type="synthetic",
            source_name="make_blobs",
            target_column=None,
            ref_filter={"sample_range": [0, 10]},
            test_filter={"sample_range": [10, 15]},
        )

        # Act
        result = ScenarioResult(
            name="unlabeled_scenario",
            X_ref=X_ref,
            X_test=X_test,
            y_ref=None,  # No labels available
            y_test=None,
            dataset_metadata=dataset_metadata,
            scenario_metadata=scenario_metadata,
            definition=definition,
        )

        # Assert
        assert result.y_ref is None
        assert result.y_test is None
        assert result.scenario_metadata.has_labels is False

    def test_should_validate_dataframe_types_when_created(self):
        """Test ScenarioResult validates DataFrame and Series types"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata, ScenarioDefinition, ScenarioMetadata
            from drift_benchmark.models.results import ScenarioResult
        except ImportError:
            pytest.skip("ScenarioResult and metadata models not implemented yet")

        # Mock minimal metadata
        from unittest.mock import Mock

        metadata = Mock()

        # Act & Assert - invalid X_ref type
        with pytest.raises(ValueError):
            ScenarioResult(
                name="test",
                X_ref="not_dataframe",  # Should be DataFrame
                X_test=pd.DataFrame({"col": [1, 2]}),
                y_ref=None,
                y_test=None,
                dataset_metadata=metadata,
                scenario_metadata=metadata,
                definition=metadata,
            )


# Integration tests for result models
class TestResultModelsIntegration:
    """Test integration between result models"""

    def test_should_work_together_in_complete_benchmark_result_when_combined(self, mock_benchmark_config):
        """Test all result models work together in complete benchmark result"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary, DatasetMetadata, ScenarioDefinition, ScenarioMetadata
            from drift_benchmark.models.results import BenchmarkResult, DetectorResult, ScenarioResult
        except ImportError:
            pytest.skip("Result models not implemented yet")

        import numpy as np

        # Create test scenario result
        X_ref = pd.DataFrame({"feature": np.random.normal(0, 1, 10)})
        X_test = pd.DataFrame({"feature": np.random.normal(0.5, 1, 5)})

        dataset_metadata = DatasetMetadata(
            name="test_dataset", data_type="continuous", dimension="univariate", n_samples_ref=10, n_samples_test=5, n_features=1
        )

        scenario_metadata = ScenarioMetadata(
            total_samples=15, ref_samples=10, test_samples=5, n_features=1, has_labels=False, data_type="continuous", dimension="univariate"
        )

        definition = ScenarioDefinition(
            description="Integration test scenario",
            source_type="synthetic",
            source_name="make_blobs",
            target_column=None,
            ref_filter={"sample_range": [0, 10]},
            test_filter={"sample_range": [10, 15]},
        )

        # Create detector results
        detector_results = [
            DetectorResult(
                detector_id="detector_1",
                method_id="ks_test",
                variant_id="scipy",
                library_id="scipy",
                scenario_name="test_scenario",
                drift_detected=True,
                execution_time=0.1,
                drift_score=0.8,
            ),
            DetectorResult(
                detector_id="detector_2",
                method_id="ks_test",
                variant_id="scipy",
                library_id="evidently",
                scenario_name="test_scenario",
                drift_detected=False,
                execution_time=0.2,
                drift_score=0.3,
            ),
        ]

        summary = BenchmarkSummary(total_detectors=2, successful_runs=2, failed_runs=0, avg_execution_time=0.15, total_scenarios=1)

        # Act
        benchmark_result = BenchmarkResult(
            config=mock_benchmark_config,
            detector_results=detector_results,
            summary=summary,
            timestamp=datetime.now(),
            output_directory="/results/integration_test",
        )

        # Assert
        assert benchmark_result is not None
        assert len(benchmark_result.detector_results) == 2
        assert benchmark_result.summary.total_detectors == 2
        assert benchmark_result.summary.successful_runs == 2
        assert benchmark_result.summary.avg_execution_time == 0.15

        # Should serialize properly
        serialized = benchmark_result.model_dump(exclude={"config"})  # Exclude mock config
        assert "detector_results" in serialized
        assert "summary" in serialized
        assert len(serialized["detector_results"]) == 2
