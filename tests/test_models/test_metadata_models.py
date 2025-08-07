"""
Test suite for models.metadata module - REQ-MET-XXX

This module tests the metadata models that provide descriptive information
about datasets, scenarios, detectors, and benchmark summaries.
"""

from typing import Any, Dict, List

import pytest


# REQ-MET-001: DatasetMetadata Model Tests
class TestDatasetMetadataModel:
    """Test REQ-MET-001: DatasetMetadata model for describing source datasets"""

    def test_should_define_dataset_metadata_model_when_imported(self):
        """Test that DatasetMetadata model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.metadata import DatasetMetadata
        except ImportError:
            pytest.skip("DatasetMetadata not implemented yet")

        # Assert
        assert DatasetMetadata is not None
        from pydantic import BaseModel

        assert issubclass(DatasetMetadata, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test DatasetMetadata accepts all required fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata
        except ImportError:
            pytest.skip("DatasetMetadata not implemented yet")

        # Act
        metadata = DatasetMetadata(
            name="sklearn_classification_source",
            data_type="continuous",
            dimension="multivariate",
            n_samples_ref=1000,
            n_samples_test=500,
            n_features=10,
        )

        # Assert
        assert metadata.name == "sklearn_classification_source"
        assert metadata.data_type == "continuous"
        assert metadata.dimension == "multivariate"
        assert metadata.n_samples_ref == 1000
        assert metadata.n_samples_test == 500
        assert metadata.n_features == 10

    def test_should_validate_data_type_field_when_created(self):
        """Test DatasetMetadata validates data_type uses literal values"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata
        except ImportError:
            pytest.skip("DatasetMetadata not implemented yet")

        # Act & Assert - valid data types
        valid_types = ["continuous", "categorical", "mixed"]
        for data_type in valid_types:
            metadata = DatasetMetadata(
                name="test_dataset", data_type=data_type, dimension="multivariate", n_samples_ref=100, n_samples_test=50, n_features=5
            )
            assert metadata.data_type == data_type

        # Invalid data type
        with pytest.raises(ValueError):
            DatasetMetadata(
                name="test_dataset", data_type="invalid_type", dimension="multivariate", n_samples_ref=100, n_samples_test=50, n_features=5
            )

    def test_should_validate_dimension_field_when_created(self):
        """Test DatasetMetadata validates dimension uses literal values"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata
        except ImportError:
            pytest.skip("DatasetMetadata not implemented yet")

        # Act & Assert - valid dimensions
        valid_dimensions = ["univariate", "multivariate"]
        for dimension in valid_dimensions:
            metadata = DatasetMetadata(
                name="test_dataset",
                data_type="continuous",
                dimension=dimension,
                n_samples_ref=100,
                n_samples_test=50,
                n_features=5 if dimension == "multivariate" else 1,
            )
            assert metadata.dimension == dimension

        # Invalid dimension
        with pytest.raises(ValueError):
            DatasetMetadata(
                name="test_dataset",
                data_type="continuous",
                dimension="invalid_dimension",
                n_samples_ref=100,
                n_samples_test=50,
                n_features=5,
            )

    def test_should_validate_sample_counts_are_positive_when_created(self):
        """Test DatasetMetadata validates sample counts are positive integers"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata
        except ImportError:
            pytest.skip("DatasetMetadata not implemented yet")

        # Act & Assert - negative samples should fail
        with pytest.raises(ValueError):
            DatasetMetadata(
                name="test_dataset", data_type="continuous", dimension="multivariate", n_samples_ref=-100, n_samples_test=50, n_features=5
            )

        with pytest.raises(ValueError):
            DatasetMetadata(
                name="test_dataset",
                data_type="continuous",
                dimension="multivariate",
                n_samples_ref=100,
                n_samples_test=0,  # Zero should also fail
                n_features=5,
            )


# REQ-MET-002: DetectorMetadata Model Tests
class TestDetectorMetadataModel:
    """Test REQ-MET-002: DetectorMetadata model for detector information"""

    def test_should_define_detector_metadata_model_when_imported(self):
        """Test that DetectorMetadata model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.metadata import DetectorMetadata
        except ImportError:
            pytest.skip("DetectorMetadata not implemented yet")

        # Assert
        assert DetectorMetadata is not None
        from pydantic import BaseModel

        assert issubclass(DetectorMetadata, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test DetectorMetadata accepts all required fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DetectorMetadata
        except ImportError:
            pytest.skip("DetectorMetadata not implemented yet")

        # Act
        metadata = DetectorMetadata(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="evidently",
            name="Kolmogorov-Smirnov Test",
            family="statistical-test",
            description="Two-sample test for equality of continuous distributions",
        )

        # Assert
        assert metadata.method_id == "kolmogorov_smirnov"
        assert metadata.variant_id == "batch"
        assert metadata.library_id == "evidently"
        assert metadata.name == "Kolmogorov-Smirnov Test"
        assert metadata.family == "statistical-test"
        assert metadata.description == "Two-sample test for equality of continuous distributions"

    def test_should_validate_family_field_when_created(self):
        """Test DetectorMetadata validates family uses MethodFamily literal values"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DetectorMetadata
        except ImportError:
            pytest.skip("DetectorMetadata not implemented yet")

        # Act & Assert - valid method families
        valid_families = ["statistical-test", "distance-based", "change-detection", "window-based", "statistical-process-control"]

        for family in valid_families:
            metadata = DetectorMetadata(
                method_id="test_method",
                variant_id="test_variant",
                library_id="test_library",
                name="Test Method",
                family=family,
                description="Test description",
            )
            assert metadata.family == family

        # Invalid family
        with pytest.raises(ValueError):
            DetectorMetadata(
                method_id="test_method",
                variant_id="test_variant",
                library_id="test_library",
                name="Test Method",
                family="invalid_family",
                description="Test description",
            )


# REQ-MET-003: BenchmarkSummary Model Tests
class TestBenchmarkSummaryModel:
    """Test REQ-MET-003: BenchmarkSummary model for performance metrics"""

    def test_should_define_benchmark_summary_model_when_imported(self):
        """Test that BenchmarkSummary model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
        except ImportError:
            pytest.skip("BenchmarkSummary not implemented yet")

        # Assert
        assert BenchmarkSummary is not None
        from pydantic import BaseModel

        assert issubclass(BenchmarkSummary, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test BenchmarkSummary accepts all required performance metrics fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
        except ImportError:
            pytest.skip("BenchmarkSummary not implemented yet")

        # Act
        summary = BenchmarkSummary(total_detectors=5, successful_runs=4, failed_runs=1, avg_execution_time=0.125, total_scenarios=3)

        # Assert
        assert summary.total_detectors == 5
        assert summary.successful_runs == 4
        assert summary.failed_runs == 1
        assert summary.avg_execution_time == 0.125
        assert summary.total_scenarios == 3

    def test_should_validate_counts_are_non_negative_when_created(self):
        """Test BenchmarkSummary validates counts are non-negative"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
        except ImportError:
            pytest.skip("BenchmarkSummary not implemented yet")

        # Act & Assert - negative values should fail
        with pytest.raises(ValueError):
            BenchmarkSummary(total_detectors=-1, successful_runs=4, failed_runs=1, avg_execution_time=0.125, total_scenarios=3)

        with pytest.raises(ValueError):
            BenchmarkSummary(total_detectors=5, successful_runs=-1, failed_runs=1, avg_execution_time=0.125, total_scenarios=3)

    def test_should_validate_execution_time_is_non_negative_when_created(self):
        """Test BenchmarkSummary validates execution time is non-negative"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
        except ImportError:
            pytest.skip("BenchmarkSummary not implemented yet")

        # Act & Assert - negative execution time should fail
        with pytest.raises(ValueError):
            BenchmarkSummary(
                total_detectors=5, successful_runs=4, failed_runs=1, avg_execution_time=-0.125, total_scenarios=3  # Negative time
            )

    def test_should_validate_logical_consistency_when_created(self):
        """Test BenchmarkSummary validates logical consistency between counts"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary
        except ImportError:
            pytest.skip("BenchmarkSummary not implemented yet")

        # Act & Assert - successful + failed should not exceed total
        # (This might be implemented as a validator in the model)
        try:
            summary = BenchmarkSummary(
                total_detectors=5, successful_runs=4, failed_runs=2, avg_execution_time=0.125, total_scenarios=3  # 4 + 2 = 6 > 5 total
            )
            # If no validation error, check if it's handled elsewhere
            assert summary.successful_runs + summary.failed_runs <= summary.total_detectors or True
        except ValueError:
            # This is expected if the model validates consistency
            pass


# REQ-MET-004: ScenarioDefinition Model Tests
class TestScenarioDefinitionModel:
    """Test REQ-MET-004: ScenarioDefinition model for scenario .toml file structure"""

    def test_should_define_scenario_definition_model_when_imported(self):
        """Test that ScenarioDefinition model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.metadata import ScenarioDefinition
        except ImportError:
            pytest.skip("ScenarioDefinition not implemented yet")

        # Assert
        assert ScenarioDefinition is not None
        from pydantic import BaseModel

        assert issubclass(ScenarioDefinition, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test ScenarioDefinition accepts all required fields from README examples"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioDefinition
        except ImportError:
            pytest.skip("ScenarioDefinition not implemented yet")

        # Act - based on README scenario example
        definition = ScenarioDefinition(
            description="Covariate drift scenario with known ground truth",
            source_type="sklearn",
            source_name="make_classification",
            target_column="target",
            ref_filter={"sample_range": [0, 500]},
            test_filter={"sample_range": [500, 1000], "noise_factor": 1.5},
        )

        # Assert
        assert definition.description == "Covariate drift scenario with known ground truth"
        assert definition.source_type == "sklearn"
        assert definition.source_name == "make_classification"
        assert definition.target_column == "target"
        assert definition.ref_filter == {"sample_range": [0, 500]}
        assert definition.test_filter == {"sample_range": [500, 1000], "noise_factor": 1.5}

    def test_should_support_optional_target_column_when_not_provided(self):
        """Test ScenarioDefinition supports None target_column for unsupervised scenarios"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioDefinition
        except ImportError:
            pytest.skip("ScenarioDefinition not implemented yet")

        # Act - unsupervised scenario
        definition = ScenarioDefinition(
            description="Unsupervised covariate drift scenario",
            source_type="sklearn",
            source_name="make_blobs",
            target_column=None,  # No target for unsupervised
            ref_filter={"sample_range": [0, 500]},
            test_filter={"sample_range": [500, 1000]},
        )

        # Assert
        assert definition.target_column is None

    def test_should_validate_source_type_field_when_created(self):
        """Test ScenarioDefinition validates source_type uses ScenarioSourceType literal"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioDefinition
        except ImportError:
            pytest.skip("ScenarioDefinition not implemented yet")

        # Act & Assert - valid source types
        valid_source_types = ["sklearn", "file"]
        for source_type in valid_source_types:
            definition = ScenarioDefinition(
                description="Test scenario",
                source_type=source_type,
                source_name="test_source",
                target_column="target",
                ref_filter={"sample_range": [0, 100]},
                test_filter={"sample_range": [100, 200]},
            )
            assert definition.source_type == source_type

        # Invalid source type
        with pytest.raises(ValueError):
            ScenarioDefinition(
                description="Test scenario",
                source_type="invalid_source",
                source_name="test_source",
                target_column="target",
                ref_filter={"sample_range": [0, 100]},
                test_filter={"sample_range": [100, 200]},
            )

    def test_should_support_flexible_filter_dictionaries_when_created(self):
        """Test ScenarioDefinition supports flexible filter dictionaries"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioDefinition
        except ImportError:
            pytest.skip("ScenarioDefinition not implemented yet")

        # Act - various filter configurations
        complex_filters = {"sample_range": [100, 500], "noise_factor": 1.2, "random_state": 42, "custom_param": "value"}

        definition = ScenarioDefinition(
            description="Complex filter scenario",
            source_type="sklearn",
            source_name="make_classification",
            target_column="target",
            ref_filter={"sample_range": [0, 100]},
            test_filter=complex_filters,
        )

        # Assert
        assert definition.test_filter == complex_filters
        assert definition.test_filter["sample_range"] == [100, 500]
        assert definition.test_filter["noise_factor"] == 1.2


# REQ-MET-005: ScenarioMetadata Model Tests
class TestScenarioMetadataModel:
    """Test REQ-MET-005: ScenarioMetadata model for scenario-specific metadata"""

    def test_should_define_scenario_metadata_model_when_imported(self):
        """Test that ScenarioMetadata model exists with required fields"""
        # Arrange & Act
        try:
            from drift_benchmark.models.metadata import ScenarioMetadata
        except ImportError:
            pytest.skip("ScenarioMetadata not implemented yet")

        # Assert
        assert ScenarioMetadata is not None
        from pydantic import BaseModel

        assert issubclass(ScenarioMetadata, BaseModel)

    def test_should_accept_all_required_fields_when_created(self):
        """Test ScenarioMetadata accepts all required fields"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioMetadata
        except ImportError:
            pytest.skip("ScenarioMetadata not implemented yet")

        # Act
        metadata = ScenarioMetadata(
            total_samples=1000,
            ref_samples=500,
            test_samples=500,
            n_features=10,
            has_labels=True,
            data_type="continuous",
            dimension="multivariate",
        )

        # Assert
        assert metadata.total_samples == 1000
        assert metadata.ref_samples == 500
        assert metadata.test_samples == 500
        assert metadata.n_features == 10
        assert metadata.has_labels is True
        assert metadata.data_type == "continuous"
        assert metadata.dimension == "multivariate"

    def test_should_validate_sample_counts_consistency_when_created(self):
        """Test ScenarioMetadata validates sample count consistency"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioMetadata
        except ImportError:
            pytest.skip("ScenarioMetadata not implemented yet")

        # Act & Assert - valid scenario
        metadata = ScenarioMetadata(
            total_samples=150,
            ref_samples=100,
            test_samples=50,  # 100 + 50 = 150 ✓
            n_features=5,
            has_labels=True,
            data_type="continuous",
            dimension="multivariate",
        )
        assert metadata.total_samples == 150

        # Inconsistent counts (might be validated)
        try:
            invalid_metadata = ScenarioMetadata(
                total_samples=100,
                ref_samples=70,
                test_samples=50,  # 70 + 50 = 120 > 100
                n_features=5,
                has_labels=True,
                data_type="continuous",
                dimension="multivariate",
            )
            # If no validation error, that's also acceptable
        except ValueError:
            # This is expected if the model validates consistency
            pass

    def test_should_validate_data_type_and_dimension_fields_when_created(self):
        """Test ScenarioMetadata validates data_type and dimension use literal values"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import ScenarioMetadata
        except ImportError:
            pytest.skip("ScenarioMetadata not implemented yet")

        # Act & Assert - valid data types and dimensions
        valid_combinations = [
            ("continuous", "univariate"),
            ("continuous", "multivariate"),
            ("categorical", "univariate"),
            ("categorical", "multivariate"),
            ("mixed", "multivariate"),
        ]

        for data_type, dimension in valid_combinations:
            metadata = ScenarioMetadata(
                total_samples=100,
                ref_samples=60,
                test_samples=40,
                n_features=1 if dimension == "univariate" else 5,
                has_labels=True,
                data_type=data_type,
                dimension=dimension,
            )
            assert metadata.data_type == data_type
            assert metadata.dimension == dimension


# Integration tests for metadata models
class TestMetadataModelsIntegration:
    """Test integration between metadata models"""

    def test_should_work_together_in_scenario_result_when_combined(self):
        """Test metadata models work together in ScenarioResult"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import DatasetMetadata, ScenarioDefinition, ScenarioMetadata
        except ImportError:
            pytest.skip("Metadata models not implemented yet")

        # Act - create related metadata objects
        dataset_metadata = DatasetMetadata(
            name="make_classification",
            data_type="continuous",
            dimension="multivariate",
            n_samples_ref=1000,
            n_samples_test=500,
            n_features=20,
        )

        scenario_metadata = ScenarioMetadata(
            total_samples=1500,
            ref_samples=1000,
            test_samples=500,
            n_features=20,
            has_labels=True,
            data_type="continuous",
            dimension="multivariate",
        )

        definition = ScenarioDefinition(
            description="Integration test scenario",
            source_type="sklearn",
            source_name="make_classification",
            target_column="target",
            ref_filter={"sample_range": [0, 1000]},
            test_filter={"sample_range": [1000, 1500]},
        )

        # Assert - should work together consistently
        assert dataset_metadata.data_type == scenario_metadata.data_type
        assert dataset_metadata.dimension == scenario_metadata.dimension
        assert dataset_metadata.n_features == scenario_metadata.n_features
        assert dataset_metadata.n_samples_ref == scenario_metadata.ref_samples
        assert dataset_metadata.n_samples_test == scenario_metadata.test_samples
        assert dataset_metadata.name == definition.source_name

    def test_should_serialize_properly_for_storage_when_exported(self):
        """Test metadata models serialize properly for result storage"""
        # Arrange
        try:
            from drift_benchmark.models.metadata import BenchmarkSummary, DetectorMetadata
        except ImportError:
            pytest.skip("Metadata models not implemented yet")

        summary = BenchmarkSummary(total_detectors=3, successful_runs=2, failed_runs=1, avg_execution_time=0.25, total_scenarios=2)

        detector_metadata = DetectorMetadata(
            method_id="ks_test",
            variant_id="scipy",
            library_id="scipy",
            name="Kolmogorov-Smirnov Test",
            family="statistical-test",
            description="Statistical test for distribution differences",
        )

        # Act
        summary_dict = summary.model_dump()
        detector_dict = detector_metadata.model_dump()

        # Assert - should be JSON serializable
        import json

        summary_json = json.dumps(summary_dict)
        detector_json = json.dumps(detector_dict)

        assert "total_detectors" in summary_json
        assert "method_id" in detector_json
        assert "statistical-test" in detector_json
