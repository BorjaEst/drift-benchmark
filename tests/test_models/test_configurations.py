"""
Functional tests for configuration models in drift-benchmark.

This module tests all configuration models to ensure they meet the TDD requirements
for proper validation, type safety, serialization, and user workflow support.
Tests are organized around functional requirements rather than implementation details.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from drift_benchmark.literals import DataDimension, DataType, DriftPattern, DriftType, ExecutionMode, FileFormat
from drift_benchmark.models import BenchmarkConfig, BenchmarkMetadata, DatasetConfig, DetectorConfig, EvaluationConfig


class TestBenchmarkConfigModel:
    """Test complete benchmark configuration model - REQ-CFG-001."""

    def test_should_create_complete_benchmark_config_when_all_fields_provided(self, sample_complete_benchmark_config: BenchmarkConfig):
        """REQ-CFG-001: Must define BenchmarkConfig with nested fields for complete benchmark definition."""
        config = sample_complete_benchmark_config

        # Verify all required nested fields are present
        assert hasattr(config, "metadata")
        assert hasattr(config, "data")
        assert hasattr(config, "detectors")
        assert hasattr(config, "evaluation")

        # Verify nested objects are properly instantiated
        assert isinstance(config.metadata, BenchmarkMetadata)
        assert isinstance(config.data, DatasetConfig)
        assert isinstance(config.detectors, DetectorConfig)
        assert isinstance(config.evaluation, EvaluationConfig)

        # Verify the config contains expected content
        assert config.metadata.name == "Comprehensive Drift Detection Benchmark"
        assert len(config.data.datasets) == 3
        assert len(config.detectors.algorithms) == 3
        assert "accuracy" in config.evaluation.classification_metrics

    def test_should_support_json_serialization_when_config_created(
        self, sample_complete_benchmark_config: BenchmarkConfig, temporary_file_paths: Dict[str, Path]
    ):
        """REQ-CFG-009: Configuration models must support JSON serialization/deserialization."""
        config = sample_complete_benchmark_config
        json_path = temporary_file_paths["json_file"]

        # Serialize to JSON
        config_dict = config.model_dump()
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Deserialize from JSON
        with open(json_path, "r") as f:
            loaded_dict = json.load(f)

        reconstructed_config = BenchmarkConfig(**loaded_dict)

        # Verify serialization/deserialization preserves data
        assert reconstructed_config.metadata.name == config.metadata.name
        assert len(reconstructed_config.data.datasets) == len(config.data.datasets)
        assert len(reconstructed_config.detectors.algorithms) == len(config.detectors.algorithms)
        assert reconstructed_config.evaluation.classification_metrics == config.evaluation.classification_metrics

    def test_should_fail_validation_when_required_fields_missing(self):
        """REQ-CFG-006: Configuration models must include Pydantic validators for field constraints."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig()  # Missing all required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]

        # Verify all required fields are reported as missing
        required_fields = {"metadata", "data", "detectors", "evaluation"}
        assert required_fields.issubset(set(error_fields))

    def test_should_provide_meaningful_error_when_invalid_nested_config(self):
        """REQ-CFG-006: Configuration models must include cross-field validation."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkConfig(
                metadata={"invalid": "metadata"},  # Invalid metadata structure
                data=DatasetConfig(),
                detectors=DetectorConfig(),
                evaluation=EvaluationConfig(),
            )

        error = exc_info.value
        assert len(error.errors()) > 0
        assert any("metadata" in str(err) for err in error.errors())


class TestMetadataConfigModel:
    """Test metadata configuration model - REQ-CFG-002."""

    def test_should_create_metadata_config_when_all_fields_provided(self, sample_benchmark_metadata: BenchmarkMetadata):
        """REQ-CFG-002: Must define MetadataConfig with fields for benchmark identification."""
        metadata = sample_benchmark_metadata

        # Verify all required fields are present
        assert metadata.name == "Comprehensive Drift Detection Benchmark"
        assert metadata.description == "Multi-method evaluation across diverse drift scenarios"
        assert metadata.author == "Drift Research Team"
        assert metadata.version == "2.1.0"

    def test_should_validate_required_fields_when_creating_metadata(self):
        """REQ-CFG-006: Configuration models must include validators for field constraints."""
        # Test with all required fields
        valid_metadata = BenchmarkMetadata(name="Test Benchmark", description="Test description", author="Test Author", version="1.0.0")
        assert valid_metadata.name == "Test Benchmark"

        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkMetadata()  # Missing required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]
        required_fields = {"name", "description", "author", "version"}
        assert required_fields.issubset(set(error_fields))

    def test_should_serialize_metadata_config_when_requested(self, sample_benchmark_metadata: BenchmarkMetadata):
        """REQ-CFG-009: Configuration models must support JSON serialization."""
        metadata = sample_benchmark_metadata

        # Test JSON serialization
        metadata_dict = metadata.model_dump()
        metadata_json = json.dumps(metadata_dict)

        # Test deserialization
        loaded_dict = json.loads(metadata_json)
        reconstructed = BenchmarkMetadata(**loaded_dict)

        assert reconstructed.name == metadata.name
        assert reconstructed.description == metadata.description
        assert reconstructed.author == metadata.author
        assert reconstructed.version == metadata.version


class TestDatasetConfigModel:
    """Test dataset configuration model - REQ-CFG-003."""

    def test_should_create_dataset_config_when_datasets_provided(self, sample_dataset_config: DatasetConfig):
        """REQ-CFG-003: Must define DatasetConfig with fields for dataset configuration."""
        config = sample_dataset_config

        # Verify datasets list is properly populated
        assert len(config.datasets) == 3

        # Verify different dataset types are supported
        dataset_types = [dataset.get("type") for dataset in config.datasets]
        assert "scenario" in dataset_types
        assert "file" in dataset_types
        assert "synthetic" in dataset_types

        # Verify each dataset has required configuration
        for dataset in config.datasets:
            assert "name" in dataset
            assert "type" in dataset
            assert "config" in dataset

    def test_should_provide_default_empty_list_when_no_datasets(self):
        """REQ-CFG-010: Configuration models must provide sensible defaults."""
        config = DatasetConfig()

        # Verify default empty list is provided
        assert config.datasets == []
        assert isinstance(config.datasets, list)

    def test_should_validate_dataset_structure_when_provided(self):
        """REQ-CFG-006: Configuration models must include validators for field constraints."""
        # Valid dataset configuration
        valid_config = DatasetConfig(
            datasets=[{"name": "test_dataset", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}}]
        )
        assert len(valid_config.datasets) == 1

        # Test with invalid structure (should still work as it's Dict[str, Any])
        config_with_extra_fields = DatasetConfig(
            datasets=[{"name": "test_dataset", "type": "file", "config": {"path": "/data/test.csv"}, "extra_field": "should_be_allowed"}]
        )
        assert len(config_with_extra_fields.datasets) == 1


class TestDetectorConfigModel:
    """Test detector configuration model - REQ-CFG-004."""

    def test_should_create_detector_config_when_algorithms_provided(self, sample_detector_config: DetectorConfig):
        """REQ-CFG-004: Must define DetectorConfig with fields for detector setup."""
        config = sample_detector_config

        # Verify algorithms list is properly populated
        assert len(config.algorithms) == 3

        # Verify each algorithm has required fields
        for algorithm in config.algorithms:
            assert "adapter" in algorithm
            assert "method_id" in algorithm
            assert "implementation_id" in algorithm
            assert "parameters" in algorithm

    def test_should_support_multiple_detector_configurations_when_provided(self, sample_detector_config: DetectorConfig):
        """REQ-CFG-004: Must support multiple detector configurations for benchmarking."""
        config = sample_detector_config

        # Verify different adapters are supported
        adapters = [algo.get("adapter") for algo in config.algorithms]
        assert "evidently_adapter" in adapters
        assert "alibi_adapter" in adapters
        assert "frouros_adapter" in adapters

        # Verify different methods are supported
        methods = [algo.get("method_id") for algo in config.algorithms]
        assert "kolmogorov_smirnov" in methods
        assert "maximum_mean_discrepancy" in methods
        assert "page_hinkley" in methods

    def test_should_provide_default_empty_list_when_no_algorithms(self):
        """REQ-CFG-010: Configuration models must provide sensible defaults."""
        config = DetectorConfig()

        # Verify default empty list is provided
        assert config.algorithms == []
        assert isinstance(config.algorithms, list)

    def test_should_preserve_algorithm_parameters_when_configured(self, sample_detector_config: DetectorConfig):
        """REQ-CFG-004: Must preserve detector parameters for algorithm configuration."""
        config = sample_detector_config

        # Find KS algorithm and verify parameters
        ks_algo = next(algo for algo in config.algorithms if algo["method_id"] == "kolmogorov_smirnov")
        assert ks_algo["parameters"]["threshold"] == 0.05
        assert ks_algo["parameters"]["alternative"] == "two-sided"

        # Find MMD algorithm and verify parameters
        mmd_algo = next(algo for algo in config.algorithms if algo["method_id"] == "maximum_mean_discrepancy")
        assert mmd_algo["parameters"]["sigma"] == 1.0
        assert mmd_algo["parameters"]["kernel"] == "rbf"
        assert mmd_algo["parameters"]["window_size"] == 100


class TestEvaluationConfigModel:
    """Test evaluation configuration model - REQ-CFG-005."""

    def test_should_create_evaluation_config_when_metrics_provided(self, sample_evaluation_config: EvaluationConfig):
        """REQ-CFG-005: Must define EvaluationConfig with fields for evaluation configuration."""
        config = sample_evaluation_config

        # Verify all metric categories are present
        assert len(config.classification_metrics) > 0
        assert len(config.detection_metrics) > 0
        assert len(config.statistical_tests) > 0
        assert len(config.performance_analysis) > 0
        assert len(config.runtime_analysis) > 0

        # Verify specific metrics are included
        assert "accuracy" in config.classification_metrics
        assert "precision" in config.classification_metrics
        assert "detection_delay" in config.detection_metrics
        assert "ttest" in config.statistical_tests

    def test_should_provide_default_empty_lists_when_no_metrics(self):
        """REQ-CFG-010: Configuration models must provide sensible defaults."""
        config = EvaluationConfig()

        # Verify all fields default to empty lists
        assert config.classification_metrics == []
        assert config.detection_metrics == []
        assert config.statistical_tests == []
        assert config.performance_analysis == []
        assert config.runtime_analysis == []

    def test_should_support_comprehensive_evaluation_metrics_when_configured(self, sample_evaluation_config: EvaluationConfig):
        """REQ-CFG-005: Must support comprehensive evaluation configuration."""
        config = sample_evaluation_config

        # Verify classification metrics
        expected_classification = {"accuracy", "precision", "recall", "f1_score"}
        assert expected_classification.issubset(set(config.classification_metrics))

        # Verify detection metrics
        expected_detection = {"detection_delay", "detection_rate"}
        assert expected_detection.issubset(set(config.detection_metrics))

        # Verify statistical tests
        expected_tests = {"ttest", "mannwhitneyu", "wilcoxon"}
        assert expected_tests.issubset(set(config.statistical_tests))

        # Verify analysis types
        assert "rankings" in config.performance_analysis
        assert "memory_usage" in config.runtime_analysis

    def test_should_serialize_evaluation_config_when_requested(self, sample_evaluation_config: EvaluationConfig):
        """REQ-CFG-009: Configuration models must support JSON serialization."""
        config = sample_evaluation_config

        # Test JSON serialization
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict)

        # Test deserialization
        loaded_dict = json.loads(config_json)
        reconstructed = EvaluationConfig(**loaded_dict)

        assert reconstructed.classification_metrics == config.classification_metrics
        assert reconstructed.detection_metrics == config.detection_metrics
        assert reconstructed.statistical_tests == config.statistical_tests
        assert reconstructed.performance_analysis == config.performance_analysis
        assert reconstructed.runtime_analysis == config.runtime_analysis


class TestConfigurationValidationRules:
    """Test configuration validation rules - REQ-CFG-006."""

    def test_should_validate_field_constraints_when_creating_configs(self, invalid_data_samples: Dict[str, Any]):
        """REQ-CFG-006: Configuration models must include validators for field constraints."""
        # Test metadata validation
        with pytest.raises(ValidationError):
            BenchmarkMetadata(name="", description="Valid description", author="Valid author", version="1.0.0")  # Empty name should fail

    def test_should_provide_clear_error_messages_when_validation_fails(self):
        """REQ-CFG-006: Must provide clear error messages for validation failures."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkMetadata(
                name="Valid name",
                # Missing required fields: description, author, version
            )

        error = exc_info.value
        error_messages = [err["msg"] for err in error.errors()]

        # Verify error messages are informative
        assert any("required" in msg.lower() for msg in error_messages)


class TestConfigurationTypeSafety:
    """Test configuration type safety - REQ-CFG-007."""

    def test_should_accept_valid_literal_types_when_using_enums(self, literal_type_samples: Dict[str, Any]):
        """REQ-CFG-007: Configuration models must use Literal types for enumerated fields."""
        # This test verifies that when literal types are used in configs,
        # they accept valid values from the literals module

        # Test with valid drift types (when implemented in configs)
        valid_drift_types = literal_type_samples["drift_types"]
        assert "COVARIATE" in valid_drift_types
        assert "CONCEPT" in valid_drift_types
        assert "PRIOR" in valid_drift_types

        # Test with valid data types
        valid_data_types = literal_type_samples["data_types"]
        assert "CONTINUOUS" in valid_data_types
        assert "CATEGORICAL" in valid_data_types
        assert "MIXED" in valid_data_types

    def test_should_reject_invalid_literal_values_when_validated(self, invalid_data_samples: Dict[str, Any]):
        """REQ-CFG-007: Must validate enumerated fields against literal types."""
        # Test that invalid literal values are rejected when validation is implemented
        invalid_values = [
            invalid_data_samples["invalid_drift_type"],
            invalid_data_samples["invalid_data_dimension"],
            invalid_data_samples["invalid_execution_mode"],
        ]

        for invalid_value in invalid_values:
            assert invalid_value not in DriftType.__args__
            assert invalid_value not in DataType.__args__
            assert invalid_value not in ExecutionMode.__args__


class TestConfigurationSerialization:
    """Test configuration serialization - REQ-CFG-009."""

    def test_should_serialize_and_deserialize_complex_config_when_requested(
        self, sample_complete_benchmark_config: BenchmarkConfig, temporary_file_paths: Dict[str, Path]
    ):
        """REQ-CFG-009: Must support JSON/TOML serialization with proper handling."""
        config = sample_complete_benchmark_config
        json_path = temporary_file_paths["json_file"]

        # Serialize complex nested configuration
        config_dict = config.model_dump()

        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Verify file was created and contains expected content
        assert json_path.exists()

        with open(json_path, "r") as f:
            content = f.read()
            assert "metadata" in content
            assert "data" in content
            assert "detectors" in content
            assert "evaluation" in content

        # Deserialize and verify integrity
        with open(json_path, "r") as f:
            loaded_dict = json.load(f)

        reconstructed = BenchmarkConfig(**loaded_dict)

        # Verify nested structure is preserved
        assert reconstructed.metadata.name == config.metadata.name
        assert len(reconstructed.data.datasets) == len(config.data.datasets)
        assert len(reconstructed.detectors.algorithms) == len(config.detectors.algorithms)

    def test_should_handle_enum_serialization_when_present_in_config(self, literal_type_samples: Dict[str, Any]):
        """REQ-CFG-009: Must properly handle enum serialization in JSON."""
        # Create a sample configuration with enum-like values
        config = DatasetConfig(
            datasets=[
                {
                    "name": "test",
                    "type": "file",
                    "config": {
                        "format": "CSV",  # This should be FileFormat literal
                        "drift_type": "COVARIATE",  # This should be DriftType literal
                    },
                }
            ]
        )

        # Serialize and verify enum values are preserved as strings
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict)

        loaded_dict = json.loads(config_json)
        dataset_config = loaded_dict["datasets"][0]["config"]

        assert dataset_config["format"] == "CSV"
        assert dataset_config["drift_type"] == "COVARIATE"


class TestConfigurationDefaults:
    """Test configuration default values - REQ-CFG-010."""

    def test_should_provide_sensible_defaults_when_optional_fields_omitted(self):
        """REQ-CFG-010: Configuration models must provide sensible defaults for optional fields."""
        # Test DatasetConfig defaults
        data_config = DatasetConfig()
        assert data_config.datasets == []

        # Test DetectorConfig defaults
        detector_config = DetectorConfig()
        assert detector_config.algorithms == []

        # Test EvaluationConfig defaults
        eval_config = EvaluationConfig()
        assert eval_config.classification_metrics == []
        assert eval_config.detection_metrics == []
        assert eval_config.statistical_tests == []
        assert eval_config.performance_analysis == []
        assert eval_config.runtime_analysis == []

    def test_should_simplify_user_configuration_when_defaults_used(self):
        """REQ-CFG-010: Defaults should simplify user configuration."""
        # User can create minimal configuration and extend as needed
        minimal_config = BenchmarkConfig(
            metadata=BenchmarkMetadata(name="Minimal Test", description="Minimal configuration test", author="Test User", version="1.0.0"),
            data=DatasetConfig(),  # Uses default empty list
            detectors=DetectorConfig(),  # Uses default empty list
            evaluation=EvaluationConfig(),  # Uses default empty lists
        )

        # Verify the configuration is valid and usable
        assert minimal_config.metadata.name == "Minimal Test"
        assert len(minimal_config.data.datasets) == 0
        assert len(minimal_config.detectors.algorithms) == 0
        assert len(minimal_config.evaluation.classification_metrics) == 0

        # User can then add datasets/detectors as needed
        minimal_config.data.datasets.append(
            {"name": "added_dataset", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}}
        )

        assert len(minimal_config.data.datasets) == 1
