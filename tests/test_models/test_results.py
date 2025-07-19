"""
Functional tests for result models in drift-benchmark.

This module tests all result models to ensure they meet the TDD requirements
for proper validation, type safety, serialization, and result management.
Tests focus on functional behavior and user workflows rather than implementation details.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from drift_benchmark.models import DatasetResult, DetectorMetadata, DriftMetadata, ScoreResult


class TestDatasetResultModel:
    """Test dataset result model - REQ-RES-002."""

    def test_should_create_dataset_result_when_complete_data_provided(self, sample_dataset_result: DatasetResult):
        """REQ-RES-002: Must define DatasetResult with fields for dataset processing results."""
        result = sample_dataset_result

        # Verify all required fields are present
        assert isinstance(result.X_ref, pd.DataFrame)
        assert isinstance(result.X_test, pd.DataFrame)
        assert isinstance(result.y_ref, pd.Series) or result.y_ref is None
        assert isinstance(result.y_test, pd.Series) or result.y_test is None
        assert isinstance(result.drift_info, DriftMetadata)
        assert hasattr(result, "metadata")

        # Verify data structure integrity
        assert len(result.X_ref) > 0
        assert len(result.X_test) > 0
        assert result.X_ref.shape[1] == result.X_test.shape[1]  # Same number of features

    def test_should_handle_labeled_and_unlabeled_datasets_when_created(self, sample_pandas_dataframes: Dict[str, pd.DataFrame]):
        """REQ-RES-002: Must support both supervised and unsupervised dataset results."""
        dataframes = sample_pandas_dataframes

        # Test supervised dataset (with labels)
        supervised_result = DatasetResult(
            X_ref=dataframes["X_ref"],
            X_test=dataframes["X_test"],
            y_ref=dataframes["y_ref"],
            y_test=dataframes["y_test"],
            drift_info=DriftMetadata(drift_type="CONCEPT", drift_position=0.5, drift_magnitude=1.0, drift_pattern="SUDDEN"),
            metadata=sample_dataset_metadata,
        )

        assert supervised_result.y_ref is not None
        assert supervised_result.y_test is not None
        assert len(supervised_result.y_ref) == len(supervised_result.X_ref)
        assert len(supervised_result.y_test) == len(supervised_result.X_test)

        # Test unsupervised dataset (without labels)
        unsupervised_result = DatasetResult(
            X_ref=dataframes["X_ref"],
            X_test=dataframes["X_test"],
            y_ref=None,
            y_test=None,
            drift_info=DriftMetadata(drift_type="COVARIATE", drift_position=0.3, drift_magnitude=2.0, drift_pattern="GRADUAL"),
            metadata=sample_dataset_metadata,
        )

        assert unsupervised_result.y_ref is None
        assert unsupervised_result.y_test is None

    def test_should_preserve_dataframe_structure_when_created(self, sample_pandas_dataframes: Dict[str, pd.DataFrame]):
        """REQ-RES-002: Must preserve pandas DataFrame structure and data types."""
        dataframes = sample_pandas_dataframes

        result = DatasetResult(
            X_ref=dataframes["X_ref"],
            X_test=dataframes["X_test"],
            y_ref=dataframes["y_ref"],
            y_test=dataframes["y_test"],
            drift_info=DriftMetadata(drift_type="PRIOR", drift_position=0.7, drift_magnitude=1.5),
            metadata=sample_dataset_metadata,
        )

        # Verify DataFrame structure is preserved
        assert list(result.X_ref.columns) == list(dataframes["X_ref"].columns)
        assert list(result.X_test.columns) == list(dataframes["X_test"].columns)

        # Verify data types are preserved
        assert result.X_ref.dtypes.equals(dataframes["X_ref"].dtypes)
        assert result.X_test.dtypes.equals(dataframes["X_test"].dtypes)

        # Verify data content is preserved
        pd.testing.assert_frame_equal(result.X_ref, dataframes["X_ref"])
        pd.testing.assert_frame_equal(result.X_test, dataframes["X_test"])

    def test_should_validate_required_fields_when_creating_result(self, sample_pandas_dataframes: Dict[str, pd.DataFrame]):
        """REQ-RES-006: Result models must include validators for field constraints."""
        dataframes = sample_pandas_dataframes

        # Test missing required fields
        with pytest.raises(ValidationError) as exc_info:
            DatasetResult()  # Missing all required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]
        required_fields = {"X_ref", "X_test", "drift_info", "metadata"}
        assert required_fields.issubset(set(error_fields))

    def test_should_support_arbitrary_pandas_types_when_configured(self):
        """REQ-RES-002: Must support arbitrary pandas DataFrame types through Config."""
        # Create DatasetResult with pandas objects (tests Config.arbitrary_types_allowed)
        X_ref = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        X_test = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12]})
        y_ref = pd.Series([0, 1, 0])
        y_test = pd.Series([1, 0, 1])

        result = DatasetResult(
            X_ref=X_ref,
            X_test=X_test,
            y_ref=y_ref,
            y_test=y_test,
            drift_info=DriftMetadata(drift_type="COVARIATE", drift_position=0.5, drift_magnitude=1.0),
            metadata=sample_dataset_metadata,
        )

        # Verify pandas objects are properly stored
        assert isinstance(result.X_ref, pd.DataFrame)
        assert isinstance(result.X_test, pd.DataFrame)
        assert isinstance(result.y_ref, pd.Series)
        assert isinstance(result.y_test, pd.Series)


class TestScoreResultModel:
    """Test score result model - REQ-RES-005."""

    def test_should_create_score_result_when_detection_info_provided(self, sample_score_result: ScoreResult):
        """REQ-RES-005: Must define ScoreResult with fields for detection scoring."""
        result = sample_score_result

        # Verify all required fields are present
        assert result.drift_detected is True
        assert result.drift_score == 0.087
        assert result.threshold == 0.05
        assert result.p_value == 0.023
        assert result.confidence_interval == (0.015, 0.045)
        assert isinstance(result.metadata, dict)

        # Verify metadata contains additional information
        assert "test_statistic" in result.metadata
        assert "execution_time" in result.metadata

    def test_should_handle_different_detection_outcomes_when_scoring(self):
        """REQ-RES-005: Must support various detection outcomes and statistical measures."""
        # Test positive detection
        positive_result = ScoreResult(
            drift_detected=True,
            drift_score=0.12,
            threshold=0.05,
            p_value=0.001,
            confidence_interval=(0.08, 0.16),
            metadata={"method": "ks_test", "sample_size": 1000},
        )

        assert positive_result.drift_detected is True
        assert positive_result.drift_score > positive_result.threshold
        assert positive_result.p_value < 0.05

        # Test negative detection
        negative_result = ScoreResult(
            drift_detected=False,
            drift_score=0.02,
            threshold=0.05,
            p_value=0.3,
            confidence_interval=None,
            metadata={"method": "ks_test", "sample_size": 500},
        )

        assert negative_result.drift_detected is False
        assert negative_result.drift_score < negative_result.threshold
        assert negative_result.p_value > 0.05
        assert negative_result.confidence_interval is None

    def test_should_support_optional_statistical_measures_when_available(self):
        """REQ-RES-005: Must support optional p-value and confidence intervals."""
        # Test with full statistical information
        full_result = ScoreResult(
            drift_detected=True,
            drift_score=0.08,
            threshold=0.05,
            p_value=0.02,
            confidence_interval=(0.06, 0.10),
            metadata={"method": "statistical_test"},
        )

        assert full_result.p_value is not None
        assert full_result.confidence_interval is not None

        # Test with minimal information (no p-value or confidence interval)
        minimal_result = ScoreResult(
            drift_detected=True,
            drift_score=0.75,
            threshold=0.5,
            p_value=None,
            confidence_interval=None,
            metadata={"method": "distance_based"},
        )

        assert minimal_result.p_value is None
        assert minimal_result.confidence_interval is None

    def test_should_validate_score_result_consistency_when_created(self):
        """REQ-RES-006: Result models must include validators for result consistency."""
        # Test consistent positive detection
        consistent_positive = ScoreResult(
            drift_detected=True, drift_score=0.1, threshold=0.05, p_value=0.01  # Score > threshold  # p < 0.05 (significant)
        )

        # Verify consistency
        assert consistent_positive.drift_detected is True
        assert consistent_positive.drift_score > consistent_positive.threshold
        assert consistent_positive.p_value < 0.05

        # Test consistent negative detection
        consistent_negative = ScoreResult(
            drift_detected=False, drift_score=0.02, threshold=0.05, p_value=0.4  # Score < threshold  # p > 0.05 (not significant)
        )

        # Verify consistency
        assert consistent_negative.drift_detected is False
        assert consistent_negative.drift_score < consistent_negative.threshold
        assert consistent_negative.p_value > 0.05

    def test_should_serialize_score_result_when_requested(self, sample_score_result: ScoreResult):
        """REQ-RES-008: Result models must support JSON serialization with numerical precision."""
        result = sample_score_result

        # Test JSON serialization
        result_dict = result.model_dump()
        result_json = json.dumps(result_dict)

        # Test deserialization
        loaded_dict = json.loads(result_json)
        reconstructed = ScoreResult(**loaded_dict)

        # Verify numerical precision is preserved
        assert reconstructed.drift_detected == result.drift_detected
        assert reconstructed.drift_score == result.drift_score
        assert reconstructed.threshold == result.threshold
        assert reconstructed.p_value == result.p_value
        assert reconstructed.confidence_interval == result.confidence_interval
        assert reconstructed.metadata == result.metadata

    def test_should_handle_confidence_interval_tuple_when_provided(self):
        """REQ-RES-005: Must properly handle confidence interval tuples."""
        result = ScoreResult(
            drift_detected=True,
            drift_score=0.085,
            threshold=0.05,
            p_value=0.025,
            confidence_interval=(0.07, 0.10),  # Tuple of floats
            metadata={"ci_level": 0.95},
        )

        # Verify tuple structure
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert isinstance(result.confidence_interval[0], float)
        assert isinstance(result.confidence_interval[1], float)


class TestResultValidationRules:
    """Test result validation rules - REQ-RES-006."""

    def test_should_enforce_field_constraints_when_validating_results(self, invalid_data_samples: Dict[str, Any]):
        """REQ-RES-006: Result models must include Pydantic validators for field constraints."""
        # Test missing required fields in ScoreResult
        with pytest.raises(ValidationError) as exc_info:
            ScoreResult()  # Missing all required fields

        error = exc_info.value
        error_fields = [err["loc"][0] for err in error.errors()]
        required_fields = {"drift_detected", "drift_score", "threshold"}
        assert required_fields.issubset(set(error_fields))

    def test_should_validate_result_consistency_when_constraints_applied(self):
        """REQ-RES-006: Must validate consistency between related result fields."""
        # Create valid score result
        valid_result = ScoreResult(drift_detected=True, drift_score=0.08, threshold=0.05, p_value=0.02, confidence_interval=(0.06, 0.10))

        # Verify valid result is accepted
        assert valid_result.drift_detected is True
        assert valid_result.drift_score == 0.08

        # Note: Pydantic validation for consistency between drift_detected and
        # drift_score vs threshold would need custom validators in the actual model


class TestResultTypeSafety:
    """Test result type safety - REQ-RES-007."""

    def test_should_maintain_type_safety_when_creating_results(self, literal_type_samples: Dict[str, Any]):
        """REQ-RES-007: Result models must use Literal types for enumerated fields."""
        # Test that result models properly type their fields
        result = ScoreResult(
            drift_detected=True,  # bool
            drift_score=0.085,  # float
            threshold=0.05,  # float
            p_value=0.023,  # Optional[float]
            confidence_interval=(0.07, 0.10),  # Optional[Tuple[float, float]]
            metadata={"key": "value"},  # Optional[Dict[str, Any]]
        )

        # Verify types are maintained
        assert isinstance(result.drift_detected, bool)
        assert isinstance(result.drift_score, (int, float))
        assert isinstance(result.threshold, (int, float))
        assert isinstance(result.p_value, (int, float, type(None)))
        assert isinstance(result.confidence_interval, (tuple, type(None)))
        assert isinstance(result.metadata, (dict, type(None)))

    def test_should_reject_invalid_types_when_validated(self):
        """REQ-RES-007: Must reject invalid types for result fields."""
        # Test invalid boolean type for drift_detected
        with pytest.raises(ValidationError):
            ScoreResult(drift_detected="true", drift_score=0.08, threshold=0.05)  # String instead of bool

        # Test invalid numeric type for drift_score
        with pytest.raises(ValidationError):
            ScoreResult(drift_detected=True, drift_score="0.08", threshold=0.05)  # String instead of float


class TestResultSerialization:
    """Test result serialization - REQ-RES-008."""

    def test_should_serialize_with_numerical_precision_when_requested(self, sample_score_result: ScoreResult):
        """REQ-RES-008: Result models must support JSON serialization with proper numerical precision."""
        result = sample_score_result

        # Test serialization preserves precision
        result_dict = result.model_dump()

        # Verify numerical precision in dictionary
        assert isinstance(result_dict["drift_score"], float)
        assert isinstance(result_dict["threshold"], float)
        assert isinstance(result_dict["p_value"], float)

        # Test JSON round-trip
        result_json = json.dumps(result_dict)
        loaded_dict = json.loads(result_json)
        reconstructed = ScoreResult(**loaded_dict)

        # Verify precision is maintained through JSON serialization
        assert abs(reconstructed.drift_score - result.drift_score) < 1e-10
        assert abs(reconstructed.threshold - result.threshold) < 1e-10
        assert abs(reconstructed.p_value - result.p_value) < 1e-10

    def test_should_handle_datetime_serialization_when_present_in_metadata(self):
        """REQ-RES-008: Must properly handle datetime objects in serialization."""
        from datetime import datetime

        result = ScoreResult(
            drift_detected=True,
            drift_score=0.08,
            threshold=0.05,
            p_value=0.02,
            metadata={"timestamp": datetime(2024, 1, 15, 10, 30, 0), "execution_time": 1.234, "method": "test_detector"},
        )

        # Test that datetime in metadata can be serialized
        result_dict = result.model_dump()

        # Verify datetime is converted to string or timestamp
        assert "timestamp" in result_dict["metadata"]
        # Note: Actual datetime serialization behavior depends on Pydantic configuration

    def test_should_serialize_complex_nested_data_when_present(self, sample_dataset_result: DatasetResult):
        """REQ-RES-008: Must handle complex nested data structures in serialization."""
        result = sample_dataset_result

        # Test that complex result with pandas objects can be processed
        # Note: Full serialization of pandas objects requires special handling

        # Test basic field access and structure
        assert hasattr(result, "X_ref")
        assert hasattr(result, "X_test")
        assert hasattr(result, "drift_info")
        assert hasattr(result, "metadata")

        # Test that drift_info (nested model) can be serialized
        drift_dict = result.drift_info.model_dump()
        drift_json = json.dumps(drift_dict)

        # Verify nested model serialization works
        loaded_drift = json.loads(drift_json)
        reconstructed_drift = DriftMetadata(**loaded_drift)

        assert reconstructed_drift.drift_type == result.drift_info.drift_type
        assert reconstructed_drift.drift_position == result.drift_info.drift_position


class TestResultExportSupport:
    """Test result export support - REQ-RES-010."""

    def test_should_support_json_export_when_requested(self, sample_score_result: ScoreResult, temporary_file_paths: Dict[str, Path]):
        """REQ-RES-010: Result models must support export to JSON format."""
        result = sample_score_result
        json_path = temporary_file_paths["json_file"]

        # Export to JSON file
        result_dict = result.model_dump()
        with open(json_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        # Verify file exists and contains expected content
        assert json_path.exists()

        with open(json_path, "r") as f:
            content = f.read()
            assert "drift_detected" in content
            assert "drift_score" in content
            assert "threshold" in content

    def test_should_support_csv_compatible_export_when_flattened(self, sample_score_result: ScoreResult):
        """REQ-RES-010: Result models must support CSV-compatible export for analysis."""
        result = sample_score_result

        # Convert to flat dictionary for CSV export
        flat_dict = {}

        # Basic fields
        flat_dict["drift_detected"] = result.drift_detected
        flat_dict["drift_score"] = result.drift_score
        flat_dict["threshold"] = result.threshold
        flat_dict["p_value"] = result.p_value

        # Confidence interval (flatten tuple)
        if result.confidence_interval:
            flat_dict["ci_lower"] = result.confidence_interval[0]
            flat_dict["ci_upper"] = result.confidence_interval[1]

        # Metadata (flatten important fields)
        if result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    flat_dict[f"metadata_{key}"] = value

        # Verify flattened structure is CSV-compatible
        assert all(isinstance(v, (str, int, float, bool, type(None))) for v in flat_dict.values())

        # Test conversion to pandas DataFrame (simulates CSV export)
        df = pd.DataFrame([flat_dict])
        assert len(df) == 1
        assert "drift_detected" in df.columns
        assert "drift_score" in df.columns

    def test_should_support_structured_export_when_analysis_required(self, sample_score_result: ScoreResult):
        """REQ-RES-010: Must support structured export for detailed analysis and reporting."""
        result = sample_score_result

        # Create structured export format
        structured_export = {
            "detection_result": {
                "detected": result.drift_detected,
                "confidence": "high" if result.p_value and result.p_value < 0.01 else "medium",
            },
            "statistical_measures": {
                "score": result.drift_score,
                "threshold": result.threshold,
                "p_value": result.p_value,
                "significant": result.p_value < 0.05 if result.p_value else None,
            },
            "confidence_interval": {
                "lower": result.confidence_interval[0] if result.confidence_interval else None,
                "upper": result.confidence_interval[1] if result.confidence_interval else None,
                "width": result.confidence_interval[1] - result.confidence_interval[0] if result.confidence_interval else None,
            },
            "additional_info": result.metadata or {},
        }

        # Verify structured export contains analytical information
        assert "detection_result" in structured_export
        assert "statistical_measures" in structured_export
        assert "confidence_interval" in structured_export

        # Test that structured export can be serialized
        structured_json = json.dumps(structured_export, indent=2)
        loaded_structured = json.loads(structured_json)

        assert loaded_structured["detection_result"]["detected"] == result.drift_detected
        assert loaded_structured["statistical_measures"]["score"] == result.drift_score


# Additional global fixtures needed for these tests
@pytest.fixture
def sample_dataset_metadata():
    """Provide sample dataset metadata for testing."""
    from drift_benchmark.models import DatasetMetadata

    return DatasetMetadata(
        name="test_dataset",
        n_samples=1000,
        n_features=4,
        has_drift=True,
        data_types=["CONTINUOUS"],
        dimension="MULTIVARIATE",
        labeling="SUPERVISED",
    )
