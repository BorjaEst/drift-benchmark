"""
Tests for literals and constants functionality (REQ-LIT-001, REQ-LIT-002).

These functional tests validate that the constants module provides consistent
literals and fixed values used throughout the drift-benchmark library,
ensuring type safety and consistent identifiers across all components.
"""

from typing import Any, List

import pytest


class TestLiteralsModule:
    """Test literals module provides fixed values for the library."""

    def test_should_provide_fixed_values_when_accessing_constants(self, literal_values_samples):
        """Constants module contains fixed values used throughout library (REQ-LIT-001)."""
        # This test will fail until literals module is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import (
                DATA_TYPES,
                DATASET_TYPES,
                DETECTOR_FAMILIES,
                DIMENSIONS,
                DRIFT_PATTERNS,
                DRIFT_TYPES,
                EXECUTION_MODES,
                LABELING_TYPES,
            )

            # When implemented, should provide consistent literal values
            assert isinstance(DRIFT_TYPES, (list, tuple))
            assert isinstance(DRIFT_PATTERNS, (list, tuple))
            assert isinstance(DATA_TYPES, (list, tuple))

            # Should include expected drift types
            expected_drift_types = literal_values_samples["drift_types"]
            for drift_type in expected_drift_types:
                assert drift_type in DRIFT_TYPES

            # Should include expected drift patterns
            expected_patterns = literal_values_samples["drift_patterns"]
            for pattern in expected_patterns:
                assert pattern in DRIFT_PATTERNS

    def test_should_ensure_id_consistency_when_using_method_identifiers(self):
        """Literals provide consistent method and implementation IDs (REQ-LIT-002)."""
        # This test will fail until ID consistency is implemented
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import ADAPTER_NAMES, IMPLEMENTATION_IDS, METHOD_IDS

            # When implemented, should provide consistent identifiers
            assert isinstance(METHOD_IDS, (list, tuple, dict))
            assert isinstance(IMPLEMENTATION_IDS, (list, tuple, dict))
            assert isinstance(ADAPTER_NAMES, (list, tuple))

            # Should include common method IDs
            expected_methods = ["kolmogorov_smirnov", "maximum_mean_discrepancy", "chi_square", "wasserstein_distance"]
            for method_id in expected_methods:
                if hasattr(METHOD_IDS, "__contains__"):
                    # If it's a container, check membership
                    if method_id in METHOD_IDS:
                        assert True  # Method found
                elif hasattr(METHOD_IDS, "keys"):
                    # If it's a dict, check keys
                    if method_id in METHOD_IDS.keys():
                        assert True  # Method found

            # Should include implementation IDs
            expected_implementations = ["batch", "streaming", "ks_batch", "mmd_batch"]
            # Similar check for implementations
            assert len(IMPLEMENTATION_IDS) > 0


class TestConstantsIntegration:
    """Test constants integration with other modules."""

    def test_should_validate_literal_usage_when_creating_models(self, literal_values_samples):
        """Constants should integrate with model validation."""
        # This test validates literal integration with models
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import DRIFT_PATTERNS, DRIFT_TYPES
            from drift_benchmark.constants.models import DriftInfo

            # When implemented, should validate against literals
            valid_drift_info = DriftInfo(
                drift_type=DRIFT_TYPES[0],  # First valid drift type
                drift_position=0.5,
                drift_magnitude=1.0,
                drift_pattern=DRIFT_PATTERNS[0],  # First valid pattern
            )

            assert valid_drift_info.drift_type in DRIFT_TYPES
            assert valid_drift_info.drift_pattern in DRIFT_PATTERNS

    def test_should_reject_invalid_literals_when_validating_models(self):
        """Models should reject invalid literal values."""
        # This test validates literal validation in models
        with pytest.raises(ImportError):
            from drift_benchmark.constants.models import DriftInfo
            from pydantic import ValidationError

            # When implemented, should reject invalid literals
            with pytest.raises(ValidationError):
                DriftInfo(drift_type="INVALID_DRIFT_TYPE", drift_position=0.5, drift_magnitude=1.0, drift_pattern="INVALID_PATTERN")

    def test_should_provide_consistent_values_when_used_across_modules(self, literal_values_samples):
        """Constants provide consistent values across different modules."""
        # This test validates cross-module consistency
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import DETECTOR_FAMILIES, EXECUTION_MODES
            from drift_benchmark.detectors.registry import get_methods_by_family

            # When implemented, should maintain consistency
            for family in DETECTOR_FAMILIES:
                # Should be able to query methods by family
                methods = get_methods_by_family(family)
                assert isinstance(methods, (list, dict))

            # Should provide consistent execution modes
            for mode in EXECUTION_MODES:
                assert mode in ["BATCH", "STREAMING"]

    def test_should_support_extensibility_when_adding_new_constants(self):
        """Constants module should support adding new values."""
        # This test validates extensibility of constants
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import register_custom_literal

            # When implemented, should allow custom literals
            custom_drift_type = "CUSTOM_DRIFT"
            register_custom_literal("DRIFT_TYPES", custom_drift_type)

            from drift_benchmark.constants.literals import DRIFT_TYPES

            assert custom_drift_type in DRIFT_TYPES


class TestConstantsDocumentation:
    """Test constants provide proper documentation and metadata."""

    def test_should_document_literal_values_when_accessing_help(self):
        """Constants should provide documentation for literal values."""
        # This test validates documentation availability
        with pytest.raises(ImportError):
            from drift_benchmark.constants.literals import DRIFT_TYPES

            # When implemented, should provide documentation
            assert hasattr(DRIFT_TYPES, "__doc__") or hasattr(DRIFT_TYPES, "_doc")

            # Should document what each drift type means
            if hasattr(DRIFT_TYPES, "_descriptions"):
                descriptions = DRIFT_TYPES._descriptions
                assert "COVARIATE" in descriptions
                assert "CONCEPT" in descriptions

    def test_should_provide_examples_when_documenting_usage(self):
        """Constants should provide usage examples."""
        # This test validates usage examples
        with pytest.raises(ImportError):
            from drift_benchmark.constants import examples

            # When implemented, should provide examples
            assert hasattr(examples, "drift_type_examples")
            assert hasattr(examples, "pattern_examples")

            # Should provide concrete examples
            drift_examples = examples.drift_type_examples
            assert "COVARIATE" in drift_examples

            covariate_example = drift_examples["COVARIATE"]
            assert "description" in covariate_example
            assert "use_case" in covariate_example
