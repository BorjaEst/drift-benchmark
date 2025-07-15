"""
Tests for the detectors module.

This module tests the drift detection methods registry system, including:
- Method loading and validation
- Filtering and searching capabilities
- Metadata access and validation
- Registry management and caching
"""

import os
import tempfile

import pytest
import tomli

from drift_benchmark.constants.models import DetectorData, ImplementationData, MethodData
from drift_benchmark.detectors import (
    METHODS_TOML,
    detector_exists,
    filter_detectors,
    filter_methods,
    get_data_dimensions,
    get_data_types,
    get_detector,
    get_detector_ids,
    get_drift_types,
    get_execution_modes,
    get_families,
    get_method,
    get_method_ids,
    get_method_implementation_count,
    get_summary,
    list_all_detectors,
    list_all_methods,
    load_methods,
    method_exists,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_methods_data():
    """Sample methods data for testing."""
    return {
        "test_method_1": {
            "name": "Test Method 1",
            "description": "A test statistical method for drift detection.",
            "drift_types": ["COVARIATE", "CONCEPT"],
            "family": "STATISTICAL_TEST",
            "data_dimension": "UNIVARIATE",
            "data_types": ["CONTINUOUS"],
            "requires_labels": False,
            "references": ["https://example.com/test1"],
            "implementations": {
                "batch_impl": {
                    "name": "Batch Implementation",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["threshold", "alpha"],
                    "references": [],
                },
                "streaming_impl": {
                    "name": "Streaming Implementation",
                    "execution_mode": "STREAMING",
                    "hyperparameters": ["threshold", "window_size"],
                    "references": [],
                },
            },
        },
        "test_method_2": {
            "name": "Test Method 2",
            "description": "A test distance-based method for multivariate drift detection.",
            "drift_types": ["COVARIATE"],
            "family": "DISTANCE_BASED",
            "data_dimension": "MULTIVARIATE",
            "data_types": ["CONTINUOUS", "CATEGORICAL"],
            "requires_labels": True,
            "references": ["https://example.com/test2"],
            "implementations": {
                "batch_impl": {
                    "name": "Batch Distance Implementation",
                    "execution_mode": "BATCH",
                    "hyperparameters": ["threshold", "metric"],
                    "references": [],
                }
            },
        },
    }


@pytest.fixture
def temp_methods_toml(sample_methods_data):
    """Create a temporary methods.toml file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        import toml

        toml.dump(sample_methods_data, f)
        temp_file = f.name

    # Patch the METHODS_TOML path
    original_path = getattr(pytest, "_original_methods_toml", None)
    if original_path is None:
        pytest._original_methods_toml = METHODS_TOML

    import drift_benchmark.detectors

    drift_benchmark.detectors.METHODS_TOML = temp_file

    # Clear the cache to ensure fresh loading
    load_methods.cache_clear()

    yield temp_file

    # Cleanup
    os.unlink(temp_file)
    drift_benchmark.detectors.METHODS_TOML = pytest._original_methods_toml
    load_methods.cache_clear()


# =============================================================================
# BASIC LOADING AND ACCESS TESTS
# =============================================================================


def test_load_methods_from_real_file():
    """Test loading methods from the real methods.toml file."""
    methods = load_methods()

    assert isinstance(methods, dict)
    assert len(methods) > 0

    # Check that all values are MethodData instances
    for method_id, method in methods.items():
        assert isinstance(method_id, str)
        assert isinstance(method, MethodData)
        assert hasattr(method, "name")
        assert hasattr(method, "implementations")
        assert len(method.implementations) > 0


def test_load_methods_caching():
    """Test that load_methods uses caching properly."""
    # Clear cache first
    load_methods.cache_clear()

    # First call should load from file
    methods1 = load_methods()

    # Second call should use cache (same object)
    methods2 = load_methods()

    assert methods1 is methods2  # Same object due to caching


def test_load_methods_with_temp_file(temp_methods_toml, sample_methods_data):
    """Test loading methods from temporary file."""
    methods = load_methods()

    assert len(methods) == 2
    assert "test_method_1" in methods
    assert "test_method_2" in methods

    method1 = methods["test_method_1"]
    assert method1.name == "Test Method 1"
    assert method1.family == "STATISTICAL_TEST"
    assert len(method1.implementations) == 2


def test_get_method_existing(temp_methods_toml):
    """Test getting an existing method."""
    method = get_method("test_method_1")

    assert method is not None
    assert isinstance(method, MethodData)
    assert method.name == "Test Method 1"
    assert method.family == "STATISTICAL_TEST"


def test_get_method_nonexistent(temp_methods_toml):
    """Test getting a non-existent method."""
    method = get_method("nonexistent_method")
    assert method is None


def test_get_detector_existing(temp_methods_toml):
    """Test getting an existing detector implementation."""
    detector = get_detector("test_method_1", "batch_impl")

    assert detector is not None
    assert isinstance(detector, DetectorData)
    assert detector.name == "Test Method 1"
    assert detector.implementation.name == "Batch Implementation"
    assert detector.implementation.execution_mode == "BATCH"


def test_get_detector_nonexistent_method(temp_methods_toml):
    """Test getting detector for non-existent method."""
    detector = get_detector("nonexistent_method", "batch_impl")
    assert detector is None


def test_get_detector_nonexistent_implementation(temp_methods_toml):
    """Test getting non-existent implementation for existing method."""
    detector = get_detector("test_method_1", "nonexistent_impl")
    assert detector is None


# =============================================================================
# LISTING AND ENUMERATION TESTS
# =============================================================================


def test_list_all_methods(temp_methods_toml):
    """Test listing all methods."""
    methods = list_all_methods()

    assert isinstance(methods, list)
    assert len(methods) == 2

    method_names = [m.name for m in methods]
    assert "Test Method 1" in method_names
    assert "Test Method 2" in method_names


def test_list_all_detectors(temp_methods_toml):
    """Test listing all detector implementations."""
    detectors = list_all_detectors()

    assert isinstance(detectors, list)
    assert len(detectors) == 3  # 2 from method_1 + 1 from method_2

    # Check that all are DetectorData instances
    for detector in detectors:
        assert isinstance(detector, DetectorData)
        assert hasattr(detector, "implementation")


def test_get_method_ids(temp_methods_toml):
    """Test getting all method IDs."""
    method_ids = get_method_ids()

    assert isinstance(method_ids, list)
    assert len(method_ids) == 2
    assert "test_method_1" in method_ids
    assert "test_method_2" in method_ids


def test_get_detector_ids(temp_methods_toml):
    """Test getting all detector IDs."""
    detector_ids = get_detector_ids()

    assert isinstance(detector_ids, list)
    assert len(detector_ids) == 3

    expected_ids = [
        ("test_method_1", "batch_impl"),
        ("test_method_1", "streaming_impl"),
        ("test_method_2", "batch_impl"),
    ]

    for expected_id in expected_ids:
        assert expected_id in detector_ids


# =============================================================================
# EXISTENCE AND VALIDATION TESTS
# =============================================================================


def test_method_exists(temp_methods_toml):
    """Test method existence checking."""
    assert method_exists("test_method_1") is True
    assert method_exists("test_method_2") is True
    assert method_exists("nonexistent_method") is False


def test_detector_exists(temp_methods_toml):
    """Test detector existence checking."""
    assert detector_exists("test_method_1", "batch_impl") is True
    assert detector_exists("test_method_1", "streaming_impl") is True
    assert detector_exists("test_method_2", "batch_impl") is True

    assert detector_exists("nonexistent_method", "batch_impl") is False
    assert detector_exists("test_method_1", "nonexistent_impl") is False


def test_get_method_implementation_count(temp_methods_toml):
    """Test getting implementation count for methods."""
    assert get_method_implementation_count("test_method_1") == 2
    assert get_method_implementation_count("test_method_2") == 1
    assert get_method_implementation_count("nonexistent_method") == 0


# =============================================================================
# FILTERING AND SEARCHING TESTS
# =============================================================================


def test_filter_methods_by_drift_type(temp_methods_toml):
    """Test filtering methods by drift type."""
    # Filter by single drift type
    covariate_methods = filter_methods(drift_types="COVARIATE")
    assert len(covariate_methods) == 2  # Both methods support COVARIATE

    concept_methods = filter_methods(drift_types="CONCEPT")
    assert len(concept_methods) == 1  # Only test_method_1 supports CONCEPT

    # Filter by multiple drift types
    multi_drift_methods = filter_methods(drift_types=["COVARIATE", "CONCEPT"])
    assert len(multi_drift_methods) == 2  # Methods supporting either type


def test_filter_methods_by_family(temp_methods_toml):
    """Test filtering methods by family."""
    statistical_methods = filter_methods(family="STATISTICAL_TEST")
    assert len(statistical_methods) == 1
    assert statistical_methods[0].name == "Test Method 1"

    distance_methods = filter_methods(family="DISTANCE_BASED")
    assert len(distance_methods) == 1
    assert distance_methods[0].name == "Test Method 2"


def test_filter_methods_by_data_dimension(temp_methods_toml):
    """Test filtering methods by data dimension."""
    univariate_methods = filter_methods(data_dimension="UNIVARIATE")
    assert len(univariate_methods) == 1
    assert univariate_methods[0].name == "Test Method 1"

    multivariate_methods = filter_methods(data_dimension="MULTIVARIATE")
    assert len(multivariate_methods) == 1
    assert multivariate_methods[0].name == "Test Method 2"


def test_filter_methods_by_data_types(temp_methods_toml):
    """Test filtering methods by data types."""
    continuous_methods = filter_methods(data_types="CONTINUOUS")
    assert len(continuous_methods) == 2  # Both support continuous

    categorical_methods = filter_methods(data_types="CATEGORICAL")
    assert len(categorical_methods) == 1  # Only test_method_2 supports categorical

    # Multiple data types
    multi_type_methods = filter_methods(data_types=["CONTINUOUS", "CATEGORICAL"])
    assert len(multi_type_methods) == 2  # Methods supporting either type


def test_filter_methods_by_requires_labels(temp_methods_toml):
    """Test filtering methods by label requirements."""
    unsupervised_methods = filter_methods(requires_labels=False)
    assert len(unsupervised_methods) == 1
    assert unsupervised_methods[0].name == "Test Method 1"

    supervised_methods = filter_methods(requires_labels=True)
    assert len(supervised_methods) == 1
    assert supervised_methods[0].name == "Test Method 2"


def test_filter_methods_by_method_ids(temp_methods_toml):
    """Test filtering methods by specific IDs."""
    specific_methods = filter_methods(method_ids="test_method_1")
    assert len(specific_methods) == 1
    assert specific_methods[0].name == "Test Method 1"

    multiple_methods = filter_methods(method_ids=["test_method_1", "test_method_2"])
    assert len(multiple_methods) == 2


def test_filter_methods_combined_filters(temp_methods_toml):
    """Test filtering methods with multiple criteria."""
    filtered_methods = filter_methods(
        drift_types="COVARIATE",
        family="STATISTICAL_TEST",
        data_dimension="UNIVARIATE",
        requires_labels=False,
    )
    assert len(filtered_methods) == 1
    assert filtered_methods[0].name == "Test Method 1"

    # Filter that should return no results
    no_results = filter_methods(
        family="STATISTICAL_TEST",
        requires_labels=True,  # test_method_1 doesn't require labels
    )
    assert len(no_results) == 0


def test_filter_detectors_by_execution_mode(temp_methods_toml):
    """Test filtering detectors by execution mode."""
    batch_detectors = filter_detectors(execution_mode="BATCH")
    assert len(batch_detectors) == 2  # One from each method

    streaming_detectors = filter_detectors(execution_mode="STREAMING")
    assert len(streaming_detectors) == 1  # Only from test_method_1

    for detector in batch_detectors:
        assert detector.implementation.execution_mode == "BATCH"


def test_filter_detectors_with_method_filters(temp_methods_toml):
    """Test filtering detectors with method-level filters."""
    supervised_detectors = filter_detectors(requires_labels=True)
    assert len(supervised_detectors) == 1
    assert supervised_detectors[0].name == "Test Method 2"

    statistical_detectors = filter_detectors(family="STATISTICAL_TEST")
    assert len(statistical_detectors) == 2  # Both implementations of test_method_1


def test_filter_detectors_by_implementation_ids(temp_methods_toml):
    """Test filtering detectors by implementation IDs."""
    batch_impls = filter_detectors(implementation_ids="batch_impl")
    assert len(batch_impls) == 2  # Both methods have batch_impl

    streaming_impls = filter_detectors(implementation_ids="streaming_impl")
    assert len(streaming_impls) == 1  # Only test_method_1 has streaming_impl


# =============================================================================
# CATEGORY EXTRACTION TESTS
# =============================================================================


def test_get_drift_types(temp_methods_toml):
    """Test getting all available drift types."""
    drift_types = get_drift_types()

    assert isinstance(drift_types, set)
    assert "COVARIATE" in drift_types
    assert "CONCEPT" in drift_types
    assert len(drift_types) == 2


def test_get_families(temp_methods_toml):
    """Test getting all available detector families."""
    families = get_families()

    assert isinstance(families, set)
    assert "STATISTICAL_TEST" in families
    assert "DISTANCE_BASED" in families
    assert len(families) == 2


def test_get_data_dimensions(temp_methods_toml):
    """Test getting all available data dimensions."""
    dimensions = get_data_dimensions()

    assert isinstance(dimensions, set)
    assert "UNIVARIATE" in dimensions
    assert "MULTIVARIATE" in dimensions
    assert len(dimensions) == 2


def test_get_data_types(temp_methods_toml):
    """Test getting all available data types."""
    data_types = get_data_types()

    assert isinstance(data_types, set)
    assert "CONTINUOUS" in data_types
    assert "CATEGORICAL" in data_types
    assert len(data_types) == 2


def test_get_execution_modes(temp_methods_toml):
    """Test getting all available execution modes."""
    execution_modes = get_execution_modes()

    assert isinstance(execution_modes, set)
    assert "BATCH" in execution_modes
    assert "STREAMING" in execution_modes
    assert len(execution_modes) == 2


# =============================================================================
# SUMMARY AND STATISTICS TESTS
# =============================================================================


def test_get_summary(temp_methods_toml):
    """Test getting summary statistics."""
    summary = get_summary()

    assert isinstance(summary, dict)

    # Basic counts
    assert summary["total_methods"] == 2
    assert summary["total_detectors"] == 3
    assert summary["batch_detectors"] == 2
    assert summary["streaming_detectors"] == 1
    assert summary["univariate_methods"] == 1
    assert summary["multivariate_methods"] == 1
    assert summary["supervised_methods"] == 1
    assert summary["unsupervised_methods"] == 1

    # Drift type counts
    assert summary["drift_type_covariate"] == 2
    assert summary["drift_type_concept"] == 1

    # Family counts
    assert summary["family_statistical_test"] == 1
    assert summary["family_distance_based"] == 1


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================


def test_invalid_methods_toml():
    """Test behavior with invalid TOML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write("invalid toml content [[[")
        temp_file = f.name

    # Patch the METHODS_TOML path
    import drift_benchmark.detectors

    original_path = drift_benchmark.detectors.METHODS_TOML
    drift_benchmark.detectors.METHODS_TOML = temp_file
    load_methods.cache_clear()

    try:
        with pytest.raises(tomli.TOMLDecodeError):
            load_methods()
    finally:
        os.unlink(temp_file)
        drift_benchmark.detectors.METHODS_TOML = original_path
        load_methods.cache_clear()


def test_missing_methods_toml():
    """Test behavior with missing TOML file."""
    import drift_benchmark.detectors

    original_path = drift_benchmark.detectors.METHODS_TOML
    drift_benchmark.detectors.METHODS_TOML = "/nonexistent/path/methods.toml"
    load_methods.cache_clear()

    try:
        with pytest.raises(FileNotFoundError):
            load_methods()
    finally:
        drift_benchmark.detectors.METHODS_TOML = original_path
        load_methods.cache_clear()


def test_filter_methods_with_none_values(temp_methods_toml):
    """Test filtering methods with None filter values."""
    # All parameters None should return all methods
    all_methods = filter_methods(
        drift_types=None,
        family=None,
        data_dimension=None,
        data_types=None,
        requires_labels=None,
        method_ids=None,
    )
    assert len(all_methods) == 2


def test_filter_detectors_with_none_values(temp_methods_toml):
    """Test filtering detectors with None filter values."""
    # All parameters None should return all detectors
    all_detectors = filter_detectors(
        drift_types=None,
        execution_mode=None,
        family=None,
        data_dimension=None,
        data_types=None,
        requires_labels=None,
        method_ids=None,
        implementation_ids=None,
    )
    assert len(all_detectors) == 3


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================


def test_backward_compatibility_aliases():
    """Test that backward compatibility aliases work."""
    from drift_benchmark.detectors import (
        get_all_detectors,
        get_available_data_dimensions,
        get_available_data_types,
        get_available_drift_types,
        get_available_execution_modes,
        get_available_families,
        get_detector_by_id,
        get_method_by_id,
        get_methods,
        get_methods_summary,
    )

    # Test that aliases point to the correct functions
    assert get_methods is load_methods
    assert get_method_by_id is get_method
    assert get_detector_by_id is get_detector
    assert get_all_detectors is list_all_detectors
    assert get_available_drift_types is get_drift_types
    assert get_available_families is get_families
    assert get_available_data_dimensions is get_data_dimensions
    assert get_available_data_types is get_data_types
    assert get_available_execution_modes is get_execution_modes
    assert get_methods_summary is get_summary


# =============================================================================
# REAL DATA INTEGRATION TESTS
# =============================================================================


def test_real_methods_toml_structure():
    """Test that the real methods.toml has expected structure."""
    methods = load_methods()

    # Should have common methods
    assert "kolmogorov_smirnov" in methods

    # Test a known method structure
    ks_method = methods["kolmogorov_smirnov"]
    assert ks_method.name == "Kolmogorov-Smirnov Test"
    assert "COVARIATE" in ks_method.drift_types
    assert ks_method.family == "STATISTICAL_TEST"
    assert ks_method.data_dimension == "UNIVARIATE"
    assert "CONTINUOUS" in ks_method.data_types
    assert ks_method.requires_labels is False

    # Should have implementations
    assert len(ks_method.implementations) >= 1
    assert "ks_batch" in ks_method.implementations

    batch_impl = ks_method.implementations["ks_batch"]
    assert batch_impl.execution_mode == "BATCH"
    assert "threshold" in batch_impl.hyperparameters


def test_real_methods_validation():
    """Test that all methods in real methods.toml are valid."""
    methods = load_methods()

    for method_id, method in methods.items():
        # Validate method structure
        assert isinstance(method, MethodData)
        assert len(method.name) > 0
        assert len(method.description) >= 10
        assert len(method.drift_types) >= 1
        assert len(method.data_types) >= 1
        assert len(method.implementations) >= 1

        # Validate implementations
        for impl_id, impl in method.implementations.items():
            assert isinstance(impl, ImplementationData)
            assert len(impl.name) > 0
            assert impl.execution_mode in ["BATCH", "STREAMING"]


def test_comprehensive_filtering_real_data():
    """Test comprehensive filtering on real data."""
    # Test various filter combinations that should return results
    batch_statistical = filter_detectors(execution_mode="BATCH", family="STATISTICAL_TEST")
    assert len(batch_statistical) > 0

    univariate_continuous = filter_methods(data_dimension="UNIVARIATE", data_types="CONTINUOUS")
    assert len(univariate_continuous) > 0

    unsupervised_covariate = filter_methods(requires_labels=False, drift_types="COVARIATE")
    assert len(unsupervised_covariate) > 0
