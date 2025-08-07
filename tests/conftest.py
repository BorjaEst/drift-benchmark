# Session and module-scoped fixtures for shared testing infrastructure
# Aligned with README examples and REQUIREMENTS.md Phase 1 implementation

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import toml

# Import test detectors to auto-register them
try:
    from drift_benchmark.adapters import test_detectors

    print("Successfully imported test detectors from drift_benchmark.adapters")
except ImportError as e:
    print(f"Failed to import test detectors from drift_benchmark.adapters: {e}")

try:
    from .assets.components import test_detectors

    print("Successfully imported test detectors from tests/assets/components")
except ImportError as e:
    print(f"Failed to import test detectors from tests/assets/components: {e}")
    pass  # Skip if not available

# Try other import paths as fallback
try:
    from . import test_detectors  # Import test detectors to register them

    print("Successfully imported test detectors from relative path")
except ImportError as e:
    print(f"Failed to import test detectors from relative path: {e}")
    pass  # Skip if not available

# Try direct import as well
try:
    import test_detectors

    print("Successfully imported test detectors from direct import")
except ImportError as e:
    print(f"Failed to import test detectors from direct import: {e}")
    pass

# Try absolute import as well
try:
    from tests import test_detectors

    print("Successfully imported test detectors from tests package")
except ImportError as e:
    print(f"Failed to import test detectors from tests package: {e}")
    pass


# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="drift_benchmark_test_")
    workspace_path = Path(temp_dir)

    # Create standard directory structure
    (workspace_path / "datasets").mkdir()
    (workspace_path / "results").mkdir()
    (workspace_path / "logs").mkdir()

    yield workspace_path

    # Cleanup after all tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_dataset():
    """Provide sample dataset for testing"""
    import numpy as np

    # Generate synthetic dataset with drift
    np.random.seed(42)

    # Reference data (normal distribution)
    ref_size = 1000
    ref_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, ref_size),
            "feature_2": np.random.normal(0, 1, ref_size),
            "categorical_feature": np.random.choice(["A", "B", "C"], ref_size),
        }
    )

    # Test data (shifted distribution - concept drift)
    test_size = 500
    test_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, test_size),  # Shifted mean
            "feature_2": np.random.normal(0, 1.2, test_size),  # Increased variance
            "categorical_feature": np.random.choice(["A", "B", "C"], test_size, p=[0.6, 0.3, 0.1]),  # Changed distribution
        }
    )

    full_dataset = pd.concat([ref_data, test_data], ignore_index=True)

    return {
        "full_dataset": full_dataset,
        "reference_data": ref_data,
        "test_data": test_data,
        "reference_split": 0.67,  # ref_size / (ref_size + test_size)
    }


@pytest.fixture(scope="session")
def mock_methods_registry():
    """Provide mock methods registry configuration"""
    return {
        "methods": {
            "ks_test": {
                "name": "Kolmogorov-Smirnov Test",
                "description": "Statistical test for distribution differences",
                "family": "statistical-test",
                "data_dimension": ["univariate", "multivariate"],
                "data_types": ["continuous"],
                "variants": {"scipy": {"name": "SciPy Variant", "execution_mode": "batch"}},
            },
            "drift_detector": {
                "name": "Basic Drift Detector",
                "description": "Simple change detection algorithm",
                "family": "change-detection",
                "data_dimension": ["univariate", "multivariate"],
                "data_types": ["continuous", "categorical"],
                "variants": {"custom": {"name": "Custom Variant", "execution_mode": "batch"}},
            },
        }
    }


# Module-scoped fixtures for shared test utilities
@pytest.fixture(scope="module")
def settings_env_vars():
    """Provide environment variable settings for testing"""
    return {
        "DRIFT_BENCHMARK_DATASETS_DIR": "test_datasets",
        "DRIFT_BENCHMARK_RESULTS_DIR": "test_results",
        "DRIFT_BENCHMARK_LOGS_DIR": "test_logs",
        "DRIFT_BENCHMARK_LOG_LEVEL": "debug",
        "DRIFT_BENCHMARK_RANDOM_SEED": "123",
    }


@pytest.fixture(scope="module")
def sample_csv_content():
    """Provide sample csv content for file-based testing"""
    return """feature_1,feature_2,categorical_feature
1.2,0.8,A
-0.5,1.1,B
2.1,-0.3,C
0.7,1.5,A
-1.2,0.2,B
1.8,-0.7,C
0.3,0.9,A
-0.8,1.3,B
1.5,-0.1,C
0.1,0.6,A"""


@pytest.fixture(scope="module")
def mock_detector_variants():
    """Provide mock detector variants for testing"""

    class MockDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str = "custom", **kwargs):
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id
            self._fitted = False
            self._last_score = None

        def preprocess(self, data, **kwargs):
            # Mock preprocessing - return numpy arrays
            if hasattr(data, "X_ref"):
                return data.X_ref.values
            elif hasattr(data, "X_test"):
                return data.X_test.values
            return data.values if hasattr(data, "values") else data

        def fit(self, preprocessed_data, **kwargs):
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data, **kwargs):
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            # Mock drift detection - always detect drift for testing
            self._last_score = 0.75
            return True

        def score(self):
            return self._last_score

    return {"ks_test": {"scipy": MockDetector}, "drift_detector": {"custom": MockDetector}}


@pytest.fixture
def mock_benchmark_config():
    """Provide mock BenchmarkConfig matching README TOML examples and REQ-CFM-002 flat structure"""

    class MockScenarioConfig:
        def __init__(self, id):
            self.id = id

    class MockDetectorConfig:
        def __init__(self, method_id, variant_id, library_id):
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id

    class MockBenchmarkConfig:
        def __init__(self):
            # Scenarios based on README examples
            self.scenarios = [
                MockScenarioConfig("covariate_drift_example"),
                MockScenarioConfig("concept_drift_example"),
            ]
            # Detectors comparing libraries as shown in README
            self.detectors = [
                MockDetectorConfig("kolmogorov_smirnov", "ks_batch", "evidently"),
                MockDetectorConfig("kolmogorov_smirnov", "ks_batch", "alibi-detect"),
                MockDetectorConfig("cramer_von_mises", "cvm_batch", "scipy"),
            ]

    return MockBenchmarkConfig()


@pytest.fixture
def mock_dataset_result():
    """Provide mock DatasetResult for testing"""
    import numpy as np

    ref_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
        }
    )

    test_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, 50),  # Shifted distribution
            "feature_2": np.random.normal(0, 1.2, 50),  # Different variance
            "categorical": np.random.choice(["A", "B", "C"], 50),
        }
    )

    metadata = Mock()
    metadata.name = "mock_dataset"
    metadata.data_type = "mixed"
    metadata.dimension = "multivariate"
    metadata.n_samples_ref = 100
    metadata.n_samples_test = 50

    class MockDatasetResult:
        def __init__(self, X_ref, X_test, metadata):
            self.X_ref = X_ref
            self.X_test = X_test
            self.metadata = metadata

    return MockDatasetResult(ref_data, test_data, metadata)


@pytest.fixture
def mock_detector():
    """Provide mock detector following REQ-ADP-005 and REQ-ADP-010 preprocessing workflow"""
    from typing import Any, Optional

    import numpy as np

    class MockDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str = "custom", **kwargs):
            # REQ-ADP-002, REQ-ADP-003, REQ-ADP-004: Properties
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id
            self._fitted = False
            self._last_score = None
            self._execution_count = 0

        def preprocess(self, data, phase: str = "detect", **kwargs) -> Any:
            """REQ-ADP-005: Extract phase-specific data from ScenarioResult"""
            if phase == "train":
                # Extract X_ref for training
                return data.X_ref.values
            elif phase == "detect":
                # Extract X_test for detection
                return data.X_test.values
            else:
                raise ValueError(f"Invalid phase: {phase}")

        def fit(self, preprocessed_data: Any, **kwargs):
            """REQ-ADP-006: Abstract fit method"""
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            """REQ-ADP-007: Abstract detect method"""
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._execution_count += 1
            self._last_score = 0.75 + (self._execution_count * 0.05)  # Varying scores
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            """REQ-ADP-008: Return drift score after detection"""
            return self._last_score

    return MockDetector


@pytest.fixture
def mock_failing_detector():
    """Provide mock detector that fails for error handling testing"""
    from typing import Any, Optional

    class FailingDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str = "custom", **kwargs):
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id

        def preprocess(self, data, **kwargs) -> Any:
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            raise RuntimeError("Mock detector fit failure")

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            raise RuntimeError("Mock detector detect failure")

        def score(self) -> Optional[float]:
            return None

    return FailingDetector


# Scenario-based fixtures for the new requirements
@pytest.fixture
def sample_scenario_config_data():
    """Provide sample ScenarioConfig data for testing"""
    return {"id": "covariate_drift_example"}


@pytest.fixture
def sample_scenario_result_data():
    """Provide sample ScenarioResult data for testing"""
    import numpy as np

    # Create scenario-based data following REQ-MDL-004 structure
    X_ref = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, 50),  # Shifted distribution
            "feature_2": np.random.normal(0, 1.2, 50),  # Different variance
        }
    )

    y_ref = pd.Series(np.random.choice([0, 1], 100))
    y_test = pd.Series(np.random.choice([0, 1], 50))

    dataset_metadata = {
        "name": "make_classification",
        "data_type": "continuous",
        "dimension": "multivariate",
        "n_samples_ref": 100,
        "n_samples_test": 50,
        "n_features": 2,
    }

    scenario_metadata = {
        "total_samples": 150,
        "ref_samples": 100,
        "test_samples": 50,
        "n_features": 2,
        "has_labels": True,
        "data_type": "continuous",
        "dimension": "multivariate",
    }

    definition = {
        "description": "Sample covariate drift scenario",
        "source_type": "sklearn",
        "source_name": "make_classification",
        "target_column": "target",
        "drift_types": ["covariate"],
        "ref_filter": {"sample_range": [0, 1000]},
        "test_filter": {"sample_range": [1000, 1500]},
    }

    return {
        "name": "covariate_drift_example",
        "X_ref": X_ref,
        "X_test": X_test,
        "y_ref": y_ref,
        "y_test": y_test,
        "dataset_metadata": dataset_metadata,
        "scenario_metadata": scenario_metadata,
        "definition": definition,
    }


@pytest.fixture
def sample_scenario_definition_data():
    """Provide sample ScenarioDefinition data for testing"""
    return {
        "description": "Covariate drift scenario with known ground truth",
        "source_type": "sklearn",
        "source_name": "make_classification",
        "target_column": "target",
        "drift_types": ["covariate"],
        "ref_filter": {"sample_indices": "range(0, 1000)"},
        "test_filter": {"sample_indices": "range(1000, 1500)"},
    }


@pytest.fixture
def sample_scenario_definition():
    """Provide sample ScenarioDefinition factory for testing"""
    from pathlib import Path

    import toml

    # Track created files for cleanup
    created_files = []

    def _create_scenario_definition(scenario_id="test_scenario", **kwargs):
        """Create a ScenarioDefinition with optional overrides and write to TOML file"""
        from pathlib import Path

        import toml

        # Use appropriate ranges based on source type
        source_type = kwargs.get("source_type", "sklearn")
        source_name = kwargs.get("source_name")

        if source_type == "file" and source_name:
            # For CSV files, determine appropriate ranges by reading the file
            try:
                import pandas as pd

                file_path = Path(source_name)
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    total_rows = len(df)

                    # Split data roughly in half
                    mid_point = total_rows // 2
                    if total_rows < 10:
                        # For small files, use all data for ref, minimal for test
                        ref_end = total_rows
                        test_start = max(0, total_rows - 2) if total_rows > 1 else 0
                        test_end = total_rows
                    else:
                        # For larger files, split more evenly
                        ref_end = mid_point
                        test_start = mid_point
                        test_end = total_rows

                    default_ref_filter = {"sample_range": [0, ref_end]}
                    default_test_filter = {"sample_range": [test_start, test_end]}
                else:
                    # File doesn't exist yet, use small ranges
                    default_ref_filter = {"sample_range": [0, 5]}
                    default_test_filter = {"sample_range": [5, 10]}
            except Exception:
                # Fall back to small ranges if file reading fails
                default_ref_filter = {"sample_range": [0, 5]}
                default_test_filter = {"sample_range": [5, 10]}
        else:
            # For sklearn, use larger ranges
            default_ref_filter = {"sample_range": [0, 500]}
            default_test_filter = {"sample_range": [500, 1000]}

        default_data = {
            "description": "Test scenario definition",
            "source_type": "sklearn",
            "source_name": "make_classification",
            "target_column": "target",
            "drift_types": ["covariate"],
            "ground_truth": {},
            "ref_filter": kwargs.get("ref_filter", default_ref_filter),
            "test_filter": kwargs.get("test_filter", default_test_filter),
        }
        default_data.update(kwargs)

        # Write to TOML file in scenarios directory
        scenarios_dir = Path("scenarios")
        scenarios_dir.mkdir(exist_ok=True)
        scenario_file = scenarios_dir / f"{scenario_id}.toml"

        with open(scenario_file, "w") as f:
            toml.dump(default_data, f)

        created_files.append(scenario_file)

        # Also create scenario files for commonly used test scenario names with the same data
        # This ensures tests that create a scenario with specific parameters but load a different ID work
        common_scenarios = [
            "numeric_scenario",
            "categorical_scenario",
            "mixed_scenario",
            "non_existent_scenario",
            "non_existent_scenario_standalone",
            "missing_data_scenario",
            "missing_variations_scenario",
            "bool_scenario",
            "single_scenario",
            "large_scenario",
            "memory_test_scenario",
        ]

        # Create files for commonly used scenario names using the same parameters
        for common_scenario in common_scenarios:
            if common_scenario != scenario_id:  # Don't duplicate the main scenario
                common_file = scenarios_dir / f"{common_scenario}.toml"
                with open(common_file, "w") as f:
                    toml.dump(default_data, f)
                created_files.append(common_file)

        return default_data

    yield _create_scenario_definition

    # Cleanup created files
    for file_path in created_files:
        if file_path.exists():
            file_path.unlink()


@pytest.fixture
def sample_scenario_result():
    """Provide sample ScenarioResult for testing"""
    import numpy as np

    # Create sample data with expected test sizes - separate features from target following REQ-MDL-004
    # Tests expect ref_data to have shape (5, 2)
    X_ref = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 5),
            "feature_2": np.random.normal(0, 1, 5),
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, 5),  # Shifted distribution
            "feature_2": np.random.normal(0, 1.2, 5),  # Different variance
        }
    )

    # Target as separate Series (following REQ-MDL-004)
    y_ref = pd.Series(np.random.choice([0, 1], 5))
    y_test = pd.Series(np.random.choice([0, 1], 5))

    # Mock metadata with required fields
    class MockMetadata:
        def __init__(self):
            self.description = "Test scenario"
            self.source_type = "sklearn"
            self.source_name = "make_classification"
            self.target_column = "target"
            self.drift_types = ["covariate"]
            self.ref_filter = {"sample_range": [0, 5]}
            self.test_filter = {"sample_range": [5, 10]}

    class MockScenarioResult:
        def __init__(self):
            self.name = "test_scenario"
            # Legacy attributes for backward compatibility
            self.ref_data = X_ref
            self.test_data = X_test
            # Modern attributes following REQ-MDL-004
            self.X_ref = X_ref
            self.X_test = X_test
            self.y_ref = y_ref
            self.y_test = y_test
            self.metadata = MockMetadata()

    return MockScenarioResult()


@pytest.fixture
def sample_dataset_metadata_data():
    """Provide sample DatasetMetadata data for testing - describes source dataset from which scenarios are generated"""
    return {
        "name": "sklearn_classification_source",
        "data_type": "continuous",
        "dimension": "multivariate",
        "n_samples_ref": 1000,
        "n_samples_test": 500,
    }


@pytest.fixture
def sample_detector_metadata_data():
    """Provide sample DetectorMetadata data for testing"""
    return {
        "method_id": "kolmogorov_smirnov",
        "variant_id": "batch",
        "library_id": "evidently",
        "name": "Kolmogorov-Smirnov Test",
        "family": "statistical-test",
        "description": "Two-sample test for equality of continuous distributions",
    }


@pytest.fixture
def sample_benchmark_summary_data():
    """Provide sample BenchmarkSummary data for testing"""
    return {
        "total_detectors": 5,
        "successful_runs": 4,  # Test expects this specific value
        "failed_runs": 1,
        "avg_execution_time": 0.0196,  # Modified to match test_models conftest.py
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.9,
    }


@pytest.fixture
def mock_scenario_result():
    """Provide mock ScenarioResult following REQ-MDL-004 structure and README scenario examples"""
    import numpy as np

    # Create scenario data based on README examples
    ref_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 500),
            "feature_2": np.random.normal(0, 1, 500),
        }
    )

    test_data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0.5, 1, 500),  # Shifted distribution (covariate drift)
            "feature_2": np.random.normal(0, 1.2, 500),  # Different variance
        }
    )

    y_ref = pd.Series(np.random.choice([0, 1], 500))
    y_test = pd.Series(np.random.choice([0, 1], 500))

    # DatasetMetadata following REQ-MET-001
    dataset_metadata = Mock()
    dataset_metadata.name = "make_classification"
    dataset_metadata.data_type = "continuous"
    dataset_metadata.dimension = "multivariate"
    dataset_metadata.n_samples_ref = 500
    dataset_metadata.n_samples_test = 500

    # ScenarioMetadata following REQ-MET-005 Phase 1 fields
    scenario_metadata = Mock()
    scenario_metadata.total_samples = 1000
    scenario_metadata.ref_samples = 500
    scenario_metadata.test_samples = 500
    scenario_metadata.has_labels = True

    # ScenarioDefinition following REQ-MET-004 Phase 1 fields
    definition = Mock()
    definition.description = "Covariate drift scenario with known ground truth"
    definition.source_type = "sklearn"
    definition.source_name = "make_classification"
    definition.target_column = "target"
    definition.ref_filter = {"sample_range": [0, 500]}
    definition.test_filter = {"sample_range": [500, 1000], "noise_factor": 1.5}

    class MockScenarioResult:
        def __init__(self, name, X_ref, X_test, y_ref, y_test, dataset_metadata, scenario_metadata, definition):
            self.name = name
            self.X_ref = X_ref
            self.X_test = X_test
            self.y_ref = y_ref
            self.y_test = y_test
            self.dataset_metadata = dataset_metadata
            self.scenario_metadata = scenario_metadata
            self.definition = definition

            # Legacy properties for backwards compatibility
            self.ref_data = X_ref
            self.test_data = X_test

    return MockScenarioResult(
        "covariate_drift_example", ref_data, test_data, y_ref, y_test, dataset_metadata, scenario_metadata, definition
    )


@pytest.fixture(autouse=True)
def clear_detector_registry():
    """Clear detector registry before each test to ensure test isolation"""
    # Clear registry before test
    try:
        from drift_benchmark.adapters.registry import _detector_registry

        original_registry = _detector_registry.copy()
        _detector_registry.clear()
        yield
        # Restore original registry after test
        _detector_registry.clear()
        _detector_registry.update(original_registry)
    except ImportError:
        # Module doesn't exist yet, just yield
        yield
