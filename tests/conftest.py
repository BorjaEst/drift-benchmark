# Session and module-scoped fixtures for shared testing infrastructure
# Aligned with README examples and REQUIREMENTS.md Phase 1 implementation
# REFACTORED: Asset-driven testing approach - all shared data loaded from tests/assets/

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import toml

# Test detectors are imported by the tests that need them.
# Automatic import in conftest was causing duplicate registrations.
# Tests should import detectors directly when needed.


# Asset loading utilities
def load_asset_csv(filename: str) -> pd.DataFrame:
    """Load CSV asset from tests/assets/datasets/"""
    assets_dir = Path(__file__).parent / "assets" / "datasets"
    return pd.read_csv(assets_dir / filename)


def load_asset_toml(filename: str, subfolder: str = "configurations") -> dict:
    """Load TOML asset from tests/assets/{subfolder}/"""
    assets_dir = Path(__file__).parent / "assets" / subfolder
    with open(assets_dir / filename, "r") as f:
        return toml.load(f)


def load_asset_config(filename: str, subfolder: str = "configurations") -> dict:
    """
    Load configuration asset from tests/assets/{subfolder}/

    Prioritizes TOML format (.toml) over JSON (.json) for configuration files.
    Includes backward compatibility support with deprecation warnings.

    Args:
        filename: The config filename (with or without extension)
        subfolder: The subfolder under tests/assets/

    Returns:
        dict: Parsed configuration data

    Raises:
        FileNotFoundError: If neither TOML nor JSON version exists
        ValueError: If JSON format is used (with backward compatibility option)
    """
    import warnings

    assets_dir = Path(__file__).parent / "assets" / subfolder

    # Remove extension if provided to allow flexible input
    base_name = Path(filename).stem

    # Priority 1: Try TOML format first
    toml_file = assets_dir / f"{base_name}.toml"
    if toml_file.exists():
        with open(toml_file, "r") as f:
            return toml.load(f)

    # Priority 2: Check for JSON format (backward compatibility)
    json_file = assets_dir / f"{base_name}.json"
    if json_file.exists():
        warnings.warn(
            f"Loading JSON configuration '{json_file.name}' is deprecated. "
            f"Please use the TOML equivalent '{toml_file.name}'. "
            "JSON support will be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
        with open(json_file, "r") as f:
            return json.load(f)

    # Neither format found
    raise FileNotFoundError(
        f"Configuration file '{base_name}' not found in {assets_dir}. " f"Looked for: {toml_file.name}, {json_file.name}"
    )


def load_asset_json(filename: str, subfolder: str = "results") -> dict:
    """
    Load JSON asset from tests/assets/{subfolder}/

    NOTE: For configuration files, use load_asset_config() instead.
    This function should only be used for result/output data that remains in JSON format.
    """
    assets_dir = Path(__file__).parent / "assets" / subfolder
    with open(assets_dir / filename, "r") as f:
        return json.load(f)


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


# Module-scoped fixtures for shared test utilities


@pytest.fixture
def mock_benchmark_config():
    """Provide mock BenchmarkConfig matching README TOML examples and REQ-CFM-002 flat structure - loaded from assets"""

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
            # Load configuration from assets
            config_data = load_asset_toml("basic_benchmark.toml")

            # Create scenarios from loaded data
            self.scenarios = [MockScenarioConfig(scenario["id"]) for scenario in config_data["scenarios"]]

            # Create detectors from loaded data
            self.detectors = [
                MockDetectorConfig(detector["method_id"], detector["variant_id"], detector["library_id"])
                for detector in config_data["detectors"]
            ]

    return MockBenchmarkConfig()


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
    """Provide sample ScenarioResult data for testing - asset-driven approach"""
    # Given: We have standard test datasets as assets
    # When: A test needs scenario result data
    # Then: Load it from assets for consistency
    X_ref = load_asset_csv("ref_data_100.csv")
    X_test = load_asset_csv("test_data_50.csv")

    # Create target data to match expected sizes
    import numpy as np

    y_ref = X_ref["target"] if "target" in X_ref.columns else pd.Series(np.random.choice([0, 1], len(X_ref)))
    y_test = X_test["target"] if "target" in X_test.columns else pd.Series(np.random.choice([0, 1], len(X_test)))

    dataset_metadata = {
        "name": "make_classification",
        "data_type": "continuous",
        "dimension": "multivariate",
        "n_samples_ref": len(X_ref),
        "n_samples_test": len(X_test),
        "n_features": len(X_ref.columns) - (1 if "target" in X_ref.columns else 0),
    }

    scenario_metadata = {
        "total_samples": len(X_ref) + len(X_test),
        "ref_samples": len(X_ref),
        "test_samples": len(X_test),
        "n_features": len(X_ref.columns) - (1 if "target" in X_ref.columns else 0),
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
        "ref_filter": {"sample_range": [0, len(X_ref) - 1]},
        "test_filter": {"sample_range": [len(X_ref), len(X_ref) + len(X_test) - 1]},
    }

    return {
        "name": "covariate_drift_example",
        "X_ref": X_ref[["feature_1", "feature_2"]],  # Features only
        "X_test": X_test[["feature_1", "feature_2"]],  # Features only
        "y_ref": y_ref,
        "y_test": y_test,
        "dataset_metadata": dataset_metadata,
        "scenario_metadata": scenario_metadata,
        "definition": definition,
        # Legacy fields for backwards compatibility
        "ref_data": X_ref,  # Include target column for legacy tests
        "test_data": X_test,  # Include target column for legacy tests
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

        # Use appropriate ranges based on source type and name
        source_type = kwargs.get("source_type", "sklearn")
        source_name = kwargs.get("source_name", "make_classification")

        # Determine dataset size based on known sklearn datasets
        if source_type == "sklearn":
            # Known dataset sizes from sklearn
            dataset_sizes = {
                "make_classification": 1000,
                "make_regression": 1000,
                "make_blobs": 600,
                "load_iris": 150,
                "load_breast_cancer": 569,
                "load_wine": 178,
                "load_diabetes": 442,
            }

            total_size = dataset_sizes.get(source_name, 1000)

            # Split roughly in half with some overlap allowed - use inclusive endpoints
            mid_point = total_size // 2

            # For small datasets, use smaller ranges with inclusive endpoints (indices 0 to total_size-1)
            if total_size <= 200:
                default_ref_filter = {"sample_range": [0, mid_point - 1]}
                default_test_filter = {"sample_range": [mid_point // 2, min(total_size - 1, mid_point + mid_point // 2 - 1)]}
            else:
                # For larger datasets, use standard splits
                default_ref_filter = {"sample_range": [0, mid_point - 1]}
                default_test_filter = {"sample_range": [mid_point // 2, min(total_size - 1, mid_point + mid_point // 2 - 1)]}

        elif source_type == "file" and source_name:
            # For CSV files, determine appropriate ranges by reading the file
            try:
                import pandas as pd

                file_path = Path(source_name)
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    total_rows = len(df)

                    # Split data roughly in half - use inclusive endpoints (indices 0 to total_rows-1)
                    mid_point = total_rows // 2
                    if total_rows < 10:
                        # For small files, use most data for ref, minimal for test
                        ref_end = min(total_rows - 1, max(0, total_rows - 2))
                        test_start = max(0, total_rows - 3) if total_rows > 2 else 0
                        test_end = total_rows - 1
                    else:
                        # For larger files, split more evenly
                        ref_end = mid_point - 1
                        test_start = mid_point
                        test_end = total_rows - 1

                    default_ref_filter = {"sample_range": [0, ref_end]}
                    default_test_filter = {"sample_range": [test_start, test_end]}
                else:
                    # File doesn't exist yet, use small ranges that work with typical test files (10 rows, indices 0-9)
                    default_ref_filter = {"sample_range": [0, 4]}
                    default_test_filter = {"sample_range": [5, 9]}
            except Exception:
                # Fall back to small ranges that work with 10-row test files (indices 0-9)
                default_ref_filter = {"sample_range": [0, 4]}
                default_test_filter = {"sample_range": [5, 9]}
        else:
            # Default fallback
            default_ref_filter = {"sample_range": [0, 500]}
            default_test_filter = {"sample_range": [250, 750]}

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
    """Provide sample ScenarioResult for testing - asset-driven approach"""
    # Given: We have standard test datasets as assets
    # When: A test needs a ScenarioResult mock
    # Then: Load it from assets for consistency
    X_ref_full = load_asset_csv("ref_data_5.csv")
    X_test_full = load_asset_csv("test_data_5.csv")

    X_ref = X_ref_full[["feature_1", "feature_2"]]
    X_test = X_test_full[["feature_1", "feature_2"]]

    # Create target data to match expected sizes
    import numpy as np

    y_ref = X_ref_full["target"] if "target" in X_ref_full.columns else pd.Series(np.random.choice([0, 1], len(X_ref)))
    y_test = X_test_full["target"] if "target" in X_test_full.columns else pd.Series(np.random.choice([0, 1], len(X_test)))

    # Mock metadata with required fields
    class MockMetadata:
        def __init__(self):
            self.description = "Test scenario"
            self.source_type = "sklearn"
            self.source_name = "make_classification"
            self.target_column = "target"
            self.drift_types = ["covariate"]
            self.ref_filter = {"sample_range": [0, len(X_ref) - 1]}
            self.test_filter = {"sample_range": [len(X_ref), len(X_ref) + len(X_test) - 1]}

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
def mock_scenario_result():
    """Provide mock ScenarioResult following REQ-MDL-004 structure and README scenario examples - asset-driven approach"""
    # Given: We have standard test datasets as assets
    # When: A test needs a mock scenario result
    # Then: Load it from assets for consistency
    ref_data = load_asset_csv("standard_ref_data.csv")[["feature_1", "feature_2"]]
    test_data = load_asset_csv("standard_test_data.csv")[["feature_1", "feature_2"]]

    import numpy as np

    y_ref = pd.Series(np.random.choice([0, 1], len(ref_data)))
    y_test = pd.Series(np.random.choice([0, 1], len(test_data)))

    # DatasetMetadata following REQ-MET-001
    dataset_metadata = Mock()
    dataset_metadata.name = "make_classification"
    dataset_metadata.data_type = "continuous"
    dataset_metadata.dimension = "multivariate"
    dataset_metadata.n_samples_ref = len(ref_data)
    dataset_metadata.n_samples_test = len(test_data)

    # ScenarioMetadata following REQ-MET-005 Phase 1 fields
    scenario_metadata = Mock()
    scenario_metadata.total_samples = len(ref_data) + len(test_data)
    scenario_metadata.ref_samples = len(ref_data)
    scenario_metadata.test_samples = len(test_data)
    scenario_metadata.has_labels = True

    # ScenarioDefinition following REQ-MET-004 Phase 1 fields
    definition = Mock()
    definition.description = "Covariate drift scenario with known ground truth"
    definition.source_type = "sklearn"
    definition.source_name = "make_classification"
    definition.target_column = "target"
    definition.ref_filter = {"sample_range": [0, len(ref_data) - 1]}
    definition.test_filter = {"sample_range": [len(ref_data), len(ref_data) + len(test_data) - 1], "noise_factor": 1.5}

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


# === ASSET-DRIVEN FIXTURES ===
# These fixtures load data from tests/assets/ for consistency and maintainability


@pytest.fixture
def dataset_assets_path():
    """Provide path to dataset assets for test data file creation"""
    return Path(__file__).parent / "assets" / "datasets"


@pytest.fixture
def configuration_assets_path():
    """Provide path to configuration assets for test configuration file creation"""
    return Path(__file__).parent / "assets" / "configurations"


@pytest.fixture
def large_dataset_generator():
    """Generate large datasets for performance testing - minimizes hardcoded content"""

    def _generate_csv(num_rows: int = 1000, categories: int = 5) -> str:
        """Generate CSV content with specified number of rows and categories"""
        lines = ["feature_1,feature_2,categorical_feature"]
        for i in range(num_rows):
            lines.append(f"{i * 1.5},{i * 2.3},category_{i % categories}")
        return "\n".join(lines)

    return _generate_csv


@pytest.fixture
def simple_dataframe_factory():
    """Factory for creating simple DataFrames from assets or minimal inline data"""

    def _create_df(df_type: str = "simple"):
        """Create DataFrame based on type - prioritizes assets, falls back to minimal inline"""
        if df_type == "simple":
            # REFACTORED: Load from asset when available, fallback to minimal inline
            try:
                return load_asset_csv("simple_test_data.csv")
            except:
                return pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        elif df_type == "mixed":
            try:
                ref_data = load_asset_csv("mixed_result_ref.csv")
                test_data = load_asset_csv("mixed_result_test.csv")
                return {"ref": ref_data, "test": test_data}
            except:
                ref_data = pd.DataFrame(
                    {"numeric_col": [1.0, 2.5, 3.7], "categorical_col": ["A", "B", "C"], "mixed_col": [1, "text", 3.14]}
                )
                test_data = pd.DataFrame(
                    {"numeric_col": [4.2, 5.1, 6.8], "categorical_col": ["D", "E", "F"], "mixed_col": ["text2", 2, 2.71]}
                )
                return {"ref": ref_data, "test": test_data}
        else:
            raise ValueError(f"Unknown df_type: {df_type}")

    return _create_df


@pytest.fixture
def standard_test_configurations():
    """Provide standard test configurations to avoid repetitive inline config creation"""

    def _get_config(config_type: str = "simple"):
        """Get standard configuration by type"""
        if config_type == "simple":
            return load_asset_config("simple_test_config", "configurations")
        elif config_type == "invalid":
            return load_asset_config("invalid_test_config", "configurations")
        elif config_type == "standard":
            return load_asset_config("standard_benchmark", "configurations")
        else:
            raise ValueError(f"Unknown config_type: {config_type}")

    return _get_config
