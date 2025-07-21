# Session and module-scoped fixtures for shared testing infrastructure

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import toml


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
                "family": "STATISTICAL_TEST",
                "data_dimension": ["UNIVARIATE", "MULTIVARIATE"],
                "data_types": ["CONTINUOUS"],
                "variants": {"scipy": {"name": "SciPy Variant", "execution_mode": "BATCH"}},
            },
            "drift_detector": {
                "name": "Basic Drift Detector",
                "description": "Simple change detection algorithm",
                "family": "CHANGE_DETECTION",
                "data_dimension": ["UNIVARIATE", "MULTIVARIATE"],
                "data_types": ["CONTINUOUS", "CATEGORICAL"],
                "variants": {"custom": {"name": "Custom Variant", "execution_mode": "BATCH"}},
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
        "DRIFT_BENCHMARK_LOG_LEVEL": "DEBUG",
        "DRIFT_BENCHMARK_RANDOM_SEED": "123",
    }


@pytest.fixture(scope="module")
def sample_csv_content():
    """Provide sample CSV content for file-based testing"""
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
        def __init__(self, method_id: str, variant_id: str, library_id: str = "CUSTOM", **kwargs):
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
    """Provide mock BenchmarkConfig for testing"""

    class MockDatasetConfig:
        def __init__(self, path, format, reference_split):
            self.path = path
            self.format = format
            self.reference_split = reference_split

    class MockDetectorConfig:
        def __init__(self, method_id, variant_id, library_id):
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id

    class MockBenchmarkConfig:
        def __init__(self):
            self.datasets = [
                MockDatasetConfig("tests/assets/datasets/test1.csv", "CSV", 0.6),
                MockDatasetConfig("tests/assets/datasets/test2.csv", "CSV", 0.7),
            ]
            self.detectors = [
                MockDetectorConfig("ks_test", "scipy", "SCIPY"),
                MockDetectorConfig("drift_detector", "custom", "CUSTOM"),
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
    metadata.data_type = "MIXED"
    metadata.dimension = "MULTIVARIATE"
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
    """Provide mock detector for testing"""
    from typing import Any, Optional

    import numpy as np

    class MockDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str = "CUSTOM", **kwargs):
            self.method_id = method_id
            self.variant_id = variant_id
            self.library_id = library_id
            self._fitted = False
            self._last_score = None
            self._execution_count = 0

        def preprocess(self, data, **kwargs) -> Any:
            # Return numeric data only for simplicity
            if hasattr(data, "X_ref"):
                return data.X_ref.select_dtypes(include=[np.number]).values
            elif hasattr(data, "X_test"):
                return data.X_test.select_dtypes(include=[np.number]).values
            return data

        def fit(self, preprocessed_data: Any, **kwargs):
            self._fitted = True
            self._reference_data = preprocessed_data
            return self

        def detect(self, preprocessed_data: Any, **kwargs) -> bool:
            if not self._fitted:
                raise RuntimeError("Detector must be fitted before detection")
            self._execution_count += 1
            self._last_score = 0.75 + (self._execution_count * 0.05)  # Varying scores
            return True  # Always detect drift for testing

        def score(self) -> Optional[float]:
            return self._last_score

    return MockDetector


@pytest.fixture
def mock_failing_detector():
    """Provide mock detector that fails for error handling testing"""
    from typing import Any, Optional

    class FailingDetector:
        def __init__(self, method_id: str, variant_id: str, library_id: str = "CUSTOM", **kwargs):
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
