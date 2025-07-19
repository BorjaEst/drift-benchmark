"""
Shared test fixtures and configuration for drift-benchmark tests.

This module provides session and module-scoped fixtures that are used across
all test modules. It includes database connections, shared services, and
common test data that supports functional testing of user workflows.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.models import (
    BenchmarkConfig,
    BenchmarkMetadata,
    DatasetConfig,
    DatasetMetadata,
    DatasetResult,
    DetectorConfig,
    DriftMetadata,
    EvaluationConfig,
    ScoreResult,
)
from drift_benchmark.settings import Settings


@pytest.fixture(scope="session")
def test_workspace() -> Generator[Path, None, None]:
    """Provide a temporary workspace directory for testing.

    Creates a temporary directory structure that mimics the production
    workspace layout for comprehensive functional testing.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="drift_benchmark_test_"))

    # Create workspace structure
    (temp_dir / "components").mkdir()
    (temp_dir / "configurations").mkdir()
    (temp_dir / "datasets").mkdir()
    (temp_dir / "results").mkdir()
    (temp_dir / "logs").mkdir()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def test_settings(test_workspace: Path) -> Settings:
    """Provide test settings configuration.

    Creates a settings instance configured for testing with isolated
    directories and appropriate logging levels for test execution.
    """
    return Settings(
        components_dir=str(test_workspace / "components"),
        configurations_dir=str(test_workspace / "configurations"),
        datasets_dir=str(test_workspace / "datasets"),
        results_dir=str(test_workspace / "results"),
        logs_dir=str(test_workspace / "logs"),
        log_level="DEBUG",
        enable_caching=False,
        max_workers=2,
        random_seed=42,
        memory_limit_mb=1024,
    )


@pytest.fixture(scope="session")
def sample_drift_dataset() -> DatasetResult:
    """Provide a realistic drift dataset for testing.

    Creates a representative dataset with known drift characteristics
    that can be used to validate drift detection workflows.
    """
    np.random.seed(42)

    # Reference data - normal distribution
    n_ref = 500
    X_ref = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_ref),
            "feature_2": np.random.normal(0, 1, n_ref),
            "feature_3": np.random.choice(["A", "B", "C"], n_ref),
        }
    )
    y_ref = pd.Series(np.random.choice([0, 1], n_ref))

    # Test data - shifted distribution (drift)
    n_test = 500
    X_test = pd.DataFrame(
        {
            "feature_1": np.random.normal(1.5, 1, n_test),  # Mean shift
            "feature_2": np.random.normal(0, 1.5, n_test),  # Variance change
            "feature_3": np.random.choice(["A", "B", "C"], n_test, p=[0.1, 0.7, 0.2]),  # Distribution change
        }
    )
    y_test = pd.Series(np.random.choice([0, 1], n_test, p=[0.3, 0.7]))  # Label shift

    drift_info = DriftMetadata(drift_type="COVARIATE", drift_position=0.5, drift_magnitude=1.5, drift_pattern="SUDDEN")

    metadata = DatasetMetadata(
        name="test_drift_dataset",
        description="Test dataset with known drift for validation",
        n_samples=n_ref + n_test,
        n_features=3,
        has_drift=True,
        data_types=["CONTINUOUS", "CATEGORICAL"],
        dimension="MULTIVARIATE",
        labeling="SUPERVISED",
    )

    return DatasetResult(X_ref=X_ref, X_test=X_test, y_ref=y_ref, y_test=y_test, drift_info=drift_info, metadata=metadata)


@pytest.fixture(scope="session")
def sample_no_drift_dataset() -> DatasetResult:
    """Provide a dataset without drift for testing.

    Creates a dataset where reference and test data come from the same
    distribution to validate no-drift scenarios.
    """
    np.random.seed(42)

    # Both reference and test data from same distribution
    n_samples = 500
    X_ref = pd.DataFrame({"feature_1": np.random.normal(0, 1, n_samples), "feature_2": np.random.normal(0, 1, n_samples)})
    X_test = pd.DataFrame({"feature_1": np.random.normal(0, 1, n_samples), "feature_2": np.random.normal(0, 1, n_samples)})

    y_ref = pd.Series(np.random.choice([0, 1], n_samples))
    y_test = pd.Series(np.random.choice([0, 1], n_samples))

    drift_info = DriftMetadata(drift_type=None, drift_position=None, drift_magnitude=0.0, drift_pattern=None)

    metadata = DatasetMetadata(
        name="test_no_drift_dataset",
        description="Test dataset without drift for validation",
        n_samples=n_samples * 2,
        n_features=2,
        has_drift=False,
        data_types=["CONTINUOUS"],
        dimension="MULTIVARIATE",
        labeling="SUPERVISED",
    )

    return DatasetResult(X_ref=X_ref, X_test=X_test, y_ref=y_ref, y_test=y_test, drift_info=drift_info, metadata=metadata)


@pytest.fixture
def mock_score_result() -> ScoreResult:
    """Provide a realistic score result for testing."""
    return ScoreResult(
        drift_detected=True,
        drift_score=0.75,
        threshold=0.5,
        p_value=0.02,
        confidence_interval=(0.65, 0.85),
        metadata={"method": "test_method", "timestamp": "2024-01-01T00:00:00"},
    )


@pytest.fixture
def sample_benchmark_config(test_workspace: Path) -> BenchmarkConfig:
    """Provide a realistic benchmark configuration for testing."""
    return BenchmarkConfig(
        metadata=BenchmarkMetadata(
            name="Test Benchmark", description="Comprehensive benchmark for testing", author="Test Author", version="1.0.0"
        ),
        data=DatasetConfig(datasets=[{"name": "test_dataset", "type": "scenario", "config": {"scenario_name": "iris_species_drift"}}]),
        detectors=DetectorConfig(
            algorithms=[
                {
                    "adapter": "test_adapter",
                    "method_id": "kolmogorov_smirnov",
                    "implementation_id": "ks_batch",
                    "parameters": {"threshold": 0.05},
                }
            ]
        ),
        evaluation=EvaluationConfig(
            classification_metrics=["accuracy", "precision", "recall"],
            detection_metrics=["detection_delay", "auc_score"],
            statistical_tests=["ttest", "mannwhitneyu"],
            performance_analysis=["rankings", "statistical_significance"],
            runtime_analysis=["memory_usage", "cpu_time"],
        ),
    )


@pytest.fixture
def mock_adapter_registry():
    """Provide a mocked adapter registry for testing."""
    mock_registry = Mock()
    mock_adapter_class = Mock()
    mock_registry.get_adapter.return_value = mock_adapter_class
    return mock_registry


@pytest.fixture
def mock_detector():
    """Provide a mocked detector instance for testing."""
    detector = Mock()
    detector.method_id = "test_method"
    detector.implementation_id = "test_implementation"
    detector.fit.return_value = detector
    detector.detect.return_value = True
    detector.score.return_value = ScoreResult(drift_detected=True, drift_score=0.8, threshold=0.5, p_value=0.01)
    detector.preprocess.return_value = "preprocessed_data"
    detector.reset.return_value = None
    return detector


@pytest.fixture
def sample_adapter_directory(test_workspace):
    """Provide a sample adapter directory with test adapter modules."""
    adapter_dir = test_workspace / "components"

    # Create a sample adapter file
    adapter_file = adapter_dir / "sample_adapter.py"
    adapter_content = '''
"""Sample adapter for testing discovery."""

from drift_benchmark.adapters.base import BaseDetector
from drift_benchmark.adapters.registry import register_detector
from drift_benchmark.constants.models import DetectorMetadata, ScoreResult


@register_detector("sample_method", "sample_impl")
class SampleDetector(BaseDetector):
    """Sample detector for testing."""

    @property
    def method_id(self) -> str:
        return "sample_method"

    @property  
    def implementation_id(self) -> str:
        return "sample_impl"

    @classmethod
    def metadata(cls):
        return DetectorMetadata(
            method_id="sample_method",
            implementation_id="sample_impl", 
            name="Sample Detector",
            description="Test detector",
            category="test",
            data_type="tabular",
            streaming=False
        )

    def fit(self, preprocessed_data, **kwargs):
        self._fitted = True
        return self

    def detect(self, preprocessed_data, **kwargs):
        return True
'''
    adapter_file.write_text(adapter_content)

    return adapter_dir
