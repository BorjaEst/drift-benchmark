"""
Adapter for Evidently AI drift detection algorithms.
This module provides adapters for various drift detectors from the Evidently library.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from drift_benchmark.constants import DetectorMetadata
from drift_benchmark.constants.enums import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode
from drift_benchmark.detectors.base import BaseDetector

try:
    # Import Evidently components
    import evidently
    from evidently import ColumnMapping
    from evidently.test_preset import DataDriftTestPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import TestCatTargetDrift, TestColumnDrift, TestNumTargetDrift

    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    logging.warning("Evidently not installed. To use these detectors, install with `pip install evidently`")


class EvidentlyBaseAdapter(BaseDetector):
    """Base adapter class for Evidently drift detectors."""

    # Add aliases class attribute
    aliases = []

    def __init__(
        self,
        test_type: str,
        test_kwargs: Dict[str, Any] = None,
        column_mapping: Optional[Dict[str, Union[str, List[str]]]] = None,
        significance_level: float = 0.05,
        **kwargs,
    ):
        """
        Initialize the Evidently adapter.

        Args:
            test_type: Type of the Evidently test to use
            test_kwargs: Keyword arguments to pass to the test constructor
            column_mapping: Column mapping for reference and current data
            significance_level: Significance level for drift detection
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        super().__init__(**kwargs)

        if not HAS_EVIDENTLY:
            raise ImportError("Evidently is required to use this detector. " "Install it with `pip install evidently`")

        self.test_type = test_type
        self.test_kwargs = test_kwargs or {}
        self.column_mapping = column_mapping
        self.significance_level = significance_level

        self.reference_data = None
        self.is_fitted = False

        # Store drift detection results
        self.test_suite = None
        self.results = None
        self.drift_detected = False
        self.p_values = {}
        self.test_stats = {}

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Evidently base adapter.
        This should be overridden by subclasses.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="EvidentlyBase",
            description="Base adapter for Evidently drift detectors",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MIXED],
            requires_labels=False,
            references=["https://github.com/evidentlyai/evidently"],
            hyperparameters={
                "test_type": "Type of Evidently test to use",
                "significance_level": "Significance level for drift detection",
            },
        )

    def _prepare_data(self, data: np.ndarray) -> pd.DataFrame:
        """
        Convert numpy array to pandas DataFrame with appropriate column names.

        Args:
            data: Input data as numpy array

        Returns:
            Pandas DataFrame with column names
        """
        if isinstance(data, pd.DataFrame):
            return data

        # Convert numpy array to pandas DataFrame
        columns = [f"feature_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=columns)

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "EvidentlyBaseAdapter":
        """
        Initialize the drift detector with reference data.

        Args:
            reference_data: Reference data used as baseline
            **kwargs: Additional arguments

        Returns:
            Self
        """
        self.reference_data = self._prepare_data(reference_data)

        # Create the test suite but don't run it yet
        if self.test_type == "DataDriftPreset":
            self.test_suite = TestSuite(tests=[DataDriftTestPreset(significance_level=self.significance_level)])
        elif self.test_type == "ColumnDrift":
            self.test_suite = TestSuite(
                tests=[
                    TestColumnDrift(
                        column_name=self.test_kwargs.get("column_name", "feature_0"),
                        significance_level=self.significance_level,
                    )
                ]
            )
        elif self.test_type == "CatTargetDrift":
            self.test_suite = TestSuite(tests=[TestCatTargetDrift(significance_level=self.significance_level)])
        elif self.test_type == "NumTargetDrift":
            self.test_suite = TestSuite(tests=[TestNumTargetDrift(significance_level=self.significance_level)])
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")

        self.is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect if drift has occurred.

        Args:
            data: New data batch to check for drift
            **kwargs: Additional arguments

        Returns:
            True if drift is detected, False otherwise
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        current_data = self._prepare_data(data)

        # Run the test suite to detect drift
        self.test_suite.run(
            reference_data=self.reference_data, current_data=current_data, column_mapping=self.column_mapping
        )

        self.results = self.test_suite.as_dict()

        # Extract test results
        tests = self.results.get("tests", [])
        self.drift_detected = False
        self.p_values = {}
        self.test_stats = {}

        for test in tests:
            test_name = test.get("name", "")
            test_result = test.get("result", {}).get("drift_detected", False)
            p_value = test.get("result", {}).get("p_value", None)
            test_stat = test.get("result", {}).get("test_stat", None)

            if test_result:
                self.drift_detected = True

            if p_value is not None:
                self.p_values[test_name] = p_value

            if test_stat is not None:
                self.test_stats[test_name] = test_stat

        return self.drift_detected

    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        if not self.results:
            raise RuntimeError("No drift detection has been performed")

        scores = {"threshold": float(self.significance_level), "drift_detected": int(self.drift_detected)}

        # Add p-values for all tests
        for test_name, p_value in self.p_values.items():
            scores[f"p_value_{test_name}"] = float(p_value)

        # Add test statistics for all tests
        for test_name, test_stat in self.test_stats.items():
            scores[f"test_stat_{test_name}"] = float(test_stat)

        return scores

    def reset(self) -> None:
        """Reset the detector state."""
        if self.is_fitted:
            # Keep the reference data and test_suite but reset detection results
            self.results = None
            self.drift_detected = False
            self.p_values = {}
            self.test_stats = {}


class DataDriftDetector(EvidentlyBaseAdapter):
    """Data Drift detector from Evidently."""

    # Add library name as an alias
    aliases = ["DataDrift"]

    def __init__(
        self,
        significance_level: float = 0.05,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the Data Drift detector.

        Args:
            significance_level: Significance level for drift detection
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        # Create column mapping if features are specified
        column_mapping = None
        if numerical_features or categorical_features:
            column_mapping = ColumnMapping()
            if numerical_features:
                column_mapping.numerical_features = numerical_features
            if categorical_features:
                column_mapping.categorical_features = categorical_features

        super().__init__(
            test_type="DataDriftPreset", column_mapping=column_mapping, significance_level=significance_level, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Data Drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="DataDriftDetector",
            description="Evidently detector for overall data drift across multiple features",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MIXED],
            requires_labels=False,
            references=["https://docs.evidentlyai.com/user-guide/test-presets/data-drift"],
            hyperparameters={
                "significance_level": "Significance level for drift detection",
                "numerical_features": "List of numerical feature names",
                "categorical_features": "List of categorical feature names",
            },
        )


class FeatureDriftDetector(EvidentlyBaseAdapter):
    """Single Feature Drift detector from Evidently."""

    # Add library name as an alias
    aliases = ["FeatureDrift"]

    def __init__(self, column_name: str, significance_level: float = 0.05, **kwargs):
        """
        Initialize the Feature Drift detector.

        Args:
            column_name: Name of the column to monitor for drift
            significance_level: Significance level for drift detection
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        test_kwargs = {"column_name": column_name}

        super().__init__(
            test_type="ColumnDrift", test_kwargs=test_kwargs, significance_level=significance_level, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Feature Drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="FeatureDriftDetector",
            description="Evidently detector for monitoring drift in a single feature",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.UNIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL],
            requires_labels=False,
            references=["https://docs.evidentlyai.com/reference/tests/test-column-drift"],
            hyperparameters={
                "column_name": "Name of the column to monitor for drift",
                "significance_level": "Significance level for drift detection",
            },
        )


class CategoricalTargetDriftDetector(EvidentlyBaseAdapter):
    """Categorical Target Drift detector from Evidently."""

    # Add library name as an alias
    aliases = ["CategoricalTargetDrift", "CatTargetDrift"]

    def __init__(self, target_column: str, significance_level: float = 0.05, **kwargs):
        """
        Initialize the Categorical Target Drift detector.

        Args:
            target_column: Name of the target column
            significance_level: Significance level for drift detection
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        # Create column mapping for target
        column_mapping = ColumnMapping()
        column_mapping.target = target_column

        super().__init__(
            test_type="CatTargetDrift", column_mapping=column_mapping, significance_level=significance_level, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Categorical Target Drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="CategoricalTargetDriftDetector",
            description="Evidently detector for monitoring drift in categorical target variable",
            drift_types=[DriftType.LABEL],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.UNIVARIATE,
            data_types=[DataType.CATEGORICAL],
            requires_labels=True,
            references=["https://docs.evidentlyai.com/reference/tests/test-cat-target-drift"],
            hyperparameters={
                "target_column": "Name of the categorical target column",
                "significance_level": "Significance level for drift detection",
            },
        )


class NumericalTargetDriftDetector(EvidentlyBaseAdapter):
    """Numerical Target Drift detector from Evidently."""

    # Add library name as an alias
    aliases = ["NumericalTargetDrift", "NumTargetDrift"]

    def __init__(self, target_column: str, significance_level: float = 0.05, **kwargs):
        """
        Initialize the Numerical Target Drift detector.

        Args:
            target_column: Name of the target column
            significance_level: Significance level for drift detection
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        # Create column mapping for target
        column_mapping = ColumnMapping()
        column_mapping.target = target_column

        super().__init__(
            test_type="NumTargetDrift", column_mapping=column_mapping, significance_level=significance_level, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Numerical Target Drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="NumericalTargetDriftDetector",
            description="Evidently detector for monitoring drift in numerical target variable",
            drift_types=[DriftType.LABEL],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.UNIVARIATE,
            data_types=[DataType.CONTINUOUS],
            requires_labels=True,
            references=["https://docs.evidentlyai.com/reference/tests/test-num-target-drift"],
            hyperparameters={
                "target_column": "Name of the numerical target column",
                "significance_level": "Significance level for drift detection",
            },
        )


# Simple tests using pytest
if __name__ == "__main__":
    import pytest

    @pytest.fixture
    def sample_data():
        # Generate reference and test data
        np.random.seed(42)
        features = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]

        # Reference data
        ref_data = pd.DataFrame(
            {
                "feature_0": np.random.normal(0, 1, size=1000),
                "feature_1": np.random.normal(0, 1, size=1000),
                "feature_2": np.random.normal(0, 1, size=1000),
                "feature_3": np.random.choice(["A", "B", "C"], size=1000),
                "feature_4": np.random.choice(["X", "Y", "Z"], size=1000),
                "target_num": np.random.normal(0, 1, size=1000),
                "target_cat": np.random.choice(["0", "1"], size=1000),
            }
        )

        # Non-drifting data
        no_drift = pd.DataFrame(
            {
                "feature_0": np.random.normal(0, 1, size=500),
                "feature_1": np.random.normal(0, 1, size=500),
                "feature_2": np.random.normal(0, 1, size=500),
                "feature_3": np.random.choice(["A", "B", "C"], size=500),
                "feature_4": np.random.choice(["X", "Y", "Z"], size=500),
                "target_num": np.random.normal(0, 1, size=500),
                "target_cat": np.random.choice(["0", "1"], size=500),
            }
        )

        # Drifting data
        drift = pd.DataFrame(
            {
                "feature_0": np.random.normal(2, 1, size=500),  # Drift in feature_0
                "feature_1": np.random.normal(0, 1, size=500),
                "feature_2": np.random.normal(0, 1, size=500),
                "feature_3": np.random.choice(["A", "B", "C"], size=500, p=[0.6, 0.3, 0.1]),  # Drift in feature_3
                "feature_4": np.random.choice(["X", "Y", "Z"], size=500),
                "target_num": np.random.normal(1, 1, size=500),  # Drift in target_num
                "target_cat": np.random.choice(["0", "1"], size=500, p=[0.8, 0.2]),  # Drift in target_cat
            }
        )

        return ref_data, no_drift, drift

    def test_data_drift_detector(sample_data):
        reference, no_drift, drift = sample_data

        # Initialize and fit detector
        detector = DataDriftDetector(
            significance_level=0.05,
            numerical_features=["feature_0", "feature_1", "feature_2"],
            categorical_features=["feature_3", "feature_4"],
        )
        detector.fit(reference)

        # Test with non-drifting data
        assert not detector.detect(no_drift)
        scores = detector.score()
        assert scores["drift_detected"] == 0

        # Test with drifting data
        detector.reset()
        assert detector.detect(drift)
        scores = detector.score()
        assert scores["drift_detected"] == 1

    def test_feature_drift_detector(sample_data):
        reference, no_drift, drift = sample_data

        # Initialize and fit detector for feature_0
        detector = FeatureDriftDetector(column_name="feature_0", significance_level=0.05)
        detector.fit(reference)

        # Test with drifting data (feature_0 has drift)
        assert detector.detect(drift)
        scores = detector.score()
        assert scores["drift_detected"] == 1

        # Initialize and fit detector for feature_1
        detector = FeatureDriftDetector(column_name="feature_1", significance_level=0.05)
        detector.fit(reference)

        # Test with drifting data (feature_1 has no drift)
        assert not detector.detect(drift)
        scores = detector.score()
        assert scores["drift_detected"] == 0

    def test_target_drift_detectors(sample_data):
        reference, no_drift, drift = sample_data

        # Test numerical target drift
        num_detector = NumericalTargetDriftDetector(target_column="target_num", significance_level=0.05)
        num_detector.fit(reference)

        assert num_detector.detect(drift)
        scores = num_detector.score()
        assert scores["drift_detected"] == 1

        # Test categorical target drift
        cat_detector = CategoricalTargetDriftDetector(target_column="target_cat", significance_level=0.05)
        cat_detector.fit(reference)

        assert cat_detector.detect(drift)
        scores = cat_detector.score()
        assert scores["drift_detected"] == 1
