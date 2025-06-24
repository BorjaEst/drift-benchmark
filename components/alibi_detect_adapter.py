"""
Adapter for Alibi Detect drift detection algorithms.
This module provides adapters for various drift detectors from the Alibi Detect library.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from drift_benchmark.constants import DetectorMetadata
from drift_benchmark.constants.enums import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode
from drift_benchmark.detectors.base import BaseDetector

try:
    # Import Alibi Detect components
    from alibi_detect.cd import ChiSquareDrift, ClassifierDrift, KSDrift, MMDDrift, TabularDrift
    from alibi_detect.cd.pytorch import preprocess_drift

    HAS_ALIBI_DETECT = True
except ImportError:
    HAS_ALIBI_DETECT = False
    logging.warning("Alibi Detect not installed. To use these detectors, install with `pip install alibi-detect`")


class AlibiDetectBaseAdapter(BaseDetector):
    """Base adapter class for Alibi Detect drift detectors."""

    # Add aliases class attribute
    aliases = []

    def __init__(self, detector_type: str, detector_kwargs: Dict[str, Any], p_val_threshold: float = 0.05, **kwargs):
        """
        Initialize the Alibi Detect adapter.

        Args:
            detector_type: Type of the Alibi Detect drift detector to use
            detector_kwargs: Keyword arguments to pass to the detector constructor
            p_val_threshold: P-value threshold for drift detection
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        super().__init__(**kwargs)

        if not HAS_ALIBI_DETECT:
            raise ImportError(
                "Alibi Detect is required to use this detector. " "Install it with `pip install alibi-detect`"
            )

        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs
        self.p_val_threshold = p_val_threshold
        self.detector = None
        self.is_fitted = False

        # Store drift detection results
        self.drift_detected = False
        self.p_value = None
        self.test_stats = None

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Alibi Detect base adapter.
        This should be overridden by subclasses.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="AlibiDetectBase",
            description="Base adapter for Alibi Detect drift detectors",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MIXED],
            requires_labels=False,
            references=["https://github.com/SeldonIO/alibi-detect"],
            hyperparameters={
                "detector_type": "Type of Alibi Detect drift detector",
                "p_val_threshold": "P-value threshold for drift detection",
            },
        )

    def fit(self, reference_data: np.ndarray, **kwargs) -> "AlibiDetectBaseAdapter":
        """
        Initialize the drift detector with reference data.

        Args:
            reference_data: Reference data used as baseline
            **kwargs: Additional arguments

        Returns:
            Self
        """
        if self.detector_type == "KSDrift":
            self.detector = KSDrift(
                x_ref=reference_data,
                p_val=self.p_val_threshold,
                **self.detector_kwargs,
            )
        elif self.detector_type == "MMDDrift":
            self.detector = MMDDrift(
                x_ref=reference_data,
                p_val=self.p_val_threshold,
                **self.detector_kwargs,
            )
        elif self.detector_type == "ChiSquareDrift":
            self.detector = ChiSquareDrift(
                x_ref=reference_data,
                p_val=self.p_val_threshold,
                **self.detector_kwargs,
            )
        elif self.detector_type == "TabularDrift":
            self.detector = TabularDrift(
                x_ref=reference_data,
                p_val=self.p_val_threshold,
                **self.detector_kwargs,
            )
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

        self.is_fitted = True
        return self

    def detect(self, data: np.ndarray, **kwargs) -> bool:
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

        # Run drift detection
        result = self.detector.predict(data)

        # Extract results
        self.drift_detected = bool(result["data"]["is_drift"])
        self.p_value = result["data"]["p_val"]
        self.test_stats = result["data"].get("distance", None)

        return self.drift_detected

    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        if self.p_value is None:
            raise RuntimeError("No drift detection has been performed")

        scores = {
            "p_value": float(self.p_value),
            "threshold": float(self.p_val_threshold),
            "drift_detected": float(self.drift_detected),
        }

        if self.test_stats is not None:
            scores["test_statistic"] = float(self.test_stats)

        return scores

    def reset(self) -> None:
        """Reset the detector state."""
        if self.is_fitted:
            # Keep the detector but reset detection results
            self.drift_detected = False
            self.p_value = None
            self.test_stats = None


class KSDriftDetector(AlibiDetectBaseAdapter):
    """Kolmogorov-Smirnov drift detector from Alibi Detect."""

    # Add library name as an alias
    aliases = ["KSDrift"]

    def __init__(self, p_val_threshold: float = 0.05, alternative: str = "two-sided", **kwargs):
        """
        Initialize the KS drift detector.

        Args:
            p_val_threshold: P-value threshold for drift detection
            alternative: Alternative hypothesis, one of 'two-sided', 'less', 'greater'
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        detector_kwargs = {
            "alternative": alternative,
        }

        super().__init__(
            detector_type="KSDrift", detector_kwargs=detector_kwargs, p_val_threshold=p_val_threshold, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the KS drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="KSDriftDetector",
            description="Kolmogorov-Smirnov statistical test for drift detection",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.UNIVARIATE,
            data_types=[DataType.CONTINUOUS],
            requires_labels=False,
            references=[
                "https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/ksdrift.html",
                "https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
            ],
            hyperparameters={
                "p_val_threshold": "P-value threshold for drift detection",
                "alternative": "Alternative hypothesis: two-sided, less, greater",
            },
        )


class MMDDriftDetector(AlibiDetectBaseAdapter):
    """Maximum Mean Discrepancy drift detector from Alibi Detect."""

    # Add library name as an alias
    aliases = ["MMDDrift"]

    def __init__(
        self,
        p_val_threshold: float = 0.05,
        kernel: str = "rbf",
        backend: str = "numpy",
        n_permutations: int = 100,
        **kwargs,
    ):
        """
        Initialize the MMD drift detector.

        Args:
            p_val_threshold: P-value threshold for drift detection
            kernel: Kernel used for the MMD computation ('rbf', 'linear')
            backend: Backend to use ('numpy', 'tensorflow', 'pytorch')
            n_permutations: Number of permutations to use for p-value computation
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        detector_kwargs = {
            "kernel": kernel,
            "backend": backend,
            "n_permutations": n_permutations,
        }

        super().__init__(
            detector_type="MMDDrift", detector_kwargs=detector_kwargs, p_val_threshold=p_val_threshold, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the MMD drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="MMDDriftDetector",
            description="Maximum Mean Discrepancy test for drift detection",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.DISTANCE_BASED,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.CONTINUOUS, DataType.CATEGORICAL, DataType.MIXED],
            requires_labels=False,
            references=[
                "https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/mmddrift.html",
                "https://arxiv.org/abs/0805.2368",
            ],
            hyperparameters={
                "p_val_threshold": "P-value threshold for drift detection",
                "kernel": "Kernel used for the MMD computation (rbf, linear)",
                "backend": "Backend to use (numpy, tensorflow, pytorch)",
                "n_permutations": "Number of permutations for p-value computation",
            },
        )


class ChiSquareDriftDetector(AlibiDetectBaseAdapter):
    """Chi-Square drift detector from Alibi Detect."""

    # Add library name as an alias
    aliases = ["ChiSquareDrift"]

    def __init__(
        self, p_val_threshold: float = 0.05, categories_per_feature: Optional[Dict[int, int]] = None, **kwargs
    ):
        """
        Initialize the Chi-Square drift detector.

        Args:
            p_val_threshold: P-value threshold for drift detection
            categories_per_feature: Dict with feature indices as keys and number of categories as values
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        detector_kwargs = {}

        if categories_per_feature is not None:
            detector_kwargs["categories_per_feature"] = categories_per_feature

        super().__init__(
            detector_type="ChiSquareDrift", detector_kwargs=detector_kwargs, p_val_threshold=p_val_threshold, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Chi-Square drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="ChiSquareDriftDetector",
            description="Chi-Square statistical test for detecting drift in categorical data",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.STATISTICAL_TEST,
            data_dimension=DataDimension.UNIVARIATE,
            data_types=[DataType.CATEGORICAL],
            requires_labels=False,
            references=[
                "https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/chisquaredrift.html",
                "https://en.wikipedia.org/wiki/Chi-squared_test",
            ],
            hyperparameters={
                "p_val_threshold": "P-value threshold for drift detection",
                "categories_per_feature": "Dict specifying number of categories per feature",
            },
        )


class TabularDriftDetector(AlibiDetectBaseAdapter):
    """Tabular drift detector from Alibi Detect."""

    # Add library name as an alias
    aliases = ["TabularDrift"]

    def __init__(
        self, p_val_threshold: float = 0.05, categories_per_feature: Optional[Dict[int, int]] = None, **kwargs
    ):
        """
        Initialize the Tabular drift detector.

        Args:
            p_val_threshold: P-value threshold for drift detection
            categories_per_feature: Dict with feature indices as keys and number of categories as values
            **kwargs: Additional keyword arguments for the BaseDetector
        """
        detector_kwargs = {}

        if categories_per_feature is not None:
            detector_kwargs["categories_per_feature"] = categories_per_feature

        super().__init__(
            detector_type="TabularDrift", detector_kwargs=detector_kwargs, p_val_threshold=p_val_threshold, **kwargs
        )

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Provide metadata for the Tabular drift detector.

        Returns:
            DetectorMetadata object with information about the detector
        """
        return DetectorMetadata(
            name="TabularDriftDetector",
            description="Combined statistical tests for detecting drift in tabular data with mixed feature types",
            drift_types=[DriftType.COVARIATE],
            execution_mode=ExecutionMode.BATCH,
            family=DetectorFamily.ENSEMBLE,
            data_dimension=DataDimension.MULTIVARIATE,
            data_types=[DataType.MIXED],
            requires_labels=False,
            references=["https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods/tabulardrift.html"],
            hyperparameters={
                "p_val_threshold": "P-value threshold for drift detection",
                "categories_per_feature": "Dict specifying number of categories per feature",
            },
        )


# Simple tests using pytest
if __name__ == "__main__":
    import pytest

    @pytest.fixture
    def sample_data():
        # Generate reference and test data
        np.random.seed(42)
        reference = np.random.normal(0, 1, size=(1000, 5))
        no_drift = np.random.normal(0, 1, size=(500, 5))
        drift = np.random.normal(0.7, 1, size=(500, 5))
        return reference, no_drift, drift

    def test_ks_drift_detector(sample_data):
        reference, no_drift, drift = sample_data

        # Initialize and fit detector
        detector = KSDriftDetector(p_val_threshold=0.05)
        detector.fit(reference)

        # Test with non-drifting data
        assert not detector.detect(no_drift)
        scores = detector.score()
        assert scores["p_value"] >= scores["threshold"]

        # Test with drifting data
        detector.reset()
        assert detector.detect(drift)
        scores = detector.score()
        assert scores["p_value"] < scores["threshold"]

    def test_mmd_drift_detector(sample_data):
        reference, no_drift, drift = sample_data

        # Initialize and fit detector
        detector = MMDDriftDetector(p_val_threshold=0.05)
        detector.fit(reference)

        # Test with non-drifting data
        assert not detector.detect(no_drift)
        scores = detector.score()
        assert scores["p_value"] >= scores["threshold"]

        # Test with drifting data
        detector.reset()
        assert detector.detect(drift)
        scores = detector.score()
        assert scores["p_value"] < scores["threshold"]
