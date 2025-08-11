"""
River streaming drift detectors.

This module implements drift detectors using the River library for online/streaming
drift detection. River specializes in concept drift detection for streaming data
with adaptive algorithms designed for continuous learning scenarios.

References:
- River Documentation: https://riverml.xyz/
- River Drift Detection: https://riverml.xyz/0.21.0/api/drift/
"""

from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from drift_benchmark.adapters.base_detector import BaseDetector
from drift_benchmark.adapters.registry import register_detector
from drift_benchmark.literals import LibraryId
from drift_benchmark.models.results import ScenarioResult
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)


class RiverStreamingDetector(BaseDetector):
    """
    Base class for River streaming drift detectors.

    Provides common functionality for streaming data simulation and multivariate
    data handling that all River detectors inherit.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self._detectors: Dict[int, Any] = {}  # One detector per feature
        self._drift_history: List[Dict] = []
        self._last_drift_count = 0

    def _simulate_stream(self, data: np.ndarray) -> Iterator[float]:
        """Convert batch data to streaming format."""
        for value in data.flatten():
            yield float(value)

    def _process_multivariate_stream(self, data: np.ndarray, phase: str = "detect") -> bool:
        """
        Handle multivariate data by processing each feature separately.

        Args:
            data: Preprocessed data array
            phase: Either "train" for reference data or "detect" for test data

        Returns:
            Boolean indicating if any feature detected drift
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        drift_signals = []

        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            feature_drift = False

            # Ensure detector exists for this feature
            if feature_idx not in self._detectors:
                continue

            detector = self._detectors[feature_idx]

            for sample_idx, value in enumerate(self._simulate_stream(feature_data)):
                detector.update(value)

                if phase == "detect" and detector.drift_detected:
                    feature_drift = True
                    self._drift_history.append({"feature": feature_idx, "sample": sample_idx, "value": value})
                    break

            drift_signals.append(feature_drift)

        # Return True if any feature shows drift
        overall_drift = any(drift_signals)
        if overall_drift:
            self._last_drift_count = sum(drift_signals)

        return overall_drift

    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> np.ndarray:
        """
        Convert pandas DataFrames to numpy arrays for streaming simulation.

        Args:
            data: ScenarioResult containing X_ref and X_test DataFrames
            phase: Either "train" for training or "detect" for detection
            **kwargs: Additional preprocessing parameters

        Returns:
            np.ndarray: Preprocessed data array suitable for streaming
        """
        if phase == "train":
            X_data = data.X_ref
        elif phase == "detect":
            X_data = data.X_test
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'train' or 'detect'.")

        if isinstance(X_data, pd.DataFrame):
            # Select numeric columns and handle missing values
            numeric_data = X_data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError(f"No numeric columns found in {phase} data")

            # Fill missing values with mean
            processed = numeric_data.fillna(numeric_data.mean()).values
        else:
            processed = np.asarray(X_data)

        # Ensure 2D array for consistent processing
        if len(processed.shape) == 1:
            processed = processed.reshape(-1, 1)

        return processed

    def score(self) -> Optional[float]:
        """Return proportion of features that detected drift."""
        if not self._detectors:
            return None

        # Return proportion of features with drift
        return float(self._last_drift_count) / len(self._detectors)


@register_detector(method_id="adaptive_windowing", variant_id="adwin_standard", library_id="river")
class RiverADWINDetector(RiverStreamingDetector):
    """
    ADWIN (Adaptive Windowing) drift detector from River.

    ADWIN maintains a variable-length window of recent items and detects drift
    when the mean of the window changes significantly.

    References:
    - Bifet & Gavaldà (2007): Learning from Time-Changing Data with Adaptive Windowing
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.delta = kwargs.get("delta", 0.002)  # Confidence level

        logger.info(f"Initialized {self.__class__.__name__} with delta={self.delta}")

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverADWINDetector":
        """
        Initialize ADWIN detectors and train on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for ADWIN detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one ADWIN detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.ADWIN(delta=self.delta)

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                if not np.isnan(value):
                    self._detectors[feature_idx].update(value)

        logger.info(f"ADWIN trained on {n_features} features with {preprocessed_data.shape[0]} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift by processing test data as a stream.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="drift_detection_method", variant_id="ddm_standard", library_id="river")
class RiverDDMDetector(RiverStreamingDetector):
    """
    DDM (Drift Detection Method) from River.

    DDM monitors the error rate of a classifier and detects drift when the
    error rate increases significantly above expected levels.

    References:
    - Gama et al. (2004): Learning with Drift Detection
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.warning_level = kwargs.get("warning_level", 2.0)
        self.drift_level = kwargs.get("drift_level", 3.0)

        logger.info(f"Initialized {self.__class__.__name__} with warning_level={self.warning_level}, drift_level={self.drift_level}")

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverDDMDetector":
        """
        Initialize DDM detectors - DDM doesn't require training on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for DDM detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one DDM detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.DDM(warning_level=self.warning_level, drift_level=self.drift_level)

        logger.info(f"DDM initialized for {n_features} features")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift by monitoring simulated error rates.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        # For demonstration, convert data values to simulated binary errors
        # In practice, this would use actual classification errors
        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="early_drift_detection_method", variant_id="eddm_standard", library_id="river")
class RiverEDDMDetector(RiverStreamingDetector):
    """
    EDDM (Early Drift Detection Method) from River.

    EDDM is an improvement over DDM that detects drift earlier by monitoring
    the distance between classification errors rather than just the error rate.

    References:
    - Baena-García et al. (2006): Early drift detection method
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.alpha = kwargs.get("alpha", 0.95)  # Confidence level for warning
        self.beta = kwargs.get("beta", 0.9)  # Confidence level for drift

        logger.info(f"Initialized {self.__class__.__name__} with alpha={self.alpha}, beta={self.beta}")

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverEDDMDetector":
        """
        Initialize EDDM detectors.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for EDDM detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one EDDM detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.EDDM(alpha=self.alpha, beta=self.beta)

        logger.info(f"EDDM initialized for {n_features} features")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift using EDDM algorithm.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="hoeffding_drift_detection_test_a", variant_id="hddm_a_standard", library_id="river")
class RiverHDDMaDetector(RiverStreamingDetector):
    """
    HDDM_A (Hoeffding Drift Detection Method - Averages) from River.

    HDDM_A uses Hoeffding bounds to detect changes in the mean of a stream.

    References:
    - Frías-Blanco et al. (2015): Online and non-parametric drift detection methods based on Hoeffding's bounds
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.drift_confidence = kwargs.get("drift_confidence", 0.001)
        self.warning_confidence = kwargs.get("warning_confidence", 0.005)

        logger.info(
            f"Initialized {self.__class__.__name__} with drift_confidence={self.drift_confidence}, warning_confidence={self.warning_confidence}"
        )

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverHDDMaDetector":
        """
        Initialize HDDM_A detectors and train on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for HDDM_A detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one HDDM_A detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.HDDM_A(drift_confidence=self.drift_confidence, warning_confidence=self.warning_confidence)

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                if not np.isnan(value):
                    self._detectors[feature_idx].update(value)

        logger.info(f"HDDM_A trained on {n_features} features with {preprocessed_data.shape[0]} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift using HDDM_A algorithm.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="hoeffding_drift_detection_test_w", variant_id="hddm_w_standard", library_id="river")
class RiverHDDMwDetector(RiverStreamingDetector):
    """
    HDDM_W (Hoeffding Drift Detection Method - Weighted) from River.

    HDDM_W uses weighted averages with Hoeffding bounds to detect changes
    that give more importance to recent data.

    References:
    - Frías-Blanco et al. (2015): Online and non-parametric drift detection methods based on Hoeffding's bounds
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.drift_confidence = kwargs.get("drift_confidence", 0.001)
        self.warning_confidence = kwargs.get("warning_confidence", 0.005)
        self.lambda_ = kwargs.get("lambda_", 0.050)  # Decay parameter

        logger.info(
            f"Initialized {self.__class__.__name__} with drift_confidence={self.drift_confidence}, warning_confidence={self.warning_confidence}, lambda_={self.lambda_}"
        )

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverHDDMwDetector":
        """
        Initialize HDDM_W detectors and train on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for HDDM_W detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one HDDM_W detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.HDDM_W(
                drift_confidence=self.drift_confidence, warning_confidence=self.warning_confidence, lambda_=self.lambda_
            )

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                if not np.isnan(value):
                    self._detectors[feature_idx].update(value)

        logger.info(f"HDDM_W trained on {n_features} features with {preprocessed_data.shape[0]} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift using HDDM_W algorithm.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="page_hinkley", variant_id="ph_standard", library_id="river")
class RiverPageHinkleyDetector(RiverStreamingDetector):
    """
    Page-Hinkley drift detector from River.

    The Page-Hinkley test is a sequential analysis technique for detecting
    changes in the mean of a Gaussian signal.

    References:
    - Page (1954): Continuous Inspection Schemes
    - Hinkley (1971): Inference about the change-point from cumulative sum tests
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.min_instances = kwargs.get("min_instances", 30)
        self.delta = kwargs.get("delta", 0.005)
        self.threshold = kwargs.get("threshold", 50)
        self.alpha = kwargs.get("alpha", 1 - 0.0001)

        logger.info(
            f"Initialized {self.__class__.__name__} with min_instances={self.min_instances}, delta={self.delta}, threshold={self.threshold}, alpha={self.alpha}"
        )

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverPageHinkleyDetector":
        """
        Initialize Page-Hinkley detectors and train on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for Page-Hinkley detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one Page-Hinkley detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.PageHinkley(
                min_instances=self.min_instances, delta=self.delta, threshold=self.threshold, alpha=self.alpha
            )

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                if not np.isnan(value):
                    self._detectors[feature_idx].update(value)

        logger.info(f"Page-Hinkley trained on {n_features} features with {preprocessed_data.shape[0]} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift using Page-Hinkley test.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")


@register_detector(method_id="kswin", variant_id="kswin_standard", library_id="river")
class RiverKSWINDetector(RiverStreamingDetector):
    """
    KSWIN (Kolmogorov-Smirnov Windowing) drift detector from River.

    KSWIN uses a sliding window and the Kolmogorov-Smirnov test to detect
    distribution changes in streaming data.

    References:
    - Raab et al. (2020): Reactive Soft Prototype Computing for Concept Drift Streams
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.alpha = kwargs.get("alpha", 0.005)
        self.window_size = kwargs.get("window_size", 100)
        self.stat_size = kwargs.get("stat_size", 30)

        logger.info(
            f"Initialized {self.__class__.__name__} with alpha={self.alpha}, window_size={self.window_size}, stat_size={self.stat_size}"
        )

    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverKSWINDetector":
        """
        Initialize KSWIN detectors and train on reference data.

        Args:
            preprocessed_data: Preprocessed reference data (numpy array)
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from river import drift
        except ImportError:
            raise ImportError("River library is required for KSWIN detector. Install with: pip install river")

        if len(preprocessed_data.shape) == 1:
            preprocessed_data = preprocessed_data.reshape(-1, 1)

        # Initialize one KSWIN detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.KSWIN(alpha=self.alpha, window_size=self.window_size, stat_size=self.stat_size)

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                if not np.isnan(value):
                    self._detectors[feature_idx].update(value)

        logger.info(f"KSWIN trained on {n_features} features with {preprocessed_data.shape[0]} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift using KSWIN algorithm.

        Args:
            preprocessed_data: Preprocessed test data (numpy array)
            **kwargs: Additional detection parameters

        Returns:
            Boolean indicating whether drift was detected
        """
        if not self._detectors:
            raise RuntimeError("Detector must be fitted before detection")

        return self._process_multivariate_stream(preprocessed_data, phase="detect")
