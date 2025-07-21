"""
Example Kolmogorov-Smirnov Drift Detector Adapter

This file demonstrates how to implement a complete drift detector adapter
following the drift-benchmark framework requirements (REQ-ADP-XXX).

This example implements:
- REQ-ADP-001 through REQ-ADP-010: BaseDetector implementation
- REQ-REG-001: Registration using decorator
- Integration with methods.toml configuration
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)


@register_detector(method_id="kolmogorov_smirnov", implementation_id="ks_batch")
class KolmogorovSmirnovDetector(BaseDetector):
    """
    Kolmogorov-Smirnov drift detector implementation.

    This detector implements the two-sample Kolmogorov-Smirnov test to detect
    covariate drift by comparing the empirical distribution functions of
    reference and test datasets.

    References:
    - Massey Jr (1951): https://doi.org/10.2307/2280095
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        """
        Initialize Kolmogorov-Smirnov detector.

        REQ-ADP-008: Accept method and implementation identifiers

        Args:
            method_id: Must be "kolmogorov_smirnov"
            implementation_id: Must be "ks_batch"
            **kwargs: Additional parameters including:
                - threshold (float): P-value threshold for drift detection (default: 0.05)
                - alternative (str): Alternative hypothesis ('two-sided', 'less', 'greater')
        """
        super().__init__(method_id, implementation_id, **kwargs)

        # Hyperparameters from methods.toml
        self.threshold = kwargs.get("threshold", 0.05)
        self.alternative = kwargs.get("alternative", "two-sided")

        # Internal state
        self._reference_data: Optional[np.ndarray] = None
        self._last_statistic: Optional[float] = None
        self._last_pvalue: Optional[float] = None
        self._last_score: Optional[float] = None

        logger.info(f"Initialized {self.__class__.__name__} with threshold={self.threshold}")

    def preprocess(self, data: DatasetResult, **kwargs) -> dict:
        """
        Convert pandas DataFrames to numpy arrays for KS test.

        REQ-ADP-004: Handle data format conversion from pandas DataFrames
        REQ-ADP-009: Extract appropriate data for training/detection phases
        REQ-ADP-010: Format flexibility for detector-specific formats

        Args:
            data: DatasetResult containing X_ref and X_test DataFrames
            **kwargs: Additional preprocessing parameters

        Returns:
            dict: Preprocessed data with numpy arrays

        Note:
            KS test works on univariate data, so we extract the first column
            if multivariate data is provided.
        """
        # Extract reference data (for training phase)
        X_ref = data.X_ref
        if isinstance(X_ref, pd.DataFrame):
            # KS test is univariate - use first column for multivariate data
            ref_values = X_ref.iloc[:, 0].values if X_ref.shape[1] > 0 else np.array([])
        else:
            ref_values = np.asarray(X_ref).flatten()

        # Extract test data (for detection phase)
        X_test = data.X_test
        if isinstance(X_test, pd.DataFrame):
            # Use same column as reference
            test_values = X_test.iloc[:, 0].values if X_test.shape[1] > 0 else np.array([])
        else:
            test_values = np.asarray(X_test).flatten()

        # Remove NaN values (required for KS test)
        ref_values = ref_values[~np.isnan(ref_values)]
        test_values = test_values[~np.isnan(test_values)]

        preprocessed = {
            "X_ref": ref_values,
            "X_test": test_values,
            "metadata": data.metadata,
            "n_ref_samples": len(ref_values),
            "n_test_samples": len(test_values),
        }

        logger.debug(f"Preprocessed data: {preprocessed['n_ref_samples']} ref samples, " f"{preprocessed['n_test_samples']} test samples")

        return preprocessed

    def fit(self, preprocessed_data: Any, **kwargs) -> "KolmogorovSmirnovDetector":
        """
        Train the detector on reference data.

        REQ-ADP-005: Abstract fit method for training on reference data

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional fitting parameters

        Returns:
            self: Fitted detector instance

        Note:
            KS test is non-parametric and doesn't require explicit training,
            but we store the reference data for later comparison.
        """
        if not isinstance(preprocessed_data, dict) or "X_ref" not in preprocessed_data:
            raise ValueError("preprocessed_data must be dict with 'X_ref' key")

        self._reference_data = preprocessed_data["X_ref"]

        if len(self._reference_data) == 0:
            logger.warning("Empty reference data provided to fit()")
        else:
            logger.info(f"Fitted KS detector on {len(self._reference_data)} reference samples")

        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Perform drift detection using KS test.

        REQ-ADP-006: Abstract detect method returning drift detection result

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional detection parameters

        Returns:
            bool: True if drift is detected, False otherwise

        Raises:
            RuntimeError: If detector hasn't been fitted
            ValueError: If test data is invalid
        """
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if not isinstance(preprocessed_data, dict) or "X_test" not in preprocessed_data:
            raise ValueError("preprocessed_data must be dict with 'X_test' key")

        test_data = preprocessed_data["X_test"]

        if len(test_data) == 0:
            logger.warning("Empty test data provided to detect()")
            self._last_statistic = 0.0
            self._last_pvalue = 1.0
            self._last_score = 0.0
            return False

        # Perform two-sample Kolmogorov-Smirnov test
        try:
            statistic, pvalue = stats.ks_2samp(self._reference_data, test_data, alternative=self.alternative)

            # Store results
            self._last_statistic = float(statistic)
            self._last_pvalue = float(pvalue)
            # Use 1 - pvalue as drift score (higher = more drift)
            self._last_score = 1.0 - float(pvalue)

            # Drift detected if p-value < threshold
            drift_detected = pvalue < self.threshold

            logger.info(f"KS test: statistic={statistic:.4f}, pvalue={pvalue:.4f}, " f"drift_detected={drift_detected}")

            return drift_detected

        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            self._last_statistic = None
            self._last_pvalue = None
            self._last_score = None
            raise

    def score(self) -> Optional[float]:
        """
        Return drift score after detection.

        REQ-ADP-007: Score method returning drift score, None if unavailable

        Returns:
            Optional[float]: Drift score (1 - p_value) or None if no detection performed
        """
        return self._last_score

    # Additional methods for enhanced functionality

    def get_test_statistics(self) -> Optional[dict]:
        """
        Get detailed test statistics from last detection.

        Returns:
            Optional[dict]: Dictionary with statistic and pvalue, or None
        """
        if self._last_statistic is not None and self._last_pvalue is not None:
            return {
                "statistic": self._last_statistic,
                "pvalue": self._last_pvalue,
                "threshold": self.threshold,
                "alternative": self.alternative,
            }
        return None

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"KolmogorovSmirnovDetector(method_id='{self.method_id}', "
            f"implementation_id='{self.implementation_id}', "
            f"threshold={self.threshold})"
        )


@register_detector(method_id="kolmogorov_smirnov", implementation_id="ks_incremental")
class IncrementalKolmogorovSmirnovDetector(BaseDetector):
    """
    Incremental Kolmogorov-Smirnov drift detector for streaming data.

    This implementation maintains a sliding window of reference data and
    performs KS tests on incoming data batches.

    References:
    - dos Reis et al. (2016): https://doi.org/10.1145/2939672.2939836
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        """
        Initialize incremental KS detector.

        Args:
            method_id: Must be "kolmogorov_smirnov"
            implementation_id: Must be "ks_incremental"
            **kwargs: Additional parameters including:
                - threshold (float): P-value threshold (default: 0.05)
                - window_size (int): Reference window size (default: 1000)
        """
        super().__init__(method_id, implementation_id, **kwargs)

        self.threshold = kwargs.get("threshold", 0.05)
        self.window_size = kwargs.get("window_size", 1000)

        # Sliding window for reference data
        self._reference_window: list = []
        self._last_score: Optional[float] = None

        logger.info(f"Initialized {self.__class__.__name__} with " f"threshold={self.threshold}, window_size={self.window_size}")

    def preprocess(self, data: DatasetResult, **kwargs) -> dict:
        """Preprocess data for incremental detection."""
        # Similar to batch version but designed for streaming
        X_ref = data.X_ref
        if isinstance(X_ref, pd.DataFrame):
            ref_values = X_ref.iloc[:, 0].values
        else:
            ref_values = np.asarray(X_ref).flatten()

        X_test = data.X_test
        if isinstance(X_test, pd.DataFrame):
            test_values = X_test.iloc[:, 0].values
        else:
            test_values = np.asarray(X_test).flatten()

        # Clean NaN values
        ref_values = ref_values[~np.isnan(ref_values)]
        test_values = test_values[~np.isnan(test_values)]

        return {"X_ref": ref_values, "X_test": test_values, "metadata": data.metadata}

    def fit(self, preprocessed_data: Any, **kwargs) -> "IncrementalKolmogorovSmirnovDetector":
        """Initialize reference window with initial data."""
        if not isinstance(preprocessed_data, dict) or "X_ref" not in preprocessed_data:
            raise ValueError("preprocessed_data must be dict with 'X_ref' key")

        ref_data = preprocessed_data["X_ref"]

        # Initialize sliding window
        self._reference_window = list(ref_data[-self.window_size :])

        logger.info(f"Initialized reference window with {len(self._reference_window)} samples")
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Perform incremental drift detection."""
        if not self._reference_window:
            raise RuntimeError("Detector must be fitted before detection")

        if not isinstance(preprocessed_data, dict) or "X_test" not in preprocessed_data:
            raise ValueError("preprocessed_data must be dict with 'X_test' key")

        test_data = preprocessed_data["X_test"]

        if len(test_data) == 0:
            self._last_score = 0.0
            return False

        # Perform KS test between current window and test data
        try:
            statistic, pvalue = stats.ks_2samp(np.array(self._reference_window), test_data, alternative="two-sided")

            self._last_score = 1.0 - float(pvalue)
            drift_detected = pvalue < self.threshold

            # Update reference window with test data (sliding window)
            self._reference_window.extend(test_data)
            if len(self._reference_window) > self.window_size:
                excess = len(self._reference_window) - self.window_size
                self._reference_window = self._reference_window[excess:]

            logger.debug(f"Incremental KS: pvalue={pvalue:.4f}, drift={drift_detected}, " f"window_size={len(self._reference_window)}")

            return drift_detected

        except Exception as e:
            logger.error(f"Error in incremental KS test: {e}")
            self._last_score = None
            raise

    def score(self) -> Optional[float]:
        """Return drift score from last detection."""
        return self._last_score

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"IncrementalKolmogorovSmirnovDetector("
            f"method_id='{self.method_id}', "
            f"implementation_id='{self.implementation_id}', "
            f"threshold={self.threshold}, window_size={self.window_size})"
        )
