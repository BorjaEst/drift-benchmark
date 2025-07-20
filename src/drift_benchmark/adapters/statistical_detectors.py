"""
Statistical test-based drift detectors implementation.

This module implements drift detectors based on statistical tests like
Kolmogorov-Smirnov and Cramér-von Mises for distribution comparison.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..models.result_models import DatasetResult
from ..settings import get_logger
from .base_detector import BaseDetector
from .registry import register_detector

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

        Args:
            method_id: Must be "kolmogorov_smirnov"
            implementation_id: Must be "ks_batch"
            **kwargs: Additional parameters including:
                - threshold (float): P-value threshold for drift detection (default: 0.05)
        """
        super().__init__(method_id, implementation_id, **kwargs)

        # Hyperparameters from methods.toml
        self.threshold = kwargs.get("threshold", 0.05)

        # Internal state
        self._reference_data: Optional[np.ndarray] = None
        self._last_statistic: Optional[float] = None
        self._last_pvalue: Optional[float] = None
        self._last_score: Optional[float] = None

        logger.info(f"Initialized {self.__class__.__name__} with threshold={self.threshold}")

    def preprocess(self, data: DatasetResult, **kwargs) -> dict:
        """
        Convert pandas DataFrames to numpy arrays for KS test.

        Args:
            data: DatasetResult containing X_ref and X_test DataFrames
            **kwargs: Additional preprocessing parameters

        Returns:
            dict: Preprocessed data with numpy arrays
        """
        # Extract reference data (for training phase)
        X_ref = data.X_ref
        if isinstance(X_ref, pd.DataFrame):
            # Select only numeric columns and use the first one
            numeric_cols = X_ref.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in reference data")
            ref_values = X_ref[numeric_cols[0]].values
        else:
            ref_values = np.asarray(X_ref).flatten()

        # Extract test data (for detection phase)
        X_test = data.X_test
        if isinstance(X_test, pd.DataFrame):
            # Use same column as reference
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in test data")
            test_values = X_test[numeric_cols[0]].values
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

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional fitting parameters

        Returns:
            self: Fitted detector instance
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

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional detection parameters

        Returns:
            bool: True if drift is detected, False otherwise
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
            statistic, pvalue = stats.ks_2samp(self._reference_data, test_data)

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

        Returns:
            Optional[float]: Drift score (1 - p_value) or None if no detection performed
        """
        return self._last_score


@register_detector(method_id="cramer_von_mises", implementation_id="cvm_batch")
class CramerVonMisesDetector(BaseDetector):
    """
    Cramér-von Mises drift detector implementation.

    This detector implements the two-sample Cramér-von Mises test to detect
    covariate drift by comparing the distribution functions of reference and test datasets.

    References:
    - Cramér (1902): https://doi.org/10.1080/03461238.1928.10416862
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        """
        Initialize Cramér-von Mises detector.

        Args:
            method_id: Must be "cramer_von_mises"
            implementation_id: Must be "cvm_batch"
            **kwargs: Additional parameters including:
                - threshold (float): P-value threshold for drift detection (default: 0.05)
        """
        super().__init__(method_id, implementation_id, **kwargs)

        # Hyperparameters from methods.toml
        self.threshold = kwargs.get("threshold", 0.05)

        # Internal state
        self._reference_data: Optional[np.ndarray] = None
        self._last_statistic: Optional[float] = None
        self._last_pvalue: Optional[float] = None
        self._last_score: Optional[float] = None

        logger.info(f"Initialized {self.__class__.__name__} with threshold={self.threshold}")

    def preprocess(self, data: DatasetResult, **kwargs) -> dict:
        """
        Convert pandas DataFrames to numpy arrays for Cramér-von Mises test.

        Args:
            data: DatasetResult containing X_ref and X_test DataFrames
            **kwargs: Additional preprocessing parameters

        Returns:
            dict: Preprocessed data with numpy arrays
        """
        # Extract reference data
        X_ref = data.X_ref
        if isinstance(X_ref, pd.DataFrame):
            # Select only numeric columns and use the first one
            numeric_cols = X_ref.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in reference data")
            ref_values = X_ref[numeric_cols[0]].values
        else:
            ref_values = np.asarray(X_ref).flatten()

        # Extract test data
        X_test = data.X_test
        if isinstance(X_test, pd.DataFrame):
            # Use same column as reference
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in test data")
            test_values = X_test[numeric_cols[0]].values
        else:
            test_values = np.asarray(X_test).flatten()

        # Remove NaN values
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

    def fit(self, preprocessed_data: Any, **kwargs) -> "CramerVonMisesDetector":
        """
        Train the detector on reference data.

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional fitting parameters

        Returns:
            self: Fitted detector instance
        """
        if not isinstance(preprocessed_data, dict) or "X_ref" not in preprocessed_data:
            raise ValueError("preprocessed_data must be dict with 'X_ref' key")

        self._reference_data = preprocessed_data["X_ref"]

        if len(self._reference_data) == 0:
            logger.warning("Empty reference data provided to fit()")
        else:
            logger.info(f"Fitted CvM detector on {len(self._reference_data)} reference samples")

        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Perform drift detection using Cramér-von Mises test.

        Args:
            preprocessed_data: Output from preprocess() method
            **kwargs: Additional detection parameters

        Returns:
            bool: True if drift is detected, False otherwise
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

        # Perform two-sample Cramér-von Mises test
        try:
            # Use scipy's implementation
            result = stats.cramervonmises_2samp(self._reference_data, test_data)
            statistic = result.statistic
            pvalue = result.pvalue

            # Store results
            self._last_statistic = float(statistic)
            self._last_pvalue = float(pvalue)
            # Use 1 - pvalue as drift score (higher = more drift)
            self._last_score = 1.0 - float(pvalue)

            # Drift detected if p-value < threshold
            drift_detected = pvalue < self.threshold

            logger.info(f"CvM test: statistic={statistic:.4f}, pvalue={pvalue:.4f}, " f"drift_detected={drift_detected}")

            return drift_detected

        except Exception as e:
            logger.error(f"Error in CvM test: {e}")
            self._last_statistic = None
            self._last_pvalue = None
            self._last_score = None
            raise

    def score(self) -> Optional[float]:
        """
        Return drift score after detection.

        Returns:
            Optional[float]: Drift score (1 - p_value) or None if no detection performed
        """
        return self._last_score
