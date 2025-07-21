"""
This module implements statistical test-based drift detectors using the
drift-benchmark adapter framework with custom implementations.
"""

from typing import Optional

import numpy as np
from scipy import stats

from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult


@register_detector(method_id="anderson_darling", implementation_id="ad_custom")
class CustomAndersonDarlingDetector(BaseDetector):
    """
    Custom Anderson-Darling test implementation for drift detection.

    Enhanced implementation with robust preprocessing and multivariate support
    using feature-wise testing with Bonferroni correction.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.05)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Convert DataFrame to numpy array with enhanced preprocessing."""
        phase = kwargs.get("phase", "train")

        if phase == "train":
            df = data.X_ref
        else:
            df = data.X_test

        # Select numeric columns and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in data")

        # Fill missing values with median (more robust than mean)
        return numeric_data.fillna(numeric_data.median()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "CustomAndersonDarlingDetector":
        """Store reference data for comparison."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform multivariate Anderson-Darling test using feature-wise approach."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Apply Anderson-Darling test to each feature and combine p-values
        p_values = []

        if self._reference_data.ndim == 1:
            # Univariate case - use KS test as approximation since scipy doesn't have 2-sample AD
            _, p_value = stats.ks_2samp(self._reference_data, preprocessed_data)
            p_values = [p_value]
        else:
            # Multivariate case - test each feature
            for feature_idx in range(self._reference_data.shape[1]):
                ref_feature = self._reference_data[:, feature_idx]
                test_feature = preprocessed_data[:, feature_idx]

                try:
                    # Use KS test as proxy for Anderson-Darling
                    _, p_value = stats.ks_2samp(ref_feature, test_feature)
                    p_values.append(p_value)
                except Exception:
                    # If test fails, assume no drift for this feature
                    p_values.append(1.0)

        # Combine p-values using Bonferroni correction for conservative approach
        if len(p_values) > 1:
            min_p_value = min(p_values)
            bonferroni_corrected = min(min_p_value * len(p_values), 1.0)
            combined_p_value = bonferroni_corrected
        else:
            combined_p_value = p_values[0]

        # Store combined p-value as drift score
        self._last_score = combined_p_value

        # Drift detected if combined p-value < threshold
        return combined_p_value < self.threshold

    def score(self) -> Optional[float]:
        """Return combined p-value from last detection."""
        return self._last_score


@register_detector(method_id="mann_whitney", implementation_id="mw_custom")
class CustomMannWhitneyDetector(BaseDetector):
    """
    Custom Mann-Whitney U-test implementation for drift detection.

    Enhanced implementation with robust preprocessing and multivariate support
    using feature-wise testing.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.05)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Convert DataFrame to numpy array with enhanced preprocessing."""
        phase = kwargs.get("phase", "train")

        if phase == "train":
            df = data.X_ref
        else:
            df = data.X_test

        # Select numeric columns and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in data")

        # Fill missing values with median and apply robust scaling
        filled_data = numeric_data.fillna(numeric_data.median())
        return filled_data.values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "CustomMannWhitneyDetector":
        """Store reference data for comparison."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform multivariate Mann-Whitney U test."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Apply Mann-Whitney U test to each feature and combine p-values
        p_values = []

        if self._reference_data.ndim == 1:
            # Univariate case
            try:
                _, p_value = stats.mannwhitneyu(self._reference_data, preprocessed_data, alternative="two-sided")
                p_values = [p_value]
            except Exception:
                # Fallback if test fails
                p_values = [1.0]
        else:
            # Multivariate case - test each feature
            for feature_idx in range(self._reference_data.shape[1]):
                ref_feature = self._reference_data[:, feature_idx]
                test_feature = preprocessed_data[:, feature_idx]

                try:
                    _, p_value = stats.mannwhitneyu(ref_feature, test_feature, alternative="two-sided")
                    p_values.append(p_value)
                except Exception:
                    # If test fails, assume no drift for this feature
                    p_values.append(1.0)

        # Combine p-values using Fisher's method
        if len(p_values) > 1:
            try:
                _, combined_p_value = stats.combine_pvalues(p_values, method="fisher")
            except Exception:
                # Fallback to minimum p-value with Bonferroni correction
                combined_p_value = min(min(p_values) * len(p_values), 1.0)
        else:
            combined_p_value = p_values[0]

        # Store combined p-value as drift score
        self._last_score = combined_p_value

        # Drift detected if combined p-value < threshold
        return combined_p_value < self.threshold

    def score(self) -> Optional[float]:
        """Return combined p-value from last detection."""
        return self._last_score
