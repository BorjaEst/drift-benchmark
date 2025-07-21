"""
This module implements distance-based drift detectors using the
drift-benchmark adapter framework with custom implementations.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from sklearn.metrics.pairwise import euclidean_distances

from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult


@register_detector(method_id="jensen_shannon_divergence", implementation_id="js_custom")
class CustomJensenShannonDetector(BaseDetector):
    """
    Custom Jensen-Shannon divergence implementation for drift detection.

    Enhanced implementation with robust preprocessing and multivariate support
    using feature-wise divergence calculation.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.1)  # Divergence threshold
        self.bins = kwargs.get("bins", 30)  # Number of bins for histogram
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

        # Fill missing values with median
        return numeric_data.fillna(numeric_data.median()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "CustomJensenShannonDetector":
        """Store reference data for comparison."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform multivariate Jensen-Shannon divergence calculation."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Calculate divergence for each feature and combine
        divergences = []

        if self._reference_data.ndim == 1:
            # Univariate case
            divergence = self._calculate_js_divergence(self._reference_data, preprocessed_data)
            divergences = [divergence]
        else:
            # Multivariate case - calculate for each feature
            for feature_idx in range(self._reference_data.shape[1]):
                ref_feature = self._reference_data[:, feature_idx]
                test_feature = preprocessed_data[:, feature_idx]

                divergence = self._calculate_js_divergence(ref_feature, test_feature)
                divergences.append(divergence)

        # Combine divergences (average for interpretability)
        average_divergence = np.mean(divergences)

        # Store divergence as drift score
        self._last_score = average_divergence

        # Drift detected if average divergence > threshold
        return average_divergence > self.threshold

    def _calculate_js_divergence(self, ref_data: np.ndarray, test_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two samples."""
        try:
            # Determine common range for histograms
            combined_data = np.concatenate([ref_data, test_data])
            data_range = (combined_data.min(), combined_data.max())

            # Avoid identical min/max (causes issues with histogram)
            if data_range[0] == data_range[1]:
                return 0.0

            # Create histograms
            ref_hist, _ = np.histogram(ref_data, bins=self.bins, range=data_range, density=True)
            test_hist, _ = np.histogram(test_data, bins=self.bins, range=data_range, density=True)

            # Normalize to probabilities (handle zero sum case)
            ref_prob = ref_hist / (ref_hist.sum() + 1e-15)
            test_prob = test_hist / (test_hist.sum() + 1e-15)

            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            ref_prob = ref_prob + epsilon
            test_prob = test_prob + epsilon

            # Renormalize after adding epsilon
            ref_prob = ref_prob / ref_prob.sum()
            test_prob = test_prob / test_prob.sum()

            # Calculate Jensen-Shannon divergence
            divergence = jensenshannon(ref_prob, test_prob)

            # Handle NaN/inf values
            if np.isnan(divergence) or np.isinf(divergence):
                return 0.0

            return float(divergence)

        except Exception:
            # Fallback: return 0 divergence if calculation fails
            return 0.0

    def score(self) -> Optional[float]:
        """Return average divergence from last detection."""
        return self._last_score


@register_detector(method_id="wasserstein_distance", implementation_id="ws_custom")
class CustomWassersteinDetector(BaseDetector):
    """
    Custom Wasserstein distance implementation for drift detection.

    Enhanced implementation with robust preprocessing and multivariate support
    using feature-wise distance calculation.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.5)  # Distance threshold
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

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "CustomWassersteinDetector":
        """Store reference data for comparison."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform multivariate Wasserstein distance calculation."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Calculate Wasserstein distance for each feature and combine
        distances = []

        if self._reference_data.ndim == 1:
            # Univariate case
            distance = self._calculate_wasserstein_distance(self._reference_data, preprocessed_data)
            distances = [distance]
        else:
            # Multivariate case - calculate for each feature
            for feature_idx in range(self._reference_data.shape[1]):
                ref_feature = self._reference_data[:, feature_idx]
                test_feature = preprocessed_data[:, feature_idx]

                distance = self._calculate_wasserstein_distance(ref_feature, test_feature)
                distances.append(distance)

        # Combine distances (average for interpretability)
        average_distance = np.mean(distances)

        # Store distance as drift score
        self._last_score = average_distance

        # Drift detected if average distance > threshold
        return average_distance > self.threshold

    def _calculate_wasserstein_distance(self, ref_data: np.ndarray, test_data: np.ndarray) -> float:
        """Calculate Wasserstein distance between two samples."""
        try:
            # Calculate Wasserstein distance using scipy
            distance = scipy_wasserstein_distance(ref_data, test_data)

            # Handle NaN/inf values
            if np.isnan(distance) or np.isinf(distance):
                return 0.0

            return float(distance)

        except Exception:
            # Fallback: return 0 distance if calculation fails
            return 0.0

    def score(self) -> Optional[float]:
        """Return average Wasserstein distance from last detection."""
        return self._last_score


@register_detector(method_id="maximum_mean_discrepancy", implementation_id="mmd_rbf")
class MaximumMeanDiscrepancyDetector(BaseDetector):
    """
    Maximum Mean Discrepancy with RBF kernel for drift detection.

    Computes the Maximum Mean Discrepancy between reference and test
    distributions using a Radial Basis Function (RBF) kernel.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.1)
        self.gamma = kwargs.get("gamma", 1.0)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Standardize data for distance calculations."""
        phase = kwargs.get("phase", "train")

        if phase == "train":
            df = data.X_ref
        else:
            df = data.X_test

        # Convert to numpy and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        processed = numeric_data.fillna(numeric_data.mean()).values

        # Standardize using reference statistics
        if phase == "train":
            self._mean = processed.mean(axis=0)
            self._std = processed.std(axis=0)
            self._std[self._std == 0] = 1  # Avoid division by zero

        # Apply standardization
        if hasattr(self, "_mean") and hasattr(self, "_std"):
            processed = (processed - self._mean) / self._std

        return processed

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "MaximumMeanDiscrepancyDetector":
        """Store reference data for MMD calculation."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        distances = euclidean_distances(X, Y)
        return np.exp(-self.gamma * distances**2)

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy."""
        m, n = X.shape[0], Y.shape[0]

        # Compute kernel matrices
        K_XX = self._rbf_kernel(X, X)
        K_YY = self._rbf_kernel(Y, Y)
        K_XY = self._rbf_kernel(X, Y)

        # Compute MMD^2
        mmd_squared = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n)

        return np.sqrt(max(mmd_squared, 0))  # Ensure non-negative

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Detect drift using MMD."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Compute MMD between reference and test data
        mmd_score = self._compute_mmd(self._reference_data, preprocessed_data)
        self._last_score = mmd_score

        return mmd_score > self.threshold

    def score(self) -> Optional[float]:
        """Return MMD score from last detection."""
        return self._last_score


@register_detector(method_id="wasserstein_distance", implementation_id="wasserstein_1d")
class WassersteinDistanceDetector(BaseDetector):
    """
    Wasserstein (Earth Mover's) distance for drift detection.

    Computes the Wasserstein distance between reference and test
    distributions for univariate drift detection.
    """

    def __init__(self, method_id: str, implementation_id: str, **kwargs):
        super().__init__(method_id, implementation_id)
        self.threshold = kwargs.get("threshold", 0.1)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Extract first numeric column for univariate analysis."""
        phase = kwargs.get("phase", "train")

        if phase == "train":
            df = data.X_ref
        else:
            df = data.X_test

        # Convert to numpy and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric columns found in data")

        # Use first column for univariate analysis
        first_column = numeric_data.iloc[:, 0]
        return first_column.fillna(first_column.mean()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "WassersteinDistanceDetector":
        """Store reference data for distance calculation."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Detect drift using Wasserstein distance."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Compute Wasserstein distance
        distance = wasserstein_distance(self._reference_data, preprocessed_data)
        self._last_score = distance

        return distance > self.threshold

    def score(self) -> Optional[float]:
        """Return Wasserstein distance from last detection."""
        return self._last_score
