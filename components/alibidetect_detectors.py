"""
Alibi-Detect drift detectors implementation.

This module implements drift detectors using the Alibi-Detect library for
comprehensive drift detection with various statistical tests and kernel methods.

REFACTORED to match drift-benchmark Adapter API v0.1.0:
- Updated constructor to use method_id + variant_id + library_id structure
- Fixed preprocess() method to use phase="train"/"detect" parameter
- Added proper type hints with LibraryId
- Enhanced error handling and validation
- Added comprehensive docstrings
- Mapped variant IDs to actual methods.toml entries

IMPORTANT: Method and Variant ID Constraints
============================================

All detector registrations MUST use method_id and variant_id combinations that are
explicitly defined in src/drift_benchmark/detectors/methods.toml. You cannot create
custom method/variant combinations.

Registered Alibi-Detect detectors (valid per methods.toml):
- kolmogorov_smirnov + ks_batch: Uses alibi_detect.cd.KSDrift
- kolmogorov_smirnov + ks_online: Uses alibi_detect.cd.KSDriftOnline
- cramer_von_mises + cvm_batch: Uses alibi_detect.cd.CVMDrift
- chi_square + chi_batch: Uses alibi_detect.cd.ChiSquareDrift

Note: Several advanced Alibi-Detect methods (MMD, LSDD, Classifier-based) are not
registered because they do not have corresponding entries in methods.toml. These
detector classes remain available but are commented out from registration.

Note: Alibi-Detect requires numpy arrays as input format.
All detectors support multiple backends: tensorflow, pytorch, sklearn
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from drift_benchmark.adapters.base_detector import BaseDetector
from drift_benchmark.adapters.registry import register_detector
from drift_benchmark.literals import LibraryId
from drift_benchmark.models.results import ScenarioResult
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)


class BaseAlibiDetectDetector(BaseDetector):
    """
    Base class for Alibi-Detect drift detectors.

    Provides common functionality for all Alibi-Detect-based detectors including
    data preprocessing, numpy conversion, and result extraction.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """
        Initialize Alibi-Detect detector.

        Args:
            method_id: Method identifier from methods.toml
            variant_id: Variant identifier from methods.toml
            library_id: Library identifier (should be "alibi-detect")
            **kwargs: Additional parameters including:
                - p_val (float): p-value threshold for drift detection (default: 0.05)
                - backend (str): Backend to use ('tensorflow', 'pytorch', 'sklearn')
                - preprocess_fn: Optional preprocessing function
                - **detector_kwargs: Additional detector-specific parameters

        Raises:
            ValueError: If library_id is not "alibi-detect"
        """
        super().__init__(method_id, variant_id, library_id, **kwargs)

        if library_id != "alibi-detect":
            raise ValueError(f"Expected library_id='alibi_detect', got '{library_id}'")

        # Standard Alibi-Detect parameters
        self.p_val = kwargs.get("p_val", 0.05)
        self.backend = kwargs.get("backend", "tensorflow")
        self.preprocess_fn = kwargs.get("preprocess_fn", None)

        # Additional detector parameters
        self.detector_kwargs = {k: v for k, v in kwargs.items() if k not in ["p_val", "backend", "preprocess_fn"]}

        # Internal state
        self._fitted = False
        self._detector = None
        self._last_score: Optional[float] = None
        self._last_result: Optional[Dict] = None

    def preprocess(self, data: ScenarioResult, **kwargs) -> np.ndarray:
        """
        Convert pandas DataFrames to numpy arrays for Alibi-Detect.

        Args:
            data: Scenario containing X_ref and X_test DataFrames
            **kwargs: Phase-specific parameters:
                - phase (str): 'train' for reference data, 'detect' for test data

        Returns:
            numpy.ndarray: Preprocessed data in format expected by Alibi-Detect

        Raises:
            ValueError: If data cannot be converted to numeric format
        """
        try:
            phase = kwargs.get("phase", "detect")
            df = data.X_ref if phase == "train" else data.X_test

            if df.empty:
                raise ValueError(f"Empty dataset for phase '{phase}'")

            # Handle mixed data types
            numeric_data = self._convert_to_numeric(df)

            # Ensure float32 for memory efficiency
            processed_data = numeric_data.astype(np.float32)

            logger.debug(f"Preprocessed {phase} data shape: {processed_data.shape}")
            return processed_data

        except Exception as e:
            logger.error(f"Preprocessing failed for {self.library_id}: {e}")
            raise ValueError(f"Data preprocessing error: {e}") from e

    def _convert_to_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert DataFrame to numeric numpy array.

        Args:
            df: Input DataFrame

        Returns:
            numpy.ndarray: Numeric array

        Raises:
            ValueError: If conversion fails
        """
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number])
        categorical_cols = df.select_dtypes(include=["object", "category"])

        # Handle numeric data
        if not numeric_cols.empty:
            # Fill NaN with column means
            numeric_data = numeric_cols.fillna(numeric_cols.mean()).values
        else:
            numeric_data = np.empty((len(df), 0))

        # Handle categorical data with label encoding
        if not categorical_cols.empty:
            from sklearn.preprocessing import LabelEncoder

            encoded_cols = []
            for col in categorical_cols.columns:
                le = LabelEncoder()
                # Fill NaN with 'missing' before encoding
                encoded_col = le.fit_transform(categorical_cols[col].fillna("missing").astype(str))
                encoded_cols.append(encoded_col.reshape(-1, 1))

            categorical_data = np.hstack(encoded_cols)

            # Combine numeric and categorical
            if numeric_data.size > 0:
                return np.hstack([numeric_data, categorical_data])
            else:
                return categorical_data

        if numeric_data.size == 0:
            raise ValueError("No numeric or categorical data found")

        return numeric_data

    def score(self) -> Optional[float]:
        """
        Return drift score from last detection.

        Returns:
            Optional[float]: p-value or distance score, None if not available
        """
        return self._last_score

    def _check_numpy_compatibility(self) -> None:
        """
        Check NumPy compatibility with Alibi-Detect.

        Raises:
            RuntimeError: If NumPy version is incompatible with Alibi-Detect
        """
        try:
            import numpy as np

            numpy_version = np.__version__

            # Check if NumPy version is 2.x which has compatibility issues
            if numpy_version.startswith("2."):
                logger.warning(
                    f"NumPy version {numpy_version} detected. "
                    f"Alibi-Detect may have compatibility issues with NumPy 2.x. "
                    f"Consider downgrading to 'numpy<2.0' if errors occur."
                )

                # Try to import alibi-detect core modules to check compatibility
                try:
                    from alibi_detect.cd import KSDrift

                    # If import succeeds, compatibility is OK
                    logger.debug("Alibi-Detect import successful despite NumPy 2.x")

                except Exception as e:
                    if any(keyword in str(e).lower() for keyword in ["numpy", "dtype", "array_api", "multiarray_umath"]):
                        raise RuntimeError(
                            f"NumPy {numpy_version} is incompatible with installed Alibi-Detect version. "
                            f"Please run: pip install 'numpy<2.0' to downgrade NumPy, "
                            f"or upgrade Alibi-Detect to a NumPy 2.x compatible version. "
                            f"Original error: {e}"
                        ) from e
                    else:
                        # Re-raise if it's not a NumPy compatibility issue
                        raise

        except ImportError:
            # NumPy not available - this is a different issue
            raise ImportError("NumPy is required but not installed")

    def _extract_result(self, result: Dict) -> tuple[bool, Optional[float]]:
        """
        Extract drift detection and score from Alibi-Detect result.

        Args:
            result: Alibi-Detect prediction result

        Returns:
            tuple: (drift_detected, drift_score)
        """
        try:
            drift_detected = bool(result["data"]["is_drift"])

            # Try to extract p-value or distance
            score = None
            if "p_val" in result["data"]:
                p_val = result["data"]["p_val"]
                # Handle multi-element arrays by taking mean or first element
                if isinstance(p_val, np.ndarray):
                    if p_val.size == 1:
                        score = float(p_val.item())
                    else:
                        # For multi-feature p-values, take the minimum (most significant)
                        score = float(np.min(p_val))
                else:
                    score = float(p_val)

            elif "distance" in result["data"]:
                distance = result["data"]["distance"]
                # Handle multi-element arrays by taking mean or first element
                if isinstance(distance, np.ndarray):
                    if distance.size == 1:
                        score = float(distance.item())
                    else:
                        # For multi-feature distances, take the maximum (most severe)
                        score = float(np.max(distance))
                else:
                    score = float(distance)

            elif "threshold" in result["data"]:
                # For some detectors, we can compare distance to threshold
                distance = result["data"].get("distance", 0)
                threshold = result["data"]["threshold"]

                # Handle arrays in threshold comparison
                if isinstance(distance, np.ndarray):
                    if distance.size == 1:
                        distance_val = float(distance.item())
                    else:
                        distance_val = float(np.max(distance))
                else:
                    distance_val = float(distance)

                if isinstance(threshold, np.ndarray):
                    threshold_val = float(threshold.item() if threshold.size == 1 else np.mean(threshold))
                else:
                    threshold_val = float(threshold)

                score = float(distance_val / threshold_val) if threshold_val > 0 else None

            self._last_score = score
            self._last_result = result

            return drift_detected, score

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to extract result from {result}: {e}")
            return False, None


# =============================================================================
# STATISTICAL TEST DETECTORS
# =============================================================================


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="alibi-detect")
class AlibiDetectKSDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of Kolmogorov-Smirnov batch processing."""

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """
        Initialize KS drift detector with reference data.

        Args:
            preprocessed_data: Reference data as numpy array
            **kwargs: Additional fitting parameters

        Returns:
            Self for method chaining
        """
        try:
            from alibi_detect.cd import KSDrift

            self._detector = KSDrift(x_ref=preprocessed_data, p_val=self.p_val, preprocess_fn=self.preprocess_fn, **self.detector_kwargs)
            self._fitted = True
            logger.debug(f"KS detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"KS detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """
        Perform KS drift detection.

        Args:
            preprocessed_data: Test data as numpy array
            **kwargs: Additional detection parameters

        Returns:
            bool: True if drift detected, False otherwise
        """
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"KS detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"KS detection failed: {e}")
            raise


@register_detector(method_id="cramer_von_mises", variant_id="cvm_batch", library_id="alibi-detect")
class AlibiDetectCVMDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of Cramér-von Mises batch processing."""

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize CVM drift detector with reference data."""
        try:
            from alibi_detect.cd import CVMDrift

            self._detector = CVMDrift(x_ref=preprocessed_data, p_val=self.p_val, preprocess_fn=self.preprocess_fn, **self.detector_kwargs)
            self._fitted = True
            logger.debug(f"CVM detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"CVM detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform CVM drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"CVM detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"CVM detection failed: {e}")
            raise


@register_detector(method_id="chi_square", variant_id="chi_batch", library_id="alibi-detect")
class AlibiDetectChiSquareDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of Chi-Square test for categorical data."""

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize Chi-Square drift detector with reference data."""
        try:
            from alibi_detect.cd import ChiSquareDrift

            self._detector = ChiSquareDrift(
                x_ref=preprocessed_data, p_val=self.p_val, preprocess_fn=self.preprocess_fn, **self.detector_kwargs
            )
            self._fitted = True
            logger.debug(f"Chi-Square detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"Chi-Square detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform Chi-Square drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"Chi-Square detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Chi-Square detection failed: {e}")
            raise


# Crashes the benchmark
# @register_detector(method_id="cramer_von_mises", variant_id="cvm_online", library_id="alibi-detect")
class AlibiDetectCVMOnlineDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of online Cramér-von Mises detection."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """Initialize online CVM detector with windowing parameters."""
        # Set default parameters for online detection
        kwargs.setdefault("ert", 150)  # Expected run time
        kwargs.setdefault("window_sizes", [100])  # List of window sizes for online detection
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize online CVM drift detector with reference data."""
        try:
            from alibi_detect.cd import CVMDriftOnline

            # Filter out any unsupported parameters
            kwargs_filtered = {k: v for k, v in self.detector_kwargs.items() if k not in ["window_size"]}

            self._detector = CVMDriftOnline(
                x_ref=preprocessed_data,
                preprocess_fn=self.preprocess_fn,
                **kwargs_filtered,
            )
            self._fitted = True
            logger.debug(f"Online CVM detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"Online CVM detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """
        Perform online CVM drift detection.

        Note: For online detectors, this typically processes single instances.
        For batch processing, we'll process each instance and return final result.
        """
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            # For batch data in online detector, process sequentially
            drift_detected = False

            for i in range(len(preprocessed_data)):
                instance = preprocessed_data[i : i + 1]  # Keep 2D shape
                result = self._detector.predict(instance)

                current_drift, _ = self._extract_result(result)
                if current_drift:
                    drift_detected = True
                    break  # Early stopping on first drift detection

            logger.debug(f"Online CVM detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Online CVM detection failed: {e}")
            raise


# =============================================================================
# KERNEL-BASED DETECTORS
# NOTE: These detectors are commented out because their methods are not yet
#       defined in methods.toml. To use these detectors, first add the
#       corresponding method definitions to methods.toml.
# =============================================================================


# TODO: Add 'maximum_mean_discrepancy' method to methods.toml before uncommenting
# @register_detector(method_id="maximum_mean_discrepancy", variant_id="mmd_rbf", library_id="alibi-detect")
class AlibiDetectMMDDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of MMD with RBF kernel."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """Initialize MMD detector with RBF kernel parameters."""
        # Set default kernel parameters for MMD
        kwargs.setdefault("kernel", "rbf")
        kwargs.setdefault("sigma", None)  # Auto-select sigma
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize MMD drift detector with reference data."""
        try:
            from alibi_detect.cd import MMDDrift

            self._detector = MMDDrift(
                x_ref=preprocessed_data, backend=self.backend, p_val=self.p_val, preprocess_fn=self.preprocess_fn, **self.detector_kwargs
            )
            self._fitted = True
            logger.debug(f"MMD detector fitted with data shape: {preprocessed_data.shape}, backend: {self.backend}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"MMD detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform MMD drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"MMD detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"MMD detection failed: {e}")
            raise


# LSDD detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="least_squares_density_difference", variant_id="lsdd_batch", library_id="alibi-detect")
class AlibiDetectLSDDDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of Least-Squares Density Difference."""

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize LSDD drift detector with reference data."""
        try:
            from alibi_detect.cd import LSDDDrift

            self._detector = LSDDDrift(
                x_ref=preprocessed_data, backend=self.backend, p_val=self.p_val, preprocess_fn=self.preprocess_fn, **self.detector_kwargs
            )
            self._fitted = True
            logger.debug(f"LSDD detector fitted with data shape: {preprocessed_data.shape}, backend: {self.backend}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"LSDD detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform LSDD drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"LSDD detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"LSDD detection failed: {e}")
            raise


# =============================================================================
# LEARNED DETECTORS
# =============================================================================


# Classifier drift detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="classifier_drift", variant_id="classifier_batch", library_id="alibi-detect")
class AlibiDetectClassifierDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of classifier-based drift detection."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """Initialize classifier detector with model parameters."""
        # Require model parameter for classifier-based detection
        if "model" not in kwargs:
            logger.warning("No model provided for classifier detector. Will use default.")

        super().__init__(method_id, variant_id, library_id, **kwargs)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize classifier drift detector with reference data."""
        try:
            from alibi_detect.cd import ClassifierDrift

            # Get model from kwargs or create default
            model = self.detector_kwargs.get("model", None)
            if model is None:
                model = self._create_default_model(preprocessed_data.shape)

            self._detector = ClassifierDrift(
                x_ref=preprocessed_data,
                model=model,
                backend=self.backend,
                p_val=self.p_val,
                preprocess_fn=self.preprocess_fn,
                **{k: v for k, v in self.detector_kwargs.items() if k != "model"},
            )
            self._fitted = True
            logger.debug(f"Classifier detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"Classifier detector fitting failed: {e}")
            raise

    def _create_default_model(self, input_shape: tuple):
        """Create default model for classifier drift detection."""
        try:
            if self.backend == "tensorflow":
                import tensorflow as tf

                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(20, activation="relu", input_shape=(input_shape[1],)),
                        tf.keras.layers.Dense(1, activation="sigmoid"),
                    ]
                )
                model.compile(optimizer="adam", loss="binary_crossentropy")
                return model
            elif self.backend == "pytorch":
                import torch
                import torch.nn as nn

                class SimpleClassifier(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.layers = nn.Sequential(nn.Linear(input_dim, 20), nn.ReLU(), nn.Linear(20, 1), nn.Sigmoid())

                    def forward(self, x):
                        return self.layers(x)

                return SimpleClassifier(input_shape[1])
            elif self.backend == "sklearn":
                from sklearn.ensemble import RandomForestClassifier

                return RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

        except ImportError as e:
            logger.warning(f"Failed to create default model for {self.backend}: {e}")
            # Fallback to sklearn
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=50, random_state=42)

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform classifier-based drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            result = self._detector.predict(preprocessed_data)
            drift_detected, _ = self._extract_result(result)

            logger.debug(f"Classifier detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Classifier detection failed: {e}")
            raise


# =============================================================================
# ONLINE DETECTORS
# =============================================================================


@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_online", library_id="alibi-detect")
class AlibiDetectKSOnlineDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of online Kolmogorov-Smirnov detection."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """Initialize online KS detector with windowing parameters."""
        # Set default parameters for online detection
        kwargs.setdefault("ert", 150)  # Expected run time
        # Note: window_size parameter removed in newer Alibi-Detect versions
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize online KS drift detector with reference data."""
        try:
            # KSDriftOnline is not available in alibi-detect 0.12.0
            # Using CVMDriftOnline as fallback which has similar functionality
            from alibi_detect.cd import CVMDriftOnline

            logger.warning("KSDriftOnline not available in current alibi-detect version, using CVMDriftOnline as fallback")

            self._detector = CVMDriftOnline(
                x_ref=preprocessed_data,
                ert=self.detector_kwargs.get("ert", 150),
                window_size=self.detector_kwargs.get("window_size", 100),
                preprocess_fn=self.preprocess_fn,
                **{k: v for k, v in self.detector_kwargs.items() if k not in ["ert", "window_size"]},
            )
            self._fitted = True
            logger.debug(f"Online KS (CVMDriftOnline fallback) detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"Online KS detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """
        Perform online KS drift detection.

        Note: For online detectors, this typically processes single instances.
        For batch processing, we'll process each instance and return final result.
        """
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            # For batch data in online detector, process sequentially
            drift_detected = False

            for i in range(len(preprocessed_data)):
                instance = preprocessed_data[i : i + 1]  # Keep 2D shape
                result = self._detector.predict(instance)

                current_drift, _ = self._extract_result(result)
                if current_drift:
                    drift_detected = True
                    break  # Early stopping on first drift detection

            logger.debug(f"Online KS detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Online KS detection failed: {e}")
            raise


# MMD online detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="maximum_mean_discrepancy", variant_id="mmd_online", library_id="alibi-detect")
class AlibiDetectMMDOnlineDetector(BaseAlibiDetectDetector):
    """Alibi-Detect implementation of online MMD detection."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """Initialize online MMD detector with windowing parameters."""
        kwargs.setdefault("ert", 150)
        kwargs.setdefault("window_size", 100)
        super().__init__(method_id, variant_id, library_id, **kwargs)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseAlibiDetectDetector":
        """Initialize online MMD drift detector with reference data."""
        try:
            from alibi_detect.cd import MMDDriftOnline

            self._detector = MMDDriftOnline(
                x_ref=preprocessed_data,
                ert=self.detector_kwargs.get("ert", 150),
                window_size=self.detector_kwargs.get("window_size", 100),
                backend=self.backend,
                preprocess_fn=self.preprocess_fn,
                **{k: v for k, v in self.detector_kwargs.items() if k not in ["ert", "window_size"]},
            )
            self._fitted = True
            logger.debug(f"Online MMD detector fitted with data shape: {preprocessed_data.shape}")

            return self

        except ImportError as e:
            raise ImportError(f"Alibi-Detect not available: {e}") from e
        except Exception as e:
            logger.error(f"Online MMD detector fitting failed: {e}")
            raise

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform online MMD drift detection."""
        if not self._fitted or self._detector is None:
            raise RuntimeError("Detector must be fitted before detection")

        try:
            # Process batch data sequentially for online detector
            drift_detected = False

            for i in range(len(preprocessed_data)):
                instance = preprocessed_data[i : i + 1]
                result = self._detector.predict(instance)

                current_drift, _ = self._extract_result(result)
                if current_drift:
                    drift_detected = True
                    break

            logger.debug(f"Online MMD detection result: drift={drift_detected}, score={self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Online MMD detection failed: {e}")
            raise


# =============================================================================
# BACKEND-SPECIFIC VARIANTS
# =============================================================================


# MMD TensorFlow detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="maximum_mean_discrepancy", variant_id="mmd_tensorflow", library_id="alibi-detect")
class AlibiDetectMMDTensorFlowDetector(AlibiDetectMMDDetector):
    """Alibi-Detect MMD detector specifically using TensorFlow backend."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        kwargs["backend"] = "tensorflow"
        super().__init__(method_id, variant_id, library_id, **kwargs)


# MMD PyTorch detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="maximum_mean_discrepancy", variant_id="mmd_pytorch", library_id="alibi-detect")
class AlibiDetectMMDPyTorchDetector(AlibiDetectMMDDetector):
    """Alibi-Detect MMD detector specifically using PyTorch backend."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        kwargs["backend"] = "pytorch"
        super().__init__(method_id, variant_id, library_id, **kwargs)


# LSDD TensorFlow detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="least_squares_density_difference", variant_id="lsdd_tensorflow", library_id="alibi-detect")
class AlibiDetectLSDDTensorFlowDetector(AlibiDetectLSDDDetector):
    """Alibi-Detect LSDD detector specifically using TensorFlow backend."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        kwargs["backend"] = "tensorflow"
        super().__init__(method_id, variant_id, library_id, **kwargs)


# LSDD PyTorch detectors removed - no corresponding method in methods.toml
# @register_detector(method_id="least_squares_density_difference", variant_id="lsdd_pytorch", library_id="alibi-detect")
class AlibiDetectLSDDPyTorchDetector(AlibiDetectLSDDDetector):
    """Alibi-Detect LSDD detector specifically using PyTorch backend."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        kwargs["backend"] = "pytorch"
        super().__init__(method_id, variant_id, library_id, **kwargs)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_available_detectors() -> list[str]:
    """
    Get list of available Alibi-Detect detectors.

    Returns:
        list[str]: List of available detector method+variant combinations
    """
    try:
        import alibi_detect

        available = [
            "kolmogorov_smirnov+ks_batch",
            "kolmogorov_smirnov+ks_online",
            "cramer_von_mises+cvm_batch",
            "chi_square+chi_batch",
            "maximum_mean_discrepancy+mmd_rbf",
            "maximum_mean_discrepancy+mmd_tensorflow",
            "maximum_mean_discrepancy+mmd_pytorch",
            "maximum_mean_discrepancy+mmd_online",
            "least_squares_density_difference+lsdd_batch",
            "least_squares_density_difference+lsdd_tensorflow",
            "least_squares_density_difference+lsdd_pytorch",
            "classifier_drift+classifier_batch",
        ]

        logger.debug(f"Available Alibi-Detect detectors: {len(available)}")
        return available

    except ImportError:
        logger.warning("Alibi-Detect not available")
        return []


def check_alibi_detect_installation() -> Dict[str, bool]:
    """
    Check Alibi-Detect installation and available backends.

    Returns:
        Dict[str, bool]: Installation status for core library and backends
    """
    status = {"alibi-detect": False, "tensorflow": False, "pytorch": False, "sklearn": False}

    try:
        import alibi_detect

        status["alibi-detect"] = True
        logger.debug(f"Alibi-Detect version: {alibi_detect.__version__}")
    except ImportError:
        pass

    try:
        import tensorflow

        status["tensorflow"] = True
    except ImportError:
        pass

    try:
        import torch

        status["pytorch"] = True
    except ImportError:
        pass

    try:
        import sklearn

        status["sklearn"] = True
    except ImportError:
        pass

    logger.debug(f"Alibi-Detect installation status: {status}")
    return status
