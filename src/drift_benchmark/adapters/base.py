"""
Base classes and interfaces for drift detection algorithms.

This module defines the common interface that all drift detectors must implement,
enabling standardized benchmarking and evaluation across different libraries.
"""

import abc
import time
from typing import Any, Dict, Optional, Type

from drift_benchmark.constants.models import DatasetResult, DetectorData
from drift_benchmark.detectors import detector_exists, get_detector


def register_method(method_id: str, implementation_id: str):
    """
    Decorator to register a detector with method and implementation IDs.

    This provides a cleaner alternative to setting class attributes manually.

    Args:
        method_id: The method ID from methods.toml
        implementation_id: The implementation ID from methods.toml

    Returns:
        Decorated class with method_id and implementation_id set

    Example:
        @register_method("kolmogorov_smirnov", "ks_batch")
        class KSDetector(BaseDetector):
            def fit(self, reference_data): ...
            def detect(self, data): ...
            def score(self): ...
            def reset(self): ...
    """

    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        # Validate that the method exists in methods.toml
        if not detector_exists(method_id, implementation_id):
            raise ValueError(
                f"Method '{method_id}' with implementation '{implementation_id}' "
                f"not found in methods.toml. Please add it to the registry first."
            )

        # Set the class attributes
        cls.method_id = method_id
        cls.implementation_id = implementation_id

        return cls

    return decorator


class BaseDetector(abc.ABC):
    """
    Base class for all drift detectors.

    This abstract class defines the interface that all drift detection implementations
    must follow. It provides common functionality for timing, validation, and metadata
    while requiring subclasses to implement the core detection logic.
    """

    # Class attributes for method and implementation IDs - must be set in subclasses
    method_id: str = ""
    implementation_id: str = ""

    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the detector.

        Args:
            name: Optional custom name for the detector
            **kwargs: Additional keyword arguments for detector configuration
        """
        self.name = name or self.__class__.__name__
        self._is_fitted = False
        self.fit_time: float = 0.0
        self.detect_time: float = 0.0

        # Store configuration parameters
        self.config_params = kwargs

    @classmethod
    def metadata(cls) -> DetectorData:
        """
        Get metadata about the detector from methods.toml.

        Returns:
            DetectorData object with information about the detector

        Raises:
            NotImplementedError: If method_id or implementation_id are not set
            ValueError: If the detector is not registered in methods.toml
        """
        if not cls.method_id or not cls.implementation_id:
            raise NotImplementedError(f"Subclass {cls.__name__} must set method_id and implementation_id class attributes.")

        metadata = get_detector(cls.method_id, cls.implementation_id)
        if metadata is None:
            raise ValueError(f"Detector {cls.method_id}.{cls.implementation_id} not found in methods.toml")

        return metadata

    @abc.abstractmethod
    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """
        Preprocess the dataset into the format required by the detector.

        This method converts the standardized DatasetResult (with pandas DataFrames)
        into the specific format needed by the detector implementation (e.g., numpy arrays,
        specific data structures, normalized data, etc.).

        Args:
            data: Dataset with pandas DataFrames to preprocess
            **kwargs: Additional arguments for preprocessing

        Returns:
            Preprocessed data in the format expected by fit() and detect() methods.
            This could be a tuple (X_ref, X_test, y_ref, y_test), numpy arrays,
            or any other format the detector needs.
        """
        pass

    @abc.abstractmethod
    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """
        Initialize the drift detector with preprocessed reference data.

        Args:
            preprocessed_data: Data in the format returned by preprocess()
            **kwargs: Additional arguments specific to the detector

        Returns:
            Self for method chaining
        """
        pass

    @abc.abstractmethod
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect if drift has occurred in the preprocessed test data.

        Args:
            preprocessed_data: Data in the format returned by preprocess()
            **kwargs: Additional arguments specific to the detector

        Returns:
            True if drift is detected, False otherwise
        """
        pass

    @abc.abstractmethod
    def score(self) -> Dict[str, float]:
        """
        Return the current detection scores/statistics.

        Returns:
            Dictionary with detection scores (e.g., p-values, distances, statistics)
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the detector to its initial unfitted state."""
        pass

    def timed_fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """
        Fit the detector and measure execution time.

        Args:
            preprocessed_data: Data in the format returned by preprocess()
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        start_time = time.time()
        result = self.fit(preprocessed_data, **kwargs)
        self.fit_time = time.time() - start_time
        return result

    def timed_detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift and measure execution time.

        Args:
            preprocessed_data: Data in the format returned by preprocess()
            **kwargs: Additional arguments

        Returns:
            True if drift is detected, False otherwise
        """
        start_time = time.time()
        result = self.detect(preprocessed_data, **kwargs)
        self.detect_time = time.time() - start_time
        return result

    def timed_preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """
        Preprocess data and measure execution time.

        Args:
            data: Dataset to preprocess
            **kwargs: Additional arguments

        Returns:
            Preprocessed data in detector-specific format
        """
        start_time = time.time()
        result = self.preprocess(data, **kwargs)
        self.preprocess_time = time.time() - start_time
        return result

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    def get_performance_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get performance metrics of the detector.

        Returns:
            Dictionary containing performance metrics like execution time
        """
        metrics = {
            "fit_time": getattr(self, "fit_time", None),
            "detect_time": getattr(self, "detect_time", None),
        }

        # Add preprocessing time if available
        if hasattr(self, "preprocess_time"):
            metrics["preprocess_time"] = self.preprocess_time

        return metrics

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration parameters of the detector.

        Returns:
            Dictionary containing the detector configuration
        """
        config = {
            "name": self.name,
            "class": self.__class__.__name__,
            "method_id": self.method_id,
            "implementation_id": self.implementation_id,
        }

        # Add configuration parameters
        if hasattr(self, "config_params"):
            config["parameters"] = self.config_params

        # Add metadata if available
        try:
            metadata = self.metadata()
            config["metadata"] = metadata.model_dump()
        except (ValueError, NotImplementedError):
            # Metadata not available - this is okay for some detectors
            pass

        return config

    @classmethod
    def get_method_info(cls) -> Dict[str, str]:
        """
        Get method and implementation IDs for this detector.

        Returns:
            Dictionary with method_id and implementation_id
        """
        return {
            "method_id": cls.method_id,
            "implementation_id": cls.implementation_id,
        }

    @classmethod
    def validate_method_registration(cls) -> bool:
        """
        Validate that this detector is properly registered in methods.toml.

        Returns:
            True if the detector is properly registered, False otherwise
        """
        if not cls.method_id or not cls.implementation_id:
            return False

        return detector_exists(cls.method_id, cls.implementation_id)

    def validate_data(self, data: DatasetResult) -> None:
        """
        Validate input data format and requirements.

        Args:
            data: Dataset to validate

        Raises:
            ValueError: If data doesn't meet detector requirements
        """
        if data.X_ref is None or data.X_ref.empty:
            raise ValueError("Reference data (X_ref) cannot be empty")

        # Check if detector requires labels
        try:
            metadata = self.metadata()
            if metadata.requires_labels and data.y_ref is None:
                raise ValueError(f"Detector {self.name} requires labels but y_ref is None")
        except (ValueError, NotImplementedError):
            # If metadata is not available, skip label validation
            pass

    def timed_preprocess(self, data: DatasetResult, **kwargs) -> DatasetResult:
        """
        Preprocess data and measure execution time.

        Args:
            data: Dataset to preprocess
            **kwargs: Additional arguments

        Returns:
            Preprocessed dataset
        """
        start_time = time.time()
        result = self.preprocess(data, **kwargs)
        self.preprocess_time = time.time() - start_time
        return result

    def fit_detect_workflow(self, data: DatasetResult, **kwargs) -> bool:
        """
        Complete workflow: preprocess, fit, and detect in one call.

        This is a convenience method that handles the typical workflow:
        1. Preprocess the data
        2. Fit the detector with reference data
        3. Detect drift in test data

        Args:
            data: Dataset with both reference and test data
            **kwargs: Additional arguments passed to all methods

        Returns:
            True if drift is detected, False otherwise
        """
        # Preprocess data
        preprocessed_data = self.preprocess(data, **kwargs)

        # Fit with preprocessed reference data
        self.fit(preprocessed_data, **kwargs)

        # Detect with preprocessed test data
        return self.detect(preprocessed_data, **kwargs)

    def timed_workflow(self, data: DatasetResult, **kwargs) -> bool:
        """
        Complete timed workflow: preprocess, fit, and detect with timing.

        Args:
            data: Dataset with both reference and test data
            **kwargs: Additional arguments passed to all methods

        Returns:
            True if drift is detected, False otherwise
        """
        # Timed preprocessing
        preprocessed_data = self.timed_preprocess(data, **kwargs)

        # Timed fitting
        self.timed_fit(preprocessed_data, **kwargs)

        # Timed detection
        return self.timed_detect(preprocessed_data, **kwargs)

    def extract_reference_data(self, preprocessed_data: Any) -> Any:
        """
        Extract reference data from preprocessed data.

        This is a helper method that detectors can override to extract
        reference data from their specific preprocessed format.

        Args:
            preprocessed_data: Data in detector-specific format

        Returns:
            Reference data portion for fitting
        """
        # Default implementation assumes preprocessed_data is a DatasetResult
        if hasattr(preprocessed_data, "X_ref"):
            return preprocessed_data
        return preprocessed_data

    def extract_test_data(self, preprocessed_data: Any) -> Any:
        """
        Extract test data from preprocessed data.

        This is a helper method that detectors can override to extract
        test data from their specific preprocessed format.

        Args:
            preprocessed_data: Data in detector-specific format

        Returns:
            Test data portion for detection
        """
        # Default implementation assumes preprocessed_data is a DatasetResult
        if hasattr(preprocessed_data, "X_test"):
            return preprocessed_data
        return preprocessed_data


class PeriodicTrigger(BaseDetector):
    """
    Periodic trigger detector that always detects drift or never detects it.
    Useful for testing the benchmarking framework and establishing baselines.
    """

    # Set method and implementation IDs for the dummy detector for baseline testing
    method_id = "periodic_trigger"
    implementation_id = "periodic_trigger_standard"

    def __init__(self, interval: int = 10, **kwargs) -> None:
        """
        Initialize the periodic trigger detector.

        Args:
            interval: Interval in cycles to trigger drift detection (default: 10)
            **kwargs: Additional keyword arguments for BaseDetector
        """
        super().__init__(**kwargs)
        self.interval = interval
        self.cycle_count = 0

    def preprocess(self, data: DatasetResult, **kwargs) -> DatasetResult:
        """
        Preprocess the data (no-op for periodic trigger, returns original DatasetResult).

        Args:
            data: Dataset to preprocess
            **kwargs: Additional arguments

        Returns:
            Original dataset (no preprocessing needed for this test detector)
        """
        return data

    def fit(self, preprocessed_data: DatasetResult, **kwargs) -> "PeriodicTrigger":
        """
        Fit the detector (minimal implementation for testing).

        Args:
            preprocessed_data: Preprocessed dataset (DatasetResult in this case)
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        self._is_fitted = True
        return self

    def detect(self, preprocessed_data: DatasetResult, **kwargs) -> bool:
        """
        Detect drift based on the configured interval.

        Args:
            preprocessed_data: Preprocessed dataset (DatasetResult in this case)
            **kwargs: Additional arguments

        Returns:
            True if current cycle count is divisible by interval, False otherwise

        Raises:
            RuntimeError: If detector hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        self.cycle_count += 1
        return self.cycle_count % self.interval == 0

    def score(self) -> Dict[str, float]:
        """
        Return detection scores for periodic trigger.

        Returns:
            Dictionary with cycle count and interval information

        Raises:
            RuntimeError: If detector hasn't been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling score()")

        return {
            "cycle_count": float(self.cycle_count),
            "interval": float(self.interval),
            "cycles_until_next_trigger": float(self.interval - (self.cycle_count % self.interval)),
        }

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._is_fitted = False
        self.cycle_count = 0
