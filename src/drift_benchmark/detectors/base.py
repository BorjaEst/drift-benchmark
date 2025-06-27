"""
Base classes and interfaces for drift detection algorithms.

This module defines the common interface that all drift detectors must implement,
enabling standardized benchmarking and evaluation across different libraries.
"""

import abc
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from drift_benchmark.constants.types import DetectorMetadata
from drift_benchmark.methods import detector_exists, get_detector_by_id


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
        self.fit_time: Optional[float] = None
        self.detect_time: Optional[float] = None
        self._is_fitted = False

    @classmethod
    def metadata(cls) -> DetectorMetadata:
        """
        Get metadata about the detector from methods.toml.

        Returns:
            DetectorMetadata object with information about the detector

        Raises:
            NotImplementedError: If method_id or implementation_id are not set
            ValueError: If the detector is not registered in methods.toml
        """
        if not cls.method_id or not cls.implementation_id:
            raise NotImplementedError(
                f"Subclass {cls.__name__} must set method_id and implementation_id class attributes."
            )

        metadata = get_detector_by_id(cls.method_id, cls.implementation_id)
        if metadata is None:
            raise ValueError(f"Detector {cls.method_id}.{cls.implementation_id} not found in methods.toml")

        return metadata

    @abc.abstractmethod
    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseDetector":
        """
        Initialize the drift detector with reference data.

        Args:
            reference_data: Reference data used as baseline for drift detection
            **kwargs: Additional arguments specific to the detector

        Returns:
            Self for method chaining
        """
        pass

    @abc.abstractmethod
    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect if drift has occurred in the provided data.

        Args:
            data: New data to check for drift against reference data
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

    def timed_fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "BaseDetector":
        """
        Fit the detector and measure execution time.

        Args:
            reference_data: Reference data used as baseline
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        start_time = time.time()
        result = self.fit(reference_data, **kwargs)
        self.fit_time = time.time() - start_time
        return result

    def timed_detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect drift and measure execution time.

        Args:
            data: New data to check for drift
            **kwargs: Additional arguments

        Returns:
            True if drift is detected, False otherwise
        """
        start_time = time.time()
        result = self.detect(data, **kwargs)
        self.detect_time = time.time() - start_time
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
        return {"fit_time": self.fit_time, "detect_time": self.detect_time}

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

        # Add metadata if available
        try:
            metadata = self.metadata()
            config["metadata"] = metadata.model_dump()
        except (ValueError, NotImplementedError):
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

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "PeriodicTrigger":
        """
        Fit the detector (minimal implementation for testing).

        Args:
            reference_data: Reference data (not actually used in this test detector)
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        self._is_fitted = True
        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect drift based on the configured interval.

        Args:
            data: New data to check for drift
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


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    def _example_usage() -> None:
        """Demonstrate usage of the BaseDetector interface."""
        print("=== BaseDetector Example Usage ===\n")

        # 1. Create sample data
        print("1. Creating sample data...")
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))
        test_data_no_drift = np.random.normal(0, 1, (50, 3))  # Same distribution
        test_data_with_drift = np.random.normal(2, 1, (50, 3))  # Shifted distribution

        print(f"   Reference data shape: {reference_data.shape}")
        print(f"   Test data (no drift) shape: {test_data_no_drift.shape}")
        print(f"   Test data (with drift) shape: {test_data_with_drift.shape}")

        # 2. Create detector instance
        print("\n2. Creating PeriodicTrigger detector...")
        detector = PeriodicTrigger(interval=3, name="TestDetector")
        print(f"   Detector name: {detector.name}")
        print(f"   Interval: {detector.interval}")
        print(f"   Is fitted: {detector.is_fitted}")

        # 3. Test metadata access
        print("\n3. Testing metadata access...")
        try:
            metadata = detector.metadata()
            print(f"   Method: {metadata.method.name}")
            print(f"   Description: {metadata.method.description}")
            print(f"   Implementation: {metadata.implementation.name}")
            print(f"   Execution mode: {metadata.implementation.execution_mode}")
        except Exception as e:
            print(f"   Metadata error: {e}")

        # 4. Test method registration validation
        print("\n4. Testing method registration validation...")
        is_valid = detector.validate_method_registration()
        print(f"   Is properly registered: {is_valid}")

        # 5. Fit the detector
        print("\n5. Fitting detector...")
        detector.fit(reference_data)
        print(f"   Is fitted after fit(): {detector.is_fitted}")

        # 6. Test detection on data without drift...
        print("\n6. Testing detection without drift...")
        results_no_drift = []
        for i in range(5):
            batch = (
                test_data_no_drift[i * 10 : (i + 1) * 10]
                if len(test_data_no_drift) >= (i + 1) * 10
                else test_data_no_drift[i * 10 :]
            )
            if len(batch) > 0:
                result = detector.detect(batch)
                results_no_drift.append(result)
                score = detector.score()
                print(
                    f"   Batch {i+1}: Drift={result}, Cycle={score['cycle_count']}, Until next trigger={score['cycles_until_next_trigger']}"
                )

        # 7. Reset detector and test again
        print("\n7. Resetting detector...")
        detector.reset()
        print(f"   Is fitted after reset: {detector.is_fitted}")

        # 8. Test timed operations
        print("\n8. Testing timed operations...")
        detector.timed_fit(reference_data)
        drift_detected = detector.timed_detect(test_data_with_drift)

        performance = detector.get_performance_metrics()
        print(f"   Fit time: {performance['fit_time']:.6f} seconds")
        print(f"   Detect time: {performance['detect_time']:.6f} seconds")
        print(f"   Drift detected: {drift_detected}")

        # 9. Test configuration retrieval
        print("\n9. Testing configuration retrieval...")
        config = detector.get_config()
        print("   Configuration:")
        for key, value in config.items():
            if key == "metadata" and isinstance(value, dict):
                print(f"     {key}: (metadata dict with {len(value)} keys)")
            else:
                print(f"     {key}: {value}")

        # 10. Test with pandas DataFrame
        print("\n10. Testing with pandas DataFrame...")
        df_reference = pd.DataFrame(reference_data, columns=["feature_1", "feature_2", "feature_3"])
        df_test = pd.DataFrame(test_data_no_drift, columns=["feature_1", "feature_2", "feature_3"])

        detector.reset()
        detector.fit(df_reference)
        drift_result = detector.detect(df_test[:10])  # Test with first 10 rows

        print(f"   DataFrame reference shape: {df_reference.shape}")
        print(f"   DataFrame test shape: {df_test.shape}")
        print(f"   Drift detected with DataFrame: {drift_result}")  # 11. Test error conditions
        print("\n11. Testing error conditions...")

        # Test detect without fit
        unfitted_detector = PeriodicTrigger(interval=5)
        try:
            unfitted_detector.detect(test_data_no_drift)
        except RuntimeError as e:
            print(f"   Expected error for unfitted detector: {e}")

        # 12. Test with univariate data
        print("\n12. Testing with univariate data...")
        univariate_ref = np.random.normal(0, 1, 100)  # 1D array
        univariate_test = np.random.normal(0, 1, 50)

        detector.reset()
        detector.fit(univariate_ref)
        drift_result_univariate = detector.detect(univariate_test)
        print(f"   Univariate drift detected: {drift_result_univariate}")

        print("\n=== Example completed successfully! ===")

    def _demonstrate_custom_detector() -> None:
        """Show how to implement a custom detector."""
        print("\n=== Custom Detector Implementation Example ===\n")

        class SimpleThresholdDetector(BaseDetector):
            """
            Example custom detector that compares mean values.
            This demonstrates how to properly implement the BaseDetector interface.
            """

            # These would normally reference an entry in methods.toml
            method_id = "simple_threshold"  # Would need to be added to methods.toml
            implementation_id = "threshold_batch"

            def __init__(self, threshold: float = 1.0, **kwargs):
                super().__init__(**kwargs)
                self.threshold = threshold
                self.reference_mean: Optional[float] = None
                self.current_mean: Optional[float] = None

            def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "SimpleThresholdDetector":
                """Fit by calculating reference mean."""
                if isinstance(reference_data, pd.DataFrame):
                    data_array = reference_data.values
                else:
                    data_array = reference_data

                self.reference_mean = float(np.mean(data_array))
                self._is_fitted = True
                return self

            def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
                """Detect drift by comparing means."""
                if not self.is_fitted:
                    raise RuntimeError("Detector must be fitted before calling detect()")

                if isinstance(data, pd.DataFrame):
                    data_array = data.values
                else:
                    data_array = data

                self.current_mean = float(np.mean(data_array))
                difference = abs(self.current_mean - self.reference_mean)

                return difference > self.threshold

            def score(self) -> Dict[str, float]:
                """Return threshold, reference mean, and current mean."""
                if not self.is_fitted:
                    raise RuntimeError("Detector must be fitted before calling score()")

                scores = {
                    "threshold": self.threshold,
                    "reference_mean": self.reference_mean,
                }

                if self.current_mean is not None:
                    scores["current_mean"] = self.current_mean
                    scores["mean_difference"] = abs(self.current_mean - self.reference_mean)

                return scores

            def reset(self) -> None:
                """Reset detector state."""
                self._is_fitted = False
                self.reference_mean = None
                self.current_mean = None

        # Demonstrate the custom detector
        print("1. Creating custom SimpleThresholdDetector...")
        custom_detector = SimpleThresholdDetector(threshold=0.5, name="CustomDetector")

        # Create test data
        np.random.seed(123)
        ref_data = np.random.normal(0, 1, (100, 2))
        test_data_similar = np.random.normal(0.2, 1, (50, 2))  # Small shift
        test_data_different = np.random.normal(1.5, 1, (50, 2))  # Large shift

        # Fit and test
        print("2. Fitting custom detector...")
        custom_detector.fit(ref_data)

        scores = custom_detector.score()
        print(f"   Reference mean: {scores['reference_mean']:.3f}")
        print(f"   Threshold: {scores['threshold']}")

        print("3. Testing detection...")
        drift_similar = custom_detector.detect(test_data_similar)
        scores_similar = custom_detector.score()

        drift_different = custom_detector.detect(test_data_different)
        scores_different = custom_detector.score()

        print(f"   Small shift data:")
        print(f"     - Current mean: {scores_similar['current_mean']:.3f}")
        print(f"     - Mean difference: {scores_similar['mean_difference']:.3f}")
        print(f"     - Drift detected: {drift_similar}")

        print(f"   Large shift data:")
        print(f"     - Current mean: {scores_different['current_mean']:.3f}")
        print(f"     - Mean difference: {scores_different['mean_difference']:.3f}")
        print(f"     - Drift detected: {drift_different}")

        print("\n=== Custom detector example completed! ===")

    # Run examples
    _example_usage()
    _demonstrate_custom_detector()
