"""
Mock adapters for testing the drift detection benchmark framework.

This module provides mock implementations of drift detectors with configurable
behaviors, useful for testing the benchmark infrastructure and validating metrics.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from drift_benchmark.detectors import BaseDetector

logger = logging.getLogger(__name__)


class MockDetector(BaseDetector):
    """
    A configurable mock detector for testing.

    This detector allows precise control over drift detection outcomes,
    enabling thorough testing of the benchmark infrastructure.
    """

    def __init__(
        self,
        drift_scenario: str = "random",
        drift_probability: float = 0.5,
        drift_threshold: float = 0.05,
        detection_delay: int = 0,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the mock detector.

        Args:
            drift_scenario: Drift detection behavior ('random', 'threshold', 'pattern', 'scheduled')
            drift_probability: Probability of detecting drift for 'random' scenario
            drift_threshold: Threshold for detecting drift in 'threshold' scenario
            detection_delay: Number of calls to detect() before signaling drift
            false_positive_rate: Probability of false positives
            false_negative_rate: Probability of false negatives
            seed: Random seed for reproducibility
            **kwargs: Additional keyword arguments for BaseDetector
        """
        super().__init__(**kwargs)

        self.drift_scenario = drift_scenario
        self.drift_probability = drift_probability
        self.drift_threshold = drift_threshold
        self.detection_delay = detection_delay
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate

        self.is_fitted = False
        self.reference_stats = {}
        self.current_stats = {}
        self.detect_count = 0
        self.drift_detected = False
        self.stats_diff = 0.0

        # Initialize random state
        self.random = random.Random(seed)

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "MockDetector":
        """
        Store reference data statistics.

        Args:
            reference_data: Reference data
            **kwargs: Additional arguments

        Returns:
            Self
        """
        # Store some basic stats about reference data
        if isinstance(reference_data, pd.DataFrame):
            data = reference_data.select_dtypes(include="number").values
        else:
            data = reference_data

        self.reference_stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "shape": data.shape,
        }

        self.is_fitted = True
        self.detect_count = 0
        logger.info(f"MockDetector fitted with reference data of shape {data.shape}")

        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Detect drift according to the configured scenario.

        Args:
            data: New data to check for drift
            **kwargs: Additional arguments

        Returns:
            Boolean indicating whether drift is detected
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        # Process data
        if isinstance(data, pd.DataFrame):
            data = data.select_dtypes(include="number").values

        # Store basic stats about current data
        self.current_stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "shape": data.shape,
        }

        # Calculate statistic difference (for threshold scenario)
        self.stats_diff = abs(self.current_stats["mean"] - self.reference_stats["mean"])

        # Increment detection counter
        self.detect_count += 1

        # Determine drift based on scenario
        base_drift_decision = False

        if self.drift_scenario == "random":
            base_drift_decision = self.random.random() < self.drift_probability

        elif self.drift_scenario == "threshold":
            base_drift_decision = self.stats_diff > self.drift_threshold

        elif self.drift_scenario == "pattern":
            # Detect drift every 3rd call
            base_drift_decision = self.detect_count % 3 == 0

        elif self.drift_scenario == "scheduled":
            # Detect drift after detection_delay calls
            base_drift_decision = self.detect_count >= self.detection_delay

        # Apply false positives/negatives
        if base_drift_decision:
            # Might turn to false negative
            self.drift_detected = self.random.random() >= self.false_negative_rate
        else:
            # Might turn to false positive
            self.drift_detected = self.random.random() < self.false_positive_rate

        logger.debug(
            f"MockDetector detection #{self.detect_count}: "
            f"base_decision={base_drift_decision}, "
            f"final_decision={self.drift_detected}, "
            f"stats_diff={self.stats_diff:.4f}"
        )

        return self.drift_detected

    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling score()")

        if not self.current_stats:
            raise RuntimeError("No drift detection has been performed")

        return {
            "drift_probability": float(self.drift_probability),
            "statistics_difference": float(self.stats_diff),
            "threshold": float(self.drift_threshold),
            "detection_count": int(self.detect_count),
            "drift_detected": int(self.drift_detected),
        }

    def reset(self) -> None:
        """Reset the detector state."""
        if self.is_fitted:
            self.current_stats = {}
            self.detect_count = 0
            self.drift_detected = False
            self.stats_diff = 0.0


class PredefinedDetector(BaseDetector):
    """
    A detector with predefined responses for testing specific scenarios.

    This detector returns predetermined drift detection results for each call,
    enabling precise testing of benchmark behavior in controlled sequences.
    """

    def __init__(
        self,
        drift_sequence: List[bool] = None,
        p_values: List[float] = None,
        cycle_when_exhausted: bool = True,
        **kwargs,
    ):
        """
        Initialize the predefined detector.

        Args:
            drift_sequence: Sequence of boolean values to return for each detect() call
            p_values: Sequence of p-values to return for each score() call
            cycle_when_exhausted: If True, start over when sequence is exhausted
            **kwargs: Additional keyword arguments for BaseDetector
        """
        super().__init__(**kwargs)

        self.drift_sequence = drift_sequence or [False, False, True, False, True]
        self.p_values = p_values or [0.8, 0.6, 0.02, 0.7, 0.01]
        self.cycle_when_exhausted = cycle_when_exhausted

        self.is_fitted = False
        self.call_index = 0
        self.last_result = False

    def fit(self, reference_data: Union[np.ndarray, pd.DataFrame], **kwargs) -> "PredefinedDetector":
        """
        Initialize the detector.

        Args:
            reference_data: Reference data (not used)
            **kwargs: Additional arguments

        Returns:
            Self
        """
        self.is_fitted = True
        self.call_index = 0
        return self

    def detect(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> bool:
        """
        Return the next predefined drift detection result.

        Args:
            data: New data to check for drift (not used)
            **kwargs: Additional arguments

        Returns:
            Next boolean in the predefined sequence
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")

        if self.call_index >= len(self.drift_sequence):
            if self.cycle_when_exhausted:
                self.call_index = 0
            else:
                # If we're out of predefined values, return the last one
                return self.last_result

        result = self.drift_sequence[self.call_index]
        self.last_result = result
        self.call_index += 1

        return result

    def score(self) -> Dict[str, float]:
        """
        Return detection scores.

        Returns:
            Dictionary with detection scores
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before calling score()")

        # Use the previous call index (since it was incremented in detect())
        idx = (self.call_index - 1) % len(self.p_values) if self.call_index > 0 else 0

        return {
            "p_value": float(self.p_values[idx]),
            "call_index": int(self.call_index - 1),
            "drift_detected": int(self.last_result),
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self.call_index = 0
        self.last_result = False
