"""
Evidently AI drift detectors implementation.

This module implements drift detectors using the Evidently AI library for
tabular data drift detection with various statistical tests and distance measures.

REFACTORED to match drift-benchmark Adapter API v0.1.0:
- Updated constructor to use method_id + variant_id + library_id structure
- Fixed preprocess() method to use phase="reference"/"test" parameter
- Added proper type hints with LibraryId
- Enhanced error handling and validation
- Added comprehensive docstrings
- Mapped variant IDs to actual methods.toml entries
- Fixed for Evidently v0.6.1 API

Mapping between Evidently AI tests and drift-benchmark methods:
=============================================================

Registered detectors (available in methods.toml):
- chi_square + chi_evidently: Uses Evidently's "chisquare" test
- jensen_shannon_divergence + js_evidently: Uses Evidently's "jensenshannon" test
- kullback_leibler_divergence + kl_evidently: Uses Evidently's "kl_div" test
- all_features_drift + all_features_evidently: Configurable test for all features
- kolmogorov_smirnov + ks_batch: Uses Evidently's "ks" test
- wasserstein_distance + wasserstein_batch: Uses Evidently's "wasserstein" test
- anderson_darling + ad_batch: Uses Evidently's "anderson" test
- mann_whitney + mw_batch: Uses Evidently's "mannw" test
- cramer_von_mises + cvm_batch: Uses Evidently's "cramer_von_mises" test
- epps_singleton + epps_batch: Uses Evidently's "es" test
- t_test + ttest_batch: Uses Evidently's "t_test" test

Additional Evidently tests (not registered, need methods.toml entries):
- Z-test for binary categorical data ("z")
- Population Stability Index ("psi")
- Fisher's Exact test ("fisher_exact")
- G-test ("g-test")
- Hellinger distance ("hellinger")
- Energy distance ("ed")
- Empirical Maximum Mean Discrepancy ("empirical_mmd")
- Total Variation Distance ("TVD")
"""

from typing import Any, Dict, Optional

import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAllFeaturesValueDrift,
    TestColumnDrift,
    TestCustomFeaturesValueDrift,
    TestEmbeddingsDrift,
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)

from drift_benchmark.adapters.base_detector import BaseDetector
from drift_benchmark.adapters.registry import register_detector
from drift_benchmark.literals import LibraryId
from drift_benchmark.models.results import DatasetResult
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)


class BaseEvidentlyDetector(BaseDetector):
    """
    Base class for Evidently AI drift detectors.

    Provides common functionality for all Evidently-based detectors including
    data preprocessing and result extraction.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        """
        Initialize Evidently detector.

        Args:
            method_id: Method identifier from methods.toml
            variant_id: Variant identifier from methods.toml
            library_id: Library identifier (should be "evidently")
            **kwargs: Additional parameters including:
                - threshold (float): Threshold for drift detection
                - stattest (str): Statistical test to use

        Raises:
            ValueError: If library_id is not "evidently"
        """
        super().__init__(method_id, variant_id, library_id, **kwargs)

        if library_id != "evidently":
            raise ValueError(f"Expected library_id='evidently', got '{library_id}'")

        self.threshold = kwargs.get("threshold", None)  # Will use Evidently defaults if None
        self.stattest = kwargs.get("stattest", None)  # Will be set by subclasses

        # Internal state
        self._fitted = False
        self._reference_data: Optional[pd.DataFrame] = None
        self._last_score: Optional[float] = None
        self._test_suite: Optional[TestSuite] = None

    @property
    def method_id(self) -> str:
        """Get the drift detection method identifier."""
        return self._method_id

    @property
    def variant_id(self) -> str:
        """Get the method variant identifier."""
        return self._variant_id

    @property
    def library_id(self) -> LibraryId:
        """Get the library implementation identifier."""
        return self._library_id

    def preprocess(self, data: DatasetResult, phase: str = "reference", **kwargs) -> pd.DataFrame:
        """
        Convert DatasetResult to format expected by Evidently.

        Evidently works best with pandas DataFrames directly.

        Args:
            data: Dataset containing reference and test data
            phase: "reference" for training, "test" for detection
            **kwargs: Additional preprocessing parameters

        Returns:
            Pandas DataFrame for the appropriate phase

        Raises:
            ValueError: If phase is not "reference" or "test"
        """
        if phase == "reference":
            return data.X_ref
        elif phase == "test":
            return data.X_test
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'reference' or 'test'.")

    def fit(self, preprocessed_data: pd.DataFrame, **kwargs) -> "BaseEvidentlyDetector":
        """
        Store reference data for later comparison.

        Args:
            preprocessed_data: Reference DataFrame from preprocess()
            **kwargs: Additional parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If preprocessed_data is None or empty
            TypeError: If preprocessed_data is not a pandas DataFrame
        """
        if preprocessed_data is None:
            raise ValueError("Reference data cannot be None")

        if not isinstance(preprocessed_data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(preprocessed_data)}")

        if len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        self._fitted = True

        logger.debug(f"Fitted {self.__class__.__name__} with {len(self._reference_data)} reference samples")
        return self

    def detect(self, preprocessed_data: pd.DataFrame, **kwargs) -> bool:
        """
        Perform drift detection using Evidently test suite.

        Args:
            preprocessed_data: Test DataFrame from preprocess()
            **kwargs: Additional parameters

        Returns:
            Boolean indicating whether drift was detected

        Raises:
            RuntimeError: If detector not fitted or reference data not available
            ValueError: If preprocessed_data is None or empty
            TypeError: If preprocessed_data is not a pandas DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if self._reference_data is None:
            raise RuntimeError("Reference data not available")

        if preprocessed_data is None:
            raise ValueError("Test data cannot be None")

        if not isinstance(preprocessed_data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(preprocessed_data)}")

        if len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        current_data = preprocessed_data

        try:
            # Create test suite with appropriate test
            test_kwargs = {}
            if self.threshold is not None:
                test_kwargs["threshold"] = self.threshold
            if self.stattest is not None:
                test_kwargs["stattest"] = self.stattest

            # Create test suite - use TestAllFeaturesValueDrift with specified stattest
            self._test_suite = TestSuite(tests=[TestAllFeaturesValueDrift(**test_kwargs)])

            # Run the test suite
            self._test_suite.run(reference_data=self._reference_data, current_data=current_data)

            # Extract results
            test_results = self._test_suite.as_dict()
            test_result = test_results["tests"][0]

            # Store drift score (varies by test type)
            self._last_score = self._extract_drift_score(test_result)

            # Return drift detection result
            drift_detected = test_result["status"] == "FAIL"

            logger.debug(f"{self.__class__.__name__} drift detection: {drift_detected}, score: {self._last_score}")
            return drift_detected

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} drift detection: {e}")
            # Return False on error to be conservative
            return False

    def score(self) -> Optional[float]:
        """
        Return drift score from last detection.

        Returns:
            Drift score if available, None otherwise
        """
        return self._last_score

    def _extract_drift_score(self, test_result: Dict[str, Any]) -> Optional[float]:
        """
        Extract drift score from Evidently test result.

        Args:
            test_result: Evidently test result dictionary

        Returns:
            Drift score if available, None otherwise
        """
        # Try to extract score from different possible locations
        parameters = test_result.get("parameters", {})

        # For p-value based tests (KS, Chi-square, Anderson-Darling, etc.)
        if "p_value" in parameters:
            return parameters["p_value"]

        # For distance/divergence based tests
        for score_key in ["distance", "divergence", "psi_value"]:
            if score_key in parameters:
                return parameters[score_key]

        # For column-specific results, try to extract from the first column
        # This handles cases where Evidently tests individual columns
        if "columns" in parameters and isinstance(parameters["columns"], dict):
            columns_data = parameters["columns"]
            # Get the first column's result
            for column_name, column_data in columns_data.items():
                if isinstance(column_data, dict):
                    # Try p_value first
                    if "p_value" in column_data:
                        return column_data["p_value"]
                    # Then distance metrics
                    for score_key in ["distance", "divergence", "psi_value", "statistic"]:
                        if score_key in column_data:
                            return column_data[score_key]

        # For aggregate results, look in different locations
        if "aggregate" in parameters:
            aggregate_data = parameters["aggregate"]
            if isinstance(aggregate_data, dict):
                for score_key in ["p_value", "distance", "divergence", "psi_value", "statistic"]:
                    if score_key in aggregate_data:
                        return aggregate_data[score_key]

        # Try to extract from test statistics directly
        if "statistic" in parameters:
            return parameters["statistic"]

        return None


@register_detector(method_id="chi_square", variant_id="chi_evidently", library_id="evidently")
class EvidentlyChiSquareDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Chi-square test.

    Uses Evidently's Chi-square test for categorical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "chisquare"  # Evidently's Chi-square test identifier


@register_detector(method_id="jensen_shannon_divergence", variant_id="js_evidently", library_id="evidently")
class EvidentlyJensenShannonDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Jensen-Shannon divergence.

    Uses Evidently's Jensen-Shannon distance for drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "jensenshannon"  # Evidently's JS distance identifier


@register_detector(method_id="kullback_leibler_divergence", variant_id="kl_evidently", library_id="evidently")
class EvidentlyKullbackLeiblerDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Kullback-Leibler divergence.

    Uses Evidently's KL divergence for drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "kl_div"  # Evidently's KL divergence identifier


@register_detector(method_id="all_features_drift", variant_id="all_features_evidently", library_id="evidently")
class EvidentlyAllFeaturesDriftDetector(BaseEvidentlyDetector):
    """
    Evidently implementation for comprehensive drift detection across all features.

    Uses configurable statistical tests for multi-feature drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        # Allow configuration of the statistical test for all features
        self.stattest = kwargs.get("stattest", None)  # Can be configured per use case

    def detect(self, preprocessed_data: pd.DataFrame, **kwargs) -> bool:
        """
        Perform drift detection across all features using Evidently test suite.

        For all features drift, we check each feature individually and return True
        if any feature shows drift.

        Args:
            preprocessed_data: Test DataFrame from preprocess()
            **kwargs: Additional parameters

        Returns:
            Boolean indicating whether drift was detected in any feature
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")

        if self._reference_data is None:
            raise RuntimeError("Reference data not available")

        current_data = preprocessed_data

        # For all features drift, we test each column individually
        drift_detected = False
        drift_scores = []

        for column in self._reference_data.columns:
            # Create test suite for this specific column
            test_kwargs = {"column": column}
            if self.threshold is not None:
                test_kwargs["threshold"] = self.threshold
            if self.stattest is not None:
                test_kwargs["stattest"] = self.stattest

            try:
                # Create test suite for this column - use TestColumnDrift
                column_test_suite = TestSuite(tests=[TestColumnDrift(**test_kwargs)])

                # Run the test suite
                column_test_suite.run(reference_data=self._reference_data, current_data=current_data)

                # Extract results
                test_results = column_test_suite.as_dict()
                test_result = test_results["tests"][0]

                # Check if this column shows drift
                column_drift = test_result["status"] == "FAIL"
                if column_drift:
                    drift_detected = True

                # Extract and store score
                score = self._extract_drift_score(test_result)
                if score is not None:
                    drift_scores.append(score)

            except Exception as e:
                logger.warning(f"Error testing column {column}: {e}")
                continue

        # Store the average score as the overall drift score
        if drift_scores:
            self._last_score = sum(drift_scores) / len(drift_scores)
        else:
            self._last_score = None

        logger.debug(f"{self.__class__.__name__} drift detection: {drift_detected}, avg_score: {self._last_score}")
        return drift_detected


# Use Evidently's KS test as an alternative implementation for Kolmogorov-Smirnov
@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch", library_id="evidently")
class EvidentlyKSDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Kolmogorov-Smirnov test.

    Uses Evidently's KS test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "ks"  # Evidently's KS test identifier


# Use Evidently's Wasserstein distance as an implementation for Wasserstein method
@register_detector(method_id="wasserstein_distance", variant_id="wasserstein_batch", library_id="evidently")
class EvidentlyWassersteinDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Wasserstein distance.

    Uses Evidently's Wasserstein distance for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "wasserstein"  # Evidently's Wasserstein distance identifier


# Use Evidently's Anderson-Darling test as an alternative implementation
@register_detector(method_id="anderson_darling", variant_id="ad_batch", library_id="evidently")
class EvidentlyAndersonDarlingDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Anderson-Darling test.

    Uses Evidently's Anderson-Darling test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "anderson"  # Evidently's Anderson-Darling test identifier


# Use Evidently's Mann-Whitney test as an alternative implementation
@register_detector(method_id="mann_whitney", variant_id="mw_batch", library_id="evidently")
class EvidentlyMannWhitneyDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Mann-Whitney U test.

    Uses Evidently's Mann-Whitney test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "mannw"  # Evidently's Mann-Whitney test identifier


# Use Evidently's Cramer-von Mises test as an alternative implementation
@register_detector(method_id="cramer_von_mises", variant_id="cvm_batch", library_id="evidently")
class EvidentlyCramerVonMisesDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Cramer-von Mises test.

    Uses Evidently's Cramer-von Mises test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "cramer_von_mises"  # Evidently's Cramer-von Mises test identifier


# Use Evidently's Epps-Singleton test as an alternative implementation
@register_detector(method_id="epps_singleton", variant_id="epps_batch", library_id="evidently")
class EvidentlyEppsSingletonDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of Epps-Singleton test.

    Uses Evidently's Epps-Singleton test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "es"  # Evidently's Epps-Singleton test identifier


# Use Evidently's T-Test as an alternative implementation
@register_detector(method_id="t_test", variant_id="ttest_batch", library_id="evidently")
class EvidentlyTTestDetector(BaseEvidentlyDetector):
    """
    Evidently implementation of T-Test.

    Uses Evidently's T-Test for numerical data drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "t_test"  # Evidently's T-Test identifier


# Additional unique Evidently methods - these would need corresponding entries in methods.toml to be registered


class EvidentlyZTestDetector(BaseEvidentlyDetector):
    """Evidently implementation of Z-test for binary categorical data."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "z"


class EvidentlyPSIDetector(BaseEvidentlyDetector):
    """Evidently implementation of Population Stability Index."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "psi"


class EvidentlyFisherExactDetector(BaseEvidentlyDetector):
    """Evidently implementation of Fisher's Exact test."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "fisher_exact"


class EvidentlyGTestDetector(BaseEvidentlyDetector):
    """Evidently implementation of G-test."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "g-test"


class EvidentlyHellingerDetector(BaseEvidentlyDetector):
    """Evidently implementation of Hellinger distance."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "hellinger"


class EvidentlyEnergyDistanceDetector(BaseEvidentlyDetector):
    """Evidently implementation of Energy distance."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "ed"


class EvidentlyEmpiricalMMDDetector(BaseEvidentlyDetector):
    """Evidently implementation of Empirical Maximum Mean Discrepancy."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "empirical_mmd"


class EvidentlyTVDDetector(BaseEvidentlyDetector):
    """Evidently implementation of Total Variation Distance."""

    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.stattest = "TVD"
