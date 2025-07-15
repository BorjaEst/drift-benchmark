"""
Tests for generators module.

This module contains comprehensive tests for the generators.py module,
which provides synthetic data generation with various types of drift
patterns and characteristics across multiple data generators.
"""

import numpy as np
import pandas as pd
import pytest

from drift_benchmark.constants.literals import DataGenerator, DriftCharacteristic, DriftPattern
from drift_benchmark.data.generators import generate_drift, generate_synthetic_data

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def base_generator_params():
    """Base parameters for synthetic data generation."""
    return {
        "n_samples": 1000,
        "n_features": 5,
        "drift_magnitude": 1.0,
        "drift_position": 0.5,
        "noise": 0.1,
        "random_state": 42,
    }


@pytest.fixture
def all_generators():
    """List of all available generators."""
    return ["GAUSSIAN", "MIXED", "MULTIMODAL", "TIME_SERIES"]


@pytest.fixture
def all_drift_patterns():
    """List of all available drift patterns."""
    return ["SUDDEN", "GRADUAL", "INCREMENTAL", "RECURRING", "SEASONAL"]


@pytest.fixture
def all_drift_characteristics():
    """List of all available drift characteristics."""
    return ["MEAN_SHIFT", "VARIANCE_SHIFT", "CORRELATION_SHIFT", "DISTRIBUTION_SHIFT"]


# =============================================================================
# TESTS FOR MAIN GENERATION FUNCTIONS
# =============================================================================


class TestGenerateSyntheticData:
    """Test suite for generate_synthetic_data function."""

    def test_basic_gaussian_generation(self, base_generator_params):
        """Test basic Gaussian data generation."""
        X_ref, X_test, y_ref, y_test, drift_info = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            **base_generator_params,
        )

        # Check basic properties
        assert X_ref.shape[0] == 500  # Half the samples (drift_position=0.5)
        assert X_test.shape[0] == 500
        assert X_ref.shape[1] == base_generator_params["n_features"]
        assert X_test.shape[1] == base_generator_params["n_features"]

        # Check that data is numeric
        assert np.isfinite(X_ref).all()
        assert np.isfinite(X_test).all()

        # Check drift info
        assert drift_info["generator"] == "GAUSSIAN"
        assert drift_info["drift_pattern"] == "SUDDEN"
        assert drift_info["drift_characteristic"] == "MEAN_SHIFT"
        assert "drift_magnitude" in drift_info

        # For unsupervised generation, labels should be None
        assert y_ref is None
        assert y_test is None

    def test_all_generators_basic_functionality(self, base_generator_params, all_generators):
        """Test that all generators produce valid output."""
        for generator in all_generators:
            X_ref, X_test, y_ref, y_test, drift_info = generate_synthetic_data(
                generator_name=generator,
                drift_pattern="SUDDEN",
                drift_characteristic="MEAN_SHIFT",
                **base_generator_params,
            )

            # Basic shape checks
            assert X_ref.shape[0] + X_test.shape[0] == base_generator_params["n_samples"]
            assert X_ref.shape[1] == base_generator_params["n_features"]
            assert X_test.shape[1] == base_generator_params["n_features"]

            # Data validity checks
            assert np.isfinite(X_ref).all()
            assert np.isfinite(X_test).all()

            # Drift info checks
            assert drift_info["generator"] == generator

    def test_all_drift_patterns(self, base_generator_params, all_drift_patterns):
        """Test all drift patterns with Gaussian generator."""
        for pattern in all_drift_patterns:
            # Add drift_duration for gradual pattern
            params = base_generator_params.copy()
            if pattern == "GRADUAL":
                params["drift_duration"] = 0.3

            X_ref, X_test, y_ref, y_test, drift_info = generate_synthetic_data(
                generator_name="GAUSSIAN",
                drift_pattern=pattern,
                drift_characteristic="MEAN_SHIFT",
                **params,
            )

            # Basic checks
            assert X_ref.shape[0] > 0
            assert X_test.shape[0] > 0
            assert drift_info["drift_pattern"] == pattern

    def test_all_drift_characteristics(self, base_generator_params, all_drift_characteristics):
        """Test all drift characteristics with Gaussian generator."""
        for characteristic in all_drift_characteristics:
            X_ref, X_test, y_ref, y_test, drift_info = generate_synthetic_data(
                generator_name="GAUSSIAN",
                drift_pattern="SUDDEN",
                drift_characteristic=characteristic,
                **base_generator_params,
            )

            # Basic checks
            assert X_ref.shape[0] > 0
            assert X_test.shape[0] > 0
            assert drift_info["drift_characteristic"] == characteristic

    def test_drift_magnitude_effect(self, base_generator_params):
        """Test that drift magnitude affects the generated data."""
        # Generate data with low and high drift magnitude
        X_ref_low, X_test_low, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_magnitude=0.1,
            **{k: v for k, v in base_generator_params.items() if k != "drift_magnitude"},
        )

        X_ref_high, X_test_high, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_magnitude=3.0,
            **{k: v for k, v in base_generator_params.items() if k != "drift_magnitude"},
        )

        # Reference data should be similar (same seed)
        np.testing.assert_allclose(X_ref_low, X_ref_high, rtol=1e-10)

        # Test data should be more different with higher drift magnitude
        diff_low = np.abs(X_test_low.mean(axis=0) - X_ref_low.mean(axis=0))
        diff_high = np.abs(X_test_high.mean(axis=0) - X_ref_high.mean(axis=0))

        # Higher drift should produce larger differences
        assert diff_high.mean() > diff_low.mean()

    def test_drift_position_effect(self, base_generator_params):
        """Test that drift position affects the split between reference and test data."""
        # Test different drift positions
        positions = [0.2, 0.5, 0.8]
        n_samples = base_generator_params["n_samples"]

        for position in positions:
            X_ref, X_test, _, _, _ = generate_synthetic_data(
                generator_name="GAUSSIAN",
                drift_pattern="SUDDEN",
                drift_characteristic="MEAN_SHIFT",
                drift_position=position,
                **{k: v for k, v in base_generator_params.items() if k != "drift_position"},
            )

            expected_ref_samples = int(n_samples * position)
            expected_test_samples = n_samples - expected_ref_samples

            assert X_ref.shape[0] == expected_ref_samples
            assert X_test.shape[0] == expected_test_samples

    def test_drift_affected_features(self, base_generator_params):
        """Test that only specified features are affected by drift."""
        affected_features = [0, 2]  # Only first and third features

        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_affected_features=affected_features,
            **base_generator_params,
        )

        # Calculate mean differences between reference and test data
        mean_diffs = np.abs(X_test.mean(axis=0) - X_ref.mean(axis=0))

        # Affected features should have larger differences
        for i in range(base_generator_params["n_features"]):
            if i in affected_features:
                assert mean_diffs[i] > 0.5  # Should have noticeable drift
            else:
                # Non-affected features should have smaller differences
                # (allowing for some random variation)
                assert mean_diffs[i] < 0.3

    def test_noise_effect(self, base_generator_params):
        """Test that noise parameter affects data variability."""
        # Generate data with different noise levels
        X_ref_low, X_test_low, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            noise=0.01,
            **{k: v for k, v in base_generator_params.items() if k != "noise"},
        )

        X_ref_high, X_test_high, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            noise=1.0,
            **{k: v for k, v in base_generator_params.items() if k != "noise"},
        )

        # Higher noise should produce higher variance
        var_low = np.var(X_ref_low, axis=0).mean()
        var_high = np.var(X_ref_high, axis=0).mean()

        assert var_high > var_low

    def test_random_state_reproducibility(self, base_generator_params):
        """Test that random_state ensures reproducible results."""
        # Generate data twice with same random state
        result1 = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            **base_generator_params,
        )

        result2 = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            **base_generator_params,
        )

        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])  # X_ref
        np.testing.assert_array_equal(result1[1], result2[1])  # X_test

    def test_mixed_data_with_categorical_features(self, base_generator_params):
        """Test mixed data generation with categorical features."""
        categorical_features = [1, 3]

        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="MIXED",
            drift_pattern="SUDDEN",
            drift_characteristic="DISTRIBUTION_SHIFT",
            categorical_features=categorical_features,
            **base_generator_params,
        )

        # Check that categorical features contain integer values
        for cat_feat in categorical_features:
            assert np.all(X_ref[:, cat_feat] == X_ref[:, cat_feat].astype(int))
            assert np.all(X_test[:, cat_feat] == X_test[:, cat_feat].astype(int))

        # Check drift info contains categorical feature information
        assert "categorical_features" in drift_info
        assert drift_info["categorical_features"] == categorical_features

    def test_gradual_drift_with_duration(self, base_generator_params):
        """Test gradual drift with specific duration."""
        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="GAUSSIAN",
            drift_pattern="GRADUAL",
            drift_characteristic="MEAN_SHIFT",
            drift_duration=0.5,  # Drift over first half of test period
            **base_generator_params,
        )

        # Basic checks
        assert X_ref.shape[0] > 0
        assert X_test.shape[0] > 0

        # Check that drift_duration is recorded
        assert "drift_duration" in drift_info
        assert drift_info["drift_duration"] == 0.5

    def test_multimodal_data_generation(self, base_generator_params):
        """Test multimodal data generation with specific parameters."""
        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="MULTIMODAL",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            n_modes=3,
            separation=2.0,
            **base_generator_params,
        )

        # Basic checks
        assert X_ref.shape[0] > 0
        assert X_test.shape[0] > 0
        assert drift_info["generator"] == "MULTIMODAL"

        # Check that generator parameters are recorded
        assert "parameters" in drift_info

    def test_time_series_data_generation(self, base_generator_params):
        """Test time series data generation."""
        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="TIME_SERIES",
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            trend=0.01,
            seasonality=0.1,
            **base_generator_params,
        )

        # Basic checks
        assert X_ref.shape[0] > 0
        assert X_test.shape[0] > 0
        assert drift_info["generator"] == "TIME_SERIES"

    def test_invalid_generator_raises_error(self, base_generator_params):
        """Test that invalid generator names raise appropriate errors."""
        with pytest.raises(ValueError, match="Unknown generator"):
            generate_synthetic_data(
                generator_name="INVALID_GENERATOR",
                drift_pattern="SUDDEN",
                drift_characteristic="MEAN_SHIFT",
                **base_generator_params,
            )


class TestGenerateDrift:
    """Test suite for legacy generate_drift function."""

    def test_basic_legacy_functionality(self):
        """Test basic functionality of legacy generate_drift function."""
        X_ref_df, X_test_df, metadata = generate_drift(
            generator_name="gaussian",
            n_samples=1000,
            n_features=5,
            drift_type="sudden",
            drift_magnitude=1.0,
            drift_ratio=0.6,
            drift_position=0.5,
            noise=0.05,
            random_state=42,
        )

        # Check that outputs are DataFrames
        assert isinstance(X_ref_df, pd.DataFrame)
        assert isinstance(X_test_df, pd.DataFrame)
        assert isinstance(metadata, dict)

        # Check shapes
        assert len(X_ref_df) == 500  # Half the samples
        assert len(X_test_df) == 500
        assert X_ref_df.shape[1] == 5
        assert X_test_df.shape[1] == 5

        # Check column names
        expected_columns = [f"feature_{i}" for i in range(5)]
        assert list(X_ref_df.columns) == expected_columns
        assert list(X_test_df.columns) == expected_columns

        # Check metadata
        assert metadata["name"] == "gaussian_drift"
        assert metadata["drift_type"] == "sudden"
        assert metadata["drift_magnitude"] == 1.0
        assert metadata["n_features"] == 5
        assert metadata["n_samples"] == 1000

    def test_legacy_generator_mapping(self):
        """Test that legacy generator names are properly mapped."""
        generators = ["gaussian", "mixed", "multimodal", "time_series"]

        for generator in generators:
            X_ref_df, X_test_df, metadata = generate_drift(
                generator_name=generator,
                n_samples=100,
                n_features=3,
                drift_type="sudden",
                random_state=42,
            )

            # Should produce valid DataFrames
            assert isinstance(X_ref_df, pd.DataFrame)
            assert isinstance(X_test_df, pd.DataFrame)
            assert len(X_ref_df) + len(X_test_df) == 100

    def test_legacy_drift_type_mapping(self):
        """Test that legacy drift types are properly mapped."""
        drift_types = ["sudden", "gradual", "incremental", "recurring", "mean_shift", "variance_shift"]

        for drift_type in drift_types:
            X_ref_df, X_test_df, metadata = generate_drift(
                generator_name="gaussian",
                n_samples=100,
                n_features=3,
                drift_type=drift_type,
                random_state=42,
            )

            # Should produce valid DataFrames
            assert isinstance(X_ref_df, pd.DataFrame)
            assert isinstance(X_test_df, pd.DataFrame)
            assert metadata["drift_type"] == drift_type

    def test_legacy_drift_ratio_calculation(self):
        """Test that drift_ratio properly calculates affected features."""
        X_ref_df, X_test_df, metadata = generate_drift(
            generator_name="gaussian",
            n_samples=1000,
            n_features=10,
            drift_type="sudden",
            drift_ratio=0.3,  # 30% of features affected
            drift_magnitude=2.0,
            random_state=42,
        )

        # Calculate differences between reference and test means
        ref_means = X_ref_df.mean()
        test_means = X_test_df.mean()
        mean_diffs = np.abs(test_means - ref_means)

        # With drift_ratio=0.3 and n_features=10, 3 features should be affected
        # Sort differences and check that top 3 are significantly larger
        sorted_diffs = sorted(mean_diffs, reverse=True)

        # Top 3 differences should be larger (indicating drift)
        assert sorted_diffs[0] > 1.0  # Strong drift
        assert sorted_diffs[1] > 1.0
        assert sorted_diffs[2] > 1.0

        # Remaining features should have smaller differences
        assert sorted_diffs[3] < 0.5
        assert sorted_diffs[4] < 0.5

    def test_legacy_categorical_features(self):
        """Test legacy function with categorical features."""
        categorical_features = [1, 3]

        X_ref_df, X_test_df, metadata = generate_drift(
            generator_name="mixed",
            n_samples=1000,
            n_features=5,
            drift_type="sudden",
            categorical_features=categorical_features,
            random_state=42,
        )

        # Check that specified features contain integer-like values
        for cat_feat in categorical_features:
            col_name = f"feature_{cat_feat}"
            ref_vals = X_ref_df[col_name].values
            test_vals = X_test_df[col_name].values

            # Should be close to integers (within floating-point precision)
            assert np.allclose(ref_vals, np.round(ref_vals))
            assert np.allclose(test_vals, np.round(test_vals))


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_minimum_samples(self):
        """Test generation with minimum number of samples."""
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=10,  # Very small
            n_features=2,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_position=0.5,
            random_state=42,
        )

        assert X_ref.shape[0] >= 1
        assert X_test.shape[0] >= 1
        assert X_ref.shape[0] + X_test.shape[0] == 10

    def test_single_feature(self):
        """Test generation with single feature."""
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=1000,
            n_features=1,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            random_state=42,
        )

        assert X_ref.shape[1] == 1
        assert X_test.shape[1] == 1

    def test_extreme_drift_positions(self):
        """Test with extreme drift positions."""
        # Test with very early drift position
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=1000,
            n_features=3,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_position=0.1,  # Very early
            random_state=42,
        )

        assert X_ref.shape[0] == 100
        assert X_test.shape[0] == 900

        # Test with very late drift position
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=1000,
            n_features=3,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_position=0.9,  # Very late
            random_state=42,
        )

        assert X_ref.shape[0] == 900
        assert X_test.shape[0] == 100

    def test_zero_noise(self):
        """Test generation with zero noise."""
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=1000,
            n_features=3,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            noise=0.0,  # No noise
            random_state=42,
        )

        # Should still produce valid data
        assert np.isfinite(X_ref).all()
        assert np.isfinite(X_test).all()

    def test_large_drift_magnitude(self):
        """Test with very large drift magnitude."""
        X_ref, X_test, _, _, _ = generate_synthetic_data(
            generator_name="GAUSSIAN",
            n_samples=1000,
            n_features=3,
            drift_pattern="SUDDEN",
            drift_characteristic="MEAN_SHIFT",
            drift_magnitude=100.0,  # Very large
            random_state=42,
        )

        # Should still produce finite values
        assert np.isfinite(X_ref).all()
        assert np.isfinite(X_test).all()

        # Drift should be clearly visible
        mean_diff = np.abs(X_test.mean(axis=0) - X_ref.mean(axis=0))
        assert mean_diff.mean() > 50.0  # Large difference expected


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_scenario(self):
        """Test a complex scenario with multiple drift features."""
        X_ref, X_test, _, _, drift_info = generate_synthetic_data(
            generator_name="MIXED",
            n_samples=2000,
            n_features=8,
            drift_pattern="GRADUAL",
            drift_characteristic="DISTRIBUTION_SHIFT",
            drift_magnitude=2.0,
            drift_position=0.3,
            drift_duration=0.4,
            drift_affected_features=[1, 3, 5],
            noise=0.2,
            categorical_features=[2, 4, 6],
            random_state=123,
        )

        # Verify complex parameters
        assert X_ref.shape[0] == 600  # 30% for reference
        assert X_test.shape[0] == 1400  # 70% for test
        assert X_ref.shape[1] == 8
        assert X_test.shape[1] == 8

        # Check drift info contains all expected fields
        expected_fields = [
            "generator",
            "drift_pattern",
            "drift_characteristic",
            "categorical_features",
            "drift_magnitude",
            "parameters",
        ]
        for field in expected_fields:
            assert field in drift_info

    def test_all_combinations_basic(self):
        """Test basic functionality across all generator/pattern combinations."""
        generators = ["GAUSSIAN", "MIXED"]  # Subset for faster testing
        patterns = ["SUDDEN", "GRADUAL"]
        characteristics = ["MEAN_SHIFT", "VARIANCE_SHIFT"]

        for generator in generators:
            for pattern in patterns:
                for characteristic in characteristics:
                    # Skip unsupported combinations
                    if generator == "MIXED" and characteristic == "VARIANCE_SHIFT":
                        continue

                    params = {
                        "n_samples": 100,  # Small for speed
                        "n_features": 3,
                        "random_state": 42,
                    }

                    if pattern == "GRADUAL":
                        params["drift_duration"] = 0.5

                    if generator == "MIXED":
                        params["categorical_features"] = [1]

                    X_ref, X_test, _, _, drift_info = generate_synthetic_data(
                        generator_name=generator,
                        drift_pattern=pattern,
                        drift_characteristic=characteristic,
                        **params,
                    )

                    # Basic validation
                    assert X_ref.shape[0] > 0
                    assert X_test.shape[0] > 0
                    assert drift_info["generator"] == generator
                    assert drift_info["drift_pattern"] == pattern
                    assert drift_info["drift_characteristic"] == characteristic
