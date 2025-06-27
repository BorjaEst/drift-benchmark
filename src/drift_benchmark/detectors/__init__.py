"""
Drift detection algorithms and interfaces.

This package provides a common interface for drift detection algorithms
and implementations of various drift detectors.
"""

from drift_benchmark.detectors.base import BaseDetector
from drift_benchmark.detectors.registry import (
    clear_registry,
    discover_and_register_detectors,
    get_detector,
    get_detector_by_criteria,
    get_detector_class,
    get_detector_info,
    initialize_detector,
    list_available_aliases,
    list_available_detectors,
    print_registry_status,
    register_detector,
    validate_registry_consistency,
)

__all__ = [
    # Base detector classes
    "BaseDetector",
    # Registry functions
    "register_detector",
    "get_detector",
    "get_detector_by_criteria",
    "get_detector_class",
    "get_detector_info",
    "initialize_detector",
    "list_available_detectors",
    "list_available_aliases",
    "discover_and_register_detectors",
    "clear_registry",
    "validate_registry_consistency",
    "print_registry_status",
]


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    def _detector_registry_example():
        """Demonstrate the detector registry system."""
        print("=== Detector Registry Example ===\n")

        # 1. Register the built-in PeriodicTrigger detector
        print("1. Registering PeriodicTrigger detector...")
        register_detector(BaseDetector)  # This would actually be PeriodicTrigger from base

        # Import and register
        from drift_benchmark.detectors.base import PeriodicTrigger

        register_detector(PeriodicTrigger)

        print(f"   Registered detectors: {list_available_detectors()}")

        # 2. Get detector information
        print("\n2. Getting detector information...")
        info = get_detector_info()
        for name, metadata in info.items():
            print(f"   {name}:")
            if "error" in metadata:
                print(f"     Error: {metadata['error']}")
            else:
                print(f"     Method: {metadata.get('method_name', 'N/A')}")
                print(f"     Description: {metadata.get('description', 'N/A')}")
                print(f"     Drift types: {metadata.get('drift_types', [])}")
                print(f"     Family: {metadata.get('family', 'N/A')}")
                print(f"     Execution mode: {metadata.get('execution_mode', 'N/A')}")

        # 3. Initialize a detector
        print("\n3. Initializing detector...")
        detector = initialize_detector("PeriodicTrigger", interval=3, name="TestDetector")
        print(f"   Created: {detector.name} with interval {detector.interval}")

        # 4. Test the detector
        print("\n4. Testing detector functionality...")
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (100, 3))
        test_data = np.random.normal(0, 1, (20, 3))

        # Fit the detector
        detector.fit(reference_data)
        print(f"   Detector fitted: {detector.is_fitted}")

        # Test detection multiple times
        print("   Detection results:")
        for i in range(5):
            batch = test_data[i * 4 : (i + 1) * 4]  # 4 samples per batch
            if len(batch) > 0:
                result = detector.detect(batch)
                scores = detector.score()
                print(f"     Batch {i+1}: Drift={result}, Cycle={int(scores['cycle_count'])}")

        # 5. Test detector discovery (would work if components directory exists)
        print("\n5. Testing detector discovery...")
        try:
            discovered_count = discover_and_register_detectors()
            print(f"   Discovered {discovered_count} additional detectors")
            if discovered_count > 0:
                print(f"   All detectors: {list_available_detectors()}")
        except Exception as e:
            print(f"   Discovery failed (expected if no components dir): {e}")

        # 6. Search by criteria
        print("\n6. Searching detectors by criteria...")
        concept_detectors = get_detector_by_criteria(drift_type="CONCEPT")
        print(f"   Concept drift detectors: {[d.__name__ for d in concept_detectors]}")

        streaming_detectors = get_detector_by_criteria()  # All detectors
        print(f"   All detectors: {[d.__name__ for d in streaming_detectors]}")

        # 7. Registry validation
        print("\n7. Registry validation...")
        validation_results = validate_registry_consistency()
        print(f"   Valid detectors: {len(validation_results['valid_detectors'])}")
        print(f"   Invalid detectors: {len(validation_results['invalid_detectors'])}")
        if validation_results["validation_errors"]:
            print(f"   Validation errors: {validation_results['validation_errors']}")

        # 8. Print full registry status
        print("\n8. Full registry status:")
        print_registry_status()

        print("\n=== Registry example completed! ===")

    def _detector_usage_example():
        """Demonstrate basic detector usage patterns."""
        print("\n=== Detector Usage Patterns Example ===\n")

        # 1. Direct instantiation
        print("1. Direct detector instantiation...")
        detector = BaseDetector.__subclasses__()[0](interval=2, name="DirectDetector")  # PeriodicTrigger
        print(f"   Created: {detector.name}")

        # 2. Using registry
        print("\n2. Using registry to create detector...")
        register_detector(detector.__class__)
        registry_detector = get_detector(detector.__class__.__name__)
        instance = registry_detector(interval=5, name="RegistryDetector")
        print(f"   Created via registry: {instance.name}")

        # 3. Performance measurement
        print("\n3. Performance measurement...")
        np.random.seed(123)
        ref_data = np.random.normal(0, 1, (200, 2))
        test_data = np.random.normal(0.5, 1, (100, 2))

        # Measure fit time
        result = detector.timed_fit(ref_data)
        print(f"   Fit time: {detector.fit_time:.6f} seconds")

        # Measure detection time
        drift_detected = detector.timed_detect(test_data)
        print(f"   Detect time: {detector.detect_time:.6f} seconds")
        print(f"   Drift detected: {drift_detected}")

        # Get performance metrics
        metrics = detector.get_performance_metrics()
        print(f"   Performance metrics: {metrics}")

        # 4. Configuration and metadata
        print("\n4. Detector configuration and metadata...")
        config = detector.get_config()
        print("   Configuration:")
        for key, value in config.items():
            if isinstance(value, dict) and len(value) > 3:
                print(f"     {key}: (dict with {len(value)} keys)")
            else:
                print(f"     {key}: {value}")

        # 5. State management
        print("\n5. Detector state management...")
        print(f"   Is fitted: {detector.is_fitted}")

        # Reset detector
        detector.reset()
        print(f"   After reset - Is fitted: {detector.is_fitted}")

        # 6. Error handling
        print("\n6. Error handling...")
        try:
            unfitted_detector = detector.__class__(interval=3)
            unfitted_detector.detect(test_data)
        except RuntimeError as e:
            print(f"   Expected error for unfitted detector: {e}")

        print("\n=== Usage patterns example completed! ===")

    def _pandas_integration_example():
        """Demonstrate pandas DataFrame integration."""
        print("\n=== Pandas Integration Example ===\n")

        # Create sample DataFrames
        np.random.seed(456)

        print("1. Creating pandas DataFrames...")
        reference_df = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 150),
                "feature_2": np.random.normal(1, 0.5, 150),
                "feature_3": np.random.exponential(1, 150),
            }
        )

        test_df = pd.DataFrame(
            {
                "feature_1": np.random.normal(0.2, 1, 75),
                "feature_2": np.random.normal(1.1, 0.5, 75),
                "feature_3": np.random.exponential(1.1, 75),
            }
        )

        print(f"   Reference DataFrame shape: {reference_df.shape}")
        print(f"   Test DataFrame shape: {test_df.shape}")
        print(f"   Columns: {list(reference_df.columns)}")

        # 2. Initialize and use detector with DataFrames
        print("\n2. Using detector with DataFrames...")
        detector = initialize_detector("PeriodicTrigger", interval=4, name="PandasDetector")

        # Fit with DataFrame
        detector.fit(reference_df)
        print(f"   Fitted with DataFrame: {detector.is_fitted}")

        # Test detection in batches
        print("\n3. Batch detection with DataFrames...")
        batch_size = 15
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i : i + batch_size]
            if len(batch) > 0:
                result = detector.detect(batch)
                scores = detector.score()
                print(
                    f"   Batch {i//batch_size + 1} (rows {i}-{i+len(batch)-1}): "
                    f"Drift={result}, Cycle={int(scores['cycle_count'])}"
                )

        # 4. Mixed data types (if detector supports)
        print("\n4. Mixed data types example...")
        mixed_df = pd.DataFrame(
            {
                "numeric_1": np.random.normal(0, 1, 50),
                "numeric_2": np.random.exponential(1, 50),
                "categorical": np.random.choice(["A", "B", "C"], 50),
                "boolean": np.random.choice([True, False], 50),
            }
        )

        print(f"   Mixed DataFrame dtypes:")
        print(f"     {dict(mixed_df.dtypes)}")

        # Note: PeriodicTrigger doesn't actually process the data content,
        # so it works with any DataFrame structure
        detector.reset()
        detector.fit(mixed_df)
        mixed_result = detector.detect(mixed_df.iloc[:10])
        print(f"   Detection with mixed types: {mixed_result}")

        print("\n=== Pandas integration example completed! ===")

    # Run all examples
    _detector_registry_example()
    _detector_usage_example()
    _pandas_integration_example()
