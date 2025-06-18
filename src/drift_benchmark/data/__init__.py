from drift_benchmark.data.datasets import load_dataset
from drift_benchmark.data.drift_generators import generate_drift

__all__ = ["load_dataset", "generate_drift"]


if __name__ == "__main__":
    # Example usage of load_dataset and generate_drift functions
    # This is just for demonstration; in practice, you would import this module and use the functions as needed.
    import pandas as pd

    # Example 1: Load a built-in dataset
    X_ref, X_test, metadata = load_dataset(
        "iris",
        test_size=0.4,
        random_state=42,
        preprocess={
            "scaling": True,
            "scaling_method": "standard",
        },
    )
    print(f"Loaded {metadata['name']} dataset with {metadata['n_features']} features")

    # Example 2: Load from a custom CSV file with time-based split
    X_ref, X_test, metadata = load_dataset(
        "datasets/example.csv",
        time_column="date",
        target_column="RainTomorrow",
        train_end_time="2020-01-01",
        preprocess={
            "handle_missing": True,
            "missing_strategy": "mean",
            "encode_categorical": True,
            "encoding_method": "onehot",
        },
    )
    print(f"Loaded custom dataset with time split at {metadata['time_column']}")

    # Example 3: Generate synthetic data with drift
    X_ref, X_test, metadata = generate_drift(
        generator_name="gaussian",
        n_samples=1000,
        n_features=10,
        drift_type="sudden",
        drift_magnitude=1.5,
        drift_position=0.7,
        random_state=123,
    )
    print(f"Generated {metadata['name']} with {metadata['drift_type']} drift")

    # Example 4: Generate mixed data with categorical features
    categorical_features = [0, 2, 5]
    X_ref, X_test, metadata = generate_drift(
        generator_name="mixed",
        n_samples=2000,
        n_features=8,
        drift_type="gradual",
        drift_position=0.5,
        categorical_features=categorical_features,
        n_categories=4,
        random_state=456,
    )
    print(f"Generated {metadata['name']} with categorical features at positions {metadata['categorical_features']}")

    # Example 5: Generate time series data with seasonality
    X_ref, X_test, metadata = generate_drift(
        generator_name="time_series",
        n_samples=365,
        n_features=5,
        drift_type="mean_shift",
        drift_position=0.6,
        seasonality=True,
        random_state=789,
    )
    print(f"Generated {metadata['name']} with seasonality={metadata['seasonality']}")
