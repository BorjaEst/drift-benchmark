"""
Value discovery utilities for drift-benchmark - REQ-DAT-018 to REQ-DAT-020

Provides utilities to discover feature thresholds and analyze datasets for filtering.
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from ..exceptions import DataLoadingError

# Real datasets supported by sklearn
REAL_DATASETS = {"load_breast_cancer", "load_diabetes", "load_iris", "load_wine"}
SYNTHETIC_DATASETS = {"make_classification", "make_regression", "make_blobs"}


def discover_feature_thresholds(dataset_name: str, feature_name: str) -> Dict[str, float]:
    """
    Returns statistical thresholds (min, max, median, q25, q75) for feature-based filtering.

    REQ-DAT-018: Threshold Discovery Interface
    """
    # Load the dataset to analyze
    data = _load_dataset_for_analysis(dataset_name)

    # Validate feature exists
    if feature_name not in data.columns:
        available_features = list(data.columns)
        raise DataLoadingError(
            f"Feature '{feature_name}' not found in dataset '{dataset_name}'. " f"Available features: {available_features}"
        )

    # Get feature values
    feature_values = data[feature_name]

    # Calculate statistical thresholds
    thresholds = {
        "min": float(feature_values.min()),
        "max": float(feature_values.max()),
        "median": float(feature_values.median()),
        "q25": float(feature_values.quantile(0.25)),
        "q75": float(feature_values.quantile(0.75)),
    }

    return thresholds


def analyze_feature_distribution(dataset_name: str, feature_name: str) -> Dict[str, Any]:
    """
    Analyze feature distributions in real datasets to suggest meaningful filtering thresholds.

    REQ-DAT-019: Dataset Feature Analysis
    """
    data = _load_dataset_for_analysis(dataset_name)

    if feature_name not in data.columns:
        available_features = list(data.columns)
        raise DataLoadingError(
            f"Feature '{feature_name}' not found in dataset '{dataset_name}'. " f"Available features: {available_features}"
        )

    feature_values = data[feature_name]

    # Calculate statistical measures
    mean_val = float(feature_values.mean())
    std_val = float(feature_values.std())
    skewness = float(feature_values.skew())
    kurtosis = float(feature_values.kurtosis())

    # Determine distribution shape
    if abs(skewness) < 0.5:
        distribution_shape = "approximately normal"
    elif skewness > 0.5:
        distribution_shape = "right-skewed"
    else:
        distribution_shape = "left-skewed"

    return {"mean": mean_val, "std": std_val, "skewness": skewness, "kurtosis": kurtosis, "distribution_shape": distribution_shape}


def suggest_filtering_thresholds(dataset_name: str, feature_name: str) -> Dict[str, Any]:
    """
    Suggest meaningful thresholds for creating data splits.

    REQ-DAT-019: Dataset Feature Analysis - threshold suggestions
    """
    thresholds = discover_feature_thresholds(dataset_name, feature_name)

    # Suggest meaningful split points (avoiding extremes)
    q25, q75 = thresholds["q25"], thresholds["q75"]
    median = thresholds["median"]

    low_threshold = q25
    high_threshold = q75

    recommended_splits = [
        {
            "name": "lower_quartile_vs_upper_quartile",
            "ref_filter": {"feature_filters": [{"column": feature_name, "condition": "<=", "value": q25}]},
            "test_filter": {"feature_filters": [{"column": feature_name, "condition": ">=", "value": q75}]},
            "description": "Split using quartiles for clear separation",
        },
        {
            "name": "below_median_vs_above_median",
            "ref_filter": {"feature_filters": [{"column": feature_name, "condition": "<=", "value": median}]},
            "test_filter": {"feature_filters": [{"column": feature_name, "condition": ">", "value": median}]},
            "description": "Split using median for balanced groups",
        },
    ]

    reasoning = (
        f"Suggested thresholds for '{feature_name}' avoid extremes and use statistical measures "
        f"(Q25={q25:.2f}, median={median:.2f}, Q75={q75:.2f}) to create meaningful population splits."
    )

    return {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "recommended_splits": recommended_splits,
        "reasoning": reasoning,
    }


def identify_feature_clusters(dataset_name: str, feature_name: str) -> Dict[str, Any]:
    """
    Identify natural clusters in feature distributions.

    REQ-DAT-019: Dataset Feature Analysis - cluster identification
    """
    data = _load_dataset_for_analysis(dataset_name)

    if feature_name not in data.columns:
        raise DataLoadingError(f"Feature '{feature_name}' not found in dataset '{dataset_name}'")

    feature_values = data[feature_name].values.reshape(-1, 1)

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Try 2-4 clusters and find best separation
        best_k = 2
        best_score = -1

        for k in range(2, 5):
            if len(feature_values) >= k:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(feature_values)
                if len(set(labels)) > 1:  # Ensure we have multiple clusters
                    score = silhouette_score(feature_values, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

        # Get cluster boundaries using best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans.fit(feature_values)
        centers = sorted(kmeans.cluster_centers_.flatten())

        # Calculate boundaries between cluster centers
        boundaries = []
        for i in range(len(centers) - 1):
            boundary = (centers[i] + centers[i + 1]) / 2
            boundaries.append(float(boundary))

        # Assess separation quality
        if best_score > 0.5:
            separation_quality = "good separation"
        elif best_score > 0.3:
            separation_quality = "moderate separation"
        else:
            separation_quality = "poor separation"

        # Suggest filtering strategies
        strategies = []
        if len(boundaries) >= 1:
            mid_boundary = boundaries[len(boundaries) // 2]
            strategies.append(
                {
                    "name": "cluster_based_split",
                    "boundary": mid_boundary,
                    "description": f"Use cluster boundary at {mid_boundary:.2f} to separate natural groups",
                }
            )

    except ImportError:
        boundaries = []
        separation_quality = "analysis unavailable (sklearn required)"
        strategies = [{"name": "quartile_fallback", "description": "Use quartile-based splitting instead"}]

    return {"cluster_boundaries": boundaries, "separation_quality": separation_quality, "filtering_strategies": strategies}


def analyze_feature_correlations(dataset_name: str, feature_names: List[str]) -> Dict[str, Any]:
    """
    Analyze correlations between features for multi-feature filtering.

    REQ-DAT-019: Dataset Feature Analysis - correlation analysis
    """
    data = _load_dataset_for_analysis(dataset_name)

    # Validate all features exist
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        raise DataLoadingError(f"Features not found in dataset '{dataset_name}': {missing_features}")

    if len(feature_names) < 2:
        raise DataLoadingError("At least 2 features required for correlation analysis")

    # Calculate correlation between first two features
    feature1, feature2 = feature_names[0], feature_names[1]
    correlation_coefficient = float(data[feature1].corr(data[feature2]))

    # Suggest combined filtering strategies
    thresholds1 = discover_feature_thresholds(dataset_name, feature1)
    thresholds2 = discover_feature_thresholds(dataset_name, feature2)

    combined_filtering = {
        "high_high_split": {
            "ref_filter": {
                "feature_filters": [
                    {"column": feature1, "condition": "<=", "value": thresholds1["q25"]},
                    {"column": feature2, "condition": "<=", "value": thresholds2["q25"]},
                ]
            },
            "test_filter": {
                "feature_filters": [
                    {"column": feature1, "condition": ">=", "value": thresholds1["q75"]},
                    {"column": feature2, "condition": ">=", "value": thresholds2["q75"]},
                ]
            },
            "description": "Compare low values vs high values for both features",
        }
    }

    return {"correlation_coefficient": correlation_coefficient, "combined_filtering": combined_filtering}


def get_feature_description(dataset_name: str, feature_name: str) -> Dict[str, Any]:
    """
    Get descriptive information about features in real datasets.

    REQ-DAT-020: Feature Documentation
    """
    # Load data to validate feature exists
    data = _load_dataset_for_analysis(dataset_name)

    if feature_name not in data.columns:
        raise DataLoadingError(f"Feature '{feature_name}' not found in dataset '{dataset_name}'")

    # Get basic information
    feature_values = data[feature_name]

    # Determine data type
    if pd.api.types.is_numeric_dtype(feature_values):
        data_type = "numeric"
    else:
        data_type = "categorical"

    # Create descriptions based on known datasets
    descriptions = _get_dataset_feature_descriptions()

    description_key = f"{dataset_name}.{feature_name}"
    description = descriptions.get(description_key, f"Feature from {dataset_name} dataset")

    # Try to infer unit from feature name
    unit = "unknown"
    if "(cm)" in feature_name:
        unit = "centimeters"
    elif "(mm)" in feature_name:
        unit = "millimeters"
    elif "age" in feature_name.lower():
        unit = "years"
    elif "radius" in feature_name.lower():
        unit = "distance measure"

    return {"feature_name": feature_name, "data_type": data_type, "description": description, "unit": unit}


def explain_filtering_implications(dataset_name: str, feature_name: str, condition: str, value: float) -> Dict[str, Any]:
    """
    Explain implications of filtering for dataset authenticity.

    REQ-DAT-020: Feature Documentation - filtering implications
    """
    data = _load_dataset_for_analysis(dataset_name)

    if feature_name not in data.columns:
        raise DataLoadingError(f"Feature '{feature_name}' not found in dataset '{dataset_name}'")

    feature_values = data[feature_name]

    # Calculate what percentage of data would be selected/excluded
    if condition == "<=":
        selected = feature_values <= value
    elif condition == ">=":
        selected = feature_values >= value
    elif condition == ">":
        selected = feature_values > value
    elif condition == "<":
        selected = feature_values < value
    elif condition == "==":
        selected = abs(feature_values - value) < 0.01
    elif condition == "!=":
        selected = abs(feature_values - value) >= 0.01
    else:
        selected = pd.Series([False] * len(feature_values))

    selected_count = selected.sum()
    excluded_count = len(feature_values) - selected_count
    selected_pct = (selected_count / len(feature_values)) * 100
    excluded_pct = (excluded_count / len(feature_values)) * 100

    # Create explanations
    selected_population = f"{selected_count} samples ({selected_pct:.1f}%) with {feature_name} {condition} {value}"
    excluded_population = f"{excluded_count} samples ({excluded_pct:.1f}%) with {feature_name} not {condition} {value}"

    # Get biological/domain meaning if available
    meanings = _get_filtering_meanings()
    meaning_key = f"{dataset_name}.{feature_name}.{condition}.{value}"
    biological_meaning = meanings.get(meaning_key, f"Filtering by {feature_name} creates subpopulations with different characteristics")

    # Assess authenticity impact
    if dataset_name.startswith("load_"):
        authenticity_impact = "Preserves data authenticity by using natural variation already present in real data"
    else:
        authenticity_impact = "Creates artificial distinction in synthetic data"

    # Provide recommendations
    recommendations = [
        "Verify that the filtering creates meaningful population differences",
        "Check that both filtered groups have sufficient sample sizes",
        "Consider the biological or domain significance of the threshold value",
    ]

    if selected_pct < 10 or selected_pct > 90:
        recommendations.append("Consider adjusting threshold - current filter creates very imbalanced groups")

    return {
        "selected_population": selected_population,
        "excluded_population": excluded_population,
        "biological_meaning": biological_meaning,
        "authenticity_impact": authenticity_impact,
        "recommendations": recommendations,
    }


def get_dataset_documentation(dataset_name: str) -> Dict[str, Any]:
    """
    Get documentation of overall dataset characteristics.

    REQ-DAT-020: Feature Documentation - dataset documentation
    """
    data = _load_dataset_for_analysis(dataset_name)

    # Get basic dataset information
    num_samples, num_features = data.shape
    features = list(data.columns)
    if "target" in features:
        features.remove("target")  # Don't include target in feature list

    # Dataset descriptions
    descriptions = {
        "load_iris": "Iris flower dataset with measurements of sepal and petal dimensions",
        "load_breast_cancer": "Breast cancer Wisconsin dataset with tumor characteristics",
        "load_wine": "Wine recognition dataset with chemical analysis results",
        "load_diabetes": "Diabetes dataset with physiological measurements",
        "make_classification": "Synthetic classification dataset",
        "make_regression": "Synthetic regression dataset",
        "make_blobs": "Synthetic clustering dataset",
    }

    description = descriptions.get(dataset_name, f"Dataset: {dataset_name}")

    # Source information
    if dataset_name.startswith("load_"):
        source = "Real-world data from scikit-learn datasets"
    else:
        source = "Synthetic data generated by scikit-learn"

    # Filtering guidance
    if dataset_name.startswith("load_"):
        filtering_guidance = {
            "allowed_operations": ["sample_range", "feature_filters"],
            "forbidden_operations": ["noise_factor", "feature_scaling", "n_samples"],
            "recommendation": "Use feature-based filtering to preserve data authenticity",
        }
    else:
        filtering_guidance = {
            "allowed_operations": ["sample_range", "feature_filters", "noise_factor", "feature_scaling", "n_samples"],
            "forbidden_operations": [],
            "recommendation": "Modification parameters can be used to introduce artificial drift",
        }

    return {
        "dataset_name": dataset_name,
        "source": source,
        "description": description,
        "num_samples": num_samples,
        "num_features": num_features,
        "features": features,
        "filtering_guidance": filtering_guidance,
    }


def get_filtering_examples(dataset_name: str) -> Dict[str, Any]:
    """
    Provide examples of meaningful filtering scenarios.

    REQ-DAT-020: Feature Documentation - filtering examples
    """
    data = _load_dataset_for_analysis(dataset_name)

    # Get feature information for examples
    features = [col for col in data.columns if col != "target"]

    examples = []

    if dataset_name == "load_breast_cancer":
        examples = [
            {
                "name": "Small vs Large Tumors",
                "description": "Compare small tumors (low risk) vs large tumors (high risk)",
                "filters": {
                    "ref_filter": {"feature_filters": [{"column": "mean radius", "condition": "<=", "value": 14.0}]},
                    "test_filter": {"feature_filters": [{"column": "mean radius", "condition": ">", "value": 14.0}]},
                },
                "expected_outcome": "Drift detection due to size-based population differences",
                "use_case": "Study how tumor size affects detection algorithm performance",
            },
            {
                "name": "Smooth vs Rough Texture",
                "description": "Compare smooth texture tumors vs rough texture tumors",
                "filters": {
                    "ref_filter": {"feature_filters": [{"column": "mean texture", "condition": "<=", "value": 20.0}]},
                    "test_filter": {"feature_filters": [{"column": "mean texture", "condition": ">", "value": 20.0}]},
                },
                "expected_outcome": "Drift detection due to texture-based population differences",
                "use_case": "Evaluate detection sensitivity to tumor texture characteristics",
            },
        ]
    elif dataset_name == "load_iris":
        examples = [
            {
                "name": "Small vs Large Flowers",
                "description": "Compare small flowers vs large flowers",
                "filters": {
                    "ref_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": "<=", "value": 5.5}]},
                    "test_filter": {"feature_filters": [{"column": "sepal length (cm)", "condition": ">", "value": 5.5}]},
                },
                "expected_outcome": "Drift detection due to size-based population differences",
                "use_case": "Study detection performance across flower sizes",
            }
        ]
    else:
        # Generic example
        if features:
            feature = features[0]
            thresholds = discover_feature_thresholds(dataset_name, feature)
            examples = [
                {
                    "name": f"Low vs High {feature}",
                    "description": f"Compare low {feature} values vs high {feature} values",
                    "filters": {
                        "ref_filter": {"feature_filters": [{"column": feature, "condition": "<=", "value": thresholds["median"]}]},
                        "test_filter": {"feature_filters": [{"column": feature, "condition": ">", "value": thresholds["median"]}]},
                    },
                    "expected_outcome": "Drift detection due to feature-based population differences",
                    "use_case": f"Evaluate detection sensitivity to {feature} variations",
                }
            ]

    return {"example_scenarios": examples}


def validate_filter_reasonableness(dataset_name: str, feature_filters: List[Dict]) -> Dict[str, Any]:
    """
    Validate reasonableness of filter configurations.

    REQ-DAT-020: Feature Documentation - filter validation
    """
    data = _load_dataset_for_analysis(dataset_name)

    warnings = []
    suggestions = []
    is_reasonable = True

    for filter_config in feature_filters:
        column = filter_config["column"]
        condition = filter_config["condition"]
        value = filter_config["value"]

        if column not in data.columns:
            warnings.append(f"Column '{column}' not found in dataset")
            is_reasonable = False
            continue

        feature_values = data[column]
        min_val, max_val = feature_values.min(), feature_values.max()

        # Check if value is within reasonable range
        if condition in ["<=", "<"] and value < min_val:
            warnings.append(f"Filter value {value} is below minimum value {min_val} for {column}")
            suggestions.append(f"Consider using a value between {min_val} and {max_val} for {column}")
            is_reasonable = False
        elif condition in [">=", ">"] and value > max_val:
            warnings.append(f"Filter value {value} is above maximum value {max_val} for {column}")
            suggestions.append(f"Consider using a value between {min_val} and {max_val} for {column}")
            is_reasonable = False
        elif condition == "==" and value not in feature_values.values:
            warnings.append(f"Exact value {value} not found in {column}")
            suggestions.append(f"Consider using a range-based condition instead of exact equality")

    reasoning = "Filter validation completed"
    if is_reasonable:
        reasoning = "All filter conditions appear reasonable and will produce meaningful results"
    else:
        reasoning = "Some filter conditions may produce empty results or are outside reasonable ranges"

    return {"is_reasonable": is_reasonable, "reasoning": reasoning, "warnings": warnings, "suggestions": suggestions}


def get_filtering_recommendations(dataset_name: str, drift_type: str = "covariate") -> Dict[str, Any]:
    """
    Get filtering recommendations for creating authentic drift scenarios.

    REQ-DAT-020: Feature Documentation - filtering recommendations
    """
    data = _load_dataset_for_analysis(dataset_name)
    features = [col for col in data.columns if col != "target"]

    # Recommend features suitable for filtering
    recommended_features = []
    if dataset_name == "load_breast_cancer":
        recommended_features = ["mean radius", "mean texture", "mean smoothness"]
    elif dataset_name == "load_iris":
        recommended_features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]
    elif dataset_name == "load_wine":
        recommended_features = ["alcohol", "total_phenols", "flavanoids"]
    elif dataset_name == "load_diabetes":
        recommended_features = ["age", "bmi", "bp"]
    else:
        recommended_features = features[:3]  # First 3 features as default

    # Create filter configurations
    filter_configurations = []

    for feature in recommended_features[:2]:  # Limit to first 2 for brevity
        if feature in data.columns:
            thresholds = discover_feature_thresholds(dataset_name, feature)

            config = {
                "ref_filter": {"feature_filters": [{"column": feature, "condition": "<=", "value": thresholds["q25"]}]},
                "test_filter": {"feature_filters": [{"column": feature, "condition": ">=", "value": thresholds["q75"]}]},
                "rationale": f"Use quartile-based split on {feature} to create distinct populations",
                "expected_drift_characteristics": f"Covariate drift based on {feature} distribution differences",
            }
            filter_configurations.append(config)

    return {"recommended_features": recommended_features, "filter_configurations": filter_configurations}


def _load_dataset_for_analysis(dataset_name: str) -> pd.DataFrame:
    """Load dataset for analysis purposes."""
    from .scenario_loader import _load_sklearn_data

    if dataset_name not in REAL_DATASETS and dataset_name not in SYNTHETIC_DATASETS:
        raise DataLoadingError(f"Unknown dataset: {dataset_name}. Supported datasets: {REAL_DATASETS | SYNTHETIC_DATASETS}")

    return _load_sklearn_data(dataset_name)


def _get_dataset_feature_descriptions() -> Dict[str, str]:
    """Get feature descriptions for known datasets."""
    return {
        "load_iris.sepal length (cm)": "Length of the sepal in centimeters",
        "load_iris.sepal width (cm)": "Width of the sepal in centimeters",
        "load_iris.petal length (cm)": "Length of the petal in centimeters",
        "load_iris.petal width (cm)": "Width of the petal in centimeters",
        "load_breast_cancer.mean radius": "Mean radius of tumor cell nuclei",
        "load_breast_cancer.mean texture": "Standard deviation of gray-scale values",
        "load_breast_cancer.mean smoothness": "Local variation in radius lengths",
        "load_wine.alcohol": "Alcohol content of the wine",
        "load_wine.total_phenols": "Total phenolic compounds in the wine",
        "load_diabetes.age": "Age of the patient",
        "load_diabetes.bmi": "Body mass index of the patient",
    }


def _get_filtering_meanings() -> Dict[str, str]:
    """Get biological/domain meanings for filtering operations."""
    return {
        "load_breast_cancer.mean radius.>.14.0": "Selects larger tumors which may indicate higher malignancy risk",
        "load_breast_cancer.mean texture.>.20.0": "Selects tumors with rougher texture indicating higher cellular irregularity",
        "load_iris.sepal length (cm).>.5.5": "Selects flowers with longer sepals, potentially different species",
        "load_wine.alcohol.>.12.0": "Selects wines with higher alcohol content",
        "load_diabetes.age.>.0.0": "Includes patients of all ages (age is normalized in this dataset)",
    }
