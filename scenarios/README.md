# Scenarios Directory

> Comprehensive collection of drift detection scenarios for benchmarking

## 🎯 Overview

This directory contains scenario definitions for the drift-benchmark framework. Each scenario represents a complete evaluation case with data sources, filtering conditions, ground truth specifications, and statistical validation requirements. Scenarios are organized by data source type and purpose to facilitate systematic benchmarking of drift detection methods.

## 📁 Directory Structure

```text
scenarios/
├── README.md                     # This file
├── baselines/                    # No-drift baseline scenarios
│   ├── no_drift_synthetic.toml   # Synthetic baseline for false positive testing
│   └── no_drift_uci.toml         # UCI baseline for false positive testing
├── synthetic/                    # Synthetic dataset scenarios
│   ├── concept_drift_gradual.toml # Gradual concept drift with synthetic data
│   ├── covariate_drift_strong.toml # Strong covariate drift with synthetic data
│   └── covariate_drift_weak.toml  # Weak covariate drift with synthetic data
├── uci/                          # UCI repository scenarios
│   ├── adult_income_age.toml      # Adult dataset with age-based drift
│   ├── iris_petal_length.toml     # Iris dataset with petal length drift
│   └── wine_quality_alcohol.toml  # Wine quality with alcohol-based drift
└── custom/                       # Custom dataset scenarios
    ├── customer_churn.toml        # Customer churn with concept drift
    └── sensor_data_drift.toml     # Sensor data with temporal drift
```

## 📊 Scenario Categories

### 🎲 Synthetic Scenarios (`synthetic/`)

Scenarios using scikit-learn synthetic datasets with controlled artificial drift. These provide ground truth with known mathematical properties.

#### Available Synthetic Scenarios

**`covariate_drift_strong.toml`**

- **Drift Type**: Covariate drift
- **Intensity**: Strong (Effect size: 0.8)
- **Data Source**: Synthetic classification dataset
- **Features**: 4 numerical features, binary classification
- **Drift Method**: Noise amplification (2.0x) + feature scaling (0.6x)
- **Use Case**: Testing detector sensitivity to obvious distributional changes

**`covariate_drift_weak.toml`**

- **Drift Type**: Covariate drift  
- **Intensity**: Weak (Effect size: 0.3)
- **Data Source**: Synthetic classification dataset
- **Features**: 4 numerical features, binary classification
- **Drift Method**: Subtle noise increase (1.2x) + mild feature scaling (0.9x)
- **Use Case**: Testing detector sensitivity to subtle distributional changes

**`concept_drift_gradual.toml`**

- **Drift Type**: Concept drift
- **Intensity**: Moderate (Effect size: 0.5)
- **Data Source**: Synthetic classification dataset
- **Features**: 6 numerical features, binary classification
- **Drift Method**: Target shift (0.3) + class imbalance (0.3)
- **Use Case**: Testing detection of changing relationships between features and target

### 🏛️ UCI Repository Scenarios (`uci/`)

Scenarios using real-world datasets from the UCI Machine Learning Repository with authentic drift patterns based on natural data variation.

#### Available UCI Scenarios

**`wine_quality_alcohol.toml`**

- **Drift Type**: Covariate drift
- **Dataset**: UCI Wine Quality Red (1,599 samples, 11 features)
- **Drift Feature**: Alcohol content (≤10.5% vs >12.0%)
- **Intensity**: Moderate (Effect size: 0.58)
- **Domain**: Food & Beverage Chemistry
- **Use Case**: Real-world covariate drift in wine chemistry measurements

**`iris_petal_length.toml`**

- **Drift Type**: Covariate drift
- **Dataset**: UCI Iris (150 samples, 4 features)
- **Drift Feature**: Petal length-based filtering
- **Intensity**: Strong (natural species separation)
- **Domain**: Botanical measurements
- **Use Case**: Classic ML dataset with natural distributional differences

**`adult_income_age.toml`**

- **Drift Type**: Covariate drift
- **Dataset**: UCI Adult Income (48,842 samples, 14 features)
- **Drift Feature**: Age-based demographic shifts
- **Intensity**: Moderate
- **Domain**: Census and demographic data
- **Use Case**: Socioeconomic drift patterns across age groups

### 📄 Custom File Scenarios (`custom/`)

Scenarios using custom CSV datasets with domain-specific drift patterns.

#### Available Custom Scenarios

**`customer_churn.toml`**

- **Drift Type**: Concept drift
- **Data Source**: Customer churn CSV file
- **Drift Pattern**: High-value vs low-value customer churn behaviors
- **Features**: 17 features including tenure, charges, services
- **Intensity**: Moderate (Effect size: 0.55)
- **Use Case**: Business analytics drift in customer behavior patterns

**`sensor_data_drift.toml`**

- **Drift Type**: Temporal drift
- **Data Source**: Sensor measurements CSV file
- **Drift Pattern**: Environmental condition changes over time
- **Features**: Multi-sensor time series data
- **Use Case**: Industrial IoT drift detection scenarios

### 📈 Baseline Scenarios (`baselines/`)

Control scenarios with no drift for statistical validation and false positive rate testing.

#### Available Baseline Scenarios

**`no_drift_synthetic.toml`**

- **Purpose**: False positive rate testing for synthetic scenarios
- **Expected Detection**: False
- **Data Source**: Synthetic classification (same as drift scenarios)
- **Configuration**: Different sample ranges, no drift modifications
- **Use Case**: Validate detector specificity on synthetic data

**`no_drift_uci.toml`**

- **Purpose**: False positive rate testing for real-world scenarios
- **Expected Detection**: False
- **Data Source**: UCI datasets with similar characteristics
- **Configuration**: Non-overlapping feature filters with no expected drift
- **Use Case**: Validate detector specificity on real-world data

## 🔧 Scenario Configuration

Each scenario follows a standardized TOML format with the following sections:

### Required Sections

- **Metadata**: `description`, `source_type`, `source_name`, `target_column`, `drift_types`
- **Ground Truth**: `[ground_truth]` - Expected drift characteristics and detection outcomes
- **Statistical Validation**: `[statistical_validation]` - Power analysis requirements
- **Data Filters**: `[ref_filter]` and `[test_filter]` - Data extraction specifications

### Optional Sections

- **Enhanced Metadata**: `[enhanced_metadata]` - Comprehensive dataset documentation
- **UCI Metadata**: `[uci_metadata]` - Scientific attribution for UCI datasets (UCI scenarios only)
- **Experimental Design**: `[experimental_design]` - Advanced experimental controls

### Example Structure

```toml
# Basic metadata
description = "Human-readable scenario description"
source_type = "synthetic"  # "synthetic", "file", "uci"  
source_name = "classification"  # Dataset identifier
target_column = "target"
drift_types = ["covariate"]

# Ground truth specification
[ground_truth]
drift_periods = [[500, 1000]]
expected_effect_size = 0.8
expected_detection = true

# Statistical validation
[statistical_validation]  
minimum_power = 0.80
significance_level = 0.05

# Data extraction filters
[ref_filter]
sample_range = [0, 500]

[test_filter]  
sample_range = [500, 1000]
```

## 📊 Drift Types and Intensities

### Drift Types

- **Covariate Drift**: Changes in input feature distributions (P(X) changes)
- **Concept Drift**: Changes in feature-target relationships (P(Y|X) changes)  
- **Prior Drift**: Changes in target distribution (P(Y) changes)
- **None**: No drift present (baseline scenarios)

### Intensity Levels

| Intensity | Effect Size | KL Divergence | Description |
|-----------|-------------|---------------|-------------|
| **Weak**  | 0.2-0.4     | 0.1-0.3       | Subtle changes requiring sensitive detectors |
| **Moderate** | 0.4-0.7  | 0.3-0.6       | Clear changes detectable by most methods |
| **Strong** | 0.7+       | 0.6+          | Obvious changes easily detected |

## 🧪 Usage Examples

### Running Individual Scenarios

```python
from drift_benchmark import BenchmarkRunner

# Run a specific scenario
config = {
    "scenarios": [{"id": "covariate_drift_strong"}],
    "detectors": [
        {
            "method_id": "kolmogorov_smirnov",
            "variant_id": "batch", 
            "library_id": "scipy"
        }
    ]
}

runner = BenchmarkRunner.from_config(config)
results = runner.run()
```

### Comparative Studies

```python
# Compare synthetic vs real-world drift
config = {
    "scenarios": [
        {"id": "covariate_drift_strong"},      # Synthetic
        {"id": "wine_quality_alcohol"}         # Real-world UCI
    ],
    "detectors": [
        {
            "method_id": "kolmogorov_smirnov",
            "variant_id": "batch",
            "library_id": "evidently"
        }
    ]
}
```

### Statistical Validation

```python
# Include baseline for false positive rate analysis
config = {
    "scenarios": [
        {"id": "covariate_drift_strong"},  # Drift scenario
        {"id": "no_drift_synthetic"}       # Baseline scenario
    ],
    "detectors": [...]
}
```

## 📈 Adding New Scenarios

To add new scenarios, create TOML files following the standardized format:

1. **Choose appropriate directory** based on data source type
2. **Follow naming convention**: `{source}_{drift_type}_{description}.toml`
3. **Include all required sections** with proper validation
4. **Document ground truth** with quantitative metrics
5. **Add to this README** with scenario description

### Scenario Design Guidelines

- **Scientific Rigor**: Include statistical validation requirements
- **Ground Truth**: Provide quantitative effect size and KL divergence estimates
- **Reproducibility**: Use fixed random seeds for synthetic scenarios
- **Documentation**: Include comprehensive metadata and feature descriptions
- **Authenticity**: Preserve real-world data characteristics for UCI/CSV scenarios

## 🔗 Related Documentation

- **[Main README](../README.md)**: Framework overview and quick start
- **[Scenarios Documentation](../docs/scenarios.md)**: Detailed scenario specification guide  
- **[Configurations Documentation](../docs/configurations.md)**: Benchmark configuration guide
- **[Requirements](../REQUIREMENTS.md)**: Complete framework requirements and specifications
