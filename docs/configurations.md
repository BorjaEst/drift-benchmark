# Configuration Documentation

> Comprehensive guide to benchmark and scenario configuration files

## 🎯 Overview

drift-benchmark uses TOML configuration files to define benchmarks, detectors, and scenarios. This document covers all available configuration options, validation rules, and best practices for creating reproducible benchmark setups.

## 📋 Benchmark Configuration

### Basic Structure

Benchmark configurations define which scenarios to evaluate and which detectors to compare:

```toml
# List of scenarios to evaluate
[[scenarios]]
id = "scenario_name"

# List of detectors to compare
[[detectors]]
method_id = "method_identifier"
variant_id = "variant_identifier" 
library_id = "library_identifier"
# Optional hyperparameters...
```

### Complete Example

```toml
# Scenario-based configuration
[[scenarios]]
id = "covariate_drift_example"

[[scenarios]]
id = "concept_drift_example"

[[scenarios]]
id = "no_drift_baseline"

# Compare different library implementations of the same method+variant
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "alibi-detect"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"
threshold = 0.05

# Compare different methods
[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"
library_id = "scipy"
threshold = 0.05

[[detectors]]
method_id = "anderson_darling"
variant_id = "ad_batch"
library_id = "scipy"
threshold = 0.05

# River streaming detectors
[[detectors]]
method_id = "kswin"
variant_id = "kswin_standard"
library_id = "river"
alpha = 0.005
window_size = 100
```

## 📊 Configuration Models

### BenchmarkConfig

Top-level configuration model for benchmark execution.

**Fields**:

- `scenarios`: List of ScenarioConfig objects defining evaluation scenarios
- `detectors`: List of DetectorConfig objects defining detectors to compare

**Validation Rules**:

- At least one scenario must be specified
- At least one detector must be specified
- All scenario IDs must reference existing scenario definition files
- All detector method+variant combinations must exist in methods.toml

**TOML Mapping**:

```toml
# Root level contains scenario and detector arrays
[[scenarios]]
# ScenarioConfig fields...

[[detectors]]  
# DetectorConfig fields...
```

### ScenarioConfig

Reference to scenario definition file.

**Fields**:

- `id` (string, required): Scenario identifier matching filename without .toml extension

**Validation Rules**:

- ID must be non-empty string
- Corresponding scenario file must exist in scenarios directory
- Scenario file must be valid TOML with required fields

**TOML Example**:

```toml
[[scenarios]]
id = "covariate_drift_example"  # Loads scenarios/covariate_drift_example.toml

[[scenarios]]
id = "concept_drift_example"    # Loads scenarios/concept_drift_example.toml
```

### DetectorConfig

Configuration for individual detector with library identification.

**Required Fields**:

- `method_id` (string): Method identifier from methods.toml registry
- `variant_id` (string): Variant identifier from methods.toml registry  
- `library_id` (string): Library implementation identifier

**Optional Hyperparameters**:

Hyperparameters are passed as additional fields and forwarded to detector constructors:

- `threshold` (float): Detection threshold (common across most methods)
- `window_size` (int): Window size for streaming methods
- `alpha` (float): Significance level for statistical tests
- `delta` (float): Sensitivity parameter for change detection methods
- Custom parameters specific to library implementations

**Validation Rules**:

- `method_id` must exist in methods.toml registry
- `variant_id` must exist under specified method in methods.toml
- `library_id` must have registered adapter for method+variant combination
- Hyperparameters must match expected types for detector

**TOML Examples**:

```toml
# Minimal detector configuration
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"

# Detector with custom hyperparameters
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "evidently"
threshold = 0.01
bootstrap_samples = 2000

# Streaming detector with window configuration
[[detectors]]
method_id = "kswin"
variant_id = "kswin_standard" 
library_id = "river"
alpha = 0.005
window_size = 100
stat_size = 30
```

## 🎯 Scenario Configuration

Scenario files define individual evaluation cases with data sources, filters, and ground truth information.

### File Structure

Scenario files are stored in the `scenarios/` directory with `.toml` extension:

```text
scenarios/
├── covariate_drift_example.toml
├── concept_drift_example.toml
├── no_drift_baseline.toml
├── uci_wine_quality.toml
└── synthetic_regression.toml
```

### Required Fields

All scenario files must include:

```toml
description = "Human-readable description of the scenario"
source_type = "synthetic"  # or "file" or "uci"
source_name = "classification"  # dataset identifier
target_column = "target"  # name of target column (if applicable)
drift_types = ["covariate"]  # list of drift types
```

### Source Types

#### Synthetic Datasets (`source_type = "synthetic"`)

Use scikit-learn synthetic dataset generators:

```toml
description = "Synthetic classification with covariate drift"
source_type = "synthetic"
source_name = "classification"  # make_classification
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 500]
random_state = 42

[test_filter]
sample_range = [500, 1000]
noise_factor = 1.5  # Artificial drift modification
random_state = 24
```

**Supported Synthetic Datasets**:

- `classification` - make_classification
- `regression` - make_regression  
- `blobs` - make_blobs
- `moons` - make_moons
- `circles` - make_circles

**Synthetic-Specific Parameters**:

- `noise_factor` (float): Scale noise to create artificial drift
- `n_samples` (int): Number of samples to generate
- `n_features` (int): Number of features to generate
- `random_state` (int): Random seed for reproducibility
- Additional sklearn-specific parameters

#### File Datasets (`source_type = "file"`)

Load data from CSV files:

```toml
description = "CSV file with feature-based drift detection"
source_type = "file"
source_name = "datasets/my_dataset.csv"  # Path to CSV file
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 1000]
feature_filters = [
    {column = "feature_1", condition = "<=", value = 0.5}
]

[test_filter]
sample_range = [0, 1000]
feature_filters = [
    {column = "feature_1", condition = ">", value = 0.5}
]
```

**File Requirements**:

- Must be valid CSV format with headers
- Target column must exist if specified
- File path relative to project root or absolute path

#### UCI Repository (`source_type = "uci"`)

Access datasets from UCI Machine Learning Repository:

```toml
description = "UCI Wine Quality with authentic drift patterns"
source_type = "uci"
source_name = "wine-quality-red"  # UCI dataset identifier
target_column = "quality"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 800]
feature_filters = [
    {column = "alcohol", condition = "<=", value = 10.5}
]

[test_filter] 
sample_range = [800, 1599]
feature_filters = [
    {column = "alcohol", condition = ">", value = 12.0}
]

# Optional: UCI metadata for scientific traceability
[uci_metadata]
dataset_id = "wine-quality-red"
domain = "food_beverage_chemistry"
original_source = "Paulo Cortez, University of Minho"
acquisition_date = "2009-10-07"
```

**Popular UCI Datasets**:

- `iris` - Iris flower classification
- `wine` - Wine classification  
- `wine-quality-red` - Wine quality regression
- `adult` - Adult income classification
- `diabetes` - Diabetes regression

### Filter Configuration

Filters define how to extract reference and test data from source datasets.

#### Sample Range Filtering

Extract specific row ranges from datasets:

```toml
[ref_filter]
sample_range = [0, 500]  # Rows 0-500 (inclusive)

[test_filter]
sample_range = [500, 1000]  # Rows 500-1000 (inclusive)
```

**Validation Rules**:

- Start index must be >= 0
- End index must be > start index
- End index must be <= total dataset size
- Ranges can overlap between ref and test

#### Feature-Based Filtering

Filter data based on feature value conditions:

```toml
[ref_filter]
sample_range = [0, 1000]
feature_filters = [
    {column = "feature_1", condition = "<=", value = 0.5},
    {column = "feature_2", condition = ">=", value = -1.0}
]

[test_filter]
sample_range = [0, 1000]  
feature_filters = [
    {column = "feature_1", condition = ">", value = 0.5},
    {column = "feature_2", condition = "<", value = 1.0}
]
```

**Supported Conditions**:

- `"<="` - Less than or equal
- `">="` - Greater than or equal
- `">"` - Greater than
- `"<"` - Less than
- `"=="` - Equal to
- `"!="` - Not equal to

**Filter Logic**:

- Multiple feature filters are combined with AND logic
- All conditions must be true for a sample to be included
- Applied after sample_range filtering

#### Synthetic Data Modifications

For synthetic datasets only, apply modifications to create artificial drift:

```toml
[test_filter]
sample_range = [500, 1000]
noise_factor = 1.5      # Scale noise by 1.5x
feature_scaling = 0.8   # Scale features by 0.8x
target_shift = 0.2      # Shift target values by +0.2
random_state = 42       # Seed for modifications
```

**Synthetic Modification Parameters**:

- `noise_factor` (float): Multiply noise by this factor
- `feature_scaling` (float): Scale all features by this factor
- `target_shift` (float): Add this value to target variable
- `class_imbalance` (float): Adjust class distribution (classification only)
- `outlier_fraction` (float): Add outliers to specified fraction of data

### Ground Truth Specification

Define expected drift characteristics for evaluation:

```toml
[ground_truth]
drift_periods = [[500, 1000]]  # List of [start, end] drift periods
drift_intensity = "moderate"   # "weak", "moderate", "strong"
expected_effect_size = 0.4     # Cohen's d effect size
kl_divergence = 0.35          # Expected KL divergence
expected_detection = true      # Should detectors find drift?
```

**Ground Truth Fields**:

- `drift_periods` (array of arrays): Time periods where drift occurs
- `drift_intensity` (string): Qualitative drift strength assessment  
- `expected_effect_size` (float): Quantitative effect size (Cohen's d)
- `kl_divergence` (float): Expected KL divergence between distributions
- `expected_detection` (boolean): Whether drift should be detected

**Statistical Metrics**:

- **Effect Size**: Standardized measure of difference between groups
  - Small: 0.2, Medium: 0.5, Large: 0.8
- **KL Divergence**: Information-theoretic measure of distribution difference
  - 0 = identical, higher values = more different

### Statistical Validation

Define statistical rigor requirements for scientific validity:

```toml
[statistical_validation]
expected_effect_size = 0.4     # Expected Cohen's d
minimum_power = 0.80           # Statistical power requirement (80%)
significance_level = 0.05      # Alpha level (5%)
cohens_d = 0.45               # Calculated Cohen's d
baseline_scenario = "no_drift_baseline"  # Companion no-drift scenario
```

**Statistical Validation Fields**:

- `expected_effect_size` (float): Target effect size for power analysis
- `minimum_power` (float): Required statistical power (typically 0.80)
- `significance_level` (float): Type I error rate (typically 0.05)
- `cohens_d` (float): Calculated Cohen's d effect size
- `baseline_scenario` (string): Reference no-drift scenario for comparison

**Power Analysis**:

Statistical power is the probability of detecting an effect when it truly exists:

- **Power = 0.80**: 80% chance of detecting true drift
- **Alpha = 0.05**: 5% chance of false positive (Type I error)
- **Effect Size**: Magnitude of difference being detected

### Enhanced Metadata

Optional metadata for comprehensive dataset documentation:

```toml
[enhanced_metadata]
total_instances = 1000
feature_descriptions = [
    "feature_1: Primary continuous feature for covariate drift",
    "feature_2: Secondary continuous feature",
    "categorical_feature: Categorical feature with 3 levels"
]
missing_data_indicators = ["none", "N/A", "-999"]
data_quality_score = 0.95      # Quality score 0-1
data_authenticity = "real"     # "real", "synthetic", "semi-synthetic"
scientific_traceability = true  # Enable scientific attribution
```

### UCI Metadata Integration

For UCI datasets, include comprehensive metadata:

```toml
[uci_metadata]
dataset_id = "wine-quality-red"
domain = "food_beverage_chemistry"
original_source = "Paulo Cortez, University of Minho"
acquisition_date = "2009-10-07"
last_updated = "2009-10-07"
collection_methodology = "Laboratory analysis of wine samples"
data_authenticity = "real"
total_instances = 1599
feature_descriptions = [
    "fixed_acidity: tartaric acid concentration (g/L)",
    "volatile_acidity: acetic acid concentration (g/L)",
    "alcohol: alcohol percentage by volume",
    "quality: wine quality score (0-10)"
]
missing_data_indicators = ["none"]
data_quality_score = 0.95
```

### Experimental Design Configuration

For rigorous statistical experimentation:

```toml
[experimental_design]
control_group_size = 500       # Reference data size
treatment_group_size = 500     # Test data size  
randomization_method = "fixed_seed"  # "fixed_seed", "random", "stratified"
blinding_level = "none"        # "none", "single", "double"
replication_count = 1          # Number of independent replications
```

## 📋 Available Methods and Variants

### Methods Registry Structure

Methods are defined in `src/drift_benchmark/detectors/methods.toml`:

```toml
[methods.method_id]
name = "Human-readable method name"
description = "Detailed description of the mathematical method"
drift_types = ["covariate", "concept", "prior"]
family = "statistical-test"  # Method family category
data_dimension = "univariate"  # "univariate" or "multivariate" 
data_types = ["continuous"]  # Supported data types
requires_labels = false  # Whether method needs target labels
references = ["https://doi.org/example", "Author (Year)"]

[methods.method_id.variants.variant_id]
name = "Human-readable variant name"
execution_mode = "batch"  # "batch" or "streaming"
hyperparameters = ["threshold", "window_size"]
references = ["Implementation-specific references"]
```

### Statistical Test Methods

**Kolmogorov-Smirnov Test**:

```toml
# Configuration example
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"  # or "evidently", "alibi-detect"
threshold = 0.05
```

**Cramér-von Mises Test**:

```toml
[[detectors]]
method_id = "cramer_von_mises"
variant_id = "cvm_batch"
library_id = "scipy"
threshold = 0.05
```

**Anderson-Darling Test**:

```toml
[[detectors]]
method_id = "anderson_darling"
variant_id = "ad_batch"  
library_id = "scipy"
threshold = 0.05
```

**Chi-Square Test** (categorical data):

```toml
[[detectors]]
method_id = "chi_square"
variant_id = "chi_batch"
library_id = "scipy"
threshold = 0.05
```

### Distance-Based Methods

**Wasserstein Distance**:

```toml
[[detectors]]
method_id = "wasserstein_distance"
variant_id = "wasserstein_batch"
library_id = "scipy"
threshold = 0.1
```

**Jensen-Shannon Divergence**:

```toml
[[detectors]]
method_id = "jensen_shannon_divergence"
variant_id = "js_batch"
library_id = "custom"
threshold = 0.05
```

**Kullback-Leibler Divergence**:

```toml
[[detectors]]
method_id = "kullback_leibler_divergence"
variant_id = "kl_batch"
library_id = "custom"
threshold = 0.1
```

### Window-Based Methods

**KSWIN** (Kolmogorov-Smirnov Windowing):

```toml
[[detectors]]
method_id = "kswin"
variant_id = "kswin_standard"
library_id = "river"
alpha = 0.005
window_size = 100
```

**ADWIN** (Adaptive Windowing):

```toml
[[detectors]]
method_id = "adaptive_windowing"
variant_id = "adwin_standard"
library_id = "river"
delta = 0.002
```

### Change Detection Methods

**Page-Hinkley Test**:

```toml
[[detectors]]
method_id = "page_hinkley"
variant_id = "ph_standard"
library_id = "river"
delta = 0.005
lambda = 50
alpha = 0.9999
```

**CUSUM** (Cumulative Sum):

```toml
[[detectors]]
method_id = "cusum"
variant_id = "cusum_standard"
library_id = "river"
threshold = 5.0
drift_threshold = 10.0
```

### Statistical Process Control

**DDM** (Drift Detection Method):

```toml
[[detectors]]
method_id = "drift_detection_method"
variant_id = "ddm_standard"
library_id = "river"
warning_level = 2.0
drift_level = 3.0
```

**EDDM** (Early Drift Detection Method):

```toml
[[detectors]]
method_id = "early_drift_detection_method"
variant_id = "eddm_standard"
library_id = "river"
warning_level = 0.95
drift_level = 0.90
```

### Multivariate Methods

**All Features Drift**:

```toml
[[detectors]]
method_id = "all_features_drift"
variant_id = "evidently"
library_id = "evidently"
threshold = 0.05
stattest = "ks"  # Statistical test for each feature
```

## 🔧 Hyperparameter Guidelines

### Common Hyperparameters

**threshold** (float, default: 0.05):

- Statistical significance level for hypothesis tests
- Distance threshold for distance-based methods
- Lower values = more sensitive to drift

**window_size** (int, streaming methods):

- Number of samples in sliding window
- Larger values = more stable but less responsive
- Smaller values = more responsive but potentially noisy

**alpha** (float, statistical tests):

- Significance level parameter
- Related to Type I error rate
- Common values: 0.01, 0.05, 0.10

### Method-Specific Hyperparameters

**Statistical Tests**:

```toml
threshold = 0.05        # P-value threshold
alternative = "two-sided"  # "two-sided", "less", "greater"
bootstrap_samples = 1000   # Number of bootstrap samples
```

**Distance-Based Methods**:

```toml
threshold = 0.1         # Distance threshold
metric = "euclidean"    # Distance metric
normalize = true        # Normalize features
```

**Window-Based Methods**:

```toml
window_size = 100       # Sliding window size
stat_size = 30         # Statistics window size
alpha = 0.005          # Significance level
```

**Change Detection**:

```toml
delta = 0.005          # Sensitivity parameter
lambda = 50            # Forgetting factor
warning_level = 2.0    # Warning threshold
drift_level = 3.0      # Drift threshold
```

### Library-Specific Parameters

**Evidently**:

```toml
# Evidently-specific parameters
stattest = "ks"                    # Statistical test type
stattest_threshold = 0.05          # Test threshold
drift_share = 0.5                  # Share of drifted features
```

**Alibi-Detect**:

```toml
# Alibi-Detect specific parameters
backend = "tensorflow"             # "tensorflow", "pytorch", "sklearn"
p_val = 0.05                      # P-value threshold
preprocess_fn = null              # Preprocessing function
```

**River**:

```toml
# River-specific parameters
grace_period = 200                # Grace period for streaming
warning_level = 2.0               # Warning threshold
drift_level = 3.0                 # Drift threshold
```

## ✅ Validation and Error Handling

### Configuration Validation

**Benchmark Config Validation**:

- All scenario IDs must reference existing files
- All method+variant combinations must exist in methods.toml  
- All library_id values must have registered adapters
- Hyperparameters must match expected types

**Scenario Validation**:

- Required fields must be present and non-empty
- Source files must exist and be readable
- Filter ranges must be valid for dataset size
- Ground truth metrics must be reasonable values

**Common Validation Errors**:

```
ConfigurationError: Method 'invalid_method' not found in methods.toml
VariantNotFoundError: Variant 'invalid_variant' not found for method 'ks_test'
DataLoadingError: Scenario file 'missing_scenario.toml' not found
DetectorNotFoundError: No adapter registered for kolmogorov_smirnov+ks_batch+custom
```

### Best Practices

**Configuration Organization**:

```text
project/
├── configs/
│   ├── quick_test.toml           # Fast development testing
│   ├── method_comparison.toml    # Compare different methods
│   ├── library_comparison.toml   # Compare same method across libraries
│   └── comprehensive.toml        # Full evaluation suite
├── scenarios/
│   ├── synthetic/               # Synthetic dataset scenarios
│   ├── uci/                    # UCI dataset scenarios
│   └── custom/                 # Custom dataset scenarios
```

**Naming Conventions**:

- **Scenario IDs**: Descriptive with drift type and data source
  - `synthetic_covariate_classification`
  - `uci_wine_covariate_alcohol`
  - `file_concept_customer_churn`

- **Configuration Files**: Purpose-based naming
  - `quick_test.toml` - Fast iteration during development
  - `baseline_comparison.toml` - Standard benchmark suite
  - `library_evaluation.toml` - Cross-library comparison

**Documentation Standards**:

```toml
# Configuration file header comment
# Purpose: Compare statistical test methods for covariate drift detection
# Author: Your Name
# Date: 2024-01-15
# Expected runtime: ~5 minutes

# Group related scenarios
[[scenarios]]
id = "covariate_drift_strong"    # High effect size scenario

[[scenarios]]
id = "covariate_drift_weak"      # Low effect size scenario

# Group detectors by method for comparison
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "ks_batch"
library_id = "scipy"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov" 
variant_id = "ks_batch"
library_id = "evidently"
threshold = 0.05  # Same threshold for fair comparison
```

## 📖 Related Documentation

- **[Benchmark API](benchmark_api.md)**: Using configuration files to run benchmarks
- **[Adapter API](adapter_api.md)**: Creating detectors referenced in configurations
- **[Scenarios](scenarios.md)**: Detailed scenario definition reference
- **[Methods Registry](../src/drift_benchmark/detectors/methods.toml)**: Complete method and variant definitions
