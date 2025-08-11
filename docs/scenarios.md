# Scenarios Documentation

> Comprehensive guide to scenario definitions, data filtering, and ground truth specification

## 🎯 Overview

Scenarios are the primary unit of evaluation in drift-benchmark. They define complete evaluation cases with data sources, filtering conditions, ground truth specifications, and statistical validation requirements. Each scenario represents a specific drift detection challenge with known characteristics.

## 🏗️ Core Concepts

### What are Scenarios?

A **scenario** is a complete specification of:

- **Data Source**: Where the data comes from (synthetic, file, UCI repository)
- **Data Extraction**: How to split data into reference (no-drift) and test (potential drift) sets
- **Ground Truth**: What drift characteristics are expected
- **Statistical Validation**: Requirements for scientific rigor

### Scenario-Based Evaluation

Each scenario represents a single evaluation unit:

- **Input**: Reference data (X_ref, y_ref) and test data (X_test, y_test)
- **Detection**: Detectors provide boolean drift prediction per scenario
- **Evaluation**: Compare predictions against ground truth across multiple scenarios

This enables meaningful accuracy, precision, and recall calculations across diverse drift conditions.

### Data Authenticity Principles

drift-benchmark maintains data authenticity by treating different source types appropriately:

**Synthetic Datasets**: Generate controlled artificial drift through parameter modifications (noise injection, feature scaling). These test detector sensitivity to known, quantified changes.

**Real Datasets** (UCI, CSV): Preserve data authenticity by using only feature-based filtering. Leverage natural variation and correlations already present in real-world data.

## 📁 File Structure and Organization

### Directory Layout

```text
scenarios/
├── synthetic/                    # Synthetic dataset scenarios
│   ├── covariate_drift_strong.toml
│   ├── covariate_drift_weak.toml
│   └── concept_drift_gradual.toml
├── uci/                         # UCI repository scenarios
│   ├── wine_quality_alcohol.toml
│   ├── iris_petal_length.toml
│   └── adult_income_age.toml
├── custom/                      # Custom dataset scenarios
│   ├── customer_churn.toml
│   └── sensor_data_drift.toml
└── baselines/                   # No-drift baseline scenarios
    ├── no_drift_synthetic.toml
    └── no_drift_uci.toml
```

### Naming Conventions

**Scenario ID Format**: `{source}_{drift_type}_{description}`

- `synthetic_covariate_strong` - Strong covariate drift with synthetic data
- `uci_wine_covariate_alcohol` - UCI wine dataset with alcohol-based covariate drift  
- `file_concept_customer_churn` - Customer churn file with concept drift
- `no_drift_baseline` - Baseline scenario with no drift

## 📋 Scenario Definition Schema

### Complete TOML Structure

```toml
# Required metadata
description = "Human-readable scenario description"
source_type = "synthetic"  # "synthetic", "file", "uci"
source_name = "classification"  # Dataset identifier
target_column = "target"  # Target column name (if applicable)
drift_types = ["covariate"]  # Types of drift present

# Data filtering specifications
[ref_filter]
sample_range = [0, 500]
# Additional filter parameters...

[test_filter]
sample_range = [500, 1000]
# Additional filter parameters...

# Ground truth specification
[ground_truth]
drift_periods = [[500, 1000]]
expected_effect_size = 0.4
# Additional ground truth metrics...

# Statistical validation requirements
[statistical_validation]
minimum_power = 0.80
significance_level = 0.05
# Additional validation parameters...

# Optional metadata sections
[enhanced_metadata]
# Dataset quality and traceability information...

[uci_metadata]  # Only for UCI datasets
# Scientific attribution and metadata...

[experimental_design]  # For rigorous experiments
# Control group specifications...
```

### Required Fields

All scenario definitions must include:

#### Basic Metadata

- **`description`** (string): Human-readable description of the scenario
- **`source_type`** (string): Data source type - "synthetic", "file", or "uci"
- **`source_name`** (string): Dataset identifier specific to source type
- **`target_column`** (string): Name of target/label column (use "target" if none)
- **`drift_types`** (array): List of drift types present - "covariate", "concept", "prior", or "none"

#### Filter Specifications

- **`ref_filter`** (table): Configuration for extracting reference (no-drift) data
- **`test_filter`** (table): Configuration for extracting test (potential drift) data

#### Ground Truth

- **`ground_truth`** (table): Expected drift characteristics for evaluation

## 🎲 Data Source Types

### Synthetic Datasets (`source_type = "synthetic"`)

Generate controlled datasets using scikit-learn synthetic data generators.

#### Available Synthetic Datasets

**Classification Datasets**:

- `classification` - make_classification with configurable features and classes
- `blobs` - make_blobs for cluster-based classification
- `moons` - make_moons for non-linear decision boundaries
- `circles` - make_circles for nested circular patterns

**Regression Datasets**:

- `regression` - make_regression with configurable noise and features
- `friedman1` - Friedman synthetic regression problem

#### Synthetic Dataset Configuration

```toml
description = "Synthetic classification with artificial covariate drift"
source_type = "synthetic"
source_name = "classification"
target_column = "target"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 500]
n_samples = 1000       # Total samples to generate
n_features = 4         # Number of features
n_classes = 2          # Number of classes (classification only)
random_state = 42      # Reproducibility seed

[test_filter]
sample_range = [500, 1000]
noise_factor = 1.5     # Scale noise to create drift
feature_scaling = 0.8  # Scale features for drift
random_state = 24      # Different seed for drift
```

#### Synthetic Drift Modifications

**Available Modifications** (test_filter only):

- **`noise_factor`** (float): Multiply noise by this factor (default: 1.0)
- **`feature_scaling`** (float): Scale all features by this factor (default: 1.0)
- **`target_shift`** (float): Add constant to target values (default: 0.0)
- **`class_imbalance`** (float): Adjust class distribution (classification only)
- **`outlier_fraction`** (float): Add outliers to specified fraction of samples

**Example with Multiple Modifications**:

```toml
[test_filter]
sample_range = [500, 1000]
noise_factor = 2.0      # Double the noise level
feature_scaling = 0.7   # Reduce feature magnitudes by 30%
outlier_fraction = 0.1  # Add outliers to 10% of samples
random_state = 999      # Ensure reproducible modifications
```

### File Datasets (`source_type = "file"`)

Load data from local CSV files with feature-based filtering for authentic drift patterns.

#### File Dataset Configuration

```toml
description = "Customer data with authentic covariate drift based on age"
source_type = "file"
source_name = "datasets/customer_data.csv"  # Path relative to project root
target_column = "churn"  # Target column for supervised scenarios
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 2000]
feature_filters = [
    {column = "age", condition = "<=", value = 35}  # Younger customers
]

[test_filter]
sample_range = [0, 2000]  # Same data source
feature_filters = [
    {column = "age", condition = ">", value = 50}   # Older customers
]
```

#### File Requirements

- **Format**: CSV files with headers
- **Path**: Relative to project root or absolute paths
- **Encoding**: UTF-8 encoding recommended
- **Missing Values**: Handled automatically (NaN, empty strings)
- **Target Column**: Must exist if specified, can be "target" if no natural target

### UCI Repository (`source_type = "uci"`)

Access curated datasets from the UCI Machine Learning Repository with comprehensive metadata.

#### UCI Dataset Configuration

```toml
description = "Wine quality dataset with authentic drift based on alcohol content"
source_type = "uci"
source_name = "wine-quality-red"  # UCI dataset identifier
target_column = "quality"
drift_types = ["covariate"]

[ref_filter]
sample_range = [0, 800]
feature_filters = [
    {column = "alcohol", condition = "<=", value = 10.5}  # Low alcohol wines
]

[test_filter]
sample_range = [800, 1599]
feature_filters = [
    {column = "alcohol", condition = ">", value = 12.0}   # High alcohol wines
]

# Optional: UCI metadata for scientific traceability
[uci_metadata]
dataset_id = "wine-quality-red"
domain = "food_beverage_chemistry"
original_source = "Paulo Cortez, University of Minho"
acquisition_date = "2009-10-07"
collection_methodology = "Laboratory analysis of wine samples"
data_authenticity = "real"
total_instances = 1599
```

#### Popular UCI Datasets

**Classification Datasets**:

- `iris` - Iris flower species (150 samples, 4 features)
- `wine` - Wine classification (178 samples, 13 features)
- `adult` - Adult income prediction (48,842 samples, 14 features)
- `breast-cancer-wisconsin` - Breast cancer diagnosis (699 samples, 9 features)

**Regression Datasets**:

- `wine-quality-red` - Red wine quality (1,599 samples, 11 features)
- `wine-quality-white` - White wine quality (4,898 samples, 11 features)
- `diabetes` - Diabetes progression (442 samples, 10 features)
- `california-housing` - California housing prices (20,640 samples, 8 features)

**Mixed Data Datasets**:

- `adult` - Income prediction with categorical and numerical features
- `car-evaluation` - Car evaluation with ordinal categorical features

## 🔧 Data Filtering System

### Sample Range Filtering

Extract specific row ranges from datasets using inclusive indexing.

```toml
[ref_filter]
sample_range = [0, 499]    # Rows 0 through 499 (500 samples)

[test_filter] 
sample_range = [500, 999]  # Rows 500 through 999 (500 samples)
```

**Validation Rules**:

- Start index must be >= 0
- End index must be > start index  
- End index must be < total dataset size
- Ranges can overlap between reference and test

**Use Cases**:

- **Temporal Drift**: Earlier samples as reference, later samples as test
- **Batch Processing**: Different batches for reference vs. test
- **Data Splitting**: Standard train/test splits

### Feature-Based Filtering

Filter samples based on feature value conditions using logical operators.

#### Basic Feature Filtering

```toml
[ref_filter]
sample_range = [0, 1000]
feature_filters = [
    {column = "age", condition = "<=", value = 30}
]

[test_filter]
sample_range = [0, 1000]
feature_filters = [
    {column = "age", condition = ">", value = 60}
]
```

#### Supported Conditions

- **`"<="`** - Less than or equal to
- **`">="`** - Greater than or equal to  
- **`">"`** - Greater than
- **`"<"`** - Less than
- **`"=="`** - Equal to
- **`"!="`** - Not equal to

#### Multiple Conditions (AND Logic)

```toml
[ref_filter]
feature_filters = [
    {column = "age", condition = ">=", value = 25},
    {column = "income", condition = "<=", value = 50000},
    {column = "education", condition = "==", value = "Bachelor"}
]
```

All conditions must be satisfied for a sample to be included (AND logic).

#### Advanced Filtering Examples

**Continuous Features**:

```toml
# Low alcohol, high acidity wines (reference)
[ref_filter]
feature_filters = [
    {column = "alcohol", condition = "<=", value = 10.5},
    {column = "volatile_acidity", condition = ">=", value = 0.6}
]

# High alcohol, low acidity wines (test - drift)
[test_filter]
feature_filters = [
    {column = "alcohol", condition = ">", value = 12.0},
    {column = "volatile_acidity", condition = "<=", value = 0.4}
]
```

**Categorical Features**:

```toml
# Specific categories for reference
[ref_filter]
feature_filters = [
    {column = "education", condition = "==", value = "High School"},
    {column = "marital_status", condition = "!=", value = "Divorced"}
]

# Different categories for test (concept drift)
[test_filter]
feature_filters = [
    {column = "education", condition = "==", value = "Graduate"},
    {column = "marital_status", condition = "==", value = "Married"}
]
```

### Filter Execution Order

1. **Sample Range**: Applied first to limit the dataset scope
2. **Feature Filters**: Applied to the sample range subset
3. **Synthetic Modifications**: Applied last (synthetic datasets only)

```toml
[test_filter]
sample_range = [0, 1000]          # Step 1: Extract first 1000 rows
feature_filters = [               # Step 2: Filter by feature conditions
    {column = "age", condition = ">", value = 30}
]
noise_factor = 1.5               # Step 3: Apply synthetic modifications
```

## 📊 Ground Truth Specification

Define expected drift characteristics for quantitative evaluation.

### Basic Ground Truth

```toml
[ground_truth]
drift_periods = [[500, 1000]]     # Periods where drift occurs
drift_intensity = "moderate"      # Qualitative assessment
expected_effect_size = 0.4        # Cohen's d effect size
expected_detection = true         # Should detectors find drift?
```

### Ground Truth Fields

#### Required Fields

- **`drift_periods`** (array of arrays): List of [start, end] periods where drift occurs
  - Empty array `[]` for no-drift scenarios
  - Multiple periods: `[[100, 200], [500, 600]]` for multiple drift episodes
  
- **`expected_detection`** (boolean): Whether drift should be detected
  - `true` for drift scenarios
  - `false` for no-drift baseline scenarios

#### Quantitative Metrics

- **`expected_effect_size`** (float): Expected Cohen's d effect size
  - Small: 0.2, Medium: 0.5, Large: 0.8
  - Standardized measure of difference between groups

- **`kl_divergence`** (float): Expected Kullback-Leibler divergence
  - 0 = identical distributions
  - Higher values = greater distributional difference
  - Typical range: 0.1 (weak) to 1.0+ (strong)

- **`drift_intensity`** (string): Qualitative drift strength
  - "weak", "moderate", "strong"
  - Provides intuitive understanding alongside quantitative metrics

#### Advanced Ground Truth

```toml
[ground_truth]
drift_periods = [[500, 1000]]
drift_intensity = "strong"
expected_effect_size = 0.8
kl_divergence = 0.65
cohens_d = 0.75                   # Calculated Cohen's d
js_divergence = 0.45              # Jensen-Shannon divergence
wasserstein_distance = 1.2        # Wasserstein distance
expected_detection = true
drift_confidence = 0.9            # Expected confidence level
```

### Effect Size Guidelines

**Cohen's d Interpretation**:

- **0.2**: Small effect - detectable by sensitive methods
- **0.5**: Medium effect - detectable by most methods
- **0.8**: Large effect - easily detectable by all methods
- **1.0+**: Very large effect - obvious drift

**Scenario Design by Effect Size**:

```toml
# Weak drift scenario (challenging)
[ground_truth]
expected_effect_size = 0.2
drift_intensity = "weak"
expected_detection = true  # Should still be detectable

# Strong drift scenario (obvious)
[ground_truth]
expected_effect_size = 1.0
drift_intensity = "strong" 
expected_detection = true  # Should be easily detected
```

## 🧪 Statistical Validation

Define requirements for scientific rigor and statistical power.

### Basic Statistical Validation

```toml
[statistical_validation]
expected_effect_size = 0.4        # Target effect size
minimum_power = 0.80              # Required statistical power (80%)
significance_level = 0.05         # Type I error rate (5%)
baseline_scenario = "no_drift_baseline"  # Companion no-drift scenario
```

### Statistical Power Analysis

**Statistical Power**: Probability of detecting an effect when it truly exists

- **Power = 0.80**: 80% chance of detecting true drift (industry standard)
- **Power = 0.90**: 90% chance (high-rigor research)
- **Alpha = 0.05**: 5% false positive rate (Type I error)

**Sample Size Considerations**:

```toml
[statistical_validation]
minimum_power = 0.80
significance_level = 0.05
sample_size_reference = 500       # Reference group size
sample_size_test = 500           # Test group size
power_analysis_method = "cohens_d"  # Method for power calculation
```

### Experimental Design Validation

For rigorous experimental scenarios:

```toml
[experimental_design]
control_group_size = 500          # Reference data size constraint
treatment_group_size = 500        # Test data size constraint
randomization_method = "fixed_seed"  # Reproducibility method
blinding_level = "none"           # Blinding level for bias control
replication_count = 1             # Number of independent replications
effect_size_calculation = "pooled_std"  # Method for effect size
```

**Randomization Methods**:

- `"fixed_seed"` - Fixed random seed for reproducibility
- `"random"` - True randomization each run
- `"stratified"` - Stratified sampling by key variables

## 📈 Enhanced Metadata

Optional comprehensive metadata for dataset documentation and quality assessment.

### Basic Enhanced Metadata

```toml
[enhanced_metadata]
total_instances = 1000
feature_descriptions = [
    "feature_1: Primary numerical feature for drift analysis",
    "feature_2: Secondary categorical feature",
    "target: Binary classification target (0/1)"
]
missing_data_indicators = ["NaN", "null", ""]
data_quality_score = 0.95         # Quality score 0-1
data_authenticity = "real"        # "real", "synthetic", "semi-synthetic"
scientific_traceability = true    # Enable scientific attribution
```

### Quality Assessment Metrics

```toml
[enhanced_metadata]
data_quality_score = 0.95         # Overall quality (0-1)
completeness_score = 0.98         # Fraction of non-missing values
consistency_score = 0.92          # Internal consistency measure
accuracy_score = 0.94            # Accuracy against ground truth (if known)
outlier_fraction = 0.05          # Fraction of detected outliers
duplicate_fraction = 0.01        # Fraction of duplicate samples
```

### Feature-Level Metadata

```toml
[enhanced_metadata]
feature_descriptions = [
    "age: Customer age in years (18-80)",
    "income: Annual income in USD (20k-200k)",
    "education: Education level (High School, Bachelor, Graduate, PhD)",
    "purchase_amount: Last purchase amount in USD (10-5000)"
]

feature_types = [
    "age: continuous",
    "income: continuous", 
    "education: categorical_ordinal",
    "purchase_amount: continuous_positive"
]

feature_distributions = [
    "age: normal(45, 15)",
    "income: lognormal(11, 0.5)",
    "education: categorical(0.3, 0.4, 0.2, 0.1)",
    "purchase_amount: gamma(2, 50)"
]
```

## 🏛️ UCI Metadata Integration

For UCI repository datasets, include comprehensive scientific metadata.

### Required UCI Metadata

```toml
[uci_metadata]
dataset_id = "wine-quality-red"           # UCI identifier
domain = "food_beverage_chemistry"        # Scientific domain
original_source = "Paulo Cortez, University of Minho"  # Original authors
acquisition_date = "2009-10-07"           # When data was collected
data_authenticity = "real"               # Data authenticity level
total_instances = 1599                   # Number of samples
```

### Extended UCI Metadata

```toml
[uci_metadata]
dataset_id = "wine-quality-red"
domain = "food_beverage_chemistry"
original_source = "Paulo Cortez, University of Minho"
acquisition_date = "2009-10-07"
last_updated = "2009-10-07"
collection_methodology = "Laboratory analysis of wine samples from northern Portugal"
data_authenticity = "real"
total_instances = 1599

# Feature documentation
feature_descriptions = [
    "fixed_acidity: tartaric acid concentration (g/dm³)",
    "volatile_acidity: acetic acid concentration (g/dm³)",
    "citric_acid: citric acid concentration (g/dm³)",
    "residual_sugar: residual sugar concentration (g/dm³)",
    "chlorides: sodium chloride concentration (g/dm³)",
    "free_sulfur_dioxide: free SO2 concentration (mg/dm³)",
    "total_sulfur_dioxide: total SO2 concentration (mg/dm³)",
    "density: wine density (g/cm³)",
    "pH: wine pH level (0-14 scale)",
    "sulphates: potassium sulphate concentration (g/dm³)",
    "alcohol: alcohol percentage by volume (%)",
    "quality: wine quality score (0-10, ordinal)"
]

# Data quality indicators
missing_data_indicators = ["none"]
data_quality_score = 0.95
outlier_handling = "preserved"           # How outliers were handled
normalization_applied = false           # Whether data was normalized

# Scientific context
research_context = "Wine quality prediction for viticulture optimization"
ethical_considerations = "none"         # Ethical issues if any
license = "Creative Commons"            # Data usage license
citation_required = true               # Whether citation is required
```

### Scientific Traceability

```toml
[uci_metadata]
# ... other fields ...

# Scientific attribution
original_publication = "Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553."
doi = "10.1016/j.dss.2009.05.016"
methodology_reference = "Wine samples analyzed using standard laboratory protocols"
validation_methodology = "Expert sensory evaluation by wine tasting panels"

# Reproducibility information
data_collection_protocol = "Standard physicochemical analysis following EU regulations"
sampling_methodology = "Representative sampling from Vinho Verde region"
measurement_uncertainty = "±0.1 for chemical measurements"
```

## 🎯 Scenario Design Patterns

### Drift vs. No-Drift Pairs

Create matched pairs of scenarios for statistical validation:

**Drift Scenario** (`covariate_drift_wine_alcohol.toml`):

```toml
description = "Wine quality covariate drift based on alcohol content"
source_type = "uci"
source_name = "wine-quality-red"
target_column = "quality"
drift_types = ["covariate"]

[ref_filter]
feature_filters = [
    {column = "alcohol", condition = "<=", value = 10.5}
]

[test_filter] 
feature_filters = [
    {column = "alcohol", condition = ">", value = 12.0}
]

[ground_truth]
expected_detection = true
expected_effect_size = 0.6

[statistical_validation]
baseline_scenario = "no_drift_wine_random"
```

**No-Drift Baseline** (`no_drift_wine_random.toml`):

```toml
description = "No-drift baseline for wine quality scenario"
source_type = "uci"
source_name = "wine-quality-red"
target_column = "quality"
drift_types = ["none"]

[ref_filter]
sample_range = [0, 400]         # Random samples, no filtering

[test_filter]
sample_range = [800, 1200]      # Different random samples

[ground_truth]
expected_detection = false      # Should NOT detect drift
drift_periods = []             # No drift periods

[statistical_validation]
minimum_power = 0.80           # Validate false positive rate < 5%
```

### Progressive Difficulty Scenarios

Design scenarios with increasing drift strength:

```toml
# scenarios/wine_weak_drift.toml
[ground_truth]
expected_effect_size = 0.2
drift_intensity = "weak"

# scenarios/wine_moderate_drift.toml  
[ground_truth]
expected_effect_size = 0.5
drift_intensity = "moderate"

# scenarios/wine_strong_drift.toml
[ground_truth]
expected_effect_size = 0.8
drift_intensity = "strong"
```

### Multi-Feature Drift Scenarios

Test multivariate drift with multiple features:

```toml
description = "Multi-feature covariate drift in wine data"
source_type = "uci"
source_name = "wine-quality-red" 
target_column = "quality"
drift_types = ["covariate"]

[ref_filter]
feature_filters = [
    {column = "alcohol", condition = "<=", value = 10.5},
    {column = "volatile_acidity", condition = "<=", value = 0.4},
    {column = "pH", condition = ">=", value = 3.3}
]

[test_filter]
feature_filters = [
    {column = "alcohol", condition = ">", value = 12.0},
    {column = "volatile_acidity", condition = ">", value = 0.6}, 
    {column = "pH", condition = "<=", value = 3.1}
]

[ground_truth]
expected_effect_size = 0.7      # Higher due to multiple features
drift_intensity = "strong"
```

### Temporal Drift Scenarios

Simulate time-based drift using sample ranges:

```toml
description = "Temporal concept drift simulation"
source_type = "file"
source_name = "datasets/temporal_data.csv"
target_column = "outcome"
drift_types = ["concept"]

[ref_filter]
sample_range = [0, 1000]        # Early time period

[test_filter] 
sample_range = [3000, 4000]     # Later time period (drift occurred)

[ground_truth]
drift_periods = [[2000, 4000]]  # Drift period specification
expected_effect_size = 0.5
temporal_drift = true           # Flag for temporal analysis
```

## ✅ Validation and Quality Assurance

### Scenario Validation Rules

**Data Source Validation**:

- Synthetic datasets must use supported sklearn generators
- File paths must exist and be readable CSV files
- UCI dataset IDs must be valid and accessible

**Filter Validation**:

- Sample ranges must be within dataset bounds
- Feature filter columns must exist in dataset
- Filter conditions must result in non-empty data subsets
- Reference and test data must be non-overlapping for fair evaluation

**Ground Truth Validation**:

- Effect sizes must be reasonable (typically 0.1-2.0)
- Drift periods must align with data extraction ranges
- Expected detection must match drift_types (true for drift, false for "none")

**Statistical Validation**:

- Minimum power must be between 0.5-0.99
- Significance level must be between 0.001-0.1
- Sample sizes must be sufficient for specified power

### Quality Assurance Checklist

**Before Committing New Scenarios**:

1. **Data Integrity**: Verify data loads correctly and has expected shape
2. **Filter Effectiveness**: Confirm filters produce meaningful data splits
3. **Effect Size**: Calculate actual effect size and compare to expected
4. **Balance**: Ensure reasonable balance between reference and test sizes
5. **Documentation**: Include comprehensive descriptions and metadata
6. **Reproducibility**: Test with fixed random seeds for consistent results

**Testing Scenarios**:

```python
# Load and validate scenario
from drift_benchmark.data import load_scenario

try:
    scenario = load_scenario("my_new_scenario")
    
    # Check data loading
    assert len(scenario.X_ref) > 0, "Reference data empty"
    assert len(scenario.X_test) > 0, "Test data empty"
    
    # Check effect size
    from scipy.stats import ttest_ind
    feature_col = scenario.X_ref.columns[0]
    stat, p_val = ttest_ind(
        scenario.X_ref[feature_col], 
        scenario.X_test[feature_col]
    )
    
    print(f"Actual p-value: {p_val:.4f}")
    print(f"Expected drift: {scenario.definition.ground_truth['expected_detection']}")
    
except Exception as e:
    print(f"Scenario validation failed: {e}")
```

## 🎯 Best Practices

### Scenario Design Principles

1. **Clear Purpose**: Each scenario should test specific drift characteristics
2. **Realistic Conditions**: Use authentic data patterns when possible
3. **Quantitative Grounding**: Specify expected effect sizes and metrics
4. **Statistical Rigor**: Include power analysis and validation requirements
5. **Comprehensive Documentation**: Enable reproducibility and understanding

### File Organization

```text
scenarios/
├── README.md                    # Scenario collection overview
├── synthetic/                  # Controlled artificial scenarios
│   ├── covariate/
│   ├── concept/
│   └── baselines/
├── uci/                        # Real-world UCI datasets
│   ├── classification/
│   ├── regression/
│   └── mixed/
└── custom/                     # Project-specific scenarios
    ├── domain_specific/
    └── use_case_specific/
```

### Documentation Standards

**Scenario Headers**:

```toml
# Purpose: Test weak covariate drift detection sensitivity
# Author: Your Name <email@example.com>
# Created: 2024-01-15
# Last Modified: 2024-01-20
# Expected Runtime: ~30 seconds
# Dependencies: ucimlrepo>=0.0.3
# Related Scenarios: uci_wine_strong_drift.toml, uci_wine_no_drift.toml

description = "Comprehensive description of what this scenario tests..."
```

**Metadata Completeness**:

- Always include enhanced_metadata for dataset profiling
- Document feature descriptions and expected ranges
- Specify data quality indicators and limitations
- Include scientific attribution for real datasets

### Testing and Validation

**Automated Scenario Testing**:

```python
# Test all scenarios load correctly
import pytest
from pathlib import Path
from drift_benchmark.data import load_scenario

def test_all_scenarios_load():
    """Test that all scenario files load without errors."""
    scenario_dir = Path("scenarios")
    
    for scenario_file in scenario_dir.rglob("*.toml"):
        scenario_id = scenario_file.stem
        
        try:
            scenario = load_scenario(scenario_id)
            assert scenario is not None
            assert len(scenario.X_ref) > 0
            assert len(scenario.X_test) > 0
        except Exception as e:
            pytest.fail(f"Scenario {scenario_id} failed to load: {e}")
```

## 📖 Related Documentation

- **[Benchmark API](benchmark_api.md)**: Using scenarios in benchmark configurations
- **[Adapter API](adapter_api.md)**: How detectors process scenario data
- **[Configurations](configurations.md)**: Complete configuration file reference
- **Test Scenarios**: See `tests/assets/scenarios/` for example scenario files
