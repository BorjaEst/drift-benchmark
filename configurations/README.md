# Drift-Benchmark Configurations

> Comprehensive benchmark configurations for drift detection method evaluation

This directory contains expertly crafted benchmark configurations for evaluating drift detection methods across different libraries (Evidently AI, Alibi-Detect, and River). All configurations are organized into logical folders based on research focus and analysis type.

## 📁 Directory Structure

```text
configurations/
├── by_method_type/           # Organized by mathematical methodology
├── by_data_type/            # Organized by dataset characteristics  
├── by_execution_mode/       # Organized by processing paradigm
└── comparative_studies/     # Cross-library comparative studies
```

## 🎯 Configuration Categories

### 1. **By Method Type** (`by_method_type/`)

Configurations grouped by mathematical approach to drift detection:

#### **`statistical_tests_comprehensive.toml`**

- **Purpose**: Compare classical statistical hypothesis testing methods
- **Libraries**: Evidently AI (primary), Alibi-Detect, SciPy
- **Methods**: Kolmogorov-Smirnov, Cramér-von Mises, Anderson-Darling, Mann-Whitney, T-tests, Chi-square, Epps-Singleton, Kuiper, Baumgartner-Weiss-Schindler
- **Scenarios**: 7 scenarios covering synthetic, UCI, and baseline data
- **Use Case**: Research on statistical test performance and library comparisons

#### **`distance_based_comprehensive.toml`**

- **Purpose**: Evaluate information-theoretic and distance-based methods
- **Libraries**: Evidently AI (primary), SciPy
- **Methods**: Jensen-Shannon divergence, Kullback-Leibler divergence, Wasserstein distance, multivariate drift detection
- **Scenarios**: 8 scenarios including custom domain-specific data
- **Use Case**: Research on distribution comparison methods

#### **`streaming_comprehensive.toml`**

- **Purpose**: Comprehensive streaming/online drift detection evaluation
- **Libraries**: River (primary), Alibi-Detect online variants
- **Methods**: ADWIN, DDM, EDDM, Page-Hinkley, HDDM variants, KSWIN, CUSUM, EWMA
- **Scenarios**: 4 scenarios optimized for concept drift and streaming data
- **Use Case**: Real-time drift detection system development

### 2. **By Data Type** (`by_data_type/`)

Configurations optimized for specific data characteristics:

#### **`synthetic_comprehensive.toml`**

- **Purpose**: Controlled evaluation on synthetic data with known drift patterns
- **Advantage**: Ground truth validation and controlled drift intensity
- **Coverage**: All three libraries with comprehensive method coverage
- **Scenarios**: 4 synthetic scenarios (strong/weak covariate, gradual concept, no drift)
- **Use Case**: Algorithm validation and performance baseline establishment

#### **`uci_comprehensive.toml`**

- **Purpose**: Evaluation on real-world UCI repository datasets
- **Advantage**: Authentic data patterns and realistic drift scenarios
- **Coverage**: Focus on batch methods suitable for structured datasets
- **Scenarios**: 4 UCI scenarios (wine quality, iris, adult income, no drift baseline)
- **Use Case**: Real-world performance assessment

#### **`custom_comprehensive.toml`**

- **Purpose**: Domain-specific practical scenarios
- **Advantage**: Real-world application patterns (customer behavior, sensor data)
- **Coverage**: Mixed batch and streaming approaches
- **Scenarios**: 2 custom scenarios (customer churn, sensor data drift)
- **Use Case**: Industry-specific drift detection development

### 3. **By Execution Mode** (`by_execution_mode/`)

Configurations focused on processing paradigms:

#### **`batch_comprehensive.toml`**

- **Purpose**: Offline batch processing methods evaluation
- **Advantage**: Complete dataset analysis with full statistical power
- **Libraries**: Evidently AI (dominant), Alibi-Detect, SciPy
- **Scenarios**: 9 scenarios covering all data types
- **Use Case**: Offline analysis systems and research applications

#### **`online_comprehensive.toml`**

- **Purpose**: Real-time streaming data processing evaluation
- **Advantage**: Sequential processing and adaptive detection capabilities
- **Libraries**: River (dominant), Alibi-Detect online variants
- **Scenarios**: 4 scenarios suitable for streaming analysis
- **Use Case**: Real-time monitoring systems and online applications

### 4. **Comparative Studies** (`comparative_studies/`)

Research-focused configurations for academic and industrial analysis:

#### **`library_comparison.toml`**

- **Purpose**: Direct library-to-library performance comparison
- **Focus**: Identical methods implemented across different libraries
- **Coverage**: Head-to-head comparisons of same algorithms
- **Scenarios**: 7 representative scenarios for fair comparison
- **Use Case**: Library selection and performance benchmarking

#### **`batch_vs_streaming.toml`**

- **Purpose**: Execution mode trade-off analysis
- **Focus**: Performance vs accuracy trade-offs between batch and streaming
- **Coverage**: Comparable methods in both execution modes
- **Scenarios**: 4 scenarios suitable for both approaches
- **Use Case**: Architecture decision support and performance analysis

#### **`ultimate_comprehensive.toml`**

- **Purpose**: Maximum coverage research study
- **Focus**: Complete evaluation across all available methods and scenarios
- **Coverage**: All libraries, all methods, all scenarios
- **Scenarios**: 10 scenarios (complete coverage)
- **Use Case**: Comprehensive research studies and complete method evaluation

## 🚀 Quick Start Guide

### Basic Usage

```bash
# Navigate to the project root
cd /path/to/drift-benchmark

# Run a specific configuration
python -m drift_benchmark configurations/[folder]/[configuration].toml

# Example: Run statistical tests comparison
python -m drift_benchmark configurations/by_method_type/statistical_tests_comprehensive.toml
```

### Recommended Workflow

#### 1. **Start with Method-Specific Analysis**

```bash
# For statistical methods research
python -m drift_benchmark configurations/by_method_type/statistical_tests_comprehensive.toml

# For distance-based methods
python -m drift_benchmark configurations/by_method_type/distance_based_comprehensive.toml

# For streaming methods
python -m drift_benchmark configurations/by_method_type/streaming_comprehensive.toml
```

#### 2. **Focus on Your Data Type**

```bash
# For controlled experiments
python -m drift_benchmark configurations/by_data_type/synthetic_comprehensive.toml

# For real-world validation
python -m drift_benchmark configurations/by_data_type/uci_comprehensive.toml

# For domain-specific applications
python -m drift_benchmark configurations/by_data_type/custom_comprehensive.toml
```

#### 3. **Compare Processing Approaches**

```bash
# For offline analysis
python -m drift_benchmark configurations/by_execution_mode/batch_comprehensive.toml

# For real-time systems
python -m drift_benchmark configurations/by_execution_mode/online_comprehensive.toml
```

#### 4. **Conduct Comparative Studies**

```bash
# Compare library implementations
python -m drift_benchmark configurations/comparative_studies/library_comparison.toml

# Analyze batch vs streaming trade-offs
python -m drift_benchmark configurations/comparative_studies/batch_vs_streaming.toml

# Complete comprehensive study
python -m drift_benchmark configurations/comparative_studies/ultimate_comprehensive.toml
```

## 📊 Configuration Overview

| Configuration | Detectors | Scenarios | Primary Libraries | Execution Time |
|--------------|-----------|-----------|------------------|----------------|
| **Statistical Tests** | ~15 | 7 | Evidently, Alibi-Detect | Medium |
| **Distance-Based** | ~12 | 8 | Evidently, SciPy | Medium |
| **Streaming** | ~14 | 4 | River, Alibi-Detect | Fast |
| **Synthetic Data** | ~25 | 4 | All Libraries | Medium |
| **UCI Data** | ~22 | 4 | Evidently, Alibi-Detect | Medium |
| **Custom Data** | ~25 | 2 | All Libraries | Fast |
| **Batch Processing** | ~22 | 9 | Evidently, Alibi-Detect | Long |
| **Online Processing** | ~15 | 4 | River, Alibi-Detect | Medium |
| **Library Comparison** | ~23 | 7 | All Libraries | Long |
| **Batch vs Streaming** | ~23 | 4 | All Libraries | Long |
| **Ultimate** | ~40+ | 10 | All Libraries | Very Long |

## 🔧 Library Coverage

### **Evidently AI** (Most Comprehensive)

- **Statistical Tests**: Kolmogorov-Smirnov, Cramér-von Mises, Anderson-Darling, Mann-Whitney, T-test, Welch's T-test, Epps-Singleton, Kuiper, Baumgartner-Weiss-Schindler, Chi-square
- **Distance Methods**: Jensen-Shannon, Kullback-Leibler, Wasserstein distance
- **Multivariate**: All-Features drift with configurable statistical tests
- **Strengths**: Rich statistical test library, excellent for tabular data

### **Alibi-Detect** (Advanced Algorithms)

- **Batch Methods**: Kolmogorov-Smirnov, Cramér-von Mises, Chi-square
- **Online Methods**: Online KS, Online Cramér-von Mises with windowing
- **Strengths**: Advanced kernel methods, online variants, deep learning support

### **River** (Streaming Specialists)

- **Adaptive Methods**: ADWIN (Adaptive Windowing)
- **Process Control**: DDM, EDDM, CUSUM, EWMA, Geometric Moving Average
- **Change Detection**: Page-Hinkley, Exponential Cumulative Drift
- **Hoeffding-based**: HDDM-A, HDDM-W
- **Window-based**: KSWIN
- **Strengths**: Real-time processing, concept drift focus, minimal memory footprint

## 📈 Scenario Coverage

### **Synthetic Scenarios** (4 scenarios)

- `synthetic/covariate_drift_strong` - Strong distribution shift
- `synthetic/covariate_drift_weak` - Subtle distribution change  
- `synthetic/concept_drift_gradual` - Gradual concept evolution
- `baselines/no_drift_synthetic` - Controlled no-drift baseline

### **UCI Real-World Scenarios** (4 scenarios)

- `uci/wine_quality_alcohol` - Covariate drift based on alcohol content
- `uci/iris_petal_length` - Classic pattern recognition drift
- `uci/adult_income_age` - Demographic shift simulation
- `baselines/no_drift_uci` - Real-world no-drift baseline

### **Custom Domain Scenarios** (2 scenarios)

- `custom/customer_churn` - Business analytics drift patterns
- `custom/sensor_data_drift` - Time-series sensor drift simulation

## ⚙️ Configuration Format

All configurations use standardized TOML format:

```toml
# Scenario specification
[[scenarios]]
id = "scenario_identifier"

# Detector specification  
[[detectors]]
method_id = "method_name"           # From methods.toml registry
variant_id = "variant_name"         # From methods.toml registry  
library_id = "library_name"        # evidently/alibi-detect/river
threshold = 0.05                    # Detection threshold
# Additional hyperparameters as needed
```

## 🔍 Choosing the Right Configuration

### **For Research & Development**

- Start with `by_method_type/` to understand method characteristics
- Use `by_data_type/synthetic_comprehensive.toml` for algorithm validation
- Use `comparative_studies/library_comparison.toml` for library selection

### **For Production Systems**

- Use `by_execution_mode/batch_comprehensive.toml` for offline systems
- Use `by_execution_mode/online_comprehensive.toml` for real-time systems
- Use `by_data_type/custom_comprehensive.toml` for domain-specific applications

### **For Academic Studies**

- Use `comparative_studies/ultimate_comprehensive.toml` for complete coverage
- Use `comparative_studies/batch_vs_streaming.toml` for execution mode analysis
- Use specific method types for focused research questions

## 📚 Additional Resources

- **Methods Registry**: See `src/drift_benchmark/detectors/methods.toml` for all available methods
- **Scenarios Documentation**: See `docs/scenarios.md` for scenario specifications
- **API Documentation**: See `docs/adapter_api.md` for detector implementation details
- **Configuration Guide**: See `docs/configurations.md` for detailed configuration syntax

## 🤝 Contributing

When adding new configurations:

1. Follow the existing directory structure
2. Use descriptive filenames with clear purposes
3. Include comprehensive comments explaining the configuration intent
4. Validate all method+variant+library combinations against `methods.toml`
5. Test configurations with representative data before committing

For questions or contributions, please refer to the project's main documentation and contribution guidelines.
