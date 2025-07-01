# GitHub Copilot Instructions for drift-benchmark

## Project Overview

drift-benchmark is a Python library designed to benchmark various drift detection frameworks and algorithms.
It provides a standardized way to measure the performance of different methods for detecting data drift in machine learning pipelines.

## Project Structure

```
drift-benchmark/
├── components/                  # Specific detector implementations
├── configurations/              # Configuration files for benchmarks
├── datasets/                    # Additional datasets for benchmarking from user
├── results/                     # Output directory for benchmark results
├── scripts/                     # Utility scripts for data generation and preprocessing
├── src/drift_benchmark/         # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── settings.py              # Configuration settings
│   ├── methods.toml             # List of methods and their metadata
│   ├── benchmark/               # Core functionality
│   │   ├── __init__.py
│   │   ├── benchmarks.py        # Benchmark runner
│   │   ├── configuration.py     # Configuration models to run benchmarks
│   │   └── metrics.py           # Evaluation metrics
│   ├── data/                    # Data handling utilities
│   │   ├── __init__.py
│   │   ├── datasets.py          # Loader for standard datasets
│   │   └── drift_generators.py  # Tools to simulate drift
│   ├── detectors/               # Drift detection algorithms
│   │   ├── __init__.py
│   │   └── base.py              # Base detector class
│   ├── methods/                 # Standardization of detector implementations
│   │   ├── __init__.py
│   │   └── methods.toml         # Base detector class
│   └── figures/                 # List of methods and their metadata
│       ├── __init__.py
│       └── plots.py             # Plotting utilities
├── tests/                       # Test directory
├── examples/                    # Example notebooks and scripts
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License information
```

## Key Components

### Settings Module

This module contains configuration settings for the benchmark, including paths to data, directories, detector implementations and other global settings.
Settings are defined using pydantic v2 models for type safety and validation.

### Benchmark Module

This module contains the main functionality. It handles the execution of experiments, the evaluation of results, and the computation of performance metrics.
benchmarks.py is the main entry point for running benchmarks
configuration.py defines configuration models using pydantic v2 for running benchmarks, including datasets, detectors, and parameters.
configuration.py also provides utilities for loading and validating benchmark configurations from TOML files.
metrics.py provides functions to compute evaluation metrics for drift detection results.
metrics.py includes functions to compute evaluation metrics for drift detection results, such as precision, recall, F1-score, and area under the ROC curve (AUC-ROC).
metrics.py is based on dataclasses for structured data handling and validation.

### Data Module

This module provides utilities for loading standard datasets and generating synthetic data with different types of drift (concept drift, feature drift, etc.).

### Detectors Module

This module contains the detector registry and its utilities functions.
It also implement the common interface defined in `base.py`.
The BaseDetector includes an `aliases` class attribute to support detector naming conventions from different libraries.

### Methods Module

This module provides a standardized way to define and register drift detection methods.
It includes a `methods.toml` file that lists all available methods and their metadata, such as aliases, categories, and implementation details.
The methods are dynamically loaded with lru_cache to improve performance.
The `methods.toml` file serves as a central registry for all drift detection methods, allowing for easy extension and modification.
A method metadata example:

```toml
[kolmogorov_smirnov]
    name            = "Kolmogorov-Smirnov Test"
    description     = "Non-parametric test that quantifies the distance between empirical distribution functions of two samples."
    drift_types     = ["COVARIATE"]
    family          = "STATISTICAL_TEST"
    data_dimension  = "UNIVARIATE"
    data_types      = ["CONTINUOUS"]
    requires_labels = false
    references      = ["https://doi.org/10.2307/2280095", "Massey Jr (1951)"]

[kolmogorov_smirnov.implementations.ks_batch]
    name            = "Batch Kolmogorov-Smirnov"
    execution_mode  = "BATCH"
    hyperparameters = ["threshold"]
    references      = []
```

### Visualization Module

Tools for visualizing drift detection results, benchmark comparisons, and performance metrics.

### Components Directory

This directory contains specific implementations of drift detection algorithms. Each detector is implemented as a class that adheres to a common interface defined in `base.py`.
This directory is by default at the CWD but can be configured at settings.
Implementations are loaded dynamically when the library is initialized, allowing for easy extension and addition of new detectors.
Implementations should not implement BaseAdapters and evoid as much boilerplate code as possible on the fit and detect methods in order to obtain feasible timings for benchmarks.

### Configurations Directory

This directory contains configuration files for running benchmarks.
Each configuration file defines the datasets, detectors, and parameters to be used in the benchmark.
Configuration files are expected to be in TOML format, which allows for easy readability and modification.
This directory is by default at the CWD but can be configured at settings.

### Datasets Directory

This directory contains additional datasets that can be used for benchmarking.
Each dataset is expected to be in a standardized format (e.g., CSV) and should include metadata about the dataset characteristics.
This directory is by default at the CWD but can be configured at settings.

### Results Directory

This directory is used to store the output of benchmark runs, including performance metrics, visualizations, and logs.
This directory is by default at the CWD but can be configured at settings.

## Detector Classes

### Detector Categories by Drift Type:

Detectors can be differentiated by the type of drift they are designed to detect:

- Concept drift – changes in the relationship between features and labels (i.e., shifts in P(y|X)).
- Data (covariate) drift – changes in the distribution of input features (P(X)) without directly affecting the target.
- Prior probability (label) drift – shifts in the distribution of the labels themselves (P(y)), even if P(X|y)

### Detector Categories by Implementation Type:

Detectors can be differentiated by their implementation type into:

- Streaming: Detectors that process data in a streaming fashion, analyzing each data point as it arrives.
- Batch: Detectors that analyze data in batches, requiring a complete dataset before performing drift detection

### Detector Categories by Family:

- Change detection: Detectors that identify changes in the data distribution over time.
- Statistical process control: Detectors that use control charts to monitor data and detect drift.
- Window-based: Detectors that analyze data within a sliding window to detect drift.
- Distance-based: Detectors that measure the distance between distributions to identify drift.
- Statistical tests: Detectors that use statistical tests to determine if a drift has occurred.
- Ensemble methods: Detectors that combine multiple drift detection algorithms to improve robustness and accuracy.
- Machine learning-based: Detectors that use machine learning models to learn patterns in the data and detect drift.

### Detector Categories by data type:

- Univariate: Detectors that analyze a single feature or variable at a time.
- Multivariate: Detectors that analyze multiple features or variables simultaneously.

### Detector Categories by value type:

- Continuous: Detectors that work with continuous numerical data.
- Categorical: Detectors that work with categorical data.

## Development Guidelines

### Adding a New Detector

1. Create a new file in `<working-dir>/implementations/` named after your detector, e.g., `mydetector_adapter.py`
2. Implement the detector by extending the `BaseDetector` class from `drift_benchmark/detectors/base.py`
3. Add an `aliases` class attribute to your detector class to support naming conventions from the original library. For example:
   ```python
   class MyDetector(BaseDetector):
       # Add original library method names as aliases
       aliases = ["original_name", "alternative_name"]
   ```
4. Implement required methods like `fit()`, `detect()` and `score()`
5. Implement the `metadata()` class method to provide information about the detector
6. Add tests for your detector at the end of the file using pytest framework

### Adding a New Dataset

1. Add the dataset file in `<working-dir>/datasets/` as csv format
2. Ensure it returns data in a standardized format
3. Add documentation about the dataset characteristics

### Running Benchmarks

Benchmarks are configured via TOML files and can be run using the CLI tool or programmatically:

```python
from drift_benchmark import BenchmarkRunner

runner = BenchmarkRunner(config_file='example.yaml')
results = runner.run()
results.visualize()
```

## Testing

Run tests using pytest:

```
pytest tests/
```

## Code Style

This project follows PEP 8 guidelines. Code formatting is handled by black and isort.
Python target version only 3.10

## Dependencies

Dependencies are managed in pyproject.toml with setuptools dynamically via:

- requirements.txt for runtime dependencies
- requirements-full.txt for all dependencies including drift detection libraries
- requirements-dev.txt for development dependencies
- requirements-test.txt for testing dependencies

## Installation for Development

```
git clone https://github.com/yourusername/drift-benchmark.git
cd drift-benchmark
pip install
```

## Frameworks adapters to implement in Componenents Directory

### Evidently

Evidently provides various statistical tests for drift detection. The following table summarizes the available tests, their applicability, and how drift scores are calculated:

<table>
    <thead>
        <tr>
            <th>StatTest</th>
            <th>Applicable to</th>
            <th>Drift score</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <code>ks</code>
                <br>Kolmogorov–Smirnov (K-S) test
            </td>
            <td>tabular data<br>only numerical <br><br><strong>Default method for numerical data, if
                ≤ 1000 objects</strong></td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>chisquare</code>
                <br>Chi-Square test
            </td>
            <td>tabular data<br>only categorical<br><br><strong>Default method for categorical with
                &gt; 2 labels, if ≤ 1000 objects</strong></td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>z</code>
                <br> Z-test
            </td>
            <td>tabular data<br>only categorical<br><br><strong>Default method for binary data, if ≤
                1000 objects</strong></td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>wasserstein</code>
                <br> Wasserstein distance (normed)
            </td>
            <td>tabular data<br>only numerical<br><br><strong>Default method for numerical data, if
                &gt; 1000 objects</strong></td>
            <td>returns <code>distance</code><br>drift detected when <code>distance</code> ≥ <code>
                threshold</code><br>default threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>kl_div</code>
                <br>Kullback-Leibler divergence
            </td>
            <td>tabular data<br>numerical and categorical</td>
            <td>returns <code>divergence</code><br>drift detected when <code>divergence</code> ≥ <code>
                threshold</code><br>default threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>psi</code>
                <br> Population Stability Index (PSI)
            </td>
            <td>tabular data<br>numerical and categorical</td>
            <td>returns <code>psi_value</code><br>drift detected when <code>psi_value</code> ≥ <code>
                threshold</code><br>default threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>jensenshannon</code>
                <br> Jensen-Shannon distance
            </td>
            <td>tabular data<br>numerical and categorical<br><br><strong>Default method for
                categorical, if &gt; 1000 objects</strong></td>
            <td>returns <code>distance</code><br>drift detected when <code>distance</code> ≥ <code>
                threshold</code><br>default threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>anderson</code>
                <br> Anderson-Darling test
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>fisher_exact</code>
                <br> Fisher's Exact test
            </td>
            <td>tabular data<br>only categorical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>cramer_von_mises</code>
                <br> Cramer-Von-Mises test
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>g-test</code>
                <br> G-test
            </td>
            <td>tabular data<br>only categorical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>hellinger</code>
                <br> Hellinger Distance (normed)
            </td>
            <td>tabular data<br>numerical and categorical</td>
            <td>returns <code>distance</code><br>drift detected when <code>distance</code> &gt;= <code>
                threshold</code><br>default threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>mannw</code>
                <br> Mann-Whitney U-rank test
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>ed</code>
                <br> Energy distance
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>distance</code><br>drift detected when <code>distance &gt;= threshold</code><br>default
                threshold: 0.1</td>
        </tr>
        <tr>
            <td>
                <code>es</code>
                <br> Epps-Singleton test
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>t_test</code>
                <br> T-Test
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>empirical_mmd</code>
                <br> Empirical-MMD
            </td>
            <td>tabular data<br>only numerical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value &lt; threshold</code><br>default
                threshold: 0.05</td>
        </tr>
        <tr>
            <td>
                <code>TVD</code>
                <br> Total-Variation-Distance
            </td>
            <td>tabular data<br>only categorical</td>
            <td>returns <code>p_value</code><br>drift detected when <code>p_value</code> &lt; <code>
                threshold</code><br>default threshold: 0.05</td>
        </tr>
    </tbody>
</table>

Evidently has the following data drift tests available:

- TestAllFeaturesValueDrift
- TestColumnDrift
- TestCustomFeaturesValueDrift
- TestEmbeddingsDrift
- TestNumberOfDriftedColumns
- TestShareOfDriftedColumns

### Alibi Detect
