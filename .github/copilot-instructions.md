# GitHub Copilot Instructions for drift-benchmark

## Project Overview

drift-benchmark is a comprehensive Python library designed to benchmark various drift detection frameworks and algorithms across multiple libraries and implementations. It provides a standardized, extensible platform for:

- **Performance Evaluation**: Compare drift detection methods across different scenarios and datasets
- **Method Discovery**: Centralized registry of 40+ drift detection methods with detailed metadata
- **Implementation Flexibility**: Support for both streaming and batch processing modes
- **Framework Integration**: Adapters for popular libraries (Evidently, Alibi Detect, Frouros, etc.)
- **Reproducible Research**: Standardized benchmarking protocols and result reporting

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

This module contains the detector registry and base implementation framework.

**Key Components:**

- **BaseDetector**: Abstract base class defining the common interface for all detectors
- **Registry System**: Dynamic loading and registration of detector implementations
- **Alias Support**: Multiple naming conventions from different libraries
- **Metadata Management**: Standardized detector information and capabilities

**BaseDetector Interface:**

```python
class BaseDetector(ABC):
    # Class attribute for method/implementation mapping
    detector = ("method_name", "implementation_name")

    @abstractmethod
    def fit(self, X_ref: ArrayLike, y_ref: Optional[ArrayLike] = None) -> "BaseDetector":
        """Fit detector on reference data"""

    @abstractmethod
    def detect(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> bool:
        """Detect drift in test data"""

    @abstractmethod
    def score(self, X_test: ArrayLike, y_test: Optional[ArrayLike] = None) -> float:
        """Return drift score/distance metric"""

    @classmethod
    @abstractmethod
    def metadata(cls) -> dict:
        """Return detector metadata and capabilities"""
```

### Methods Module

This module provides a centralized registry for drift detection methods through the `methods.toml` configuration file.

**Key Features:**

- **Comprehensive Coverage**: 40+ methods across statistical tests, distance-based measures, and streaming algorithms
- **Rich Metadata**: Each method includes family classification, data requirements, hyperparameters, and academic references
- **Implementation Variants**: Multiple implementations per method (batch, streaming, online)
- **Dynamic Loading**: Methods are loaded with `lru_cache` for optimal performance
- **Extensible Design**: Easy addition of new methods and implementations

**Method Categories:**

- **Statistical Tests**: KS test, Anderson-Darling, Chi-square, T-tests, etc.
- **Distance-Based**: MMD, Wasserstein, Energy distance, KL divergence, etc.
- **Streaming Methods**: DDM, EDDM, ADWIN, Page-Hinkley, CUSUM, etc.
- **Control Charts**: EWMA, Geometric Moving Average, etc.

**Method Metadata Structure:**

```toml
[method_name]
    name            = "Human-readable method name"
    description     = "Detailed description of the method"
    drift_types     = ["COVARIATE", "CONCEPT", "LABEL"]  # Types of drift detected
    family          = "STATISTICAL_TEST"  # Method family classification
    data_dimension  = "UNIVARIATE"  # UNIVARIATE or MULTIVARIATE
    data_types      = ["CONTINUOUS", "CATEGORICAL"]  # Supported data types
    requires_labels = false  # Whether method needs labeled data
    references      = ["DOI links", "Citation strings"]  # Academic references

    [method_name.implementations.impl_name]
        name            = "Implementation-specific name"
        execution_mode  = "BATCH"  # BATCH or STREAMING
        hyperparameters = ["param1", "param2"]  # Configurable parameters
        references      = []  # Implementation-specific references
```

**Drift Type Categories:**

- **COVARIATE**: Changes in input feature distributions P(X)
- **CONCEPT**: Changes in target relationship P(y|X)
- **LABEL**: Changes in target distribution P(y)

**Family Classifications:**

- **STATISTICAL_TEST**: Hypothesis testing approaches
- **DISTANCE_BASED**: Distribution distance measures
- **STATISTICAL_PROCESS_CONTROL**: Control chart methods
- **CHANGE_DETECTION**: Sequential change detection
- **WINDOW_BASED**: Sliding window approaches

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

**Step-by-Step Process:**

1. **Create Adapter File**: Create `<working-dir>/components/{framework}_adapter.py` (use single underscore)
2. **Implement Base Class**: Extend `BaseDetector` from `drift_benchmark.detectors.base`

   ```python
   from drift_benchmark.detectors.base import BaseDetector

   class MyFrameworkDetector(BaseDetector):
       # Map to methods.toml entries
       detector = ("method_name", "implementation_name")
   ```

3. **Method Mapping**: Ensure `detector` tuple matches entries in `methods.toml`:

   - First element: method key (e.g., "kolmogorov_smirnov")
   - Second element: implementation key (e.g., "ks_batch")

4. **Implement Required Methods**:

   ```python
   def fit(self, X_ref, y_ref=None):
       """Initialize with reference data"""
       # Minimize computation here for benchmark performance

   def detect(self, X_test, y_test=None):
       """Return boolean drift detection result"""

   def score(self, X_test, y_test=None):
       """Return numerical drift score/distance"""

   @classmethod
   def metadata(cls):
       """Return detector capabilities and requirements"""
       return {
           "requires_fit": True,
           "supports_streaming": False,
           "data_types": ["continuous"],
           "multivariate": False
       }
   ```

5. **Performance Optimization**:

   - Avoid heavy computation in `fit()` method
   - Cache expensive operations
   - Use efficient data structures
   - Minimize external library overhead

6. **Testing**: Add comprehensive tests at file end:

   ```python
   import pytest
   import numpy as np

   def test_detector_basic_functionality():
       detector = MyFrameworkDetector()
       X_ref = np.random.normal(0, 1, (100, 2))
       X_test = np.random.normal(0.5, 1, (100, 2))

       detector.fit(X_ref)
       drift_detected = detector.detect(X_test)
       drift_score = detector.score(X_test)

       assert isinstance(drift_detected, bool)
       assert isinstance(drift_score, (int, float))
   ```

7. **Documentation**: Include docstrings with:
   - Method description and references
   - Parameter specifications
   - Return value descriptions
   - Usage examples

### Adding a New Method to methods.toml

**Complete Method Entry:**

```toml
[new_method_name]
    name            = "Human-Readable Method Name"
    description     = "Comprehensive description of the method and its characteristics"
    drift_types     = ["COVARIATE", "CONCEPT", "LABEL"]  # Applicable drift types
    family          = "STATISTICAL_TEST"  # Method family (see classifications above)
    data_dimension  = "UNIVARIATE"  # UNIVARIATE or MULTIVARIATE
    data_types      = ["CONTINUOUS", "CATEGORICAL"]  # Supported data types
    requires_labels = false  # Boolean: needs labeled data
    references      = ["https://doi.org/...", "Author et al. (Year)"]  # Academic refs

    [new_method_name.implementations.impl_batch]
        name            = "Batch Implementation Name"
        execution_mode  = "BATCH"  # BATCH or STREAMING
        hyperparameters = ["param1", "param2", "threshold"]  # Configurable parameters
        references      = ["Implementation-specific references"]

    [new_method_name.implementations.impl_streaming]
        name            = "Streaming Implementation Name"
        execution_mode  = "STREAMING"
        hyperparameters = ["param1", "window_size", "threshold"]
        references      = []
```

**Validation Checklist:**

- [ ] Method name is descriptive and follows snake_case
- [ ] All required fields are present
- [ ] Drift types are from: COVARIATE, CONCEPT, LABEL
- [ ] Family is from established categories
- [ ] Data dimension is UNIVARIATE or MULTIVARIATE
- [ ] Data types are CONTINUOUS and/or CATEGORICAL
- [ ] References include DOI links when available
- [ ] Implementation names are unique within method
- [ ] Hyperparameters list is comprehensive

### Adding a New Dataset

**Dataset Requirements:**

1. **Format**: CSV files with proper headers
2. **Structure**: Features in columns, samples in rows
3. **Metadata**: Include dataset characteristics documentation
4. **Quality**: Clean data without missing values (or proper handling)

**Dataset Placement:**

```
<working-dir>/datasets/
├── my_dataset.csv           # Main dataset file
├── my_dataset_metadata.json # Optional metadata file
└── README.md               # Dataset documentation
```

**CSV Format Standards:**

```csv
feature_1,feature_2,feature_3,target,timestamp
1.23,0.45,category_a,1,2024-01-01
2.34,0.67,category_b,0,2024-01-02
...
```

**Metadata Documentation:**

```json
{
    "name": "My Dataset",
    "description": "Description of the dataset and its characteristics",
    "features": {
        "feature_1": {"type": "continuous", "range": [0, 10]},
        "feature_2": {"type": "continuous", "range": [-1, 1]},
        "feature_3": {"type": "categorical", "categories": ["a", "b", "c"]}
    },
    "target": {"type": "binary", "classes": [0, 1]},
    "drift_points": [1000, 2000],  # Known drift locations
    "source": "Original data source or reference"
}
```

### Running Benchmarks

**Programmatic Usage:**

```python
from drift_benchmark import BenchmarkRunner
from drift_benchmark.benchmark.configuration import BenchmarkConfig

# Load configuration from TOML file
config = BenchmarkConfig.from_toml('configurations/my_benchmark.toml')

# Initialize and run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Analyze results
results.summary()
results.visualize()
results.export('results/my_benchmark_results.json')
```

**Configuration File Structure:**

```toml
[benchmark]
name = "My Benchmark"
description = "Benchmark description"
output_dir = "results/my_benchmark"

[dataset]
name = "synthetic_drift"
parameters = { n_samples = 1000, drift_magnitude = 0.5 }

[[detectors]]
method = "kolmogorov_smirnov"
implementation = "ks_batch"
parameters = { threshold = 0.05 }

[[detectors]]
method = "maximum_mean_discrepancy"
implementation = "mmd_batch"
parameters = { kernel = "rbf", gamma = 1.0, threshold = 0.1 }

[evaluation]
metrics = ["precision", "recall", "f1_score", "auc_roc"]
cross_validation = { n_splits = 5, random_state = 42 }
```

**Command Line Interface:**

```bash
# Run single benchmark
python -m drift_benchmark run configurations/example.toml

# Run multiple benchmarks
python -m drift_benchmark run configurations/*.toml

# Generate benchmark report
python -m drift_benchmark report results/my_benchmark/
```

## Testing

**Test Structure:**

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_benchmark.py              # Core benchmark functionality tests
├── test_detectors/               # Detector-specific tests
│   ├── test_base.py              # BaseDetector interface tests
│   ├── test_evidently.py         # Evidently adapter tests
│   └── test_alibi_detect.py      # Alibi Detect adapter tests
├── test_methods/                 # Method registry tests
│   ├── test_methods_loading.py   # methods.toml loading tests
│   └── test_method_validation.py # Method metadata validation
└── assets/                       # Test data and mock components
    ├── components/
    ├── configurations/
    └── datasets/
```

**Running Tests:**

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_benchmark.py
pytest tests/test_detectors/

# Run with coverage
pytest tests/ --cov=drift_benchmark --cov-report=html

# Run performance tests
pytest tests/ -m "performance"

# Run integration tests
pytest tests/ -m "integration"
```

**Test Categories:**

- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmark timing and memory usage
- **Validation Tests**: Configuration and data validation

## Code Style and Standards

**Python Standards:**

- **Target Version**: Python 3.10+
- **Code Style**: PEP 8 compliance
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all modules, classes, and functions

**Formatting Tools:**

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/
mypy src/

# Check imports
isort --check-only src/ tests/
```

**Documentation Standards:**

```python
def detect_drift(X_ref: ArrayLike, X_test: ArrayLike, method: str) -> bool:
    """Detect drift between reference and test datasets.

    Args:
        X_ref: Reference dataset for comparison
        X_test: Test dataset to analyze for drift
        method: Name of drift detection method to use

    Returns:
        True if drift is detected, False otherwise

    Raises:
        ValueError: If method is not supported

    Example:
        >>> X_ref = np.random.normal(0, 1, (100, 2))
        >>> X_test = np.random.normal(0.5, 1, (100, 2))
        >>> detect_drift(X_ref, X_test, "kolmogorov_smirnov")
        True
    """
```

**Pre-commit Hooks:**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Dependencies Management

**Dependency Structure:**

- `requirements.txt` - Core runtime dependencies (minimal)
- `requirements-full.txt` - All dependencies including drift detection libraries
- `requirements-dev.txt` - Development tools (testing, linting, formatting)
- `requirements-test.txt` - Testing-specific dependencies

**Installation Modes:**

```bash
# Minimal installation (core functionality only)
pip install -r requirements.txt

# Full installation (all supported libraries)
pip install -r requirements-full.txt

# Development installation
pip install -r requirements-dev.txt
pip install -r requirements-test.txt

# Editable development install
pip install -e .
```

**Dependency Categories:**

- **Core**: numpy, pandas, scikit-learn, pydantic, toml
- **Drift Detection**: evidently, alibi-detect, frouros, menelaus, river
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: pytest, black, isort, flake8, mypy
- **Documentation**: sphinx, sphinx-rtd-theme

**Version Management:**

- Pin major versions for stability
- Use compatible release specifiers (~=) for flexibility
- Regular dependency updates with testing

## Installation for Development

**Quick Setup:**

```bash
# Clone repository
git clone https://github.com/yourusername/drift-benchmark.git
cd drift-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Verify installation
python -c "import drift_benchmark; print('Installation successful!')"
```

**Development Workflow:**

```bash
# 1. Install pre-commit hooks
pre-commit install

# 2. Run tests to verify setup
pytest tests/

# 3. Check code style
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/

# 4. Type checking
mypy src/

# 5. Run full validation
make test  # or equivalent command
```

**Docker Development:**

```dockerfile
# Dockerfile for development
FROM python:3.10-slim

WORKDIR /app
COPY requirements-full.txt .
RUN pip install -r requirements-full.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "pytest"]
```

## Frameworks adapters to implement in Componenents Directory

### Evidently Framework

**Overview:**
Evidently provides statistical tests and distance-based methods for data drift detection with built-in visualization capabilities.

**Key Features:**

- Automatic test selection based on data characteristics
- Rich statistical test portfolio (20+ methods)
- Built-in data preprocessing and binning
- Comprehensive reporting and visualization

**Implementation Strategy:**

```python
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
from drift_benchmark.detectors.base import BaseDetector

class EvidentlyDetector(BaseDetector):
    detector = ("kolmogorov_smirnov", "ks_batch")  # Map to methods.toml

    def __init__(self, stattest='ks', threshold=0.05):
        self.stattest = stattest
        self.threshold = threshold
        self.test_suite = None

    def fit(self, X_ref, y_ref=None):
        # Prepare test suite configuration
        self.test_suite = TestSuite(tests=[
            TestColumnDrift(column='feature', stattest=self.stattest)
        ])
        # Store reference data efficiently
        self.X_ref = X_ref
        return self

    def detect(self, X_test, y_test=None):
        # Run test suite and extract results
        self.test_suite.run(reference_data=self.X_ref, current_data=X_test)
        results = self.test_suite.as_dict()
        return results['tests'][0]['parameters']['drift_detected']

    def score(self, X_test, y_test=None):
        # Extract numerical score (p-value or distance)
        self.test_suite.run(reference_data=self.X_ref, current_data=X_test)
        results = self.test_suite.as_dict()
        return results['tests'][0]['parameters']['drift_score']
```

**Available Tests Mapping:**

| Evidently Test     | methods.toml Key              | Data Type   | Execution Mode |
| ------------------ | ----------------------------- | ----------- | -------------- |
| `ks`               | `kolmogorov_smirnov`          | CONTINUOUS  | BATCH          |
| `chisquare`        | `chi_square`                  | CATEGORICAL | BATCH          |
| `z`                | `z_test`                      | CATEGORICAL | BATCH          |
| `wasserstein`      | `wasserstein_distance`        | CONTINUOUS  | BATCH          |
| `kl_div`           | `kullback_leibler_divergence` | BOTH        | BATCH          |
| `psi`              | `population_stability_index`  | BOTH        | BATCH          |
| `jensenshannon`    | `jensen_shannon_distance`     | BOTH        | BATCH          |
| `anderson`         | `anderson_darling`            | CONTINUOUS  | BATCH          |
| `fisher_exact`     | `fisher_exact`                | CATEGORICAL | BATCH          |
| `cramer_von_mises` | `cramer_von_mises`            | CONTINUOUS  | BATCH          |
| `mannw`            | `mann_whitney`                | CONTINUOUS  | BATCH          |
| `t_test`           | `t_test`                      | CONTINUOUS  | BATCH          |
| `empirical_mmd`    | `maximum_mean_discrepancy`    | CONTINUOUS  | BATCH          |

**Integration Notes:**

- Handle both univariate and multivariate data
- Support automatic and manual test selection
- Extract drift_detected boolean and numerical scores
- Consider memory usage with large test suites

### Alibi Detect Framework

**Overview:**
Alibi Detect provides advanced drift detection methods with uncertainty quantification and online/offline capabilities.

**Key Features:**

- Advanced statistical methods (MMD, LSDD, Chi-Square)
- Online and offline drift detection modes
- Deep learning-based detectors
- Uncertainty quantification
- TensorFlow/PyTorch backend support

**Implementation Example:**

```python
from alibi_detect.cd import KSDrift, MMDDrift
from drift_benchmark.detectors.base import BaseDetector

class AlibiDetectDetector(BaseDetector):
    detector = ("kolmogorov_smirnov", "ks_batch")

    def __init__(self, p_val=0.05, alternative='two-sided'):
        self.p_val = p_val
        self.alternative = alternative
        self.detector_instance = None

    def fit(self, X_ref, y_ref=None):
        # Initialize Alibi detector with reference data
        self.detector_instance = KSDrift(
            x_ref=X_ref,
            p_val=self.p_val,
            alternative=self.alternative
        )
        return self

    def detect(self, X_test, y_test=None):
        # Run drift detection
        result = self.detector_instance.predict(X_test)
        return bool(result['data']['is_drift'])

    def score(self, X_test, y_test=None):
        # Return p-value or distance metric
        result = self.detector_instance.predict(X_test)
        return float(result['data']['p_val'])
```

**Available Detectors:**

- **KSDrift**: Kolmogorov-Smirnov test with multi-dimensional support
- **MMDDrift**: Maximum Mean Discrepancy with learned kernels
- **LSDDDrift**: Least-Squares Density Difference estimation
- **Chi2Drift**: Chi-squared test for categorical data
- **TabularDrift**: Comprehensive tabular data drift detection
- **ClassifierDrift**: Classifier-based drift detection

**Advanced Features:**

- **Online Detection**: Incremental processing with `update_ref()`
- **Preprocessing**: Built-in PCA, UAE (Untrained AutoEncoder)
- **Backend Flexibility**: NumPy, TensorFlow, PyTorch support
- **Uncertainty**: Confidence intervals and p-value corrections

### Frouros Framework

**Overview:**
Frouros specializes in concept drift detection with streaming algorithms and statistical methods.

**Key Features:**

- Comprehensive streaming drift detection
- Statistical tests and distance measures
- Change point detection algorithms
- Memory-efficient online processing

**Implementation Approach:**

```python
from frouros.detectors.concept_drift import DDM, EDDM, ADWIN
from drift_benchmark.detectors.base import BaseDetector

class FrourosDetector(BaseDetector):
    detector = ("drift_detection_method", "ddm_streaming")

    def __init__(self, warning_level=2.0, out_control_level=3.0):
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.detector_instance = DDM(
            warning_level=warning_level,
            out_control_level=out_control_level
        )

    def fit(self, X_ref, y_ref=None):
        # Streaming detectors often don't require explicit fitting
        return self

    def detect(self, X_test, y_test=None):
        # Process each sample and check for drift
        for sample in X_test:
            # Assuming binary classification or error rate input
            error = self._compute_error(sample)
            status = self.detector_instance.update(error)
            if status.drift:
                return True
        return False
```

### River Framework

**Overview:**
River provides online machine learning algorithms including drift detection for streaming data.

**Key Features:**

- Pure streaming/online algorithms
- Integration with online ML models
- Efficient memory usage
- Real-time processing capabilities

### Menelaus Framework

**Overview:**
Menelaus focuses on data drift detection with statistical tests and distance-based methods.

**Implementation Focus:**

- Statistical hypothesis testing
- Distribution-free methods
- Multivariate drift detection
- Batch processing optimization
