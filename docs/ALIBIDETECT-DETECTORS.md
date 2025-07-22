# Drift Detection Benchmark Adapter Development Guide

> A comprehensive guide for building drift detection benchmark adapters based on the drift-benchmark framework and Alibi-Detect methods

## 🎯 Overview

This guide consolidates information from the drift-benchmark framework and Alibi-Detect documentation to help developers create standardized adapters for comparing drift detection implementations across different libraries (Evidently, Alibi-Detect, scikit-learn, etc.).

## 🏗️ Framework Architecture

### Core Concepts

- **🔬 Method**: Mathematical methodology for drift detection (e.g., Kolmogorov-Smirnov Test, MMD)
- **⚙️ Variant**: Standardized algorithmic approach defined by drift-benchmark (e.g., batch, incremental, sliding window)
- **🔌 Detector**: How a specific library implements a method+variant combination
- **🔄 Adapter**: Your custom class that maps a library's implementation to our standardized interface

### Framework Roles

**drift-benchmark defines standardized variants** - libraries don't define variants themselves. Users create adapters that map their library's implementation to match our variant specifications.

## 📚 Drift Detection Methods Overview

### Types of Drift

1. **Covariate Drift** (Input Drift): $P(\mathbf{X}) \ne P_{ref}(\mathbf{X})$ while $P(\mathbf{Y}|\mathbf{X}) = P_{ref}(\mathbf{Y}|\mathbf{X})$
2. **Prior Drift** (Label Drift): $P(\mathbf{Y}) \ne P_{ref}(\mathbf{Y})$ while $P(\mathbf{X}|\mathbf{Y}) = P_{ref}(\mathbf{X}|\mathbf{Y})$
3. **Concept Drift**: $P(\mathbf{Y}|\mathbf{X}) \ne P_{ref}(\mathbf{Y}|\mathbf{X})$

### Statistical Test Categories

#### Univariate Tests (Feature-wise)

- **Chi-Squared Test**: For categorical data
- **Kolmogorov-Smirnov Test**: For continuous distributions
- **Cramér-von Mises Test**: Alternative to K-S, more sensitive to variance changes
- **Fisher's Exact Test**: For binary/Bernoulli data

#### Multivariate Tests

- **Maximum Mean Discrepancy (MMD)**: Kernel-based method using RKHS
- **Least-Squares Density Difference (LSDD)**: Direct density difference estimation

#### Advanced/Learned Methods

- **Classifier-based**: Train classifier to discriminate reference vs test data
- **Learned Kernel**: Extend MMD with trainable kernels
- **Context-Aware MMD**: Consider context variables
- **Spot-the-diff**: Interpretable classifier-based detection
- **Model Uncertainty**: Monitor model prediction uncertainty

## 🔧 Available Drift Detection Methods

### 1. Chi-Squared Drift

- **Purpose**: Categorical feature drift detection
- **Test Statistic**: Feature-wise Chi-Squared tests
- **Data Type**: Categorical
- **Key Parameters**: `p_val`, `categories_per_feature`
- **Aggregation**: Bonferroni or FDR correction

### 2. Kolmogorov-Smirnov Drift

- **Purpose**: Continuous distribution drift detection
- **Test Statistic**: Maximum distance between empirical CDFs
- **Data Type**: Continuous
- **Key Parameters**: `p_val`
- **Preprocessing**: Often requires dimensionality reduction

### 3. Cramér-von Mises Drift

- **Purpose**: Alternative to K-S, better for variance changes
- **Test Statistic**: $W = \sum_{z\in k} |F(z) - F_{ref}(z)|^2$
- **Data Type**: Continuous
- **Key Parameters**: `p_val`

### 4. Fisher's Exact Test Drift

- **Purpose**: Binary/Bernoulli data drift
- **Test Statistic**: Contingency table analysis
- **Data Type**: Binary (0/1, True/False)
- **Key Parameters**: `p_val`, `alternative`
- **Use Case**: Model accuracy monitoring

### 5. Maximum Mean Discrepancy (MMD) Drift

- **Purpose**: Multivariate drift detection
- **Test Statistic**: $MMD(F, p, q) = ||\mu_p - \mu_q||^2_F$
- **Data Type**: Continuous (multivariate)
- **Key Parameters**: `p_val`, `kernel`, `backend`
- **Backends**: TensorFlow, PyTorch, KeOps

### 6. Least-Squares Density Difference (LSDD) Drift

- **Purpose**: Multivariate density difference detection
- **Test Statistic**: $LSDD(p,q) = \int_{\mathcal{X}} (p(x)-q(x))^2 dx$
- **Data Type**: Continuous (requires probability density)
- **Key Parameters**: `p_val`, `backend`
- **Backends**: TensorFlow, PyTorch

### 7. Classifier-based Drift

- **Purpose**: Learned drift detection
- **Method**: Train classifier to distinguish reference vs test data
- **Data Type**: Any (depends on model)
- **Key Parameters**: `p_val`, `model`, `backend`, `preds_type`
- **Backends**: TensorFlow, PyTorch, sklearn

### 8. Learned Kernel Drift

- **Purpose**: MMD with optimized kernel
- **Method**: Train kernel to maximize test power
- **Data Type**: Continuous
- **Key Parameters**: `p_val`, `kernel`, `backend`
- **Training Required**: Yes

### 9. Context-Aware MMD Drift

- **Purpose**: MMD considering context variables
- **Method**: Conditional mean embeddings
- **Data Type**: Continuous + context
- **Key Parameters**: `p_val`, `backend`
- **Special**: Requires context data

### 10. Spot-the-diff Drift

- **Purpose**: Interpretable classifier-based detection
- **Method**: Classifier with interpretable structure
- **Data Type**: Any
- **Key Parameters**: `p_val`, `kernel`, `n_diffs`
- **Output**: Feature-level interpretability

### 11. Tabular Drift

- **Purpose**: Mixed-type tabular data
- **Method**: K-S for continuous, Chi-Squared for categorical
- **Data Type**: Mixed (continuous + categorical)
- **Key Parameters**: `p_val`, `categories_per_feature`

### 12. Model Uncertainty Drift

- **Purpose**: Model performance-focused detection
- **Method**: Monitor model prediction uncertainty
- **Data Type**: Model predictions
- **Types**:
  - **ClassifierUncertaintyDrift**: Entropy or margin-based
  - **RegressorUncertaintyDrift**: MC dropout or ensemble

## 🔄 Online Detection Methods

For sequential/streaming data:

### Online MMD Drift

- **Method**: Incremental MMD computation
- **Key Parameters**: `ert`, `window_size`
- **Configuration**: Bootstrap simulation for thresholds

### Online LSDD Drift

- **Method**: Incremental LSDD computation
- **Key Parameters**: `ert`, `window_size`

### Online Cramér-von Mises Drift

- **Method**: Incremental CVM computation
- **Key Parameters**: `ert`, `window_sizes`

### Online Fisher's Exact Test Drift

- **Method**: Incremental FET for binary data
- **Key Parameters**: `ert`, `window_sizes`

## 🛠️ Adapter Development Framework

### BaseDetector Interface

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from typing import Any, Dict, Optional
import pandas as pd

@register_detector(method_id="method_name", variant_id="variant_name", library_id="library_name")
class YourDetector(BaseDetector):
    """Your library's implementation of method+variant."""

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        # Initialize library-specific parameters
        self.threshold = kwargs.get('threshold', 0.05)
        # Store other hyperparameters

    def preprocess(self, data: pd.DataFrame, **kwargs) -> Any:
        """Convert data to library-specific format."""
        # Transform pandas DataFrame to library's expected format
        # Handle categorical encoding, normalization, etc.
        return processed_data

    def fit(self, preprocessed_data: Any, **kwargs) -> 'YourDetector':
        """Fit/initialize the detector on reference data."""
        # Initialize library's drift detector
        # Store reference data or trained models
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Perform drift detection on test data."""
        # Run library's drift detection
        # Return boolean drift detection result
        return drift_detected

    def get_drift_score(self, preprocessed_data: Any, **kwargs) -> Optional[float]:
        """Get continuous drift score if available."""
        # Return p-value, test statistic, or confidence score
        # Return None if not available
        return score
```

### Common Preprocessing Patterns

#### Data Format Conversion

```python
def preprocess(self, data: pd.DataFrame, **kwargs) -> Any:
    """Convert pandas DataFrame to numpy array."""
    if isinstance(data, pd.DataFrame):
        return data.values
    return data
```

#### Categorical Encoding

```python
def preprocess(self, data: pd.DataFrame, **kwargs) -> Any:
    """Handle categorical features."""
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        # Apply label encoding or one-hot encoding
        processed = data.copy()
        for col in categorical_cols:
            processed[col] = pd.Categorical(processed[col]).codes
        return processed.values
    return data.values
```

#### Dimensionality Reduction

```python
def preprocess(self, data: pd.DataFrame, **kwargs) -> Any:
    """Apply PCA for high-dimensional data."""
    if hasattr(self, '_pca') and self._pca is not None:
        return self._pca.transform(data.values)
    return data.values
```

## 📋 Method+Variant Examples

### Kolmogorov-Smirnov Batch Implementation

```python
# Evidently implementation
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
class EvidentlyKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.05)

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return data

    def fit(self, preprocessed_data: pd.DataFrame, **kwargs):
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: pd.DataFrame, **kwargs) -> bool:
        # Evidently's specific KS implementation
        # Use DataDriftPreset or specific metrics
        return drift_detected

# Alibi-Detect implementation
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi_detect")
class AlibiDetectKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.05)

    def preprocess(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        return data.values

    def fit(self, preprocessed_data: np.ndarray, **kwargs):
        from alibi_detect.cd import KSDrift
        self._detector = KSDrift(preprocessed_data, p_val=self.threshold)
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']
```

### MMD Implementations

```python
# Alibi-Detect MMD with TensorFlow backend
@register_detector(method_id="maximum_mean_discrepancy", variant_id="batch", library_id="alibi_detect_tf")
class AlibiDetectMMDTF(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.p_val = kwargs.get('p_val', 0.05)
        self.backend = 'tensorflow'

    def fit(self, preprocessed_data: np.ndarray, **kwargs):
        from alibi_detect.cd import MMDDrift
        self._detector = MMDDrift(
            preprocessed_data,
            backend=self.backend,
            p_val=self.p_val
        )
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']

# Alibi-Detect MMD with PyTorch backend
@register_detector(method_id="maximum_mean_discrepancy", variant_id="batch", library_id="alibi_detect_pytorch")
class AlibiDetectMMDPyTorch(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.p_val = kwargs.get('p_val', 0.05)
        self.backend = 'pytorch'

    def fit(self, preprocessed_data: np.ndarray, **kwargs):
        from alibi_detect.cd import MMDDrift
        self._detector = MMDDrift(
            preprocessed_data,
            backend=self.backend,
            p_val=self.p_val
        )
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']
```

### Online Detection Example

```python
@register_detector(method_id="maximum_mean_discrepancy", variant_id="online", library_id="alibi_detect")
class AlibiDetectMMDOnline(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.ert = kwargs.get('ert', 150)
        self.window_size = kwargs.get('window_size', 20)

    def fit(self, preprocessed_data: np.ndarray, **kwargs):
        from alibi_detect.cd import MMDDriftOnline
        self._detector = MMDDriftOnline(
            preprocessed_data,
            ert=self.ert,
            window_size=self.window_size,
            backend='tensorflow'
        )
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        # Online detectors work on single instances
        # preprocessed_data should be a single sample
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']
```

## ⚙️ Configuration Examples

### methods.toml Structure

```toml
[methods.kolmogorov_smirnov]
name = "Kolmogorov-Smirnov Test"
description = "Two-sample test for equality of continuous distributions"
drift_types = ["COVARIATE"]
family = "STATISTICAL_TEST"
data_dimension = "UNIVARIATE"
data_types = ["CONTINUOUS"]
requires_labels = false
references = ["https://doi.org/10.2307/2281868"]

[methods.kolmogorov_smirnov.variants.batch]
name = "Batch Processing"
execution_mode = "BATCH"
hyperparameters = ["threshold", "correction_method"]

[methods.kolmogorov_smirnov.variants.online]
name = "Online Processing"
execution_mode = "ONLINE"
hyperparameters = ["ert", "window_size"]
```

### Benchmark Configuration

```toml
[[datasets]]
path = "datasets/example.csv"
format = "CSV"
reference_split = 0.5

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "alibi_detect"
threshold = 0.05

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "batch"
library_id = "alibi_detect_tf"
p_val = 0.05

[[detectors]]
method_id = "maximum_mean_discrepancy"
variant_id = "batch"
library_id = "alibi_detect_pytorch"
p_val = 0.05
```

## 🎯 Best Practices

### 1. Consistent Parameter Mapping

- Map library-specific parameters to standardized names
- Use kwargs for flexibility
- Provide sensible defaults

### 2. Robust Preprocessing

- Handle different data types gracefully
- Implement proper categorical encoding
- Consider dimensionality reduction needs

### 3. Error Handling

```python
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    try:
        result = self._detector.predict(preprocessed_data)
        return result['data']['is_drift']
    except Exception as e:
        # Log error for debugging
        self.logger.error(f"Detection failed: {e}")
        raise
```

### 4. Performance Considerations

- Cache preprocessing steps when possible
- Use appropriate backends (CPU/GPU)
- Consider memory usage for large datasets

### 5. Testing Strategy

```python
def test_detector():
    # Create synthetic drift scenario
    X_ref = np.random.normal(0, 1, (100, 5))
    X_test = np.random.normal(1, 1, (100, 5))  # Shifted mean

    detector = YourDetector("method_id", "variant_id")
    detector.fit(detector.preprocess(pd.DataFrame(X_ref)))

    drift_detected = detector.detect(detector.preprocess(pd.DataFrame(X_test)))
    assert drift_detected == True  # Should detect drift
```

## 📊 Comparison Framework

### Library Performance Metrics

- **Execution Time**: Time for fit() and detect() operations
- **Memory Usage**: Peak memory consumption
- **Accuracy**: True positive/negative rates
- **Robustness**: Performance across different data types

### Expected Benchmarking Outcomes

- **Speed Comparison**: Which library implements KS test faster?
- **Accuracy Comparison**: Which MMD implementation has better power?
- **Backend Comparison**: TensorFlow vs PyTorch performance
- **Method Comparison**: MMD vs LSDD effectiveness

## 🔍 Implementation Checklist

### Required Methods

- [ ] `__init__()` with proper parameter handling
- [ ] `preprocess()` for data format conversion
- [ ] `fit()` for detector initialization
- [ ] `detect()` for drift detection
- [ ] `get_drift_score()` for continuous scores (optional)

### Registration

- [ ] Proper `@register_detector` decorator
- [ ] Correct method_id, variant_id, library_id
- [ ] Verify method exists in methods.toml

### Testing

- [ ] Unit tests for each method
- [ ] Integration tests with benchmark framework
- [ ] Performance benchmarks
- [ ] Edge case handling

### Documentation

- [ ] Clear docstrings
- [ ] Parameter descriptions
- [ ] Usage examples
- [ ] Library-specific requirements

## 🚀 Getting Started

1. **Choose a drift detection method** from the available options
2. **Identify libraries** that implement this method
3. **Create adapter classes** for each library implementation
4. **Register adapters** using the decorator system
5. **Configure benchmarks** to compare implementations
6. **Run comparisons** and analyze results

This guide provides the foundation for creating comprehensive drift detection benchmarks that enable fair comparison of different library implementations of the same mathematical methods.
