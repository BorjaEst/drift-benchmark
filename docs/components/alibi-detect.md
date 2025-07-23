# Alibi-Detect Drift Detection Methods

> Library-specific implementation guide for Alibi-Detect drift detection methods

## 🎯 Overview

This document outlines the drift detection methods available in the Alibi-Detect library. For framework concepts, adapter development patterns, and complete implementation guides, see [Adapter API Documentation](_adapter_api.md).

**Use this guide when**: Creating Alibi-Detect adapters or understanding Alibi-Detect's specific drift detection capabilities.

## 📋 Available Drift Detection Methods

### Batch Detection Methods

| Method                                           | Data Types                | Backend Support              | Drift Score                                                | Key Parameters                            |
| ------------------------------------------------ | ------------------------- | ---------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| `chi_square` (Chi-Squared)                       | Categorical only          | NumPy                        | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `categories_per_feature`         |
| `kolmogorov_smirnov` (KS Test)                   | Continuous only           | NumPy                        | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`                                   |
| `cramer_von_mises` (CvM Test)                    | Continuous only           | NumPy                        | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`                                   |
| `fisher_exact` (Fisher's Exact)                  | Binary data only          | NumPy                        | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `alternative`                    |
| `maximum_mean_discrepancy` (MMD)                 | Continuous (multivariate) | TensorFlow, PyTorch, KeOps   | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `kernel`, `backend`              |
| `lsdd` (Least-Squares Density Difference)        | Continuous (multivariate) | TensorFlow, PyTorch          | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `backend`                        |
| `classifier_drift` (Classifier-based)            | Any data type             | TensorFlow, PyTorch, sklearn | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `model`, `backend`, `preds_type` |
| `learned_kernel_drift` (Learned Kernel)          | Continuous only           | TensorFlow, PyTorch          | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `kernel`, `backend`              |
| `context_aware_drift` (Context-Aware MMD)        | Continuous + context      | TensorFlow, PyTorch          | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `backend`                        |
| `spot_the_diff` (Interpretable Classifier)       | Any data type             | TensorFlow, PyTorch          | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `kernel`, `n_diffs`              |
| `tabular_drift` (Mixed Data)                     | Numerical & categorical   | NumPy                        | `p_value`; Drift when `p_value < threshold`; Default: 0.05 | `p_val`, `categories_per_feature`         |
| `classifier_uncertainty` (Model Uncertainty)     | Model predictions         | TensorFlow, PyTorch          | Uncertainty score; Custom threshold                        | `uncertainty_type`, `model`               |
| `regressor_uncertainty` (Regression Uncertainty) | Model predictions         | TensorFlow, PyTorch          | Uncertainty score; Custom threshold                        | `uncertainty_type`, `model`               |

### Online Detection Methods

| Method                               | Data Types                | Backend Support     | Drift Score                                  | Key Parameters                  |
| ------------------------------------ | ------------------------- | ------------------- | -------------------------------------------- | ------------------------------- |
| `mmd_online` (Online MMD)            | Continuous (multivariate) | TensorFlow, PyTorch | Test statistic; Drift when exceeds threshold | `ert`, `window_size`, `backend` |
| `lsdd_online` (Online LSDD)          | Continuous (multivariate) | TensorFlow, PyTorch | Test statistic; Drift when exceeds threshold | `ert`, `window_size`, `backend` |
| `cvm_online` (Online CvM)            | Continuous only           | NumPy               | Test statistic; Drift when exceeds threshold | `ert`, `window_sizes`           |
| `fet_online` (Online Fisher's Exact) | Binary data only          | NumPy               | Test statistic; Drift when exceeds threshold | `ert`, `window_sizes`           |

## 🔧 Integration Patterns

### Basic Alibi-Detect Adapter Structure

```python
from drift_benchmark.adapters import BaseDetector, register_detector
import numpy as np

@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi-detect")
class AlibiDetectKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.p_val = kwargs.get('p_val', 0.05)
        self._detector = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Alibi-Detect works with numpy arrays."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Convert to numeric and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values.astype(np.float32)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        from alibi_detect.cd import KSDrift

        self._detector = KSDrift(preprocessed_data, p_val=self.p_val)
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        result = self._detector.predict(preprocessed_data)

        self._last_score = result['data']['p_val']
        return result['data']['is_drift']
```

### Method-Specific Configurations

```python
# MMD with TensorFlow backend
@register_detector(method_id="maximum_mean_discrepancy", variant_id="batch", library_id="alibi-detect-tf")
class AlibiDetectMMDTensorFlow(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kwargs.get('kernel', 'rbf')
        self.backend = 'tensorflow'

# MMD with PyTorch backend
@register_detector(method_id="maximum_mean_discrepancy", variant_id="batch", library_id="alibi-detect-pytorch")
class AlibiDetectMMDPyTorch(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kwargs.get('kernel', 'rbf')
        self.backend = 'pytorch'

# Online MMD detector
@register_detector(method_id="maximum_mean_discrepancy", variant_id="online", library_id="alibi-detect")
class AlibiDetectMMDOnline(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ert = kwargs.get('ert', 25)
        self.window_size = kwargs.get('window_size', 20)
```

## 📚 Library-Specific Features

### Strengths

- **Advanced Methods**: State-of-the-art learned and multivariate methods
- **Backend Flexibility**: Support for TensorFlow, PyTorch, and NumPy backends
- **Online Detection**: Real-time streaming drift detection capabilities
- **Model-Agnostic**: Works with any machine learning model
- **Research-Grade**: Implementation of latest academic research

### Limitations

- **Complexity**: More complex setup and configuration
- **Dependencies**: Heavy deep learning dependencies for advanced methods
- **Learning Curve**: Requires understanding of statistical concepts

### Recommended Use Cases

- **Production Systems**: When you need robust online drift detection
- **Advanced Analytics**: Complex multivariate drift detection scenarios
- **Research Projects**: Experimenting with cutting-edge drift detection methods
- **Deep Learning**: When working with neural network models
- **High-Stakes Applications**: Where advanced statistical rigor is required

## 🔄 Implementation Examples

For complete implementation examples, testing patterns, and comparative benchmarking, refer to the [Adapter API Documentation](_adapter_api.md).

## 📖 References

- [Alibi-Detect Documentation](https://docs.seldon.io/projects/alibi-detect/)
- [GitHub Repository](https://github.com/SeldonIO/alibi-detect)
- [Academic Papers](https://docs.seldon.io/projects/alibi-detect/en/stable/overview/algorithms.html)
