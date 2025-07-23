# Evidently AI Drift Detection Methods

> Library-specific implementation guide for Evidently AI drift detection methods

## 🎯 Overview

This document outlines the drift detection methods available in the Evidently AI library. For framework concepts, adapter development patterns, and complete implementation guides, see [Adapter API Documentation](_adapter_api.md).

**Use this guide when**: Creating Evidently AI adapters or understanding Evidently's specific drift detection capabilities.

## 📋 Available Drift Detection Methods

The following methods are available for tabular data in Evidently AI. Configure them using the `stattest` (or `num_stattest`, `cat_stattest`) parameter.

| Method                                 | Data Types              | Default Usage                                       | Drift Score                                                     | Key Parameters |
| -------------------------------------- | ----------------------- | --------------------------------------------------- | --------------------------------------------------------------- | -------------- |
| `ks` (Kolmogorov-Smirnov)              | Numerical only          | **Default for numerical ≤1000 objects**             | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `chisquare` (Chi-Square)               | Categorical only        | **Default for categorical >2 labels ≤1000 objects** | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `z` (Z-test)                           | Categorical only        | **Default for binary data ≤1000 objects**           | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `wasserstein` (Earth Mover's Distance) | Numerical only          | **Default for numerical >1000 objects**             | `distance`; Drift when `distance ≥ threshold`; Default: 0.1     | `threshold`    |
| `kl_div` (Kullback-Leibler)            | Numerical & categorical | Manual selection                                    | `divergence`; Drift when `divergence ≥ threshold`; Default: 0.1 | `threshold`    |
| `psi` (Population Stability Index)     | Numerical & categorical | Manual selection                                    | `psi_value`; Drift when `psi_value ≥ threshold`; Default: 0.1   | `threshold`    |
| `jensenshannon` (Jensen-Shannon)       | Numerical & categorical | **Default for categorical >1000 objects**           | `distance`; Drift when `distance ≥ threshold`; Default: 0.1     | `threshold`    |
| `anderson` (Anderson-Darling)          | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `fisher_exact` (Fisher's Exact)        | Categorical only        | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `cramer_von_mises` (Cramér-von Mises)  | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `g-test` (G-test)                      | Categorical only        | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `hellinger` (Hellinger Distance)       | Numerical & categorical | Manual selection                                    | `distance`; Drift when `distance ≥ threshold`; Default: 0.1     | `threshold`    |
| `mannw` (Mann-Whitney U)               | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `ed` (Energy Distance)                 | Numerical only          | Manual selection                                    | `distance`; Drift when `distance ≥ threshold`; Default: 0.1     | `threshold`    |
| `es` (Epps-Singleton)                  | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `t_test` (T-Test)                      | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `empirical_mmd` (Empirical MMD)        | Numerical only          | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |
| `TVD` (Total Variation Distance)       | Categorical only        | Manual selection                                    | `p_value`; Drift when `p_value < threshold`; Default: 0.05      | `threshold`    |

## 🔧 Integration Patterns

### Basic Evidently Adapter Structure

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from evidently.metrics import DataDriftPreset

@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
class EvidentlyKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.stattest = kwargs.get('stattest', 'ks')
        self.threshold = kwargs.get('threshold', 0.05)

    def preprocess(self, data: DatasetResult, **kwargs) -> pd.DataFrame:
        """Evidently works with pandas DataFrames."""
        phase = kwargs.get('phase', 'train')
        return data.X_ref if phase == 'train' else data.X_test

    def fit(self, preprocessed_data: pd.DataFrame, **kwargs) -> "BaseDetector":
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: pd.DataFrame, **kwargs) -> bool:
        report = DataDriftPreset().run(
            reference_data=self._reference_data,
            current_data=preprocessed_data,
            stattest=self.stattest,
            stattest_threshold=self.threshold
        )

        result = report.metrics[0].result
        self._last_score = getattr(result, 'drift_score', None)
        return result.dataset_drift
```

### Method-Specific Configurations

```python
# Kolmogorov-Smirnov for continuous data
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
class EvidentlyKSDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stattest = 'ks'

# Chi-Square for categorical data
@register_detector(method_id="chi_square", variant_id="batch", library_id="evidently")
class EvidentlyChiSquareDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stattest = 'chisquare'

# Wasserstein for large numerical datasets
@register_detector(method_id="wasserstein", variant_id="batch", library_id="evidently")
class EvidentlyWassersteinDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stattest = 'wasserstein'
```

## 📚 Library-Specific Features

### Strengths

- **Pandas Integration**: Native support for pandas DataFrames
- **Automatic Method Selection**: Smart defaults based on data size and type
- **Rich Reporting**: Comprehensive drift analysis reports
- **Visualization**: Built-in plotting capabilities
- **Mixed Data Types**: Handles numerical and categorical data seamlessly

### Limitations

- **New version under development**: Evidently AI is transitioning to a new version with significant changes
- **Dependencies**: Heavy dependency stack

### Recommended Use Cases

- **Exploratory Analysis**: When you need detailed drift analysis
- **Mixed Data**: Datasets with both numerical and categorical features
- **Reporting**: When visualization and reporting are important
- **Development**: During model development and validation

## 🔄 Implementation Examples

For complete implementation examples, testing patterns, and comparative benchmarking, refer to the [Adapter API Documentation](_adapter_api.md).

## 📖 References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Statistical Tests Reference](https://docs-old.evidentlyai.com/reference/data-drift-algorithm)
- [Method Selection Guide](https://docs-old.evidentlyai.com/user-guide/customization/options-for-statistical-tests#drift-detection-methods-tabular)
