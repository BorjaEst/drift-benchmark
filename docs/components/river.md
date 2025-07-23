# River Drift Detection Methods

> Library-specific implementation guide for River drift detection methods

## 🎯 Overview

This document outlines the drift detection methods available in the River library. For framework concepts, adapter development patterns, and complete implementation guides, see [Adapter API Documentation](_adapter_api.md).

**Use this guide when**: Creating River adapters or understanding River's online drift detection capabilities for streaming data scenarios.

## 📚 River Library Concepts

### Online Processing Philosophy

River is designed for online machine learning, processing data streams one element at a time. This approach differs fundamentally from batch processing and offers unique advantages for drift detection:

- **Reactive and Proactive Streams**: Handle both user-driven events and controlled data feeds
- **Stateful Models**: Continuous learning without revisiting past data
- **Dictionary-Based**: Native Python data structures for optimal performance
- **Stream Evaluation**: Reproduce production scenarios with high fidelity

### Concept Drift in Streaming Context

River addresses concept drift as a core challenge in online machine learning:

- **Virtual Drift**: Changes in feature distribution P(X) while P(Y|X) remains constant
- **Real Drift**: Changes in the joint probability P(X,Y), affecting the target relationship
- **Gradual vs Abrupt**: Different temporal patterns of drift occurrence
- **Adaptive Methods**: Built-in robustness against concept drift

## 📋 Available Drift Detection Methods

### Online Detection Methods

| Method                                          | Data Types                   | Execution Mode | Drift Score                                  | Key Parameters                                      |
| ----------------------------------------------- | ---------------------------- | -------------- | -------------------------------------------- | --------------------------------------------------- |
| `ADWIN` (Adaptive Windowing)                    | Univariate continuous        | **Streaming**  | Test statistic; Drift when exceeds threshold | `delta` (confidence level)                          |
| `DDM` (Drift Detection Method)                  | Binary classification errors | **Streaming**  | Warning/Drift levels; Boolean signals        | `warning_level`, `drift_level`                      |
| `EDDM` (Early Drift Detection Method)           | Binary classification errors | **Streaming**  | Warning/Drift levels; Boolean signals        | `alpha`, `beta`                                     |
| `HDDM_A` (Hoeffding Drift Detection - Averages) | Univariate continuous        | **Streaming**  | Test statistic; Boolean drift signal         | `drift_confidence`, `warning_confidence`            |
| `HDDM_W` (Hoeffding Drift Detection - Weighted) | Univariate continuous        | **Streaming**  | Test statistic; Boolean drift signal         | `drift_confidence`, `warning_confidence`, `lambda_` |
| `PageHinkley` (Page-Hinkley Test)               | Univariate continuous        | **Streaming**  | Cumulative sum; Boolean drift signal         | `min_instances`, `delta`, `threshold`, `alpha`      |
| `KSWIN` (Kolmogorov-Smirnov Windowing)          | Univariate continuous        | **Streaming**  | KS test p-value; Boolean drift signal        | `alpha`, `window_size`, `stat_size`                 |

### Performance Monitoring Methods

| Method                      | Data Types                    | Execution Mode | Drift Score                    | Key Parameters                 |
| --------------------------- | ----------------------------- | -------------- | ------------------------------ | ------------------------------ |
| `binary.DDM` (Binary DDM)   | Binary classification metrics | **Streaming**  | Performance-based drift signal | `warning_level`, `drift_level` |
| `binary.EDDM` (Binary EDDM) | Binary classification metrics | **Streaming**  | Performance-based drift signal | `alpha`, `beta`                |

## 🔧 Integration Patterns

### Basic River Adapter Structure

```python
from drift_benchmark.adapters import BaseDetector, register_detector
import numpy as np

@register_detector(method_id="adwin", variant_id="streaming", library_id="river")
class RiverADWINDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.delta = kwargs.get('delta', 0.002)
        self._detector = None
        self._drift_detected = False

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """River works with individual samples in streaming fashion."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Convert to numpy array for streaming simulation
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values.flatten()

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        from river import drift

        self._detector = drift.ADWIN(delta=self.delta)

        # Initialize detector with reference data
        for value in preprocessed_data:
            self._detector.update(value)

        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Process streaming data and detect drift."""
        drift_detected = False

        for value in preprocessed_data:
            self._detector.update(value)
            if self._detector.drift_detected:
                drift_detected = True
                break

        self._drift_detected = drift_detected
        return drift_detected

    def score(self) -> Optional[float]:
        """River detectors typically return boolean signals."""
        return float(self._drift_detected) if hasattr(self, '_drift_detected') else None
```

### Method-Specific Configurations

```python
# ADWIN for continuous streaming data
@register_detector(method_id="adwin", variant_id="streaming", library_id="river")
class RiverADWINDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delta = kwargs.get('delta', 0.002)

# DDM for classification performance monitoring
@register_detector(method_id="ddm", variant_id="streaming", library_id="river")
class RiverDDMDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warning_level = kwargs.get('warning_level', 2.0)
        self.drift_level = kwargs.get('drift_level', 3.0)

# Page-Hinkley for change point detection
@register_detector(method_id="page_hinkley", variant_id="streaming", library_id="river")
class RiverPageHinkleyDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min_instances = kwargs.get('min_instances', 30)
        self.delta = kwargs.get('delta', 0.005)
        self.threshold = kwargs.get('threshold', 50)

# KSWIN for distribution change detection
@register_detector(method_id="kswin", variant_id="streaming", library_id="river")
class RiverKSWINDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alpha = kwargs.get('alpha', 0.005)
        self.window_size = kwargs.get('window_size', 100)
        self.stat_size = kwargs.get('stat_size', 30)
```

### Streaming Data Simulation

```python
class RiverStreamingAdapter(BaseDetector):
    """Base class for River streaming adapters."""

    def _simulate_stream(self, data: np.ndarray) -> Iterator[float]:
        """Convert batch data to streaming format."""
        for value in data:
            yield float(value)

    def _process_multivariate_stream(self, data: np.ndarray) -> bool:
        """Handle multivariate data by processing each feature separately."""
        drift_signals = []

        for feature_idx in range(data.shape[1] if len(data.shape) > 1 else 1):
            feature_data = data[:, feature_idx] if len(data.shape) > 1 else data

            feature_drift = False
            for value in self._simulate_stream(feature_data):
                self._detector.update(value)
                if self._detector.drift_detected:
                    feature_drift = True
                    break

            drift_signals.append(feature_drift)

        # Return True if any feature shows drift
        return any(drift_signals)
```

## 📚 Library-Specific Features

### Strengths

- **True Online Processing**: Genuine one-sample-at-a-time processing
- **Adaptive Algorithms**: Built-in robustness against concept drift
- **Performance Optimized**: Native Python structures for streaming efficiency
- **Production Ready**: Designed for real-time reactive data streams
- **Memory Efficient**: Constant memory usage regardless of stream length
- **Interpretable**: Clear drift signals and warning mechanisms

### Limitations

- **Univariate Focus**: Most methods work on single features
- **Streaming Only**: Not optimized for batch processing scenarios
- **Learning Curve**: Requires understanding of online learning concepts
- **Limited Deep Learning**: No integration with neural network backends

### Recommended Use Cases

- **Real-Time Systems**: Production environments with continuous data streams
- **Resource-Constrained Environments**: Limited memory and computational resources
- **Performance Monitoring**: Continuous model performance tracking
- **IoT and Sensor Data**: High-frequency streaming data scenarios
- **Online Learning**: When models need to adapt continuously to new data

### River vs Batch Libraries Comparison

| Aspect              | River                     | Alibi-Detect          | Evidently           |
| ------------------- | ------------------------- | --------------------- | ------------------- |
| **Processing Mode** | True streaming            | Batch/Window          | Batch               |
| **Memory Usage**    | Constant                  | Varies with data size | High for reports    |
| **Latency**         | Real-time                 | Batch delays          | Report generation   |
| **Drift Types**     | Performance, distribution | Statistical, learned  | Statistical         |
| **Use Case**        | Production streaming      | Research, validation  | Analysis, reporting |

## 🔄 Implementation Examples

### Complete ADWIN Streaming Example

```python
@register_detector(method_id="adwin", variant_id="streaming", library_id="river")
class RiverADWINStreamingDetector(BaseDetector):
    """Complete implementation of ADWIN for streaming drift detection."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.delta = kwargs.get('delta', 0.002)
        self._detectors = {}  # One detector per feature
        self._drift_history = []

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Handle both univariate and multivariate data
        numeric_data = df.select_dtypes(include=[np.number])
        processed = numeric_data.fillna(numeric_data.mean()).values

        # Ensure 2D array for consistent processing
        if len(processed.shape) == 1:
            processed = processed.reshape(-1, 1)

        return processed

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        from river import drift

        # Initialize one ADWIN detector per feature
        n_features = preprocessed_data.shape[1]
        for feature_idx in range(n_features):
            self._detectors[feature_idx] = drift.ADWIN(delta=self.delta)

            # Train on reference data
            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                self._detectors[feature_idx].update(value)

        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Detect drift by processing test data as a stream."""
        overall_drift = False

        for feature_idx in range(preprocessed_data.shape[1]):
            feature_drift_detected = False

            for sample_idx in range(preprocessed_data.shape[0]):
                value = preprocessed_data[sample_idx, feature_idx]
                self._detectors[feature_idx].update(value)

                if self._detectors[feature_idx].drift_detected:
                    feature_drift_detected = True
                    self._drift_history.append({
                        'feature': feature_idx,
                        'sample': sample_idx,
                        'value': value
                    })
                    break

            if feature_drift_detected:
                overall_drift = True

        return overall_drift

    def score(self) -> Optional[float]:
        """Return proportion of features that detected drift."""
        if not self._detectors:
            return None

        drift_count = sum(1 for detector in self._detectors.values()
                         if detector.drift_detected)
        return drift_count / len(self._detectors)
```

For complete implementation examples, testing patterns, and comparative benchmarking, refer to the [Adapter API Documentation](_adapter_api.md).

## 📖 References

- [River Documentation](https://riverml.xyz/)
- [River Drift Detection Module](https://riverml.xyz/0.21.0/api/drift/)
- [GitHub Repository](https://github.com/online-ml/river)
- [Concept Drift in Online Learning](https://riverml.xyz/0.21.0/examples/concept-drift/)
- [Online Learning Fundamentals](https://riverml.xyz/0.21.0/introduction/why-river/)
