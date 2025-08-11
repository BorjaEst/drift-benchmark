# Adapter API Documentation

> Comprehensive guide to creating custom detector adapters for drift-benchmark

## 🎯 Overview

The Adapter API enables integration of any drift detection library with the drift-benchmark framework through a standardized `BaseDetector` interface. Users create adapter classes that map their preferred libraries' implementations to our unified method+variant definitions.

## 🏗️ Architecture Concepts

### Core Framework Concepts

- **🔬 Method**: Mathematical methodology (e.g., Kolmogorov-Smirnov Test, Maximum Mean Discrepancy)
- **⚙️ Variant**: Standardized algorithmic approach (e.g., batch processing, streaming, sliding window)
- **🔌 Detector**: Library-specific implementation of a method+variant combination
- **🔄 Adapter**: Your custom class extending `BaseDetector` to integrate a library

**Key Insight**: drift-benchmark defines standardized variants, and you create adapters that map your library's implementation to match our variant specifications.

## 🚀 Quick Start

### Basic Adapter Creation

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from scipy import stats
from typing import Any

@register_detector(
    method_id="kolmogorov_smirnov", 
    variant_id="ks_batch", 
    library_id="scipy"
)
class ScipyKSDetector(BaseDetector):
    """SciPy implementation of Kolmogorov-Smirnov batch processing."""
    
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        self._fitted = False
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "ScipyKSDetector":
        """Train detector on reference data."""
        self._reference_data = preprocessed_data
        self._fitted = True
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Detect drift on test data."""
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before detection")
            
        # Perform KS test
        statistic, p_value = stats.ks_2samp(
            self._reference_data, 
            preprocessed_data
        )
        
        # Store drift score for analysis
        self._drift_score = p_value
        
        # Return drift decision
        return p_value < self.threshold
```

### Registration and Usage

```python
# Adapter is automatically registered via decorator
# Now available for benchmarking:

# In benchmark_config.toml:
# [[detectors]]
# method_id = "kolmogorov_smirnov"
# variant_id = "ks_batch" 
# library_id = "scipy"
# threshold = 0.01  # Custom hyperparameter
```

## 📚 BaseDetector Interface

### Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseDetector(ABC):
    """Abstract base class for all drift detectors."""
    
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        """Initialize detector with identifiers and hyperparameters."""
        
    @property
    def method_id(self) -> str:
        """Get the method identifier (read-only)."""
        
    @property  
    def variant_id(self) -> str:
        """Get the variant identifier (read-only)."""
        
    @property
    def library_id(self) -> str:
        """Get the library identifier (read-only)."""
        
    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> Any:
        """Convert scenario data to detector-specific format (concrete method)."""
        
    @abstractmethod
    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """Train detector on reference data (must implement)."""
        
    @abstractmethod
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Detect drift on test data (must implement)."""
        
    def score(self) -> Optional[float]:
        """Get drift confidence score (concrete method)."""
```

### Required Methods (Abstract)

#### `fit(preprocessed_data: Any, **kwargs) -> BaseDetector`

**Purpose**: Train the detector on reference (non-drift) data.

**Parameters**:

- `preprocessed_data`: Data in detector-specific format from `preprocess()`
- `**kwargs`: Additional configuration parameters

**Returns**: Self (for method chaining)

**Requirements**:

- Must store reference data or trained model state
- Must return `self` for fluent interface
- Should set internal fitted state flag
- Must handle detector-specific training requirements

**Example**:

```python
def fit(self, preprocessed_data: Any, **kwargs) -> "MyDetector":
    # Store reference data for statistical tests
    self._reference_data = preprocessed_data
    
    # Or train a model for learned methods
    self._model.fit(preprocessed_data)
    
    # Set fitted state
    self._fitted = True
    
    return self
```

#### `detect(preprocessed_data: Any, **kwargs) -> bool`

**Purpose**: Perform drift detection on test data.

**Parameters**:

- `preprocessed_data`: Test data in detector-specific format
- `**kwargs`: Runtime configuration parameters

**Returns**: Boolean drift detection result (`True` = drift detected, `False` = no drift)

**Requirements**:

- Must return boolean drift decision
- Should store drift score via `self._drift_score` for `score()` method
- Must validate that detector is fitted before detection
- Should handle detector-specific drift detection logic

**Example**:

```python  
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    if not self._fitted:
        raise RuntimeError("Must fit detector before detection")
        
    # Perform drift detection
    drift_score = self._calculate_drift_score(preprocessed_data)
    
    # Store score for analysis
    self._drift_score = drift_score
    
    # Return boolean decision
    return drift_score > self.threshold
```

### Concrete Methods (Provided)

#### `preprocess(data: ScenarioResult, phase: str = "detect", **kwargs) -> Any`

**Purpose**: Convert scenario data to detector-specific format.

**Parameters**:

- `data`: ScenarioResult containing X_ref, y_ref, X_test, y_test
- `phase`: "train" for reference data, "detect" for test data
- `**kwargs`: Additional preprocessing parameters

**Returns**: Data in format expected by your detector library

**Default Behavior**:

- `phase="train"`: Returns `data.X_ref` (or `data.y_ref` if requires_labels=True)
- `phase="detect"`: Returns `data.X_test` (or `data.y_test` if requires_labels=True)
- Handles pandas DataFrame to library-specific format conversion

**Override When**:

- Library requires specific data format (e.g., dictionaries, custom objects)
- Need custom preprocessing logic
- Multiple data sources required

**Example Override**:

```python
def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs) -> dict:
    """Convert to Evidently format."""
    if phase == "train":
        return {
            "reference_data": data.X_ref,
            "reference_target": data.y_ref
        }
    else:
        return {
            "current_data": data.X_test, 
            "current_target": data.y_test
        }
```

#### `score() -> Optional[float]`

**Purpose**: Return drift confidence score from last detection.

**Returns**: Optional drift score (None if no score available)

**Default Behavior**: Returns `self._drift_score` if set by `detect()` method

**Usage**: Enable drift score analysis and comparison across detectors

```python
# In detect() method, store score:
self._drift_score = p_value  # or distance, or probability

# Framework automatically calls score() after detect()
drift_score = detector.score()  # Returns stored p_value
```

## 🔧 Registration System

### Decorator-Based Registration

```python
@register_detector(method_id: str, variant_id: str, library_id: str)
def register_detector(cls) -> cls:
    """Register detector class for method+variant+library combination."""
```

**Parameters**:

- `method_id`: Must exist in methods.toml registry
- `variant_id`: Must exist under specified method in methods.toml
- `library_id`: Unique identifier for your library (e.g., "scipy", "evidently", "custom")

**Validation**:

- Method+variant must be defined in methods.toml
- No duplicate registrations allowed
- Class must extend BaseDetector

**Example**:

```python
@register_detector("kolmogorov_smirnov", "ks_batch", "evidently")
class EvidentlyKSDetector(BaseDetector):
    pass

@register_detector("kolmogorov_smirnov", "ks_batch", "alibi-detect") 
class AlibiKSDetector(BaseDetector):
    pass
```

### Manual Registration

```python
from drift_benchmark.adapters.registry import register_detector_class

# Register without decorator
register_detector_class(
    method_id="kolmogorov_smirnov",
    variant_id="ks_batch", 
    library_id="custom",
    detector_class=CustomKSDetector
)
```

## 📊 Data Flow and Format Handling

### Scenario Data Format

Adapters receive `ScenarioResult` objects containing:

```python
class ScenarioResult(BaseModel):
    """Loaded scenario data."""
    
    X_ref: pd.DataFrame    # Reference features (no drift)
    y_ref: pd.Series       # Reference targets (if available)
    X_test: pd.DataFrame   # Test features (potential drift)
    y_test: pd.Series      # Test targets (if available)
    definition: ScenarioDefinition  # Metadata and configuration
```

### Format Conversion Patterns

#### Pandas to NumPy

```python
def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
    if phase == "train":
        return data.X_ref.values  # Convert to numpy array
    else:
        return data.X_test.values
```

#### Univariate Extraction

```python
def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
    # Extract single column for univariate methods
    column_name = self._kwargs.get('column', data.X_ref.columns[0])
    
    if phase == "train":
        return data.X_ref[column_name].values
    else:
        return data.X_test[column_name].values
```

#### Library-Specific Formats

```python
def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
    """Evidently format example."""
    if phase == "train":
        return {
            "reference_data": data.X_ref,
            "metadata": {"source": "training"}
        }
    else:
        return {
            "current_data": data.X_test,
            "metadata": {"source": "production"}
        }
```

## 🎯 Implementation Patterns

### Statistical Test Pattern

For methods based on statistical hypothesis testing:

```python
@register_detector("cramer_von_mises", "cvm_batch", "scipy")
class ScipyCVMDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "ScipyCVMDetector":
        self._reference_data = preprocessed_data
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        from scipy.stats import cramervonmises_2samp
        
        statistic, p_value = cramervonmises_2samp(
            self._reference_data, 
            preprocessed_data
        )
        
        self._drift_score = p_value
        return p_value < self.threshold
```

### Distance-Based Pattern

For methods based on distribution distance metrics:

```python
@register_detector("wasserstein_distance", "wasserstein_batch", "scipy")
class WassersteinDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.1)
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "WassersteinDetector":
        self._reference_data = preprocessed_data
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        from scipy.stats import wasserstein_distance
        
        distance = wasserstein_distance(
            self._reference_data,
            preprocessed_data
        )
        
        self._drift_score = distance
        return distance > self.threshold
```

### Learned Model Pattern

For machine learning-based drift detection:

```python
@register_detector("classifier_drift", "clf_batch", "sklearn")
class ClassifierDriftDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        from sklearn.ensemble import RandomForestClassifier
        self._classifier = RandomForestClassifier(random_state=42)
        self.threshold = kwargs.get('threshold', 0.5)
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "ClassifierDriftDetector":
        # Train classifier to distinguish reference from drift data
        # This is a simplified example - real implementation would need
        # to generate artificial drift samples for training
        self._reference_data = preprocessed_data
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Create binary classification problem
        # Label reference as 0, test as 1
        import numpy as np
        
        X_combined = np.vstack([self._reference_data, preprocessed_data])
        y_combined = np.hstack([
            np.zeros(len(self._reference_data)), 
            np.ones(len(preprocessed_data))
        ])
        
        # Train classifier
        self._classifier.fit(X_combined, y_combined)
        
        # Predict probabilities
        test_probs = self._classifier.predict_proba(preprocessed_data)[:, 1]
        drift_score = np.mean(test_probs)
        
        self._drift_score = drift_score
        return drift_score > self.threshold
```

### Streaming/Online Pattern

For incremental drift detection methods:

```python
@register_detector("page_hinkley", "ph_standard", "river")
class PageHinkleyDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        from river.drift import PageHinkley
        self._detector = PageHinkley(
            delta=kwargs.get('delta', 0.005),
            lambda_=kwargs.get('lambda', 50)
        )
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "PageHinkleyDetector":
        # Initialize with reference data
        for value in preprocessed_data:
            self._detector.update(value)
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Process test data incrementally
        drift_detected = False
        for value in preprocessed_data:
            self._detector.update(value)
            if self._detector.drift_detected:
                drift_detected = True
                break
                
        self._drift_score = float(drift_detected)
        return drift_detected
```

## 🔍 Methods Registry Integration

### Verifying Method Definitions

Before creating adapters, verify the method+variant exists:

```python
from drift_benchmark.detectors import get_method, get_variant

# Check method exists
try:
    method_info = get_method("kolmogorov_smirnov")
    print("Method found:", method_info['name'])
except MethodNotFoundError:
    print("Method not found in registry")

# Check variant exists  
try:
    variant_info = get_variant("kolmogorov_smirnov", "ks_batch")
    print("Variant found:", variant_info['name'])
except VariantNotFoundError:
    print("Variant not found in registry")
```

### Registry Structure

Methods are defined in `src/drift_benchmark/detectors/methods.toml`:

```toml
[methods.kolmogorov_smirnov]
name = "Kolmogorov-Smirnov Test"
description = "Non-parametric test for distribution equality"
drift_types = ["covariate"]
family = "statistical-test"
data_dimension = "univariate"
data_types = ["continuous"]
requires_labels = false
references = ["https://doi.org/10.2307/2280095"]

[methods.kolmogorov_smirnov.variants.ks_batch]
name = "Batch Kolmogorov-Smirnov"
execution_mode = "batch"
hyperparameters = ["threshold"]
references = []
```

### Adding New Methods

To add new methods to the registry:

1. **Define Method**: Add method definition with required metadata
2. **Define Variants**: Add one or more variant implementations
3. **Create Adapters**: Implement adapters for your preferred libraries
4. **Validate**: Test that method+variant combinations work correctly

## ⚙️ Advanced Features

### Hyperparameter Handling

```python
@register_detector("kolmogorov_smirnov", "ks_batch", "custom")
class CustomKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        
        # Extract hyperparameters with defaults
        self.threshold = kwargs.get('threshold', 0.05)
        self.bootstrap_samples = kwargs.get('bootstrap_samples', 1000)
        self.alternative = kwargs.get('alternative', 'two-sided')
        
        # Store for debugging/logging
        self._hyperparameters = {
            'threshold': self.threshold,
            'bootstrap_samples': self.bootstrap_samples,
            'alternative': self.alternative
        }
```

### Error Handling and Validation

```python
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    if not self._fitted:
        raise RuntimeError("Detector must be fitted before detection")
        
    if len(preprocessed_data) < 10:
        raise ValueError("Test data too small for reliable detection")
        
    try:
        # Perform detection logic
        drift_result = self._detect_drift(preprocessed_data)
        self._drift_score = drift_result.score
        return drift_result.detected
        
    except Exception as e:
        # Log error for debugging
        logger.error(f"Detection failed: {e}")
        # Re-raise to be handled by framework
        raise
```

### Multi-Column Support

```python
def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
    """Handle multivariate data."""
    # Extract specific columns if specified
    columns = kwargs.get('columns', None)
    
    if phase == "train":
        ref_data = data.X_ref
    else:
        ref_data = data.X_test
        
    if columns:
        return ref_data[columns].values
    else:
        return ref_data.values
        
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    # Handle multivariate detection
    if preprocessed_data.ndim == 1:
        # Univariate case
        return self._detect_univariate(preprocessed_data)
    else:
        # Multivariate case - aggregate results
        p_values = []
        for i in range(preprocessed_data.shape[1]):
            _, p_val = self._test_column(preprocessed_data[:, i])
            p_values.append(p_val)
            
        # Use Bonferroni correction or other aggregation
        combined_p = min(p_values) * len(p_values)
        self._drift_score = combined_p
        return combined_p < self.threshold
```

### Logging and Debugging

```python
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)

class MyDetector(BaseDetector):
    def fit(self, preprocessed_data: Any, **kwargs) -> "MyDetector":
        logger.debug(f"Fitting {self.library_id} detector with {len(preprocessed_data)} samples")
        self._reference_data = preprocessed_data
        logger.debug("Fitting completed successfully")
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        logger.debug(f"Detecting drift on {len(preprocessed_data)} samples")
        
        drift_score = self._calculate_drift_score(preprocessed_data)
        drift_detected = drift_score < self.threshold
        
        logger.info(f"Drift detection result: {drift_detected} (score: {drift_score:.4f})")
        
        self._drift_score = drift_score
        return drift_detected
```

## 🧪 Testing Your Adapters

### Unit Testing Pattern

```python
import pytest
from drift_benchmark.adapters import BaseDetector
from drift_benchmark.models import ScenarioResult
import pandas as pd
import numpy as np

class TestMyDetector:
    @pytest.fixture
    def sample_scenario_result(self):
        """Create sample scenario data for testing."""
        np.random.seed(42)
        
        # Reference data (no drift)
        X_ref = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100)
        })
        
        # Test data (with drift)
        X_test = pd.DataFrame({
            'feature_1': np.random.normal(0.5, 1, 100),  # Shifted mean
            'feature_2': np.random.normal(0, 1, 100)
        })
        
        return ScenarioResult(
            X_ref=X_ref, y_ref=None,
            X_test=X_test, y_test=None,
            definition=None
        )
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = MyDetector("test_method", "test_variant", "test_lib")
        
        assert detector.method_id == "test_method"
        assert detector.variant_id == "test_variant"
        assert detector.library_id == "test_lib"
    
    def test_fit_and_detect_workflow(self, sample_scenario_result):
        """Test complete fit->detect workflow."""
        detector = MyDetector("test_method", "test_variant", "test_lib")
        
        # Preprocess and fit
        ref_data = detector.preprocess(sample_scenario_result, phase="train")
        detector.fit(ref_data)
        
        # Preprocess and detect
        test_data = detector.preprocess(sample_scenario_result, phase="detect")
        drift_detected = detector.detect(test_data)
        
        # Should detect drift due to shifted mean
        assert isinstance(drift_detected, bool)
        assert detector.score() is not None
    
    def test_hyperparameter_handling(self):
        """Test hyperparameter configuration."""
        detector = MyDetector(
            "test_method", "test_variant", "test_lib",
            threshold=0.01, custom_param=42
        )
        
        assert detector.threshold == 0.01
        assert hasattr(detector, 'custom_param')
```

### Integration Testing

```python
def test_detector_registration():
    """Test detector is properly registered."""
    from drift_benchmark.adapters.registry import get_registered_detectors
    
    registered = get_registered_detectors()
    
    # Check your detector is registered
    key = ("test_method", "test_variant", "test_lib")
    assert key in registered
    assert registered[key] == MyDetector

def test_benchmark_integration(tmp_path):
    """Test detector works in full benchmark."""
    from drift_benchmark import BenchmarkRunner
    
    # Create test configuration
    config_content = """
    [[scenarios]]
    id = "test_scenario"
    
    [[detectors]]
    method_id = "test_method" 
    variant_id = "test_variant"
    library_id = "test_lib"
    threshold = 0.05
    """
    
    config_file = tmp_path / "test_config.toml"
    config_file.write_text(config_content)
    
    # Run benchmark
    runner = BenchmarkRunner.from_config(config_file)
    results = runner.run()
    
    # Verify results
    assert len(results.detector_results) == 1
    assert results.detector_results[0].library_id == "test_lib"
```

## 📚 Library-Specific Examples

### Evidently Integration

```python
@register_detector("all_features_drift", "all_features_evidently", "evidently")
class EvidentlyAllFeaturesDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        
    def preprocess(self, data: ScenarioResult, phase: str = "detect", **kwargs):
        """Evidently expects dictionary format."""
        if phase == "train":
            return {"reference": data.X_ref}
        else:
            return {"current": data.X_test}
            
    def fit(self, preprocessed_data: dict, **kwargs) -> "EvidentlyAllFeaturesDetector":
        self._reference_data = preprocessed_data["reference"]
        return self
        
    def detect(self, preprocessed_data: dict, **kwargs) -> bool:
        from evidently.metrics import DataDriftPreset
        from evidently.report import Report
        
        # Create Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self._reference_data,
            current_data=preprocessed_data["current"]
        )
        
        # Extract drift results
        drift_result = report.as_dict()
        dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]
        
        self._drift_score = float(dataset_drift)
        return dataset_drift
```

### Alibi-Detect Integration

```python
@register_detector("kolmogorov_smirnov", "ks_batch", "alibi-detect")
class AlibiDetectKSDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "AlibiDetectKSDetector":
        from alibi_detect.cd import KSDrift
        
        # Initialize Alibi-Detect detector
        self._detector = KSDrift(
            p_val=self.threshold,
            x_ref=preprocessed_data
        )
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Predict using Alibi-Detect
        prediction = self._detector.predict(preprocessed_data)
        
        # Extract results
        drift_detected = prediction['data']['is_drift']
        p_value = prediction['data']['p_val']
        
        self._drift_score = p_value
        return drift_detected
```

### River Integration

```python
@register_detector("kswin", "kswin_standard", "river")
class RiverKSWINDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        from river.drift import KSWIN
        
        self._detector = KSWIN(
            alpha=kwargs.get('alpha', 0.005),
            window_size=kwargs.get('window_size', 100)
        )
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "RiverKSWINDetector":
        # Initialize with reference data
        for value in preprocessed_data:
            self._detector.update(value)
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Process test data incrementally
        drift_detected = False
        
        for value in preprocessed_data:
            self._detector.update(value)
            if self._detector.drift_detected:
                drift_detected = True
                
        self._drift_score = float(drift_detected)
        return drift_detected
```

## 🎯 Best Practices

### Adapter Design Principles

1. **Single Responsibility**: Each adapter handles one method+variant+library combination
2. **Error Handling**: Validate inputs and provide descriptive error messages
3. **State Management**: Clearly track fitted state and reference data
4. **Score Reporting**: Always store drift scores when available
5. **Logging**: Use framework logging for debugging and monitoring

### Performance Optimization

```python
class OptimizedDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        
        # Pre-compile computationally expensive components
        self._precomputed_stats = None
        
    def fit(self, preprocessed_data: Any, **kwargs) -> "OptimizedDetector":
        # Precompute statistics to avoid recalculation
        self._precomputed_stats = self._compute_reference_statistics(preprocessed_data)
        return self
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Use precomputed statistics for faster detection
        test_stats = self._compute_test_statistics(preprocessed_data)
        return self._compare_statistics(self._precomputed_stats, test_stats)
```

### Configuration Validation

```python
def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
    super().__init__(method_id, variant_id, library_id, **kwargs)
    
    # Validate hyperparameters
    self.threshold = kwargs.get('threshold', 0.05)
    if not 0 < self.threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")
        
    # Validate library-specific requirements
    try:
        import required_library
    except ImportError:
        raise ImportError(f"Library '{library_id}' requires 'required_library' to be installed")
```

### Documentation Standards

```python
@register_detector("method_name", "variant_name", "library_name")
class WellDocumentedDetector(BaseDetector):
    """
    Brief description of what this detector does.
    
    This detector implements [method_name] using [library_name] for [variant_name]
    variant processing. It is suitable for [data_types] data and [drift_types] drift.
    
    Hyperparameters:
        threshold (float): Detection threshold, default 0.05
        window_size (int): Window size for processing, default 100
        
    References:
        - Original paper: [citation]
        - Library documentation: [url]
    """
    
    def fit(self, preprocessed_data: Any, **kwargs) -> "WellDocumentedDetector":
        """
        Train detector on reference data.
        
        Args:
            preprocessed_data: Reference data in numpy array format
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If reference data is invalid
        """
        
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """
        Detect drift on test data.
        
        Args:
            preprocessed_data: Test data in same format as training data
            **kwargs: Additional detection parameters
            
        Returns:
            True if drift detected, False otherwise
            
        Raises:
            RuntimeError: If detector not fitted
        """
```

## 📖 Related Documentation

- **[Benchmark API](benchmark_api.md)**: Running benchmarks with your adapters
- **[Configurations](configurations.md)**: Configuration file reference
- **[Scenarios](scenarios.md)**: Understanding scenario data format
- **[Methods Registry](../src/drift_benchmark/detectors/methods.toml)**: Available methods and variants
