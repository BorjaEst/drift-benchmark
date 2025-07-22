# Drift Detection Adapter API Documentation

> **Version**: 0.1.0  
> **Date**: July 22, 2025  
> **Purpose**: Complete API reference for creating drift detection adapters to enable library comparison

## 📖 Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [BaseDetector Abstract Class](#basedetector-abstract-class)
4. [Creating Library Adapters](#creating-library-adapters)
5. [Registration System](#registration-system)
6. [Data Flow and Lifecycle](#data-flow-and-lifecycle)
7. [Library Comparison Examples](#library-comparison-examples)
8. [Type System and Error Handling](#type-system-and-error-handling)
9. [Testing Adapters](#testing-adapters)
10. [Best Practices](#best-practices)

---

## 📋 Overview

The drift-benchmark framework enables **fair comparison of how different libraries implement the same mathematical drift detection methods**. This document shows how to create adapter classes that integrate your preferred libraries with our standardized interface.

### 🎯 Primary Purpose

**Compare library implementations** rather than just methods. For example:

- How does **Evidently's** Kolmogorov-Smirnov implementation compare to **Alibi-Detect's**?
- Which library provides better performance for Maximum Mean Discrepancy: **scikit-learn** or **River**?
- Is **SciPy's** statistical tests faster than custom implementations?

### Key Concepts

- **Method+Variant Standardization**: We define consistent algorithmic approaches (variants) for each mathematical method
- **Library-Agnostic Interface**: Compare implementations across Evidently, Alibi-Detect, scikit-learn, River, etc.
- **Adapter Pattern**: Your code bridges library-specific APIs to our unified interface
- **Performance Benchmarking**: Measure speed, accuracy, and resource usage across libraries

---

## 🏗️ Core Concepts

### Framework Architecture

The drift-benchmark framework acts as a **standardization layer** between different drift detection libraries:

```text
┌─────────────────────────────────────────────────────────┐
│                   BenchmarkRunner                       │
│              (Orchestrates Comparisons)                 │
├─────────────────────────────────────────────────────────┤
│                 Benchmark Core                          │
│              (Executes All Detectors)                   │
├─────────────────────────────────────────────────────────┤
│                  Adapter Layer                          │
│              BaseDetector Interface                     │
├─────────────────────────────────────────────────────────┤
│               Library Implementations                   │
│        Evidently | Alibi-Detect | scikit-learn          │
│           River | SciPy | Custom Libraries              │
└─────────────────────────────────────────────────────────┘
```

### Method+Variant+Library Hierarchy

**drift-benchmark** organizes drift detection into a three-level hierarchy:

1. **🔬 Method**: Mathematical approach (e.g., "kolmogorov_smirnov", "maximum_mean_discrepancy")
2. **⚙️ Variant**: Algorithmic implementation defined by us (e.g., "batch", "streaming", "sliding_window")
3. **🔌 Library**: Specific library implementation (e.g., "evidently", "alibi_detect", "scipy")

### Registry Structure

```text
methods.toml defines:
├── kolmogorov_smirnov (method)
│   ├── batch (variant)          ← drift-benchmark defines this
│   └── streaming (variant)      ← drift-benchmark defines this
└── maximum_mean_discrepancy (method)
    ├── rbf_kernel (variant)     ← drift-benchmark defines this
    └── linear_kernel (variant)  ← drift-benchmark defines this

Your adapters implement:
├── kolmogorov_smirnov + batch + evidently      ← You implement this
├── kolmogorov_smirnov + batch + alibi_detect   ← You implement this
├── kolmogorov_smirnov + batch + scipy          ← You implement this
└── ... (any library you want to compare)
```

### Key Framework Roles

- **drift-benchmark provides**: Standardized method+variant definitions in `methods.toml`
- **You provide**: Adapter classes that map library implementations to our variants
- **Framework enables**: Fair comparison of different libraries implementing the same method+variant

---

## 🧬 BaseDetector Abstract Class

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from drift_benchmark.models.results import DatasetResult

class BaseDetector(ABC):
    """Abstract base class for all drift detection library adapters."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        """
        Initialize detector with standardized identifiers.

        Args:
            method_id: Method from methods.toml (e.g., "kolmogorov_smirnov")
            variant_id: Variant from methods.toml (e.g., "batch")
            library_id: Library identifier (e.g., "evidently", "alibi_detect")
            **kwargs: Library-specific parameters
        """
        self._method_id = method_id
        self._variant_id = variant_id
        self._library_id = library_id
        # Store any library-specific configuration
        self._config = kwargs
```

### Required Properties

```python
@property
def method_id(self) -> str:
    """Mathematical method identifier from methods.toml registry."""
    return self._method_id

@property
def variant_id(self) -> str:
    """Algorithmic variant identifier from methods.toml registry."""
    return self._variant_id

@property
def library_id(self) -> str:
    """Library implementation identifier for comparison."""
    return self._library_id
```

### Required Methods

#### Data Preprocessing

```python
def preprocess(self, data: DatasetResult, **kwargs) -> Any:
    """
    Convert pandas DataFrames to library-specific format.

    Args:
        data: Contains X_ref and X_test DataFrames plus metadata
        **kwargs: Phase-specific parameters (e.g., phase='train'/'detect')

    Returns:
        Data in format expected by your library (numpy arrays, etc.)
    """
    # Default implementation - override for library-specific formats
    return {"X_ref": data.X_ref, "X_test": data.X_test}
```

#### Training (Abstract)

```python
@abstractmethod
def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
    """
    Train detector on reference data using your library.

    Args:
        preprocessed_data: Output from preprocess() containing reference data
        **kwargs: Training parameters

    Returns:
        Self for method chaining
    """
    pass
```

#### Detection (Abstract)

```python
@abstractmethod
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    """
    Perform drift detection using your library.

    Args:
        preprocessed_data: Output from preprocess() containing test data
        **kwargs: Detection parameters

    Returns:
        Boolean indicating whether drift was detected
    """
    pass
```

#### Scoring (Optional)

```python
def score(self) -> Optional[float]:
    """
    Return drift score after detection, if available.

    Returns:
        Continuous drift score or None if library doesn't provide scores
    """
    return getattr(self, "_last_score", None)
```

---

## 🧬 BaseDetector Abstract Class

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Any, Optional
from ..models.results import DatasetResult

class BaseDetector(ABC):
    """Abstract base class for all drift detectors."""

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        """Initialize base detector with identifiers and parameters."""

    @property
    def method_id(self) -> str:
        """Get the drift detection method identifier."""

    @property
    def variant_id(self) -> str:
        """Get the variants variant identifier."""

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Convert pandas DataFrames to detector-specific format."""

    @abstractmethod
    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """Train the detector on reference data."""

    @abstractmethod
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Perform drift detection and return boolean result."""

    def score(self) -> Optional[float]:
        """Return drift score after detection (if available)."""
```

### Method Specifications

#### Constructor: `__init__(self, method_id: str, variant_id: str, **kwargs)`

**Purpose**: Initialize detector with identifiers and optional parameters.

**Parameters**:

- `method_id` (str): Method identifier from `methods.toml` registry
- `variant_id` (str): Variants variant identifier
- `**kwargs`: Additional parameters (hyperparameters, configuration options)

**Requirements**:

- Store identifiers as read-only properties
- Accept arbitrary keyword arguments for flexibility
- Initialize drift score storage

**Example**:

```python
def __init__(self, method_id: str, variant_id: str, **kwargs):
    super().__init__(method_id, variant_id)
    self.threshold = kwargs.get('threshold', 0.05)
    self.alpha = kwargs.get('alpha', 0.01)
    self._fitted = False
```

#### Properties: `method_id` and `variant_id`

**Purpose**: Provide read-only access to detector identifiers.

**Returns**: String identifiers used for registration and lookup.

**Requirements**:

- Must be read-only (no setters)
- Must return the values passed during initialization
- Used for result tracking and error reporting

#### Data Preprocessing: `preprocess(self, data: DatasetResult, **kwargs) -> Any`

**Purpose**: Convert pandas DataFrames to detector-specific format.

**Parameters**:

- `data` (DatasetResult): Contains X_ref, X_test DataFrames and metadata
- `**kwargs`: Additional preprocessing parameters

**Returns**: Any format required by the detector library

**Default Behavior**: Returns dict with DataFrames for basic compatibility

```python
def preprocess(self, data: DatasetResult, **kwargs) -> Any:
    return {"X_ref": data.X_ref, "X_test": data.X_test, "metadata": data.metadata}
```

**Common variants**:

```python
# For numpy-based libraries
def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
    phase = kwargs.get('phase', 'train')  # 'train' or 'detect'
    if phase == 'train':
        return data.X_ref.values
    else:
        return data.X_test.values

# For specific libraries with custom formats
def preprocess(self, data: DatasetResult, **kwargs) -> CustomFormat:
    return CustomFormat.from_pandas(data.X_ref)
```

#### Training: `fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector"`

**Purpose**: Train the detector on reference data.

**Parameters**:

- `preprocessed_data` (Any): Data in detector-specific format from preprocess()
- `**kwargs`: Additional training parameters

**Returns**: Self (for method chaining)

**Requirements**:

- Must be abstract (implemented by subclasses)
- Should store trained model/parameters internally
- Must return self for fluent interface
- Should handle training failures gracefully

**Example**:

```python
@abstractmethod
def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
    # Extract reference data based on preprocessing format
    if isinstance(preprocessed_data, dict):
        X_ref = preprocessed_data["X_ref"]
    else:
        X_ref = preprocessed_data

    # Train the underlying detector
    self._detector.fit(X_ref)
    self._fitted = True
    return self
```

#### Detection: `detect(self, preprocessed_data: Any, **kwargs) -> bool`

**Purpose**: Perform drift detection and return boolean result.

**Parameters**:

- `preprocessed_data` (Any): Data in detector-specific format from preprocess()
- `**kwargs**: Additional detection parameters

**Returns**: Boolean indicating whether drift was detected

**Requirements**:

- Must be abstract (implemented by subclasses)
- Should store drift score internally for score() method
- Must return boolean (True = drift detected, False = no drift)
- Should handle detection failures gracefully

**Example**:

```python
@abstractmethod
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    if not self._fitted:
        raise RuntimeError("Detector must be fitted before detection")

    # Extract test data based on preprocessing format
    if isinstance(preprocessed_data, dict):
        X_test = preprocessed_data["X_test"]
    else:
        X_test = preprocessed_data

    # Perform detection and store score
    self._last_score = self._detector.score(X_test)
    return self._last_score > self.threshold
```

#### Scoring: `score(self) -> Optional[float]`

**Purpose**: Return drift score after detection (optional).

**Returns**: Float score or None if not available

**Requirements**:

- Called after detect() to retrieve continuous score
- Return None if detector doesn't provide scores
- Should not perform detection itself

**Default Variants**:

```python
def score(self) -> Optional[float]:
    return getattr(self, "_last_score", None)
```

---

## � Creating Library Adapters

### Step 1: Choose Method+Variant from Registry

First, check which methods and variants are available in `methods.toml`:

```toml
[methods.kolmogorov_smirnov]
name = "Kolmogorov-Smirnov Test"
description = "Two-sample test for equality of continuous distributions"
drift_types = ["COVARIATE"]
family = "STATISTICAL_TEST"
data_dimension = "UNIVARIATE"
data_types = ["CONTINUOUS"]
requires_labels = false

[methods.kolmogorov_smirnov.variants.batch]
name = "Batch Processing"
execution_mode = "BATCH"
hyperparameters = ["threshold"]
```

### Step 2: Create Library-Specific Adapters

Create multiple adapters for the **same method+variant** to enable library comparison:

```python
from drift_benchmark.adapters import BaseDetector, register_detector
import numpy as np

# Evidently's implementation
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="evidently")
class EvidentlyKSDetector(BaseDetector):
    """Evidently's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        self._reference_data = None

    def preprocess(self, data, **kwargs):
        """Convert to Evidently's expected format."""
        # Evidently typically expects pandas DataFrames
        phase = kwargs.get('phase', 'detect')
        if phase == 'train':
            return data.X_ref
        else:
            return data.X_test

    def fit(self, preprocessed_data, **kwargs):
        """Store reference data for Evidently."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs):
        """Use Evidently's drift detection."""
        from evidently.metrics import DataDriftPreset

        # Evidently-specific implementation
        # ... implementation details ...

        return drift_detected

# Alibi-Detect's implementation of the SAME method+variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi_detect")
class AlibiDetectKSDetector(BaseDetector):
    """Alibi-Detect's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        self._detector = None

    def preprocess(self, data, **kwargs):
        """Convert to Alibi-Detect's expected format."""
        # Alibi-Detect typically expects numpy arrays
        phase = kwargs.get('phase', 'detect')
        if phase == 'train':
            return data.X_ref.values
        else:
            return data.X_test.values

    def fit(self, preprocessed_data, **kwargs):
        """Initialize Alibi-Detect detector."""
        from alibi_detect.cd import KSDrift

        self._detector = KSDrift(preprocessed_data, p_val=self.threshold)
        return self

    def detect(self, preprocessed_data, **kwargs):
        """Use Alibi-Detect's drift detection."""
        result = self._detector.predict(preprocessed_data)
        self._last_score = result['data']['p_val']
        return result['data']['is_drift']

# SciPy's implementation of the SAME method+variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="scipy")
class SciPyKSDetector(BaseDetector):
    """SciPy's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id, **kwargs)
        self.threshold = kwargs.get('threshold', 0.05)
        self._reference_data = None

    def preprocess(self, data, **kwargs):
        """Convert to SciPy's expected format."""
        # SciPy expects 1D numpy arrays for KS test
        phase = kwargs.get('phase', 'detect')
        if phase == 'train':
            return data.X_ref.iloc[:, 0].values  # First column only
        else:
            return data.X_test.iloc[:, 0].values

    def fit(self, preprocessed_data, **kwargs):
        """Store reference data for SciPy."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data, **kwargs):
        """Use SciPy's KS test."""
        from scipy.stats import ks_2samp

        statistic, p_value = ks_2samp(self._reference_data, preprocessed_data)
        self._last_score = p_value
        return p_value < self.threshold
```

### Step 3: Configure Benchmark to Compare Libraries

```toml
# benchmark_config.toml
[[datasets]]
path = "datasets/example.csv"
format = "CSV"
reference_split = 0.5

# Compare the same method+variant across different libraries
[[detectors]]
method_id = "kolmogorov_smirnov"
variant_id = "batch"
library_id = "evidently"
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"  # Same method
variant_id = "batch"              # Same variant
library_id = "alibi_detect"       # Different library
threshold = 0.05

[[detectors]]
method_id = "kolmogorov_smirnov"  # Same method
variant_id = "batch"              # Same variant
library_id = "scipy"              # Different library
threshold = 0.05
```

### Library-Specific Patterns

#### Evidently Integration

```python
def preprocess(self, data, **kwargs):
    """Evidently usually works with pandas DataFrames."""
    return data.X_ref if kwargs.get('phase') == 'train' else data.X_test

def detect(self, preprocessed_data, **kwargs):
    """Use Evidently's built-in metrics."""
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=self._reference_data, current_data=preprocessed_data)
    result = report.as_dict()
    return result['metrics'][0]['result']['dataset_drift']
```

#### Alibi-Detect Integration

```python
def preprocess(self, data, **kwargs):
    """Alibi-Detect usually expects numpy arrays."""
    data_array = data.X_ref.values if kwargs.get('phase') == 'train' else data.X_test.values
    return data_array.astype(np.float32)

def fit(self, preprocessed_data, **kwargs):
    """Initialize Alibi-Detect detector with reference data."""
    from alibi_detect.cd import MMDDrift

    self._detector = MMDDrift(preprocessed_data, backend='pytorch')
    return self
```

#### SciPy/scikit-learn Integration

```python
def preprocess(self, data, **kwargs):
    """Convert to numpy arrays for scikit-learn/SciPy."""
    df = data.X_ref if kwargs.get('phase') == 'train' else data.X_test
    return df.select_dtypes(include=[np.number]).values

def detect(self, preprocessed_data, **kwargs):
    """Use SciPy statistical tests."""
    from scipy.stats import ks_2samp

    # For multivariate data, test each feature
    p_values = []
    for i in range(preprocessed_data.shape[1]):
        _, p_val = ks_2samp(self._reference_data[:, i], preprocessed_data[:, i])
        p_values.append(p_val)

    # Combine p-values (simple minimum for this example)
    combined_p = min(p_values)
    self._last_score = combined_p
    return combined_p < self.threshold
```

    numeric_data = df.select_dtypes(include=[np.number])
    return numeric_data.fillna(numeric_data.mean()).values

````

#### Categorical Data

```python
def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
    """Handle categorical data with encoding."""
    from sklearn.preprocessing import LabelEncoder

    phase = kwargs.get('phase', 'train')
    df = data.X_ref if phase == 'train' else data.X_test

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoded_data = df.copy()

    for col in categorical_cols:
        if not hasattr(self, f'_encoder_{col}'):
            encoder = LabelEncoder()
            encoder.fit(data.X_ref[col].astype(str))
            setattr(self, f'_encoder_{col}', encoder)

        encoder = getattr(self, f'_encoder_{col}')
        encoded_data[col] = encoder.transform(df[col].astype(str))

    return encoded_data.values
````

#### Multivariate Data

```python
def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
    """Handle multivariate drift detection."""
    # Apply univariate test to each feature and combine results
    p_values = []

    for feature_idx in range(preprocessed_data.shape[1]):
        ref_feature = self._reference_data[:, feature_idx]
        test_feature = preprocessed_data[:, feature_idx]

        _, p_value = stats.ks_2samp(ref_feature, test_feature)
        p_values.append(p_value)

    # Combine p-values using Fisher's method
    combined_statistic, combined_p_value = stats.combine_pvalues(p_values, method='fisher')

    self._last_score = combined_p_value
    return combined_p_value < self.threshold
```

---

## 🗂️ Registration System

### Decorator Registration

The framework uses decorator-based registration for automatic discovery:

```python
from drift_benchmark.adapters import register_detector

@register_detector(method_id="my_method", variant_id="my_impl")
class MyDetector(BaseDetector):
    # Variants here
    pass
```

### Registration Requirements

1. **Method ID**: Must exist in `methods.toml` registry
2. **Variants ID**: Must be listed under the method in `methods.toml`
3. **Class Inheritance**: Must inherit from `BaseDetector`
4. **Import Side Effects**: Registration happens at import time

### Registry Functions

```python
from drift_benchmark.adapters import get_detector_class, list_detectors

# Get specific detector class
DetectorClass = get_detector_class("kolmogorov_smirnov", "ks_batch")

# List all registered detectors
available = list_detectors()  # Returns [(method_id, impl_id), ...]

# Instantiate detector
detector = DetectorClass("kolmogorov_smirnov", "ks_batch", threshold=0.01)
```

### Method Registry Schema

The `methods.toml` file defines available methods and variants:

```toml
[methods.my_method]
name = "My Custom Method"
description = "Description of the custom drift detection method"
drift_types = ["COVARIATE"]  # COVARIATE, CONCEPT, PRIOR
family = "STATISTICAL_TEST"  # See MethodFamily in literals.py
data_dimension = "UNIVARIATE"  # UNIVARIATE, MULTIVARIATE
data_types = ["CONTINUOUS"]  # CONTINUOUS, CATEGORICAL, MIXED
requires_labels = false
references = ["https://doi.org/example", "Author (Year)"]

[methods.my_method.variants.my_impl]
name = "My Variants"
execution_mode = "BATCH"  # BATCH, STREAMING
hyperparameters = ["threshold", "alpha"]
references = ["Variants reference"]
```

---

## 🔄 Data Flow and Lifecycle

### Complete Execution Flow

```text
1. Configuration Loading
   ├── Parse TOML configuration
   ├── Validate dataset and detector configurations
   └── Load dataset files

2. Detector Instantiation
   ├── Registry lookup by (method_id, variant_id)
   ├── Class instantiation with parameters
   └── Validation of required methods

3. Benchmark Execution (per dataset)
   ├── Preprocessing Phase
   │   ├── detector.preprocess(dataset, phase='train')
   │   └── Return reference data in detector format
   │
   ├── Training Phase
   │   ├── detector.fit(preprocessed_reference_data)
   │   └── Store trained model/parameters
   │
   ├── Detection Phase
   │   ├── detector.preprocess(dataset, phase='detect')
   │   ├── detector.detect(preprocessed_test_data)
   │   └── Return boolean drift result
   │
   └── Scoring Phase
       ├── detector.score()
       └── Return optional drift score

4. Result Collection
   ├── Aggregate all detector results
   ├── Calculate summary statistics
   └── Save results to timestamped directory
```

### Data Transformations

```text
CSV File → pandas.DataFrame → DatasetResult → Preprocessed Format → Library API
    ↓              ↓              ↓               ↓               ↓
file_loader   X_ref/X_test   preprocess()   fit()/detect()   External Lib
```

### Error Handling Points

1. **Registration**: `DuplicateDetectorError`, `DetectorNotFoundError`
2. **Instantiation**: `TypeError`, `ValueError`
3. **Preprocessing**: Format conversion errors
4. **Training**: Library-specific training errors
5. **Detection**: Library-specific detection errors

---

## 🏷️ Type System and Literals

### Import Required Types

```python
from drift_benchmark.literals import (
    DriftType,      # "COVARIATE", "CONCEPT", "PRIOR"
    MethodFamily,   # "STATISTICAL_TEST", "DISTANCE_BASED", etc.
    DataType,       # "CONTINUOUS", "CATEGORICAL", "MIXED"
    DataDimension,  # "UNIVARIATE", "MULTIVARIATE"
    ExecutionMode,  # "BATCH", "STREAMING"
)
```

### Type Annotations

```python
from typing import Optional, Union, Any, Dict, List
import numpy as np
import pandas as pd

class MyDetector(BaseDetector):
    def preprocess(self, data: DatasetResult, **kwargs) -> Union[np.ndarray, Dict[str, Any]]:
        # Variants
        pass

    def fit(self, preprocessed_data: Any, **kwargs) -> "MyDetector":
        # Variants
        pass

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Variants
        pass

    def score(self) -> Optional[float]:
        # Variants
        pass
```

---

## ⚠️ Error Handling

### Exception Hierarchy

```python
from drift_benchmark.exceptions import (
    DriftBenchmarkError,        # Base exception
    DetectorNotFoundError,      # Registry lookup failures
    DuplicateDetectorError,     # Registration conflicts
    BenchmarkExecutionError,    # Runtime execution errors
)
```

### Error Handling Patterns

#### Registration Errors

```python
try:
    detector_class = get_detector_class("unknown_method", "unknown_impl")
except DetectorNotFoundError as e:
    logger.error(f"Detector not found: {e}")
    # Handle gracefully
```

#### Runtime Errors

```python
def detect(self, preprocessed_data: Any, **kwargs) -> bool:
    try:
        # Perform detection
        result = self._library_detect(preprocessed_data)
        return result > self.threshold
    except Exception as e:
        # Log error and re-raise with context
        logger.error(f"Detection failed in {self.method_id}.{self.variant_id}: {e}")
        raise BenchmarkExecutionError(f"Detection failed: {e}") from e
```

#### Validation Errors

```python
def fit(self, preprocessed_data: Any, **kwargs) -> "MyDetector":
    if preprocessed_data is None or len(preprocessed_data) == 0:
        raise ValueError("Training data cannot be empty")

    if not isinstance(preprocessed_data, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(preprocessed_data)}")

    # Proceed with training
    return self
```

---

## 📋 Complete Examples

### Example 1: Statistical Test Detector

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult
import numpy as np
from scipy import stats
from typing import Optional

@register_detector(method_id="kolmogorov_smirnov", variant_id="ks_batch")
class KolmogorovSmirnovDetector(BaseDetector):
    """
    Kolmogorov-Smirnov test for drift detection.

    Tests whether two samples come from the same distribution.
    """

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.05)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Convert DataFrame to numpy array for scipy functions."""
        phase = kwargs.get('phase', 'train')

        if phase == 'train':
            return data.X_ref.values
        else:
            return data.X_test.values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "KolmogorovSmirnovDetector":
        """Store reference data for comparison."""
        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Reference data cannot be empty")

        self._reference_data = preprocessed_data.copy()
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Perform KS test for drift detection."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        if preprocessed_data is None or len(preprocessed_data) == 0:
            raise ValueError("Test data cannot be empty")

        # Flatten arrays for univariate test
        ref_flat = self._reference_data.flatten()
        test_flat = preprocessed_data.flatten()

        # Perform two-sample Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(ref_flat, test_flat)

        # Store p-value as drift score
        self._last_score = p_value

        # Drift detected if p-value < threshold
        return p_value < self.threshold

    def score(self) -> Optional[float]:
        """Return p-value from last detection."""
        return self._last_score
```

### Example 2: Distance-Based Detector

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import Optional

@register_detector(method_id="maximum_mean_discrepancy", variant_id="mmd_rbf")
class MaximumMeanDiscrepancyDetector(BaseDetector):
    """
    Maximum Mean Discrepancy with RBF kernel for drift detection.
    """

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.threshold = kwargs.get('threshold', 0.1)
        self.gamma = kwargs.get('gamma', 1.0)
        self._reference_data: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Standardize data for distance calculations."""
        phase = kwargs.get('phase', 'train')

        if phase == 'train':
            df = data.X_ref
        else:
            df = data.X_test

        # Convert to numpy and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        processed = numeric_data.fillna(numeric_data.mean()).values

        # Standardize using reference statistics
        if phase == 'train':
            self._mean = processed.mean(axis=0)
            self._std = processed.std(axis=0)
            self._std[self._std == 0] = 1  # Avoid division by zero

        # Apply standardization
        if hasattr(self, '_mean') and hasattr(self, '_std'):
            processed = (processed - self._mean) / self._std

        return processed

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "MaximumMeanDiscrepancyDetector":
        """Store reference data for MMD calculation."""
        self._reference_data = preprocessed_data.copy()
        return self

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        distances = euclidean_distances(X, Y)
        return np.exp(-self.gamma * distances ** 2)

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy."""
        m, n = X.shape[0], Y.shape[0]

        # Compute kernel matrices
        K_XX = self._rbf_kernel(X, X)
        K_YY = self._rbf_kernel(Y, Y)
        K_XY = self._rbf_kernel(X, Y)

        # Compute MMD^2
        mmd_squared = (
            K_XX.sum() / (m * m) +
            K_YY.sum() / (n * n) -
            2 * K_XY.sum() / (m * n)
        )

        return np.sqrt(max(mmd_squared, 0))  # Ensure non-negative

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Detect drift using MMD."""
        if self._reference_data is None:
            raise RuntimeError("Detector must be fitted before detection")

        # Compute MMD between reference and test data
        mmd_score = self._compute_mmd(self._reference_data, preprocessed_data)
        self._last_score = mmd_score

        return mmd_score > self.threshold

    def score(self) -> Optional[float]:
        """Return MMD score from last detection."""
        return self._last_score
```

### Example 3: Library Integration (scikit-learn)

```python
from drift_benchmark.adapters import BaseDetector, register_detector
from drift_benchmark.models.results import DatasetResult
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from typing import Optional

@register_detector(method_id="isolation_forest", variant_id="sklearn")
class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest-based drift detector using scikit-learn.
    """

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self.contamination = kwargs.get('contamination', 0.1)
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.random_state = kwargs.get('random_state', 42)

        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self._last_score: Optional[float] = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Preprocess data for scikit-learn."""
        phase = kwargs.get('phase', 'train')

        if phase == 'train':
            df = data.X_ref
        else:
            df = data.X_test

        # Select numeric columns and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "IsolationForestDetector":
        """Train Isolation Forest on reference data."""
        self._model.fit(preprocessed_data)
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Detect drift using anomaly scores."""
        # Get anomaly scores for test data
        test_scores = self._model.score_samples(preprocessed_data)

        # Get anomaly scores for reference data (from training)
        ref_scores = self._model.score_samples(
            self._model._get_reference_data() if hasattr(self._model, '_get_reference_data')
            else preprocessed_data  # Fallback
        )

        # Use mean score difference as drift indicator
        mean_test_score = np.mean(test_scores)
        mean_ref_score = np.mean(ref_scores) if len(ref_scores) > 0 else 0

        score_diff = abs(mean_test_score - mean_ref_score)
        self._last_score = score_diff

        # Detect drift if score difference exceeds threshold
        threshold = 0.1  # Can be made configurable
        return score_diff > threshold

    def score(self) -> Optional[float]:
        """Return score difference from last detection."""
        return self._last_score
```

---

## 🧪 Testing Adapters

### Unit Test Structure

```python
import pytest
import numpy as np
import pandas as pd
from drift_benchmark.adapters import register_detector
from drift_benchmark.models.results import DatasetResult
from drift_benchmark.models.metadata import DatasetMetadata

class TestMyDetector:
    """Test suite for custom detector variants."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        # Reference data (normal distribution)
        X_ref = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })

        # Test data (shifted distribution - drift)
        X_test = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 100),  # Mean shift
            'feature2': np.random.normal(0, 1, 100)
        })

        metadata = DatasetMetadata(
            name="test_dataset",
            data_type="CONTINUOUS",
            dimension="MULTIVARIATE",
            n_samples_ref=100,
            n_samples_test=100
        )

        return DatasetResult(X_ref=X_ref, X_test=X_test, metadata=metadata)

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return MyDetector("my_method", "my_impl", threshold=0.05)

    def test_initialization(self, detector):
        """Test proper initialization."""
        assert detector.method_id == "my_method"
        assert detector.variant_id == "my_impl"
        assert detector.threshold == 0.05

    def test_preprocess(self, detector, sample_data):
        """Test data preprocessing."""
        # Test training preprocessing
        train_data = detector.preprocess(sample_data, phase='train')
        assert isinstance(train_data, np.ndarray)
        assert train_data.shape[0] == 100

        # Test detection preprocessing
        test_data = detector.preprocess(sample_data, phase='detect')
        assert isinstance(test_data, np.ndarray)
        assert test_data.shape[0] == 100

    def test_fit(self, detector, sample_data):
        """Test detector training."""
        train_data = detector.preprocess(sample_data, phase='train')
        fitted_detector = detector.fit(train_data)

        # Should return self
        assert fitted_detector is detector

        # Should store reference data
        assert hasattr(detector, '_reference_data')
        assert detector._reference_data is not None

    def test_detect_drift(self, detector, sample_data):
        """Test drift detection."""
        # Train detector
        train_data = detector.preprocess(sample_data, phase='train')
        detector.fit(train_data)

        # Detect on shifted data (should detect drift)
        test_data = detector.preprocess(sample_data, phase='detect')
        drift_detected = detector.detect(test_data)

        assert isinstance(drift_detected, bool)
        assert drift_detected is True  # Should detect drift in shifted data

    def test_detect_no_drift(self, detector):
        """Test no drift detection with same distribution."""
        # Create data with no drift
        X_same = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })

        metadata = DatasetMetadata(
            name="no_drift_dataset",
            data_type="CONTINUOUS",
            dimension="MULTIVARIATE",
            n_samples_ref=100,
            n_samples_test=100
        )

        no_drift_data = DatasetResult(X_ref=X_same, X_test=X_same, metadata=metadata)

        # Train and detect
        train_data = detector.preprocess(no_drift_data, phase='train')
        detector.fit(train_data)

        test_data = detector.preprocess(no_drift_data, phase='detect')
        drift_detected = detector.detect(test_data)

        assert drift_detected is False  # Should not detect drift

    def test_score(self, detector, sample_data):
        """Test drift score retrieval."""
        # Train and detect first
        train_data = detector.preprocess(sample_data, phase='train')
        detector.fit(train_data)

        test_data = detector.preprocess(sample_data, phase='detect')
        detector.detect(test_data)

        # Get score
        score = detector.score()
        assert score is not None
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Assuming normalized score

    def test_error_handling(self, detector, sample_data):
        """Test error handling."""
        # Test detection without fitting
        test_data = detector.preprocess(sample_data, phase='detect')

        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.detect(test_data)

        # Test fitting with invalid data
        with pytest.raises(ValueError):
            detector.fit(None)

    def test_registration(self):
        """Test detector registration."""
        from drift_benchmark.adapters import get_detector_class, list_detectors

        # Should be able to retrieve registered detector
        DetectorClass = get_detector_class("my_method", "my_impl")
        assert DetectorClass is MyDetector

        # Should appear in detector list
        detectors = list_detectors()
        assert ("my_method", "my_impl") in detectors
```

### Integration Test

```python
def test_full_integration():
    """Test complete benchmark integration."""
    from drift_benchmark import BenchmarkRunner
    import tempfile
    import os

    # Create temporary configuration
    config_content = """
    [[datasets]]
    path = "test_data.csv"
    format = "CSV"
    reference_split = 0.5

    [[detectors]]
    method_id = "my_method"
    variant_id = "my_impl"
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write config file
        config_path = os.path.join(tmpdir, "config.toml")
        with open(config_path, 'w') as f:
            f.write(config_content)

        # Create test data
        data_path = os.path.join(tmpdir, "test_data.csv")
        test_df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200)
        })
        test_df.to_csv(data_path, index=False)

        # Run benchmark
        runner = BenchmarkRunner.from_config_file(config_path)
        results = runner.run()

        # Verify results
        assert len(results.detector_results) == 1
        result = results.detector_results[0]
        assert result.detector_id == "my_method.my_impl"
        assert isinstance(result.drift_detected, bool)
        assert result.execution_time > 0
```

---

## 📖 Best Practices

### 1. Error Handling and Logging

```python
from drift_benchmark.settings import get_logger

logger = get_logger(__name__)

class MyDetector(BaseDetector):
    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        try:
            # Detection logic
            logger.info(f"Starting detection with {self.method_id}.{self.variant_id}")
            result = self._perform_detection(preprocessed_data)
            logger.info(f"Detection completed: drift_detected={result}")
            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise  # Re-raise for framework handling
```

### 2. Parameter Validation

```python
def __init__(self, method_id: str, variant_id: str, **kwargs):
    super().__init__(method_id, variant_id)

    # Validate parameters
    self.threshold = kwargs.get('threshold', 0.05)
    if not 0 < self.threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")

    self.window_size = kwargs.get('window_size', 100)
    if self.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {self.window_size}")
```

### 3. Resource Management

```python
class ResourceAwareDetector(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, **kwargs):
        super().__init__(method_id, variant_id)
        self._model = None
        self._fitted = False

    def fit(self, preprocessed_data: Any, **kwargs) -> "ResourceAwareDetector":
        # Clean up previous model if exists
        if self._model is not None:
            del self._model

        # Create new model
        self._model = SomeHeavyModel()
        self._model.fit(preprocessed_data)
        self._fitted = True
        return self

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
```

### 4. Reproducibility

```python
def __init__(self, method_id: str, variant_id: str, **kwargs):
    super().__init__(method_id, variant_id)

    # Use consistent random seed
    self.random_state = kwargs.get('random_state', 42)
    np.random.seed(self.random_state)

    # Initialize with deterministic parameters
    self.threshold = kwargs.get('threshold', 0.05)
```

### 5. Performance Optimization

```python
def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
    """Optimized preprocessing with caching."""
    phase = kwargs.get('phase', 'train')

    # Cache preprocessing results to avoid recomputation
    cache_key = f"{phase}_{id(data)}"
    if hasattr(self, '_preprocess_cache') and cache_key in self._preprocess_cache:
        return self._preprocess_cache[cache_key]

    # Perform preprocessing
    if phase == 'train':
        result = self._preprocess_training_data(data.X_ref)
    else:
        result = self._preprocess_test_data(data.X_test)

    # Cache result
    if not hasattr(self, '_preprocess_cache'):
        self._preprocess_cache = {}
    self._preprocess_cache[cache_key] = result

    return result
```

### 6. Documentation and Type Hints

```python
from typing import Any, Optional, Union
import numpy as np

class WellDocumentedDetector(BaseDetector):
    """
    A well-documented drift detector variants.

    This detector uses [algorithm name] to detect [drift type] in [data type].

    Parameters:
        threshold (float): Detection threshold (default: 0.05)
        window_size (int): Rolling window size for streaming mode (default: 100)

    References:
        - Author et al. (Year). "Paper Title". Journal Name.
        - https://doi.org/example
    """

    def __init__(self, method_id: str, variant_id: str, **kwargs):
        """Initialize detector with parameters."""
        super().__init__(method_id, variant_id)
        self.threshold: float = kwargs.get('threshold', 0.05)
        self.window_size: int = kwargs.get('window_size', 100)

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """
        Preprocess data for detection algorithm.

        Args:
            data: Dataset containing reference and test data
            **kwargs: Additional preprocessing parameters
                - phase (str): 'train' or 'detect'

        Returns:
            Preprocessed data as numpy array

        Raises:
            ValueError: If data is empty or invalid format
        """
        # Variants with clear steps
        pass

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "WellDocumentedDetector":
        """
        Train the detector on reference data.

        Args:
            preprocessed_data: Reference data in detector format
            **kwargs: Additional training parameters

        Returns:
            Self for method chaining

        Raises:
            ValueError: If training data is invalid
            RuntimeError: If training fails
        """
        # Variants
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """
        Detect drift in test data.

        Args:
            preprocessed_data: Test data in detector format
            **kwargs: Additional detection parameters

        Returns:
            True if drift is detected, False otherwise

        Raises:
            RuntimeError: If detector not fitted or detection fails
        """
        # Variants
        pass

    def score(self) -> Optional[float]:
        """
        Return drift score from last detection.

        Returns:
            Drift score in range [0, 1] or None if not available
            Higher scores indicate stronger drift evidence
        """
        return self._last_score
```

---

This completes the comprehensive Adapter API documentation. The framework provides a flexible, extensible system for integrating various drift detection libraries while maintaining consistency and ease of use.
