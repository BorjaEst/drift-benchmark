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
7. [Library Comparison Usage](#library-comparison-usage)
8. [Type System and Error Handling](#type-system-and-error-handling)
9. [Testing Adapters](#testing-adapters)
10. [Best Practices](#best-practices)

---

## 📋 Overview

This document is for **developers who want to integrate new drift detection libraries** into the drift-benchmark framework. If you need to compare existing library implementations, see [BENCHMARK-API.md](BENCHMARK-API.md).

### 🎯 Primary Purpose

**Create adapter classes** that enable your preferred drift detection library to participate in benchmarks. This document shows how to:

- **Extend BaseDetector**: Implement the required interface for your library
- **Map Library APIs**: Bridge your library's specific API to our standardized interface
- **Handle Data Formats**: Convert between pandas DataFrames and your library's expected format
- **Register Adapters**: Make your adapters discoverable by the benchmark framework

### When to Use This Document

Use this guide when:

- ✅ You want to add support for a **new library** (e.g., custom implementation)
- ✅ No existing adapter exists for your **method+variant+library** combination
- ✅ You need to **implement BaseDetector** for your specific library
- ✅ You want to **extend framework capabilities** with new libraries

**Prerequisites**: Basic understanding of your target library's API and Python inheritance patterns.

**Next Steps**: After creating adapters, use [BENCHMARK-API.md](BENCHMARK-API.md) to configure and run library comparisons.

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

## 🔌 Creating Library Adapters

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
        super().__init__(method_id, variant_id, library_id)
        self.threshold = kwargs.get('threshold', 0.05)
        self._reference_data = None
        self._last_score = None

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Evidently prefers pandas DataFrames."""
        phase = kwargs.get('phase', 'train')
        return data.X_ref if phase == 'train' else data.X_test

    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """Store reference data for Evidently."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Use Evidently's data drift detection."""
        from evidently.metrics import DataDriftPreset

        report = DataDriftPreset().run(self._reference_data, preprocessed_data)
        result = report.metrics[0].result

        self._last_score = result.drift_score
        return result.dataset_drift

# Alibi-Detect's implementation of the SAME method+variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="alibi_detect")
class AlibiDetectKSDetector(BaseDetector):
    """Alibi-Detect's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.threshold = kwargs.get('threshold', 0.05)
        self._detector = None
        self._last_score = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Alibi-Detect prefers numpy arrays."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Convert to numeric and handle missing values
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values.astype(np.float32)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Initialize Alibi-Detect KS detector."""
        from alibi_detect.cd import KSDrift

        self._detector = KSDrift(preprocessed_data, p_val=self.threshold)
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Use Alibi-Detect's KS drift detection."""
        result = self._detector.predict(preprocessed_data)

        self._last_score = result['data']['p_val']
        return result['data']['is_drift']

# SciPy's implementation of the SAME method+variant
@register_detector(method_id="kolmogorov_smirnov", variant_id="batch", library_id="scipy")
class SciPyKSDetector(BaseDetector):
    """SciPy's implementation of Kolmogorov-Smirnov batch processing."""

    def __init__(self, method_id: str, variant_id: str, library_id: str, **kwargs):
        super().__init__(method_id, variant_id, library_id)
        self.threshold = kwargs.get('threshold', 0.05)
        self._reference_data = None
        self._last_score = None

    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """SciPy works with numpy arrays."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Convert to numeric data for statistical tests
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Store reference data for SciPy comparison."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Use SciPy's Kolmogorov-Smirnov test."""
        from scipy.stats import ks_2samp

        # Compare each feature independently for multivariate data
        p_values = []
        for i in range(min(self._reference_data.shape[1], preprocessed_data.shape[1])):
            _, p_val = ks_2samp(self._reference_data[:, i], preprocessed_data[:, i])
            p_values.append(p_val)

        # Use minimum p-value (most significant drift)
        min_p_value = min(p_values)
        self._last_score = min_p_value

        return min_p_value < self.threshold
```

### Step 3: Test Your Adapters

Before using your adapters in benchmarks, test them individually:

```python
# Test your adapter implementation
from drift_benchmark.data import DatasetResult
import pandas as pd

# Create test data
test_data = DatasetResult(
    X_ref=pd.DataFrame({'feature1': [1, 2, 3, 4, 5]}),
    X_test=pd.DataFrame({'feature1': [6, 7, 8, 9, 10]})
)

# Test your adapter
detector = EvidentlyKSDetector(
    method_id="kolmogorov_smirnov",
    variant_id="batch",
    library_id="evidently",
    threshold=0.05
)

# Test preprocessing
ref_data = detector.preprocess(test_data, phase='train')
test_data_processed = detector.preprocess(test_data, phase='detect')

# Test training and detection
detector.fit(ref_data)
drift_detected = detector.detect(test_data_processed)
print(f"Drift detected: {drift_detected}")
```

### Step 4: Configure and Run Benchmarks

Once your adapters are implemented and tested, configure comparative benchmarks using [BENCHMARK-API.md](BENCHMARK-API.md). The benchmark configuration will automatically discover your registered adapters and enable library comparison.

For complete examples of configuring library comparisons, result analysis, and performance evaluation, see the [BENCHMARK-API documentation](BENCHMARK-API.md).

---

## 🔧 Library-Specific Integration Patterns

### Evidently Integration

```python
class EvidentlyAdapter(BaseDetector):
    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Evidently usually works with pandas DataFrames."""
        phase = kwargs.get('phase', 'train')
        return data.X_ref if phase == 'train' else data.X_test

    def fit(self, preprocessed_data: Any, **kwargs) -> "BaseDetector":
        """Evidently often doesn't require explicit training."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Use Evidently's built-in metrics."""
        from evidently.metrics import DataDriftPreset

        report = DataDriftPreset().run(self._reference_data, preprocessed_data)
        result = report.metrics[0].result

        self._last_score = getattr(result, 'drift_score', None)
        return result['metrics'][0]['result']['dataset_drift']
```

### Alibi-Detect Integration

```python
class AlibiDetectAdapter(BaseDetector):
    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Alibi-Detect usually expects numpy arrays."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Convert to float32 for memory efficiency
        numeric_data = df.select_dtypes(include=[np.number])
        data_array = numeric_data.fillna(numeric_data.mean()).values
        return data_array.astype(np.float32)

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Initialize Alibi-Detect detector with reference data."""
        from alibi_detect.cd import KSDrift  # or other detectors

        self._detector = KSDrift(
            x_ref=preprocessed_data,
            p_val=self.threshold,
            **kwargs
        )
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Use Alibi-Detect's prediction interface."""
        result = self._detector.predict(preprocessed_data)

        self._last_score = result['data'].get('p_val')
        return result['data']['is_drift']
```

### SciPy/scikit-learn Integration

```python
class SciPyAdapter(BaseDetector):
    def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
        """Convert to numpy arrays for scikit-learn/SciPy."""
        phase = kwargs.get('phase', 'train')
        df = data.X_ref if phase == 'train' else data.X_test

        # Handle categorical data with encoding if needed
        numeric_data = df.select_dtypes(include=[np.number])
        return numeric_data.fillna(numeric_data.mean()).values

    def fit(self, preprocessed_data: np.ndarray, **kwargs) -> "BaseDetector":
        """Store reference data for statistical tests."""
        self._reference_data = preprocessed_data
        return self

    def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
        """Use SciPy statistical tests."""
        from scipy.stats import ks_2samp

        # Handle multivariate data
        p_values = []
        for i in range(min(self._reference_data.shape[1], preprocessed_data.shape[1])):
            _, p_val = ks_2samp(
                self._reference_data[:, i],
                preprocessed_data[:, i]
            )
            p_values.append(p_val)

        # Combine p-values (using minimum for conservative approach)
        combined_p = min(p_values) if p_values else 1.0
        self._last_score = combined_p

        return combined_p < self.threshold
```

### Common Data Preprocessing Patterns

#### Handling Mixed Data Types

```python
def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
    """Handle mixed continuous/categorical data."""
    phase = kwargs.get('phase', 'train')
    df = data.X_ref if phase == 'train' else data.X_test

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number])
    categorical_cols = df.select_dtypes(include=['object', 'category'])

    # Handle numeric data
    numeric_data = numeric_cols.fillna(numeric_cols.mean()).values

    # Handle categorical data (simple label encoding)
    if not categorical_cols.empty:
        from sklearn.preprocessing import LabelEncoder
        encoded_categorical = []
        for col in categorical_cols.columns:
            le = LabelEncoder()
            encoded_col = le.fit_transform(categorical_cols[col].fillna('missing'))
            encoded_categorical.append(encoded_col)

        categorical_data = np.column_stack(encoded_categorical)
        return np.column_stack([numeric_data, categorical_data])

    return numeric_data.fillna(numeric_data.mean()).values
```

#### Categorical Data

```python
def preprocess(self, data: DatasetResult, **kwargs) -> np.ndarray:
    """Handle categorical data with encoding."""
    phase = kwargs.get('phase', 'train')
    df = data.X_ref if phase == 'train' else data.X_test

    # Use pandas get_dummies for one-hot encoding
    encoded_data = pd.get_dummies(df, dummy_na=True)
    return encoded_data.values
```

#### Multivariate Data

```python
def detect(self, preprocessed_data: np.ndarray, **kwargs) -> bool:
    """Handle multivariate drift detection."""
    # Example using multiple univariate tests
    p_values = []

    for i in range(min(self._reference_data.shape[1], preprocessed_data.shape[1])):
        _, p_val = self._statistical_test(
            self._reference_data[:, i],
            preprocessed_data[:, i]
        )
        p_values.append(p_val)

    # Apply Bonferroni correction for multiple testing
    corrected_threshold = self.threshold / len(p_values)
    combined_p_value = min(p_values)

    self._last_score = combined_p_value
    return combined_p_value < corrected_threshold
```

---

## 🗂️ Registration System

### Decorator Registration

The framework uses decorator-based registration for automatic discovery:

```python
from drift_benchmark.adapters import register_detector

@register_detector(method_id="my_method", variant_id="my_impl", library_id="my_library")
class MyDetector(BaseDetector):
    # Implementation here
    pass
```

### Registration Requirements

1. **Method ID**: Must exist in `methods.toml` registry
2. **Variant ID**: Must be listed under the method in `methods.toml`
3. **Library ID**: Must be a valid library identifier ("evidently", "alibi_detect", "scipy", etc.)
4. **Class Inheritance**: Must inherit from `BaseDetector`
5. **Import Side Effects**: Registration happens at import time

### Registry Validation

```python
# The registration system validates:
# 1. Method exists in methods.toml
# 2. Variant exists under the method
# 3. No duplicate registrations for same method+variant+library combination
# 4. BaseDetector inheritance

from drift_benchmark.exceptions import (
    MethodNotFoundError,
    VariantNotFoundError,
    DuplicateDetectorError
)

try:
    @register_detector(method_id="nonexistent", variant_id="test", library_id="custom")
    class BadDetector(BaseDetector):
        pass
except MethodNotFoundError:
    print("Method 'nonexistent' not found in registry")
```

### Registry Lookup

```python
from drift_benchmark.adapters import get_detector_class, list_detectors

# Get specific detector class
DetectorClass = get_detector_class(
    method_id="kolmogorov_smirnov",
    variant_id="batch",
    library_id="evidently"
)

# List all registered detectors
all_detectors = list_detectors()
print(f"Available detectors: {len(all_detectors)}")

# Filter by method
ks_detectors = [d for d in all_detectors if d[0] == "kolmogorov_smirnov"]
print(f"KS implementations: {[d[2] for d in ks_detectors]}")  # Show library IDs
```

---

## 🔄 Data Flow and Lifecycle

### Complete Execution Flow

```text
1. BenchmarkRunner loads configuration
2. For each detector config:
   a. Lookup detector class by (method_id, variant_id, library_id)
   b. Instantiate detector with hyperparameters
3. For each dataset:
   a. Load and split data into X_ref/X_test
4. For each detector-dataset combination:
   a. preprocess(dataset, phase='train') → ref_data
   b. detector.fit(ref_data)
   c. preprocess(dataset, phase='detect') → test_data
   d. detector.detect(test_data) → drift_detected
   e. detector.score() → drift_score
   f. Record execution time and results
5. Aggregate results and save to timestamped directory
```

### Preprocessing Phases

The `preprocess()` method is called twice per detector-dataset pair:

```python
# Phase 1: Training data preparation
reference_data = detector.preprocess(dataset_result, phase='train')
detector.fit(reference_data)

# Phase 2: Test data preparation
test_data = detector.preprocess(dataset_result, phase='detect')
drift_detected = detector.detect(test_data)
```

### Error Handling Flow

```python
# Benchmark execution continues even if individual detectors fail
for detector_config in benchmark_config.detectors:
    try:
        # Instantiate detector
        detector = create_detector(detector_config)

        for dataset in datasets:
            try:
                # Execute detector on dataset
                result = run_detector(detector, dataset)
                results.append(result)
            except Exception as e:
                # Log error and continue with next dataset
                logger.error(f"Detector {detector.library_id} failed on {dataset.name}: {e}")
                results.append(create_error_result(detector, dataset, e))

    except Exception as e:
        # Log error and continue with next detector
        logger.error(f"Failed to create detector {detector_config.library_id}: {e}")
```

---

## 📊 Library Comparison Usage

Once you've created adapters for your libraries, you can use them in comparative benchmarks. For complete examples of:

- **Configuration Setup**: How to configure benchmarks to compare your adapters
- **Performance Analysis**: Statistical comparison of library implementations
- **Result Interpretation**: Understanding which library performs better
- **Visualization**: Plotting comparative results

See the [BENCHMARK-API documentation](BENCHMARK-API.md#library-comparison-examples).

**Quick Reference**: Your registered adapters will automatically be discoverable by the benchmark framework when you specify the corresponding `method_id`, `variant_id`, and `library_id` in your benchmark configuration.

---

## 🧪 Type System and Error Handling

### Type Annotations

```python
from typing import Any, Optional, Dict, List
from drift_benchmark.models.results import DatasetResult
from drift_benchmark.literals import LibraryId

class LibraryAdapter(BaseDetector):
    def __init__(self, method_id: str, variant_id: str, library_id: LibraryId, **kwargs):
        super().__init__(method_id, variant_id, library_id)

    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        # Return type depends on library requirements
        pass

    def fit(self, preprocessed_data: Any, **kwargs) -> "LibraryAdapter":
        # Always return self for chaining
        pass

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        # Always return boolean
        pass

    def score(self) -> Optional[float]:
        # Return score or None
        pass
```

### Error Handling Patterns

```python
from drift_benchmark.exceptions import (
    DetectorNotFoundError,
    DataValidationError,
    BenchmarkExecutionError
)

class RobustAdapter(BaseDetector):
    def preprocess(self, data: DatasetResult, **kwargs) -> Any:
        """Robust preprocessing with error handling."""
        try:
            phase = kwargs.get('phase', 'train')
            df = data.X_ref if phase == 'train' else data.X_test

            if df.empty:
                raise DataValidationError(f"Empty dataset for phase '{phase}'")

            # Convert to library format with validation
            processed_data = self._convert_to_library_format(df)

            if processed_data is None:
                raise DataValidationError("Data conversion failed")

            return processed_data

        except Exception as e:
            logger.error(f"Preprocessing failed for {self.library_id}: {e}")
            raise DataValidationError(f"Preprocessing error: {e}") from e

    def detect(self, preprocessed_data: Any, **kwargs) -> bool:
        """Detection with error handling and fallback."""
        try:
            # Primary detection logic
            result = self._run_detection(preprocessed_data)
            return result

        except Exception as e:
            logger.warning(f"Detection failed for {self.library_id}: {e}")

            # Fallback: return conservative result
            return False  # or True, depending on use case

    def _convert_to_library_format(self, df):
        """Library-specific conversion with validation."""
        # Implementation depends on library requirements
        pass

    def _run_detection(self, data):
        """Core detection logic with library-specific implementation."""
        # Implementation depends on library
        pass
```

### Validation Helpers

```python
def validate_data_shape(data: Any, expected_dims: int = 2) -> bool:
    """Validate data has expected dimensions."""
    if hasattr(data, 'shape'):
        return len(data.shape) == expected_dims
    return False

def validate_library_availability(library_name: str) -> bool:
    """Check if required library is available."""
    try:
        __import__(library_name)
        return True
    except ImportError:
        return False

class ValidationMixin:
    """Mixin for common validation patterns."""

    def validate_preprocessing_output(self, data: Any) -> bool:
        """Validate preprocessed data format."""
        return (data is not None and
                hasattr(data, '__len__') and
                len(data) > 0)

    def validate_threshold(self, threshold: float) -> bool:
        """Validate threshold is in valid range."""
        return 0.0 < threshold < 1.0
```

---

## 🧪 Testing Adapters

### Unit Testing Structure

```python
import pytest
import numpy as np
import pandas as pd
from drift_benchmark.models.results import DatasetResult
from drift_benchmark.adapters import get_detector_class

class TestLibraryAdapter:
    """Test suite for library-specific adapters."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)

        # Reference data (normal distribution)
        X_ref = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })

        # Test data (shifted distribution to simulate drift)
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1, 100),  # Mean shift
            'feature2': np.random.normal(0, 1.2, 100)   # Variance change
        })

        return DatasetResult(X_ref=X_ref, X_test=X_test, metadata={})

    @pytest.fixture
    def detector_class(self):
        """Get detector class for testing."""
        return get_detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"  # or whichever library you're testing
        )

    def test_detector_initialization(self, detector_class):
        """Test detector can be initialized correctly."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy",
            threshold=0.05
        )

        assert detector.method_id == "kolmogorov_smirnov"
        assert detector.variant_id == "batch"
        assert detector.library_id == "scipy"

    def test_preprocessing(self, detector_class, sample_dataset):
        """Test data preprocessing produces expected format."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"
        )

        # Test training data preprocessing
        ref_data = detector.preprocess(sample_dataset, phase='train')
        assert ref_data is not None
        assert len(ref_data) > 0

        # Test detection data preprocessing
        test_data = detector.preprocess(sample_dataset, phase='detect')
        assert test_data is not None
        assert len(test_data) > 0

        # Data should have same shape for same dataset
        if hasattr(ref_data, 'shape') and hasattr(test_data, 'shape'):
            assert ref_data.shape[1] == test_data.shape[1]  # Same features

    def test_training(self, detector_class, sample_dataset):
        """Test detector training completes successfully."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"
        )

        ref_data = detector.preprocess(sample_dataset, phase='train')
        trained_detector = detector.fit(ref_data)

        # Should return self for chaining
        assert trained_detector is detector

    def test_detection(self, detector_class, sample_dataset):
        """Test drift detection produces boolean result."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy",
            threshold=0.05
        )

        # Train detector
        ref_data = detector.preprocess(sample_dataset, phase='train')
        detector.fit(ref_data)

        # Run detection
        test_data = detector.preprocess(sample_dataset, phase='detect')
        drift_detected = detector.detect(test_data)

        # Should return boolean
        assert isinstance(drift_detected, bool)

        # With shifted data, should likely detect drift
        # (though this depends on random seed and threshold)

    def test_scoring(self, detector_class, sample_dataset):
        """Test drift scoring after detection."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"
        )

        # Train and detect
        ref_data = detector.preprocess(sample_dataset, phase='train')
        detector.fit(ref_data)

        test_data = detector.preprocess(sample_dataset, phase='detect')
        detector.detect(test_data)

        # Get score
        score = detector.score()

        # Score should be float or None
        assert score is None or isinstance(score, (int, float))

        if score is not None:
            assert 0.0 <= score <= 1.0  # Assuming p-value or similar

    def test_error_handling(self, detector_class):
        """Test detector handles errors gracefully."""
        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"
        )

        # Test with empty data
        empty_data = pd.DataFrame()
        empty_dataset = DatasetResult(X_ref=empty_data, X_test=empty_data, metadata={})

        with pytest.raises((ValueError, IndexError, Exception)):
            ref_data = detector.preprocess(empty_dataset, phase='train')
            detector.fit(ref_data)
```

### Integration Testing

```python
class TestLibraryComparison:
    """Integration tests for comparing different library implementations."""

    @pytest.fixture
    def drift_dataset(self):
        """Create dataset with known drift for comparison testing."""
        np.random.seed(42)

        # Clear distribution shift
        X_ref = pd.DataFrame({'x': np.random.normal(0, 1, 200)})
        X_test = pd.DataFrame({'x': np.random.normal(2, 1, 200)})  # Mean shift of 2

        return DatasetResult(X_ref=X_ref, X_test=X_test, metadata={})

    def test_library_consistency(self, drift_dataset):
        """Test that different libraries detect the same obvious drift."""
        libraries = ["scipy", "evidently", "alibi_detect"]  # Available libraries
        results = {}

        for library_id in libraries:
            try:
                detector_class = get_detector_class(
                    method_id="kolmogorov_smirnov",
                    variant_id="batch",
                    library_id=library_id
                )

                detector = detector_class(
                    method_id="kolmogorov_smirnov",
                    variant_id="batch",
                    library_id=library_id,
                    threshold=0.05
                )

                # Run detection
                ref_data = detector.preprocess(drift_dataset, phase='train')
                detector.fit(ref_data)

                test_data = detector.preprocess(drift_dataset, phase='detect')
                drift_detected = detector.detect(test_data)

                results[library_id] = {
                    'drift_detected': drift_detected,
                    'score': detector.score()
                }

            except DetectorNotFoundError:
                # Library not available, skip
                continue

        # All available libraries should detect this obvious drift
        detected_counts = sum(1 for r in results.values() if r['drift_detected'])
        assert detected_counts >= len(results) * 0.8  # At least 80% should detect

    def test_performance_comparison(self, drift_dataset):
        """Test relative performance of different libraries."""
        import time

        libraries = ["scipy", "evidently"]  # Fast libraries
        timings = {}

        for library_id in libraries:
            try:
                detector_class = get_detector_class(
                    method_id="kolmogorov_smirnov",
                    variant_id="batch",
                    library_id=library_id
                )

                detector = detector_class(
                    method_id="kolmogorov_smirnov",
                    variant_id="batch",
                    library_id=library_id
                )

                # Measure execution time
                start_time = time.perf_counter()

                ref_data = detector.preprocess(drift_dataset, phase='train')
                detector.fit(ref_data)

                test_data = detector.preprocess(drift_dataset, phase='detect')
                detector.detect(test_data)

                end_time = time.perf_counter()

                timings[library_id] = end_time - start_time

            except DetectorNotFoundError:
                continue

        # Basic performance sanity check
        for library_id, timing in timings.items():
            assert timing < 1.0  # Should complete within 1 second

        print(f"Performance comparison: {timings}")
```

### Mock Testing for Library Dependencies

```python
from unittest.mock import Mock, patch

class TestWithMockedLibraries:
    """Test adapters without requiring actual library installations."""

    @patch('scipy.stats.ks_2samp')
    def test_scipy_adapter_mocked(self, mock_ks_2samp):
        """Test SciPy adapter with mocked scipy."""
        # Mock scipy function
        mock_ks_2samp.return_value = (0.5, 0.03)  # statistic, p_value

        detector_class = get_detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy"
        )

        detector = detector_class(
            method_id="kolmogorov_smirnov",
            variant_id="batch",
            library_id="scipy",
            threshold=0.05
        )

        # Create sample data
        sample_data = DatasetResult(
            X_ref=pd.DataFrame({'x': [1, 2, 3]}),
            X_test=pd.DataFrame({'x': [4, 5, 6]}),
            metadata={}
        )

        # Run detection
        ref_data = detector.preprocess(sample_data, phase='train')
        detector.fit(ref_data)

        test_data = detector.preprocess(sample_data, phase='detect')
        drift_detected = detector.detect(test_data)

        # With p_value=0.03 < threshold=0.05, should detect drift
        assert drift_detected is True
        assert detector.score() == 0.03

        # Verify scipy was called
        mock_ks_2samp.assert_called()
```

---

## 🏆 Best Practices

### Adapter Design Guidelines

1. **Library-Specific Optimization**: Leverage each library's strengths

   ```python
   # Good: Use library's preferred data format
   def preprocess(self, data, **kwargs):
       # For evidently: keep as pandas
       if self.library_id == "evidently":
           return data.X_ref if kwargs.get('phase') == 'train' else data.X_test

       # For alibi-detect: convert to numpy
       elif self.library_id == "alibi_detect":
           df = data.X_ref if kwargs.get('phase') == 'train' else data.X_test
           return df.values.astype(np.float32)
   ```

2. **Consistent Interface**: Maintain same behavior across libraries

   ```python
   # Good: Consistent boolean output
   def detect(self, preprocessed_data, **kwargs) -> bool:
       # All libraries should return boolean, regardless of internal format
       result = self._library_specific_detection(preprocessed_data)
       return bool(result)  # Ensure boolean conversion
   ```

3. **Error Isolation**: Don't let library-specific errors break the benchmark
   ```python
   def detect(self, preprocessed_data, **kwargs) -> bool:
       try:
           return self._run_library_detection(preprocessed_data)
       except ImportError:
           raise DetectorNotFoundError(f"Library {self.library_id} not available")
       except Exception as e:
           logger.error(f"Detection failed for {self.library_id}: {e}")
           # Return conservative result instead of crashing
           return False
   ```

### Performance Optimization

1. **Lazy Imports**: Only import heavy libraries when needed

   ```python
   def fit(self, preprocessed_data, **kwargs):
       # Import only when method is called
       from alibi_detect.cd import KSDrift

       self._detector = KSDrift(preprocessed_data, p_val=self.threshold)
       return self
   ```

2. **Data Format Efficiency**: Minimize conversions

   ```python
   def preprocess(self, data, **kwargs):
       # Avoid multiple conversions
       df = data.X_ref if kwargs.get('phase') == 'train' else data.X_test

       # Convert once to most efficient format for library
       if self.library_id == "alibi_detect":
           return df.values.astype(np.float32)  # Memory efficient

       return df  # Keep as pandas if library prefers it
   ```

3. **Caching Reference Data**: Avoid reprocessing

   ```python
   def fit(self, preprocessed_data, **kwargs):
       # Cache expensive computations
       if not hasattr(self, '_reference_statistics'):
           self._reference_statistics = self._compute_statistics(preprocessed_data)

       self._reference_data = preprocessed_data
       return self
   ```

### Library Comparison Strategies

1. **Fair Benchmarking**: Ensure identical conditions

   ```python
   # Configure all libraries with equivalent parameters
   detectors_config = [
       {
           "method_id": "kolmogorov_smirnov",
           "variant_id": "batch",
           "library_id": "scipy",
           "threshold": 0.05
       },
       {
           "method_id": "kolmogorov_smirnov",
           "variant_id": "batch",
           "library_id": "evidently",
           "threshold": 0.05  # Same threshold
       }
   ]
   ```

2. **Meaningful Metrics**: Track relevant performance indicators

   ```python
   # After benchmark execution
   for result in results.detector_results:
       print(f"Library: {result.library_id}")
       print(f"Accuracy: {result.accuracy:.3f}")
       print(f"Speed: {result.execution_time:.4f}s")
       print(f"Memory: {result.memory_usage:.1f}MB")  # If available
   ```

3. **Statistical Significance**: Consider multiple runs

   ```python
   # Run benchmark multiple times for stable comparisons
   import statistics

   timings_by_library = defaultdict(list)

   for _ in range(10):  # Multiple runs
       results = runner.run()
       for result in results.detector_results:
           timings_by_library[result.library_id].append(result.execution_time)

   # Report mean and standard deviation
   for library_id, timings in timings_by_library.items():
       mean_time = statistics.mean(timings)
       std_time = statistics.stdev(timings)
       print(f"{library_id}: {mean_time:.4f}s ± {std_time:.4f}s")
   ```

### Documentation and Maintenance

1. **Clear Library Dependencies**: Document requirements

   ```python
   class EvidentlyKSDetector(BaseDetector):
       """
       Evidently's implementation of Kolmogorov-Smirnov batch processing.

       Requirements:
           - evidently >= 0.3.0
           - pandas >= 1.3.0

       Strengths:
           - Excellent pandas integration
           - Rich visualization capabilities
           - Easy configuration

       Limitations:
           - Slower on large datasets
           - Limited to pandas DataFrames
       """
   ```

2. **Version Compatibility**: Handle library updates

   ```python
   def _get_evidently_version(self):
       import evidently
       return evidently.__version__

   def detect(self, preprocessed_data, **kwargs):
       version = self._get_evidently_version()

       if version >= "0.4.0":
           # Use new API
           return self._detect_v4(preprocessed_data)
       else:
           # Use legacy API
           return self._detect_legacy(preprocessed_data)
   ```

3. **Comprehensive Examples**: Show library-specific usage

   ```python
   # examples/evidently_vs_alibi_detect.py
   """
   Compare Evidently and Alibi-Detect implementations of KS test.

   This example shows:
   1. How to set up identical configurations
   2. Performance comparison methodology
   3. Result interpretation guidelines
   """
   ```

This comprehensive adapter API documentation provides everything needed to create library-specific adapters that enable fair comparison of drift detection implementations across different libraries. The focus is on enabling researchers and practitioners to choose the best library for their specific use case based on empirical performance data.
