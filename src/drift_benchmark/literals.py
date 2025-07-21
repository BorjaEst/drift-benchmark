"""
Literal type definitions for drift-benchmark - REQ-LIT-XXX

This module defines literal types that provide type safety throughout
the drift-benchmark library.
"""

from typing_extensions import Literal

# REQ-LIT-001: Drift Type Literals
DriftType = Literal["COVARIATE", "CONCEPT", "PRIOR"]

# REQ-LIT-002: Data Type Literals
DataType = Literal["CONTINUOUS", "CATEGORICAL", "MIXED"]

# REQ-LIT-003: Dimension Literals
DataDimension = Literal["UNIVARIATE", "MULTIVARIATE"]

# REQ-LIT-004: Labeling Literals
DataLabeling = Literal["SUPERVISED", "UNSUPERVISED", "SEMI_SUPERVISED"]

# REQ-LIT-005: Execution Mode Literals
ExecutionMode = Literal["BATCH", "STREAMING"]

# REQ-LIT-006: Method Family Literals
MethodFamily = Literal["STATISTICAL_TEST", "DISTANCE_BASED", "CHANGE_DETECTION", "WINDOW_BASED"]

# REQ-LIT-007: Dataset Source Literals
DatasetSource = Literal["FILE", "SYNTHETIC"]

# REQ-LIT-008: File Format Literals
FileFormat = Literal["CSV"]

# REQ-LIT-009: Log Level Literals
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
