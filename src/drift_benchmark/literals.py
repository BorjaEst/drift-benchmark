"""
Literal type definitions for drift-benchmark - REQ-LIT-XXX

This module defines literal types that provide type safety throughout
the drift-benchmark library.
"""

from typing_extensions import Literal

# REQ-LIT-001: Drift Type Literals
DriftType = Literal["covariate", "concept", "prior"]

# REQ-LIT-002: Data Type Literals
DataType = Literal["continuous", "categorical", "mixed"]

# REQ-LIT-003: Dimension Literals
DataDimension = Literal["univariate", "multivariate"]

# REQ-LIT-004: Labeling Literals
DataLabeling = Literal["supervised", "unsupervised", "semi-supervised"]

# REQ-LIT-005: Execution Mode Literals
ExecutionMode = Literal["batch", "streaming"]

# REQ-LIT-006: Method Family Literals
MethodFamily = Literal["statistical-test", "distance-based", "change-detection", "window-based", "statistical-process-control"]

# REQ-LIT-007: Dataset Source Literals (DEPRECATED)
DatasetSource = Literal["file", "synthetic"]

# REQ-LIT-011: Scenario Source Type Literals
ScenarioSourceType = Literal["sklearn", "file"]

# REQ-LIT-008: File Format Literals
FileFormat = Literal["csv"]

# REQ-LIT-009: Log Level Literals
LogLevel = Literal["debug", "info", "warning", "error", "critical"]

# REQ-LIT-010: Library ID Literals
LibraryId = Literal["evidently", "alibi-detect", "scikit-learn", "river", "scipy", "custom"]
