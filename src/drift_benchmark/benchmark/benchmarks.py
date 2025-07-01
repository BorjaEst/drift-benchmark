"""
Legacy benchmark module - DEPRECATED

This module contains the original monolithic BenchmarkRunner implementation.
For new code, please use the modular execution system from:
- drift_benchmark.benchmark.execution.BenchmarkRunner (new modular runner)
- drift_benchmark.benchmark.execution.ExecutionStrategy (for different execution strategies)

The legacy BenchmarkRunner is maintained for backward compatibility but will
be removed in a future version.
"""

import warnings

# Import the new modular components
from drift_benchmark.benchmark.execution import BenchmarkRunner as NewBenchmarkRunner
from drift_benchmark.benchmark.execution import ExecutionContext, ParallelExecutionStrategy, SequentialExecutionStrategy

# Show deprecation warning
warnings.warn(
    "The benchmarks.py module is deprecated. Please use the new modular execution system "
    "from drift_benchmark.benchmark.execution instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BenchmarkRunner(NewBenchmarkRunner):
    """
    Legacy BenchmarkRunner for backward compatibility.

    This class wraps the new modular BenchmarkRunner to maintain API compatibility.
    All new features should be implemented in the modular system.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Using legacy BenchmarkRunner. Please migrate to " "drift_benchmark.benchmark.execution.BenchmarkRunner",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# Maintain backward compatibility by re-exporting the main class
__all__ = ["BenchmarkRunner"]
