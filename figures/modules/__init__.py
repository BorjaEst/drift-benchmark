"""
Figure modules for drift-benchmark analysis.

This package provides specialized visualization modules for different aspects
of drift detection benchmark analysis. Each module focuses on a specific
comparison type and provides research-quality figures suitable for publication.

Modules:
    - performance_comparison: Library and method performance analysis
    - scenario_analysis: Drift scenarios and complexity analysis
    - execution_mode_analysis: Batch vs streaming comparison
    - method_family_analysis: Statistical vs distance vs streaming methods
    - adapter_comparison: Cross-library adapter implementation analysis
"""

try:
    from .adapter_comparison import create_adapter_comparison_figure
    from .execution_mode_analysis import create_execution_mode_figure
    from .method_family_analysis import create_method_family_figure
    from .performance_comparison import create_performance_comparison_figure
    from .scenario_analysis import create_scenario_analysis_figure
except ImportError as e:
    # Handle missing dependencies gracefully
    def create_performance_comparison_figure(*args, **kwargs):
        raise ImportError(f"Performance comparison module not available: {e}")

    def create_scenario_analysis_figure(*args, **kwargs):
        raise ImportError(f"Scenario analysis module not available: {e}")

    def create_execution_mode_figure(*args, **kwargs):
        raise ImportError(f"Execution mode module not available: {e}")

    def create_method_family_figure(*args, **kwargs):
        raise ImportError(f"Method family module not available: {e}")

    def create_adapter_comparison_figure(*args, **kwargs):
        raise ImportError(f"Adapter comparison module not available: {e}")


__all__ = [
    "create_performance_comparison_figure",
    "create_scenario_analysis_figure",
    "create_execution_mode_figure",
    "create_method_family_figure",
    "create_adapter_comparison_figure",
]
