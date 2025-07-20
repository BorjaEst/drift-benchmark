"""
Results module for drift-benchmark - REQ-RST-XXX

This module provides basic results management for storing benchmark results.
"""

from .results_core import Results
from .results_manager import save_results

__all__ = [
    "Results",
    "save_results",
]
