"""
Configuration loading module for drift-benchmark - REQ-CFG-XXX

This module provides configuration loading utilities that return validated
BenchmarkConfig instances from TOML files. Implements REQ-CFG-007 separation
of concerns by keeping file I/O logic separate from model definitions.
"""

from .loader import load_config

__all__ = [
    "load_config",
]
