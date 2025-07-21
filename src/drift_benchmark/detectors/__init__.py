"""
Detectors module for drift-benchmark - REQ-DET-XXX

This module provides a registry for drift detection methods through
the methods.toml configuration file.
"""

# Import settings for test mocking
from ..settings import settings
from .registry import get_method, get_variant, list_methods, load_methods

__all__ = [
    "load_methods",
    "get_method",
    "get_variant",
    "list_methods",
    "settings",  # Expose for test mocking
]
