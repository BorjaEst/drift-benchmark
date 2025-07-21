"""
Adapters module for drift-benchmark - REQ-ADP-XXX

This module provides the adapter framework for integrating drift detection
libraries with the drift-benchmark framework.
"""

from .base_detector import BaseDetector
from .registry import get_detector_class, list_detectors, register_detector

# Import test detectors to register them
try:
    from . import test_detectors  # This will execute the decorators
except ImportError:
    pass  # Test detectors are optional

# Import real detector implementations
try:
    from . import statistical_detectors  # Statistical test-based detectors
except ImportError:
    pass  # Real detectors are optional

__all__ = [
    "BaseDetector",
    "register_detector",
    "get_detector_class",
    "list_detectors",
]
