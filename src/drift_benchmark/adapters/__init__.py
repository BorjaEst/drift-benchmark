"""
Adapters module for drift-benchmark - REQ-ADP-XXX

This module provides the adapter framework for integrating drift detection
libraries with the drift-benchmark framework.
"""

from .base_detector import BaseDetector
from .registry import get_detector_class, list_available_detectors, list_detectors, register_detector

__all__ = [
    "BaseDetector",
    "register_detector",
    "get_detector_class",
    "list_detectors",
    "list_available_detectors",
]
