"""
Test detectors for registry testing.

These detectors are imported from tests/assets/components/test_detectors.py
to avoid duplication. This file is kept for backward compatibility.
"""

try:
    # Import the actual test detectors from the components directory
    from .assets.components.test_detectors import TestAlibiKSDetector, TestEvidentlyKSDetector, TestScipyDetector

    print(f"Successfully imported test detectors from tests/assets/components")

except ImportError as e:
    print(f"Failed to import test detectors from tests/assets/components: {e}")
    # Fallback: define minimal test detectors if import fails
    pass
