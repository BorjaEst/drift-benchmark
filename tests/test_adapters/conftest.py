# Feature-specific fixtures for adapters module testing
# REFACTORED: Enhanced with Given-When-Then patterns and improved maintainability

import sys
from abc import ABC, abstractmethod

# Import asset loaders from main conftest
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

# Add parent path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from conftest import load_asset_config, load_asset_csv
