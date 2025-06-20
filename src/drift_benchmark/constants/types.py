from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from drift_benchmark.constants.enums import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode


class DetectorMetadata(BaseModel):
    """Metadata for drift detectors."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(
        ...,
        description="Name of the detector",
    )
    description: str = Field(
        ...,
        description="Brief description of the detector's functionality",
    )
    drift_types: List[DriftType] = Field(
        ...,
        description="List of drift types that the detector can handle",
    )
    execution_mode: ExecutionMode = Field(
        ...,
        description="Execution mode of the detector (e.g., streaming or batch)",
    )
    family: DetectorFamily = Field(
        ...,
        description="Family of the drift detection algorithm (e.g., statistical, machine learning)",
    )
    data_dimension: DataDimension = Field(
        ...,
        description="Data dimensionality the detector can handle (e.g., univariate, multivariate)",
    )
    data_types: List[DataType] = Field(
        ...,
        description="List of data types the detector can handle (e.g., continuous, categorical, mixed)",
    )
    requires_labels: bool = Field(
        False,
        description="Whether the detector requires labels for drift detection",
    )
    references: Optional[List[str]] = Field(
        None,
        description="List of references or links to documentation for the detector",
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters for the detector, if applicable",
    )
