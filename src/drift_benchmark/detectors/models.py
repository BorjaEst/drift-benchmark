from typing import Dict, List

from pydantic import BaseModel, Field

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode


class ImplementationData(BaseModel):
    """Metadata for drift detector implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the implementation")
    execution_mode: ExecutionMode = Field(..., description="Execution mode of the implementation")
    hyperparameters: List[str] = Field(default_factory=list, description="Allowed configuration hyperparameters")
    references: List[str] = Field(default_factory=list, description="List of references for the implementation")


class _BaseMethodMetadata(BaseModel):
    """Base class for method metadata with common fields."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the drift detection method")
    description: str = Field(..., min_length=10, description="Description of the drift detection method")
    drift_types: List[DriftType] = Field(..., min_length=1, description="List of drift types the method can detect")
    family: DetectorFamily = Field(..., description="Family of the drift detection method")
    data_dimension: DataDimension = Field(..., description="Dimensionality of the data")
    data_types: List[DataType] = Field(..., min_length=1, description="List of data types the method can operate on")
    requires_labels: bool = Field(..., description="Whether the method requires labels for drift detection")
    references: List[str] = Field(..., description="List of references for the method")


class MethodData(_BaseMethodMetadata):
    """Metadata for drift detection methods with multiple implementations."""

    implementations: Dict[str, ImplementationData] = Field(..., description="Dictionary of implementations for the method")


class DetectorData(_BaseMethodMetadata):
    """Metadata for a specific drift detector implementation."""

    implementation: ImplementationData = Field(..., description="implementation data for the method")
