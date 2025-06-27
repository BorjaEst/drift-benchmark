from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode


class ImplementationData(BaseModel):
    """Metadata for drift detector implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., description="Name of the implementation")
    execution_mode: ExecutionMode = Field(..., description="Execution mode of the implementation")
    hyperparameters: List[str] = Field(..., description="Allowed configuration hyperparameters")
    references: List[str] = Field(..., description="List of references for the implementation")


class MethodData(BaseModel):
    """Metadata for drift detection methods."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., description="Name of the drift detection method")
    description: str = Field(..., description="Description of the drift detection method")
    drift_types: List[DriftType] = Field(..., description="List of drift types the method can detect")
    family: DetectorFamily = Field(..., description="Family of the drift detection method")
    data_dimension: DataDimension = Field(..., description="Dimensionality of the data")
    data_types: List[DataType] = Field(..., description="List of data types the method can operate on")
    requires_labels: bool = Field(..., description="Whether the method requires labels for drift detection")
    references: List[str] = Field(..., description="List of references for the method")


class MethodMetadata(MethodData):
    """Data model for drift detection methods with implementations."""

    implementations: Dict[str, ImplementationData] = Field(..., description="List of implementations for the method")

    def __getitem__(self, implementation_id: str) -> ImplementationData:
        """Get an implementation by its ID."""
        return self.implementations[implementation_id]


class DetectorMetadata(BaseModel):
    """Metadata for a drift detection method and its implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    method: MethodData = Field(..., description="Method metadata")
    implementation: ImplementationData = Field(..., description="Implementation metadata")
