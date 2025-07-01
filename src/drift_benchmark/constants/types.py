from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from drift_benchmark.constants.literals import DataDimension, DataType, DetectorFamily, DriftType, ExecutionMode


class ImplementationData(BaseModel):
    """Metadata for drift detector implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the implementation")
    execution_mode: ExecutionMode = Field(..., description="Execution mode of the implementation")
    hyperparameters: List[str] = Field(default_factory=list, description="Allowed configuration hyperparameters")
    references: List[str] = Field(default_factory=list, description="List of references for the implementation")


class MethodData(BaseModel):
    """Metadata for drift detection methods."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    name: str = Field(..., min_length=1, description="Name of the drift detection method")
    description: str = Field(..., min_length=10, description="Description of the drift detection method")
    drift_types: List[DriftType] = Field(..., min_length=1, description="List of drift types the method can detect")
    family: DetectorFamily = Field(..., description="Family of the drift detection method")
    data_dimension: DataDimension = Field(..., description="Dimensionality of the data")
    data_types: List[DataType] = Field(..., min_length=1, description="List of data types the method can operate on")
    requires_labels: bool = Field(..., description="Whether the method requires labels for drift detection")
    references: List[str] = Field(default_factory=list, description="List of references for the method")


class MethodMetadata(MethodData):
    """Data model for drift detection methods with implementations."""

    implementations: Dict[str, ImplementationData] = Field(
        ..., description="Dictionary of implementations for the method"
    )

    def __getitem__(self, implementation_id: str) -> ImplementationData:
        """Get an implementation by its ID."""
        if implementation_id not in self.implementations:
            raise KeyError(
                f"Implementation '{implementation_id}' not found. Available: {list(self.implementations.keys())}"
            )
        return self.implementations[implementation_id]


class DetectorMetadata(BaseModel):
    """Metadata for a drift detection method and its implementations."""

    model_config = {"extra": "forbid"}  # Forbid extra fields not defined in the model

    method: MethodData = Field(..., description="Method metadata")
    implementation: ImplementationData = Field(..., description="Implementation metadata")

    def get_full_id(self) -> str:
        """Get the full identifier for this detector (method.implementation)."""
        return f"{self.method.name}.{self.implementation.name}"

    def is_compatible_with_data(self, data_type: DataType, data_dimension: DataDimension, has_labels: bool) -> bool:
        """Check if this detector is compatible with the given data characteristics."""
        # Check data type compatibility
        if data_type not in self.method.data_types:
            return False

        # Check data dimension compatibility
        if data_dimension != self.method.data_dimension:
            return False

        # Check label requirements
        if self.method.requires_labels and not has_labels:
            return False

        return True
