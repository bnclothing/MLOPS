from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoleculeInput(StrictModel):
    smiles: str = Field(
        min_length=1,
        max_length=10_000,
        description="Molecule encoded as a SMILES string.",
        examples=["Cn1c(=O)c2c(ncn2C)n(C)c1=O"],
    )

    @field_validator("smiles")
    @classmethod
    def strip_smiles(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SMILES must not be blank")
        return value


class BatchMoleculeInput(MoleculeInput):
    id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional caller-provided molecule identifier.",
        examples=["compound-001"],
    )


class BatchPredictionRequest(StrictModel):
    molecules: list[BatchMoleculeInput] = Field(min_length=1, max_length=1_000)


class PredictionResponse(StrictModel):
    id: str | None = None
    smiles: str
    standardized_smiles: str | None
    valid: bool
    probability_active: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction: Literal["active", "inactive", "invalid_smiles"]
    decision_threshold: float = Field(ge=0.0, le=1.0)
    target_chembl_id: str
    target_name: str
    error: str | None = None


class BatchSummary(StrictModel):
    total: int
    valid: int
    invalid: int
    predicted_active: int
    predicted_inactive: int


class BatchPredictionResponse(StrictModel):
    predictions: list[PredictionResponse]
    summary: BatchSummary


class HealthResponse(StrictModel):
    status: Literal["ok"]
    model_loaded: bool
    device: str
    cuda_available: bool
    target_chembl_id: str
    model_format: str

