from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import BioactivityPredictor
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    HealthResponse,
    MoleculeInput,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = BioactivityPredictor()
    yield
    del app.state.predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="EGFR Bioactivity Prediction API",
    summary="Virtual screening of molecules against human EGFR (CHEMBL203).",
    description=(
        "Scores SMILES using the exported Morgan-fingerprint MLP. "
        "Predictions are for in-silico prioritization only and require experimental validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={"name": "MLOps Virtual Screening Project"},
    license_info={"name": "Research and educational use"},
)

cors_origins = [value.strip() for value in os.getenv("CORS_ORIGINS", "").split(",") if value.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def get_predictor(request: Request) -> BioactivityPredictor:
    return request.app.state.predictor


@app.get("/", tags=["Service"])
def root() -> dict:
    return {
        "service": "EGFR Bioactivity Prediction API",
        "version": app.version,
        "documentation": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
        "batch_prediction_endpoint": "/predict/batch",
    }


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health(request: Request) -> HealthResponse:
    predictor = get_predictor(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(predictor.device),
        cuda_available=torch.cuda.is_available(),
        target_chembl_id=predictor.target_chembl_id,
        model_format="TorchScript",
    )


@app.get("/model-info", tags=["Service"])
def model_info(request: Request) -> dict:
    predictor = get_predictor(request)
    return {
        **predictor.metadata,
        "serving": {
            "device": str(predictor.device),
            "cuda_available": torch.cuda.is_available(),
            "model_format": "TorchScript",
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict one molecule",
)
def predict_one(payload: MoleculeInput, request: Request) -> PredictionResponse:
    result = get_predictor(request).predict_one(payload.smiles)
    if not result["valid"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"message": result["error"], "smiles": payload.smiles},
        )
    return PredictionResponse.model_validate(result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Predict up to 1,000 molecules",
)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    predictor = get_predictor(request)
    inputs = [(molecule.id, molecule.smiles) for molecule in payload.molecules]
    raw_predictions = predictor.predict_many(inputs)
    predictions = [PredictionResponse.model_validate(item) for item in raw_predictions]
    summary = BatchSummary(
        total=len(predictions),
        valid=sum(item.valid for item in predictions),
        invalid=sum(not item.valid for item in predictions),
        predicted_active=sum(item.prediction == "active" for item in predictions),
        predicted_inactive=sum(item.prediction == "inactive" for item in predictions),
    )
    return BatchPredictionResponse(predictions=predictions, summary=summary)
