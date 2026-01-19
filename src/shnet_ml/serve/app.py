from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from shnet_ml.serve.predictor import load_artifact, predict_rows
from shnet_ml.serve.schemas import HealthResponse, PredictRequest

app = FastAPI(title="SHNet-ML-Edge Inference API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        artifact = load_artifact()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}") from e

    model_path = os.environ.get("SHNET_MODEL_PATH", "models/model.joblib")
    return HealthResponse(
        status="ok",
        model_path=model_path,
        task=artifact.task,
        n_features=len(artifact.feature_names),
    )


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        artifact = load_artifact()
        return predict_rows(req.rows, artifact=artifact)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
