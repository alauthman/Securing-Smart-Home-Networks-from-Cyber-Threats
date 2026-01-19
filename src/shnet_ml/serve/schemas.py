from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries (one per event/row).")


class HealthResponse(BaseModel):
    status: str
    model_path: str
    task: str
    n_features: int
