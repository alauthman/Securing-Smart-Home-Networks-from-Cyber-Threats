from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shnet_ml.modeling.artifact import ModelArtifact


def _default_model_path() -> str:
    return os.environ.get("SHNET_MODEL_PATH", "models/model.joblib")


@lru_cache(maxsize=1)
def load_artifact(path: Optional[str] = None) -> ModelArtifact:
    model_path = path or _default_model_path()
    return ModelArtifact.load(model_path)


def predict_rows(rows: List[Dict[str, Any]], artifact: Optional[ModelArtifact] = None) -> Dict[str, Any]:
    art = artifact or load_artifact()
    pipe = art.pipeline

    df = pd.DataFrame(rows)

    # Align to training feature names; fill missing columns with NaN
    for c in art.feature_names:
        if c not in df.columns:
            df[c] = np.nan
    df = df[art.feature_names]

    if art.task == "classification":
        pred = pipe.predict(df)
        # Convert to labels if label encoder exists
        labels = pred.tolist()
        if art.label_encoder is not None:
            labels = art.label_encoder.inverse_transform(pred).tolist()

        out: Dict[str, Any] = {"task": art.task, "predictions": labels}

        # Add probabilities if available
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)
            out["probabilities"] = proba.tolist()
        return out

    # anomaly
    pred = pipe.predict(df)  # 1 inlier, -1 outlier
    anomalies = (pred == -1).astype(int).tolist()
    out = {"task": art.task, "is_anomaly": anomalies}

    # Optional score
    if hasattr(pipe, "decision_function"):
        # Higher score -> more normal; we invert for 'anomaly_score'
        scores = (-pipe.decision_function(df)).tolist()
        out["anomaly_score"] = scores
    return out
