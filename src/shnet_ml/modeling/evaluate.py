from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from shnet_ml.config import AppConfig
from shnet_ml.features.build_features import FeatureSpec, split_xy
from shnet_ml.modeling.artifact import ModelArtifact


def _to_binary_labels(y: pd.Series) -> np.ndarray:
    # 0/normal/benign => 0, otherwise => 1
    lowered = y.astype(str).str.lower()
    normal_values = {"0", "normal", "benign", "false", "no", "ok"}
    return (~lowered.isin(normal_values)).astype(int).to_numpy()


@dataclass(frozen=True)
class EvalOutput:
    metrics_path: Path


def evaluate_on_processed_csv(cfg: AppConfig, processed_csv: Path, artifact_path: Path) -> EvalOutput:
    df = pd.read_csv(processed_csv, low_memory=False)
    feat_spec = FeatureSpec(
        numeric_cols=cfg.features.numeric_cols,
        categorical_cols=cfg.features.categorical_cols,
        label_col=cfg.dataset.label_col,
    )
    x, y = split_xy(df, feat_spec)

    artifact = ModelArtifact.load(str(artifact_path))
    pipe = artifact.pipeline

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.paths.reports_dir / "eval_metrics.json"

    # Split for evaluation so metrics are not computed on full training set
    if cfg.train.task == "classification":
        if y is None:
            raise ValueError("Classification evaluation requires labels.")
        le = artifact.label_encoder
        if le is None:
            raise ValueError("Missing label_encoder in artifact.")

        y_enc = le.transform(y.astype(str))
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_enc, test_size=cfg.train.test_size, random_state=cfg.train.random_state, stratify=y_enc
        )

        t0 = time.perf_counter()
        y_pred = pipe.predict(x_test)
        infer_seconds = time.perf_counter() - t0

        report = classification_report(
            y_test, y_pred, target_names=list(le.classes_), output_dict=True
        )

        auc = None
        try:
            if len(le.classes_) == 2 and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = None

        payload = {
            "task": "classification",
            "n_test": int(len(x_test)),
            "classification_report": report,
            "auc": auc,
            "inference_seconds": infer_seconds,
            "inference_rows_per_sec": float(len(x_test) / max(infer_seconds, 1e-9)),
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return EvalOutput(metrics_path=metrics_path)

    # anomaly evaluation
    t0 = time.perf_counter()
    pred = pipe.predict(x)  # 1 inlier, -1 outlier
    infer_seconds = time.perf_counter() - t0
    pred_anom = (pred == -1).astype(int)

    scores = None
    try:
        scores = pipe.decision_function(x)  # higher -> more normal
    except Exception:
        scores = None

    payload: Dict[str, Any] = {
        "task": "anomaly",
        "n_rows": int(len(x)),
        "predicted_anomaly_rate": float(pred_anom.mean()),
        "inference_seconds": infer_seconds,
        "inference_rows_per_sec": float(len(x) / max(infer_seconds, 1e-9)),
    }

    if y is not None:
        y_true = _to_binary_labels(y)
        report = classification_report(y_true, pred_anom, output_dict=True)
        payload["classification_report"] = report

        # AUC: invert scores so higher -> more anomalous
        if scores is not None:
            try:
                auc = float(roc_auc_score(y_true, -scores))
                payload["auc"] = auc
            except Exception:
                pass

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return EvalOutput(metrics_path=metrics_path)
