from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from shnet_ml.config import AppConfig
from shnet_ml.features.build_features import FeatureSpec, split_xy
from shnet_ml.modeling.artifact import ModelArtifact


def _make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    # Note: with_mean=False so StandardScaler can work with sparse matrices
    num_pipe = Pipeline([("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No features configured. Set features.numeric_cols and/or features.categorical_cols.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_model(cfg: AppConfig) -> Any:
    kind = cfg.model.kind.lower().strip()

    if cfg.train.task == "anomaly":
        params = dict(cfg.model.params)
        if cfg.train.contamination is not None and "contamination" not in params:
            params["contamination"] = cfg.train.contamination
        return IsolationForest(**params)

    if kind == "rf":
        params = {"n_estimators": 300, "random_state": cfg.train.random_state, "n_jobs": -1}
        params.update(cfg.model.params)
        return RandomForestClassifier(**params)

    if kind == "logreg":
        params = {
            "max_iter": 1000,
            "n_jobs": -1,
            "random_state": cfg.train.random_state,
        }
        params.update(cfg.model.params)
        return LogisticRegression(**params)

    raise ValueError(f"Unsupported model kind: {cfg.model.kind} (use rf/logreg for classification, iforest for anomaly)")


def _infer_normal_mask(y: pd.Series) -> np.ndarray:
    # Heuristics: 0 / "0" / "normal" / "benign" treated as normal
    lowered = y.astype(str).str.lower()
    normal_values = {"0", "normal", "benign", "false", "no", "ok"}
    return lowered.isin(normal_values).to_numpy()


@dataclass(frozen=True)
class TrainOutput:
    artifact_path: Path
    train_report_path: Path


def train_from_processed_csv(cfg: AppConfig, processed_csv: Path) -> TrainOutput:
    """Train a model from processed CSV and save a ModelArtifact in cfg.paths.model_dir."""
    df = pd.read_csv(processed_csv, low_memory=False)

    feat_spec = FeatureSpec(
        numeric_cols=cfg.features.numeric_cols,
        categorical_cols=cfg.features.categorical_cols,
        label_col=cfg.dataset.label_col,
    )
    x, y = split_xy(df, feat_spec)

    if cfg.train.task == "classification":
        if y is None:
            raise ValueError("Classification task requires a label column. Set dataset.label_col in config.")
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))

        x_train, x_test, y_train, y_test = train_test_split(
            x, y_enc, test_size=cfg.train.test_size, random_state=cfg.train.random_state, stratify=y_enc
        )

        pre = _make_preprocessor(cfg.features.numeric_cols, cfg.features.categorical_cols)
        model = _make_model(cfg)

        pipe = Pipeline([("preprocess", pre), ("model", model)])
        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=list(le.classes_), output_dict=True)

        # Optional AUC if binary
        auc = None
        try:
            if len(le.classes_) == 2 and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(x_test)[:, 1]
                auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = None

        payload = {
            "task": "classification",
            "classes": list(le.classes_),
            "classification_report": report,
            "auc": auc,
            "n_rows": int(len(df)),
            "n_features": int(x.shape[1]),
        }

        cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = cfg.paths.reports_dir / "train_report.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = cfg.paths.model_dir / "model.joblib"
        artifact = ModelArtifact(
            task="classification",
            pipeline=pipe,
            feature_names=list(x.columns),
            label_encoder=le,
            metadata={"train_report": payload},
        )
        artifact.save(str(artifact_path))
        return TrainOutput(artifact_path=artifact_path, train_report_path=report_path)

    # anomaly detection
    pre = _make_preprocessor(cfg.features.numeric_cols, cfg.features.categorical_cols)
    model = _make_model(cfg)
    pipe = Pipeline([("preprocess", pre), ("model", model)])

    # Train on "normal" only if labels exist and can be inferred; otherwise use all rows
    train_x = x
    if y is not None:
        mask = _infer_normal_mask(y)
        if mask.any():
            train_x = x.loc[mask]

    pipe.fit(train_x)

    cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = cfg.paths.model_dir / "model.joblib"
    artifact = ModelArtifact(
        task="anomaly",
        pipeline=pipe,
        feature_names=list(x.columns),
        label_encoder=None,
        metadata={"trained_on_rows": int(len(train_x)), "total_rows": int(len(x))},
    )
    artifact.save(str(artifact_path))

    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.paths.reports_dir / "train_report.json"
    report_payload = {
        "task": "anomaly",
        "trained_on_rows": int(len(train_x)),
        "total_rows": int(len(x)),
        "model_kind": cfg.model.kind,
        "params": cfg.model.params,
        "note": "IsolationForest score/prediction should be evaluated separately on labeled data if available.",
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return TrainOutput(artifact_path=artifact_path, train_report_path=report_path)
