from __future__ import annotations

from pathlib import Path

import pandas as pd

from shnet_ml.config import load_config
from shnet_ml.data.preprocess import preprocess_to_csv
from shnet_ml.modeling.train import train_from_processed_csv
from shnet_ml.modeling.evaluate import evaluate_on_processed_csv


def test_anomaly_pipeline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "synth"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "pkt_size_mean": [100, 110, 120, 130, 140, 1500],
            "conn_rate": [0.2, 0.25, 0.3, 0.2, 0.18, 12.0],
            "dns_qps": [0.1, 0.12, 0.08, 0.09, 0.11, 5.0],
            "proto": ["TCP", "TCP", "TCP", "TCP", "TCP", "UDP"],
            "label": ["normal", "normal", "normal", "normal", "normal", "attack"],
        }
    )
    df.to_csv(raw_dir / "synth.csv", index=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """paths:
  raw_dir: {raw}
  processed_dir: {processed}
  model_dir: {models}
  reports_dir: {reports}

dataset:
  raw_glob:
    - "synth/*.csv"
  label_col: "label"

features:
  numeric_cols: ["pkt_size_mean","conn_rate","dns_qps"]
  categorical_cols: ["proto"]

model:
  kind: "iforest"
  params:
    n_estimators: 50
    random_state: 42

train:
  task: "anomaly"
  contamination: 0.2
""".format(
            raw=tmp_path / "data" / "raw",
            processed=tmp_path / "data" / "processed",
            models=tmp_path / "models",
            reports=tmp_path / "reports",
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    prep = preprocess_to_csv(cfg)
    train_out = train_from_processed_csv(cfg, processed_csv=prep.processed_path)
    eval_out = evaluate_on_processed_csv(cfg, processed_csv=prep.processed_path, artifact_path=train_out.artifact_path)
    assert eval_out.metrics_path.exists()
