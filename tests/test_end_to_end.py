from __future__ import annotations

from pathlib import Path

import pandas as pd

from shnet_ml.config import load_config
from shnet_ml.data.preprocess import preprocess_to_csv
from shnet_ml.modeling.artifact import ModelArtifact
from shnet_ml.modeling.evaluate import evaluate_on_processed_csv
from shnet_ml.modeling.train import train_from_processed_csv
from shnet_ml.serve.predictor import predict_rows


def test_pipeline_synth(tmp_path: Path) -> None:
    # Arrange: create synthetic dataset
    raw_dir = tmp_path / "data" / "raw" / "synth"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "pkt_size_mean": [100, 1200, 340, 900, 360, 1100],
            "conn_rate": [0.2, 6.0, 1.1, 3.2, 0.8, 5.4],
            "dns_qps": [0.1, 2.2, 0.2, 1.8, 0.15, 2.4],
            "proto": ["TCP", "UDP", "TCP", "UDP", "TCP", "UDP"],
            "label": ["normal", "attack", "normal", "attack", "normal", "attack"],
        }
    )
    raw_csv = raw_dir / "synth.csv"
    df.to_csv(raw_csv, index=False)

    # Config file in tmp
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
  time_col: null
  drop_cols: []
  sample_frac: 1.0

features:
  numeric_cols: ["pkt_size_mean","conn_rate","dns_qps"]
  categorical_cols: ["proto"]

model:
  kind: "rf"
  params:
    n_estimators: 20

train:
  task: "classification"
  test_size: 0.33
  random_state: 42
""".format(
            raw=tmp_path / "data" / "raw",
            processed=tmp_path / "data" / "processed",
            models=tmp_path / "models",
            reports=tmp_path / "reports",
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    # Act: preprocess -> train -> eval
    prep = preprocess_to_csv(cfg, out_name="processed.csv")
    train_out = train_from_processed_csv(cfg, processed_csv=prep.processed_path)
    eval_out = evaluate_on_processed_csv(
        cfg, processed_csv=prep.processed_path, artifact_path=train_out.artifact_path
    )

    assert train_out.artifact_path.exists()
    assert eval_out.metrics_path.exists()

    # Inference test
    artifact = ModelArtifact.load(str(train_out.artifact_path))
    res = predict_rows(
        rows=[{"pkt_size_mean": 1200, "conn_rate": 6.0, "dns_qps": 2.0, "proto": "UDP"}],
        artifact=artifact,
    )
    assert "predictions" in res
