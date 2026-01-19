#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.config import ensure_dirs, load_config
from shnet_ml.modeling.evaluate import evaluate_on_processed_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ML model on processed CSV.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--processed",
        default=None,
        help="Path to processed CSV (default: <processed_dir>/processed.csv).",
    )
    ap.add_argument(
        "--artifact",
        default=None,
        help="Path to model artifact (default: <model_dir>/model.joblib).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed = Path(args.processed) if args.processed else (cfg.paths.processed_dir / "processed.csv")
    artifact = Path(args.artifact) if args.artifact else (cfg.paths.model_dir / "model.joblib")

    out = evaluate_on_processed_csv(cfg, processed_csv=processed, artifact_path=artifact)
    print(f"[OK] Metrics: {out.metrics_path}")


if __name__ == "__main__":
    main()
