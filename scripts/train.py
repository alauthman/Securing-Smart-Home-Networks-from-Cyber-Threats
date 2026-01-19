#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.config import ensure_dirs, load_config
from shnet_ml.modeling.train import train_from_processed_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ML model from processed CSV.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--processed",
        default=None,
        help="Path to processed CSV (default: <processed_dir>/processed.csv).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed = Path(args.processed) if args.processed else (cfg.paths.processed_dir / "processed.csv")
    out = train_from_processed_csv(cfg, processed_csv=processed)
    print(f"[OK] Model artifact: {out.artifact_path}")
    print(f"[OK] Train report: {out.train_report_path}")


if __name__ == "__main__":
    main()
