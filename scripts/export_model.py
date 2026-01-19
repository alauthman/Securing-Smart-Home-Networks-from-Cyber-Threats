#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from shnet_ml.config import ensure_dirs, load_config
from shnet_ml.modeling.export import export_artifact


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained model artifact with a timestamped name.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument(
        "--artifact",
        default=None,
        help="Path to model artifact (default: <model_dir>/model.joblib).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    artifact = Path(args.artifact) if args.artifact else (cfg.paths.model_dir / "model.joblib")
    out = export_artifact(cfg, artifact_path=artifact)

    print(f"[OK] Exported artifact: {out.exported_path}")
    print(f"[OK] Model card: {out.model_card_path}")


if __name__ == "__main__":
    main()
