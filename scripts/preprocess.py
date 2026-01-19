#!/usr/bin/env python
from __future__ import annotations

import argparse

from shnet_ml.config import ensure_dirs, load_config
from shnet_ml.data.preprocess import preprocess_to_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess raw dataset into a single processed CSV.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--out-name", default="processed.csv", help="Output file name under processed_dir.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    result = preprocess_to_csv(cfg, out_name=args.out_name)
    print(f"[OK] Processed CSV: {result.processed_path} (rows={result.rows})")
    print(f"[OK] Meta: {result.meta_path}")


if __name__ == "__main__":
    main()
