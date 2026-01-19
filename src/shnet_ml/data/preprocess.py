from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from shnet_ml.config import AppConfig, resolve_raw_files
from shnet_ml.data.ingest import IngestSpec, infer_columns_from_first_chunk, iter_tabular_chunks


@dataclass(frozen=True)
class PreprocessResult:
    processed_path: Path
    meta_path: Path
    rows: int
    columns: List[str]


def _default_keep_cols(cfg: AppConfig, available: List[str]) -> List[str]:
    # If features are specified, keep only those (+ label/time)
    cols: List[str] = []
    cols.extend(cfg.features.numeric_cols)
    cols.extend(cfg.features.categorical_cols)
    if cfg.dataset.time_col:
        cols.append(cfg.dataset.time_col)
    if cfg.dataset.label_col:
        cols.append(cfg.dataset.label_col)

    cols = [c for c in cols if c]  # remove None/empty
    if cols:
        # Keep only columns that exist in data
        return [c for c in cols if c in available]
    return available  # fallback: keep all columns


def preprocess_to_csv(cfg: AppConfig, out_name: str = "processed.csv") -> PreprocessResult:
    """Stream preprocess raw tabular files into a single processed CSV.

    - Uses chunked reading so it can handle large datasets.
    - Keeps only configured columns when provided.
    """
    paths = resolve_raw_files(cfg)
    if not paths:
        raise FileNotFoundError(
            f"No raw files found under {cfg.paths.raw_dir} using patterns: {cfg.dataset.raw_glob}"
        )

    available_cols = infer_columns_from_first_chunk(paths)
    keep_cols = _default_keep_cols(cfg, available_cols)

    spec = IngestSpec(
        keep_cols=keep_cols,
        drop_cols=cfg.dataset.drop_cols,
        time_col=cfg.dataset.time_col,
        label_col=cfg.dataset.label_col,
        sample_frac=cfg.dataset.sample_frac,
    )

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = cfg.paths.processed_dir / out_name
    meta_path = cfg.paths.processed_dir / (processed_path.stem + ".meta.json")

    # Stream to CSV
    rows = 0
    wrote_header = False
    col_order: List[str] = list(keep_cols)

    # Put label at the end for readability
    if cfg.dataset.label_col and cfg.dataset.label_col in col_order:
        col_order = [c for c in col_order if c != cfg.dataset.label_col] + [cfg.dataset.label_col]

    # Overwrite any previous file
    if processed_path.exists():
        processed_path.unlink()

    for chunk in iter_tabular_chunks(paths, spec):
        if len(chunk) == 0:
            continue

        # Ensure consistent column order; missing columns will be filled with NaN
        chunk = chunk.reindex(columns=col_order)

        # Optional sorting by time (within chunk)
        if cfg.dataset.time_col and cfg.dataset.time_col in chunk.columns:
            chunk = chunk.sort_values(cfg.dataset.time_col)

        chunk.to_csv(processed_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        rows += len(chunk)

    meta = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "raw_patterns": cfg.dataset.raw_glob,
        "raw_file_count": len(paths),
        "rows": rows,
        "columns": col_order,
        "label_col": cfg.dataset.label_col,
        "time_col": cfg.dataset.time_col,
        "drop_cols": cfg.dataset.drop_cols,
        "sample_frac": cfg.dataset.sample_frac,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return PreprocessResult(
        processed_path=processed_path, meta_path=meta_path, rows=rows, columns=col_order
    )
