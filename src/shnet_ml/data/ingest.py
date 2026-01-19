from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class IngestSpec:
    keep_cols: Optional[List[str]] = None
    drop_cols: Optional[List[str]] = None
    time_col: Optional[str] = None
    label_col: Optional[str] = None
    sample_frac: float = 1.0
    chunk_size: int = 200_000


def _is_json(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".jsonl"}


def _is_csv(path: Path) -> bool:
    return path.suffix.lower() in {".csv"}


def iter_tabular_chunks(paths: Sequence[Path], spec: IngestSpec) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from CSV/JSON files.

    Notes for very large datasets:
    - Uses pandas chunking (chunksize) to avoid loading entire files into memory.
    - Keeps only requested columns (if provided) to reduce memory footprint.
    """
    for p in paths:
        if _is_csv(p):
            reader = pd.read_csv(p, chunksize=spec.chunk_size, low_memory=False)
        elif _is_json(p):
            reader = pd.read_json(p, lines=True, chunksize=spec.chunk_size)
        else:
            # Unsupported file types are ignored by design
            continue

        for chunk in reader:
            # Column filtering
            if spec.keep_cols:
                keep = [c for c in spec.keep_cols if c in chunk.columns]
                chunk = chunk[keep]
            if spec.drop_cols:
                drop = [c for c in spec.drop_cols if c in chunk.columns]
                if drop:
                    chunk = chunk.drop(columns=drop)

            # Timestamp parsing (best effort)
            if spec.time_col and spec.time_col in chunk.columns:
                chunk[spec.time_col] = pd.to_datetime(chunk[spec.time_col], errors="coerce")

            # Optional sampling
            if 0 < spec.sample_frac < 1.0 and len(chunk) > 0:
                chunk = chunk.sample(frac=spec.sample_frac, random_state=42)

            yield chunk


def infer_columns_from_first_chunk(paths: Sequence[Path], chunk_size: int = 50_000) -> List[str]:
    """Infer available columns from the first readable file/chunk."""
    for p in paths:
        try:
            if _is_csv(p):
                df = next(pd.read_csv(p, chunksize=chunk_size, low_memory=False))
                return list(df.columns)
            if _is_json(p):
                df = next(pd.read_json(p, lines=True, chunksize=chunk_size))
                return list(df.columns)
        except StopIteration:
            continue
        except Exception:
            continue
    return []
