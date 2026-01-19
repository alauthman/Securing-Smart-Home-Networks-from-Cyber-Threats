from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    numeric_cols: List[str]
    categorical_cols: List[str]
    label_col: Optional[str] = None


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Split DataFrame into X and y using the feature spec.

    - If spec.label_col is None or not present, returns y=None.
    """
    cols = [c for c in (spec.numeric_cols + spec.categorical_cols) if c in df.columns]
    x = df[cols].copy()
    y = None
    if spec.label_col and spec.label_col in df.columns:
        y = df[spec.label_col].copy()
    return x, y
