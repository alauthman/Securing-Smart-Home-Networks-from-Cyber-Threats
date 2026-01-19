from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class DatasetConfig:
    # One or more glob patterns relative to raw_dir, e.g. ["cardiff/**/*.csv"]
    raw_glob: List[str] = field(default_factory=lambda: ["**/*.csv"])
    # Optional label column for supervised training; if None, anomaly mode must be used.
    label_col: Optional[str] = "label"
    # Optional timestamp column used for sorting / time-based operations
    time_col: Optional[str] = None
    # Optional columns to drop before training
    drop_cols: List[str] = field(default_factory=list)
    # Optional fraction of data to sample for quick experiments (0 < sample_frac <= 1)
    sample_frac: float = 1.0


@dataclass(frozen=True)
class FeaturesConfig:
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    # "rf" (RandomForest), "logreg" (LogisticRegression), "iforest" (IsolationForest)
    kind: str = "rf"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    task: str = "classification"  # "classification" or "anomaly"
    test_size: float = 0.2
    random_state: int = 42
    # For anomaly training, contamination is important (used by IsolationForest)
    contamination: Optional[float] = None


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig = DatasetConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def _as_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    return Path(str(value))


def load_config(path: str | Path) -> AppConfig:
    '''Load YAML config into an AppConfig with safe defaults.'''
    path = _as_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    paths_d = data.get("paths", {}) or {}
    dataset_d = data.get("dataset", {}) or {}
    features_d = data.get("features", {}) or {}
    model_d = data.get("model", {}) or {}
    train_d = data.get("train", {}) or {}

    paths = PathsConfig(
        raw_dir=_as_path(paths_d.get("raw_dir", "data/raw")),
        processed_dir=_as_path(paths_d.get("processed_dir", "data/processed")),
        model_dir=_as_path(paths_d.get("model_dir", "models")),
        reports_dir=_as_path(paths_d.get("reports_dir", "reports")),
    )

    dataset = DatasetConfig(
        raw_glob=list(dataset_d.get("raw_glob", ["**/*.csv"])),
        label_col=dataset_d.get("label_col", "label"),
        time_col=dataset_d.get("time_col", None),
        drop_cols=list(dataset_d.get("drop_cols", [])),
        sample_frac=float(dataset_d.get("sample_frac", 1.0)),
    )

    features = FeaturesConfig(
        numeric_cols=list(features_d.get("numeric_cols", [])),
        categorical_cols=list(features_d.get("categorical_cols", [])),
    )

    model = ModelConfig(
        kind=str(model_d.get("kind", "rf")),
        params=dict(model_d.get("params", {})),
    )

    train = TrainConfig(
        task=str(train_d.get("task", "classification")),
        test_size=float(train_d.get("test_size", 0.2)),
        random_state=int(train_d.get("random_state", 42)),
        contamination=train_d.get("contamination", None),
    )

    return AppConfig(paths=paths, dataset=dataset, features=features, model=model, train=train)


def ensure_dirs(cfg: AppConfig) -> None:
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)


def resolve_raw_files(cfg: AppConfig) -> List[Path]:
    '''Resolve raw files based on the configured glob patterns.'''
    files: List[Path] = []
    for pattern in cfg.dataset.raw_glob:
        files.extend(sorted(cfg.paths.raw_dir.glob(pattern)))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen and f.is_file():
            seen.add(f)
            uniq.append(f)
    return uniq
