from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from shnet_ml.config import AppConfig
from shnet_ml.modeling.artifact import ModelArtifact


@dataclass(frozen=True)
class ExportOutput:
    exported_path: Path
    model_card_path: Path


def export_artifact(cfg: AppConfig, artifact_path: Path) -> ExportOutput:
    cfg.paths.model_dir.mkdir(parents=True, exist_ok=True)
    artifact = ModelArtifact.load(str(artifact_path))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    exported_path = cfg.paths.model_dir / f"model-{stamp}.joblib"
    model_card_path = cfg.paths.model_dir / f"model_card-{stamp}.md"

    # Copy the artifact
    exported_path.write_bytes(Path(artifact_path).read_bytes())

    md = f"""# Model Card

- Export time (UTC): {datetime.now(timezone.utc).isoformat()}
- Task: {artifact.task}
- Features: {len(artifact.feature_names)}

## Feature names
{', '.join(artifact.feature_names)}

## Metadata (training/evaluation)
```json
{json.dumps(artifact.metadata or {}, indent=2)}
```

## Notes
- This artifact was exported by SHNet-ML-Edge.
- Ensure dataset licensing requirements are satisfied (e.g., CC BY attribution where applicable).
"""
    model_card_path.write_text(md, encoding="utf-8")

    return ExportOutput(exported_path=exported_path, model_card_path=model_card_path)
