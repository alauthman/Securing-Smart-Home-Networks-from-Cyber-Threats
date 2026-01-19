from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import joblib


@dataclass
class ModelArtifact:
    task: str
    pipeline: Any
    feature_names: List[str]
    label_encoder: Optional[Any] = None
    metadata: Dict[str, Any] = None

    def save(self, path: str) -> None:
        payload = {
            "task": self.task,
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "metadata": self.metadata or {},
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "ModelArtifact":
        payload = joblib.load(path)
        return ModelArtifact(
            task=payload["task"],
            pipeline=payload["pipeline"],
            feature_names=list(payload.get("feature_names", [])),
            label_encoder=payload.get("label_encoder", None),
            metadata=dict(payload.get("metadata", {})),
        )
