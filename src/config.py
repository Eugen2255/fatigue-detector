from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

face_path = str(MODELS_DIR / "face_landmarker.task")
pose_path = str(MODELS_DIR / "pose_landmarker_lite.task") # Lite быстрее и стабильнее для real-time

@dataclass
class MetricConfig:
    blink: Dict[str, Any] = field(default_factory=lambda: {"ear_threshold": 0.22})
    rubbing: Dict[str, Any] = field(default_factory=lambda: {"distance_threshold_px": 45.0})
    head_tilt: Dict[str, Any] = field(default_factory=lambda: {"mild_tilt_deg": 12.0, "severe_tilt_deg": 35.0})
    yawn: Dict[str, Any] = field(default_factory=lambda: {"mar_threshold": 0.45, "mar_close_threshold": 0.35, "min_duration_frames": 12, "min_opening_px": 10.0, "min_face_height_ratio": 0.10})
    
    smoothing_alpha: float = 0.2
    fps_target: int = 30

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetricConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f: data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"blink": self.blink, "rubbing": self.rubbing, "head_tilt": self.head_tilt, "yawn": self.yawn, "smoothing_alpha": self.smoothing_alpha, "fps_target": self.fps_target}
