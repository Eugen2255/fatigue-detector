# config.py
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pathlib import Path
import logging 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Преобразуем в абсолютные строки (MediaPipe C-бэкенд надёжнее работает с ними)
face_path = str(MODELS_DIR / "face_landmarker.task")
pose_path = str(MODELS_DIR / "pose_landmarker_full.task")

@dataclass
class MetricConfig:
    """Централизованные пороги и параметры метрик усталости."""
    blink: Dict[str, Any] = field(default_factory=lambda: {"ear_threshold": 0.18})
    rubbing: Dict[str, Any] = field(default_factory=lambda: {"distance_threshold_px": 50.0})
    head_tilt: Dict[str, Any] = field(default_factory=lambda: {
        "mild_tilt_deg": 10.0,
        "severe_tilt_deg": 35.0
    })
    
    # Общие параметры
    smoothing_alpha: float = 0.3  # Коэффициент экспоненциального сглаживания (0.0–1.0)
    fps_target: int = 30

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetricConfig":
        """Загрузка конфигурации из YAML-файла."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь (для логирования или API)."""
        return {
            "blink": self.blink,
            "rubbing": self.rubbing,
            "head_tilt": self.head_tilt,
            "smoothing_alpha": self.smoothing_alpha,
            "fps_target": self.fps_target
        }