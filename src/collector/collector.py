"""Сборщик метрик для real-time обработки."""
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FrameRecord:
    frame: int
    blink: int
    rubbing: bool
    head_tilt_state: int
    head_tilt_angle: Optional[float]
    face_detected: bool
    pose_detected: bool

class MetricsCollector:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._history: List[FrameRecord] = []
        self._reset_counters()

    def _reset_counters(self):
        self.total_blinks = 0
        self.total_rubbing = 0
        self.total_tilt_light = 0
        self.total_tilt_heavy = 0

    def add_frame(
        self, 
        frame_idx: int, 
        blink: Optional[int] = None, 
        rubbing: Optional[bool] = None,
        head_tilt_state: Optional[int] = None, 
        head_tilt_angle: Optional[float] = None,
        face_kps: Optional[np.ndarray] = None, 
        pose_kps: Optional[np.ndarray] = None
    ):
        """Добавляет запись о кадре. Безопасно обрабатывает None от метрик."""
        
        # === Безопасное приведение типов ===
        blink_val = 1 if blink == 1 else 0
        rubbing_val = rubbing is True
        tilt_state_val = head_tilt_state if head_tilt_state in (1, 2) else 0
        
        is_face = face_kps is not None and face_kps.size > 0
        is_pose = pose_kps is not None and pose_kps.size > 0

        # === Обновление счётчиков ===
        self.total_blinks += blink_val
        if rubbing_val:
            self.total_rubbing += 1
        if tilt_state_val == 1:
            self.total_tilt_light += 1
        elif tilt_state_val == 2:
            self.total_tilt_heavy += 1

        # === Сохранение в историю ===
        self._history.append(FrameRecord(
            frame=frame_idx, 
            blink=blink_val, 
            rubbing=rubbing_val,
            head_tilt_state=tilt_state_val, 
            head_tilt_angle=head_tilt_angle,
            face_detected=is_face, 
            pose_detected=is_pose
        ))

    def to_dataframe(self):
        """Конвертирует историю в DataFrame ТОЛЬКО когда нужен анализ."""
        import pandas as pd
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._history])

    def get_summary(self) -> Dict[str, Any]:
        total_frames = len(self._history)
        if total_frames == 0: return {}
        
        return {
            "total_frames": total_frames,
            "duration_sec": total_frames / self.fps,
            "total_blinks": self.total_blinks,
            "blink_rate_per_min": (self.total_blinks / (total_frames / self.fps)) * 60,
            "rubbing_frames": self.total_rubbing,
            "rubbing_pct": (self.total_rubbing / total_frames) * 100,
            "tilt_light": self.total_tilt_light,
            "tilt_heavy": self.total_tilt_heavy,
            "tilt_pct": ((self.total_tilt_light + self.total_tilt_heavy) / total_frames) * 100
        }

    def save(self, base_path: str):
        """Сохраняет все данные ОДНИМ файлом по завершении сессии."""
        import pandas as pd
        import json
        import os

        df = self.to_dataframe()
        if df.empty:
            logger.warning("Нет данных для сохранения")
            return

        os.makedirs(base_path, exist_ok=True)
        
        # CSV с метриками
        csv_path = os.path.join(base_path, "metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # JSON со сводкой
        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
            
        logger.info(f"Данные сохранены: {csv_path}, {summary_path}")

    def reset(self):
        self._history.clear()
        self._reset_counters()