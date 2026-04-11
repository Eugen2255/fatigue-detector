"""Визуализация метрик поверх кадра."""
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsVisualizer:
    # Константы вынесены для быстрого изменения
    POS_METRICS = (15, 30)
    POS_SUMMARY = (400, 30)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.55
    THICK = 1
    LINE_H = 22
    
    COLORS = {
        "ok": (0, 255, 0),
        "warn": (0, 0, 255),
        "info": (255, 255, 0),
        "text": (220, 220, 220)
    }

    def draw(self, frame: np.ndarray, current: Dict[str, Any], 
             summary: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Рисует метрики и предупреждения. Модифицирует кадр in-place."""
        h, w, _ = frame.shape
        x, y = self.POS_METRICS

        # 1. Основные метрики (всегда актуальные)
        lines = [
            f"Frame: {current.get('frame', 0)}",
            f"Blinks: {current.get('blink_total', 0)}",
            f"Rubbing: {'ACTIVE' if current.get('rubbing_current') else 'OK'}",
            f"Tilt: {self._tilt_label(current)}"
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (x, y + i * self.LINE_H),
                        self.FONT, self.SCALE, self.COLORS["text"], self.THICK, cv2.LINE_AA)

        # 2. Предупреждения (только аномалии, без спама от морганий)
        warnings = self._get_active_warnings(current)
        if warnings:
            warn_y = y + len(lines) * self.LINE_H + 15
            for i, txt in enumerate(warnings):
                cv2.putText(frame, f"⚠ {txt}", (x, warn_y + i * self.LINE_H),
                            self.FONT, self.SCALE, self.COLORS["warn"], self.THICK + 1, cv2.LINE_AA)

        # 3. Сводка (правый верхний угол)
        if summary:
            self._draw_summary(frame, summary, self.POS_SUMMARY)

        return frame

    def _tilt_label(self, m: Dict[str, Any]) -> str:
        state = m.get("head_tilt_state", 0)
        labels = {0: "Normal", 1: "Light", 2: "Heavy"}
        return f"{labels.get(state, 'Err')} (total: {m.get('head_tilt_total', 0)})"

    def _get_active_warnings(self, m: Dict[str, Any]) -> list:
        warnings = []
        if m.get("head_tilt_state") == 2: warnings.append("Heavy Head Tilt")
        if m.get("rubbing_current"): warnings.append("Eye Rubbing")
        return warnings

    def _draw_summary(self, frame: np.ndarray, summary: Dict[str, Any], pos: tuple):
        x, y = pos
        cv2.putText(frame, "SUMMARY:", (x, y), self.FONT, self.SCALE, self.COLORS["info"], self.THICK + 1, cv2.LINE_AA)
        
        summary_lines = [
            f"Duration: {summary.get('duration_sec', 0):.1f}s",
            f"Blink Rate: {summary.get('blink_rate_per_min', 0):.1f}/min",
            f"Rubbing: {summary.get('rubbing_pct', 0):.1f}%",
            f"Tilt: {summary.get('tilt_pct', 0):.1f}%"
        ]
        for i, txt in enumerate(summary_lines):
            cv2.putText(frame, txt, (x, y + (i + 1) * self.LINE_H),
                        self.FONT, self.SCALE - 0.05, self.COLORS["text"], self.THICK, cv2.LINE_AA)