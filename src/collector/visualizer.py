"""Визуализация метрик поверх кадра."""
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MetricsVisualizer:
    POS_METRICS = (20, 40)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.6
    THICK = 1
    LINE_H = 24
    TEXT_COLOR = (190, 190, 190)
    BG_COLOR = (30, 30, 30)
    BORDER_COLOR = (60, 60, 60)

    def draw(self, frame: np.ndarray, current: Dict[str, Any], show_counters: bool = True) -> np.ndarray:
        if not show_counters: return frame
        
        h, w, _ = frame.shape
        x, y = self.POS_METRICS
        lines = [
            f"Frame: {current.get('frame', 0)}",
            f"Blinks: {current.get('blink_total', 0)}",
            f"Yawns: {current.get('yawn_total', 0)}",
            f"Rubbing: {current.get('rubbing_total', 0)}",
            f"PERCLOS: {current.get('perclos_pct', 0):.1f}%",
            f"Tilt: {self._tilt_label(current)}",
            f"Keys/min: {current.get('key_rate', 0.0):.1f}",
            f"Clicks/min: {current.get('mouse_click_rate', 0.0):.1f}",
            f"Idle: {current.get('idle_sec', 0.0):.1f}s",
        ]
        
        # Размер подложки
        max_w = max(cv2.getTextSize(txt, self.FONT, self.SCALE, self.THICK)[0][0] for txt in lines)
        box_w, box_h = max_w + 30, len(lines) * self.LINE_H + 20
        
        # Рисуем полупрозрачную рамку
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-25), (x-10+box_w, y+box_h-15), self.BG_COLOR, -1)
        cv2.rectangle(overlay, (x-10, y-25), (x-10+box_w, y+box_h-15), self.BORDER_COLOR, 1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (x, y + i * self.LINE_H), self.FONT, self.SCALE, self.TEXT_COLOR, self.THICK, cv2.LINE_AA)
            
        return frame

    def _tilt_label(self, m: Dict[str, Any]) -> str:
        state = m.get("head_tilt_state", 0)
        labels = {0: "Normal", 1: "Light", 2: "Heavy"}
        return f"{labels.get(state, 'Err')} ({m.get('head_tilt_total', 0)})"
