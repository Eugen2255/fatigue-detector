"""Анализ собранных метрик. Работает ТОЛЬКО с DataFrame из collector."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Возвращает структурированный анализ паттернов."""
        if df.empty:
            return {}
            
        return {
            "blinks": self._analyze_blinks(df),
            "rubbing": self._analyze_rubbing(df),
            "head_tilt": self._analyze_head_tilt(df),
            "input": self._analyze_input(df),
        }

    def _analyze_blinks(self, df: pd.DataFrame) -> Dict[str, Any]:
        blink_frames = df[df["blink"] == 1]["frame"].values
        if len(blink_frames) < 2:
            return {"total": len(blink_frames), "avg_interval_sec": 0.0, "min_interval_sec": 0.0}
        
        intervals = np.diff(blink_frames) / self.fps
        return {
            "total": len(blink_frames),
            "avg_interval_sec": float(np.mean(intervals)),
            "min_interval_sec": float(np.min(intervals)),
            "max_interval_sec": float(np.max(intervals))
        }

    def _analyze_rubbing(self, df: pd.DataFrame) -> Dict[str, Any]:
        state = df["rubbing"].astype(int)
        if state.sum() == 0:
            return {"total_events": 0, "total_frames": 0, "avg_duration_sec": 0.0}
            
        # Векторизованная группировка последовательных 1
        changes = state.diff().fillna(0) != 0
        groups = changes.cumsum()
        event_lengths = df.groupby(groups)["rubbing"].sum()
        events = event_lengths[event_lengths > 0]
        
        return {
            "total_events": len(events),
            "total_frames": int(state.sum()),
            "avg_duration_sec": float(events.mean() / self.fps) if len(events) > 0 else 0.0
        }

    def _analyze_head_tilt(self, df: pd.DataFrame) -> Dict[str, Any]:
        tilted = df[df["head_tilt_state"] > 0]
        if tilted.empty:
            return {"total_events": 0, "light": 0, "heavy": 0, "avg_angle": None, "max_angle": None}
            
        angles = tilted["head_tilt_angle"].dropna()
        return {
            "total_events": len(tilted),
            "light": int((df["head_tilt_state"] == 1).sum()),
            "heavy": int((df["head_tilt_state"] == 2).sum()),
            "avg_angle": float(angles.mean()) if not angles.empty else None,
            "max_angle": float(angles.max()) if not angles.empty else None
        }

    def _analyze_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "key_presses" not in df.columns:
            return {}

        return {
            "key_press_total": int(df["key_presses"].sum()),
            "key_press_rate_per_min": float(df["key_presses"].sum() / len(df) * self.fps * 60.0),
            "mouse_click_total": int(df["mouse_clicks"].sum()) if "mouse_clicks" in df.columns else 0,
            "mouse_click_rate_per_min": float(df["mouse_clicks"].sum() / len(df) * self.fps * 60.0) if "mouse_clicks" in df.columns else 0.0,
            "mouse_distance_px_total": float(df["mouse_distance_px"].sum()) if "mouse_distance_px" in df.columns else 0.0,
            "idle_pct": float((df["idle_ms"] >= 2000.0).mean() * 100.0) if "idle_ms" in df.columns else 0.0,
        }
