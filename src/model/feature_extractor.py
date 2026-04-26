"""Извлечение признаков из окна метрик для классификации."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .model_config import ModelConfig

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Преобразует окно сырых метрик в вектор признаков для модели."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
    
    def extract(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Извлекает признаки из окна метрик.
        
        Args:
            window: DataFrame с метриками за последние N кадров.
                   Ожидает колонки: blink, rubbing, head_tilt_state, head_tilt_angle, face_detected, pose_detected
        Returns:
            Dict с признаками, готовыми к масштабированию и предсказанию.
        """
        if window.empty:
            logger.warning("Пустое окно метрик, возвращаем нулевые признаки")
            return {f: 0.0 for f in self.cfg.expected_features}
        
        features = {}
        
        # === Моргания ===
        blinks = window[window["blink"] == 1]
        features["blink_count"] = len(blinks)
        features["blink_frequency"] = len(blinks) / len(window) * 30  # нормализация к 30 FPS
        
        if len(blinks) >= 2:
            intervals = np.diff(blinks["frame"].values) if "frame" in blinks.columns else np.arange(len(blinks))
            features["blink_max_interval"] = float(np.max(intervals))
            features["blink_avg_interval"] = float(np.mean(intervals))
        else:
            features["blink_max_interval"] = 0.0
            features["blink_avg_interval"] = 0.0

        # === Зевки и PERCLOS ===
        yawns = window[window["yawn"] == 1] if "yawn" in window.columns else window.iloc[0:0]
        features["yawn_count"] = len(yawns)
        features["yawn_pct"] = len(yawns) / len(window) * 100 if len(window) > 0 else 0.0
        features["perclos_pct"] = window["perclos"].mean() * 100 if "perclos" in window.columns else 0.0
        
        # === Наклон головы ===
        tilted = window[window["head_tilt_state"] > 0]
        angles = window["head_tilt_angle"].dropna()
        
        features["head_tilt_max_angle"] = float(angles.max()) if not angles.empty else 0.0
        features["head_tilt_avg_angle"] = float(angles.mean()) if not angles.empty else 0.0
        features["head_tilt_std_angle"] = float(angles.std()) if len(angles) > 1 else 0.0
        
        features["head_tilt_light_pct"] = (window["head_tilt_state"] == 1).sum() / len(window) * 100
        features["head_tilt_heavy_pct"] = (window["head_tilt_state"] == 2).sum() / len(window) * 100
        features["head_tilt_events"] = len(tilted)
        
        # === Потирание глаз ===
        rubbing = window[window["rubbing"] == True]
        features["rubbing_count"] = len(rubbing)
        features["rubbing_duration_avg"] = float(len(rubbing) / len(window) * 30) if len(rubbing) > 0 else 0.0
        features["rubbing_pct"] = len(rubbing) / len(window) * 100

        # === Ввод с клавиатуры и мыши ===
        key_series = window["key_presses"] if "key_presses" in window.columns else pd.Series(0.0, index=window.index)
        active_key_series = window["active_keys"] if "active_keys" in window.columns else pd.Series(0.0, index=window.index)
        keyboard_burst_series = window["keyboard_burst"] if "keyboard_burst" in window.columns else pd.Series(0.0, index=window.index)
        mouse_click_series = window["mouse_clicks"] if "mouse_clicks" in window.columns else pd.Series(0.0, index=window.index)
        mouse_distance_series = window["mouse_distance_px"] if "mouse_distance_px" in window.columns else pd.Series(0.0, index=window.index)
        mouse_speed_series = window["mouse_speed_px_s"] if "mouse_speed_px_s" in window.columns else pd.Series(0.0, index=window.index)
        mouse_active_series = window["mouse_active"] if "mouse_active" in window.columns else pd.Series(0.0, index=window.index)
        idle_series = window["idle_ms"] if "idle_ms" in window.columns else pd.Series(0.0, index=window.index)

        features["key_press_count"] = float(key_series.sum())
        features["key_press_rate"] = float(key_series.sum() / len(window) * 30.0)
        features["active_keys_avg"] = float(active_key_series.mean())
        features["keyboard_burst_pct"] = float(keyboard_burst_series.mean() * 100.0)
        features["mouse_click_count"] = float(mouse_click_series.sum())
        features["mouse_click_rate"] = float(mouse_click_series.sum() / len(window) * 30.0)
        features["mouse_distance_total"] = float(mouse_distance_series.sum())
        features["mouse_speed_avg"] = float(mouse_speed_series.mean())
        features["mouse_active_pct"] = float(mouse_active_series.mean() * 100.0)
        features["idle_pct"] = float((idle_series >= 2000.0).mean() * 100.0)
        features["idle_avg_sec"] = float(idle_series.mean() / 1000.0)
        
        # === Детекция ===
        features["face_detected_pct"] = window["face_detected"].mean() * 100 if "face_detected" in window.columns else 100.0
        features["pose_detected_pct"] = window["pose_detected"].mean() * 100 if "pose_detected" in window.columns else 100.0
        features["frames_no_face"] = int((~window["face_detected"]).sum()) if "face_detected" in window.columns else 0

        # === Синтетические/агрегированные признаки ===
        features["activity_score"] = (features["blink_frequency"] * 0.4 + 
                                      features["head_tilt_avg_angle"] * 0.3 + 
                                      features["rubbing_pct"] * 0.3) / 100
        features["input_activity_score"] = (
            min(features["key_press_rate"], 180.0) / 180.0 * 0.4 +
            min(features["mouse_click_rate"], 60.0) / 60.0 * 0.2 +
            min(features["mouse_active_pct"], 100.0) / 100.0 * 0.4
        )
        features["fatigue_score_1"] = features["blink_frequency"] * 0.35 + features["yawn_pct"] * 0.25 + features["perclos_pct"] * 0.15
        features["fatigue_score_2"] = features["head_tilt_heavy_pct"] * 0.45 + features["idle_pct"] * 0.20
        features["fatigue_score_3"] = features["rubbing_pct"] * 0.25 + features["idle_avg_sec"] * 4.0 - features["input_activity_score"] * 10.0
        
        # === Временные признаки (опционально, если есть временная метка) ===
        if "timestamp" in window.columns and len(window) > 1:
            # Простая синусоида от нормализованного времени окна
            t_norm = (window.index[-1] - window.index[0]) / len(window)
            features["time_sin"] = float(np.sin(2 * np.pi * t_norm))
            features["time_cos"] = float(np.cos(2 * np.pi * t_norm))
        else:
            features["time_sin"] = 0.0
            features["time_cos"] = 0.0
        
        # === Валидация: все признаки должны быть числовыми и конечными ===
        for key in self.cfg.expected_features:
            val = features.get(key, 0.0)
            features[key] = float(val) if np.isfinite(val) else 0.0
        
        return features
