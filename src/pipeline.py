"""Единый пайплайн вычисления метрик усталости."""
from typing import Optional, Dict, Any
import numpy as np
import logging

from config import MetricConfig
from metrics.blinking import BlinkMetric
from metrics.perclos import PerclosMetric
from metrics.eyes_rub import RubbingMetric
from metrics.head_tilt import HeadTiltMetric
from metrics.yawning import YawnMetric

logger = logging.getLogger(__name__)

class FatiguePipeline:
    def __init__(self, config: MetricConfig):
        self.cfg = config
        self.metrics = [
            BlinkMetric("blink", config.blink, config.smoothing_alpha),
            PerclosMetric("perclos", config.blink, config.smoothing_alpha), # Используем тот же порог EAR
            RubbingMetric("rubbing", config.rubbing, config.smoothing_alpha),
            HeadTiltMetric("head_tilt", config.head_tilt, config.smoothing_alpha),
            YawnMetric("yawn", config.yawn, config.smoothing_alpha)
        ]
        logger.info(f"FatiguePipeline инициализирован. Метрик: {len(self.metrics)}")

    def step(self, face_kp: np.ndarray, pose_kp: np.ndarray) -> dict:
        results = {}
        for metric in self.metrics:
            data = metric(face_kp, pose_kp)
            
            if metric.name == "head_tilt":
                results["head_tilt"] = (data.get("raw"), data["state"])
            elif metric.name in ["yawn", "perclos"]:
                results[metric.name] = data["state"]
            else:
                results[metric.name] = data["state"]
                
        return results

    def reset(self):
        for m in self.metrics:
            m.reset()
        logger.debug("Состояние пайплайна сброшено")