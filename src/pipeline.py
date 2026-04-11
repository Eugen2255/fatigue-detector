"""Единый пайплайн вычисления метрик усталости."""
from typing import Optional, Dict, Any
import numpy as np
import logging

from config import MetricConfig
from metrics.blinking import BlinkMetric
from metrics.eyes_rub import RubbingMetric
from metrics.head_tilt import HeadTiltMetric

logger = logging.getLogger(__name__)

class FatiguePipeline:
    """Координатор всех метрик. Принимает ключевые точки, возвращает словарь состояний."""
    
    def __init__(self, config: MetricConfig):
        """
        Args:
            config: Готовый объект конфигурации (из MetricConfig.from_yaml или дефолтный)
        """
        self.cfg = config
        
        # Инициализируем метрики, передавая им соответствующие секции конфига
        self.metrics = [
            BlinkMetric("blink", config.blink, config.smoothing_alpha),
            RubbingMetric("rubbing", config.rubbing, config.smoothing_alpha),
            HeadTiltMetric("head_tilt", config.head_tilt, config.smoothing_alpha)
        ]
        logger.info(f"FatiguePipeline инициализирован. Метрик: {len(self.metrics)}")

    def step(self, face_kp: np.ndarray, pose_kp: np.ndarray) -> dict:
        """Вычисляет все метрики и возвращает единый словарь состояний."""
        results = {}
        for metric in self.metrics:
            data = metric(face_kp, pose_kp)
            
            # Для head_tilt сохраняем и сырой угол (raw), и классифицированное состояние
            if metric.name == "head_tilt":
                results["head_tilt"] = (data.get("raw"), data["state"])
            else:
                # Для blink и rubbing достаточно состояния (0/1 или True/False)
                results[metric.name] = data["state"]
                
        return results

    def reset(self):
        """Сбрасывает внутреннее состояние (EMA-сглаживание) всех метрик."""
        for m in self.metrics:
            m.reset()
        logger.debug("Состояние пайплайна сброшено")