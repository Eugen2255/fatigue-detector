# base_metric.py
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseFatigueMetric(ABC):
    """Базовый класс для всех метрик усталости."""
    
    def __init__(self, name: str, config: Dict[str, Any], smoothing_alpha: float = 0.3):
        self.name = name
        self.config = config
        self.alpha = max(0.0, min(1.0, smoothing_alpha))
        self._smoothed_value: Optional[float] = None

    def __call__(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Dict[str, Any]:
        """Вычисляет сырое значение, сглаживает его и мапит в состояние."""
        raw = self._compute_raw(face_kp, pose_kp)
        
        if raw is None:
            self._smoothed_value = None
            return {"metric": self.name, "state": None, "raw": None, "smoothed": None}

        smoothed = self._update_smoothing(float(raw))
        state = self._map_to_state(raw, smoothed)
        
        logger.debug(f"[{self.name}] raw={raw:.3f} | smoothed={smoothed:.3f} | state={state}")
        return {"metric": self.name, "state": state, "raw": raw, "smoothed": smoothed}

    def _update_smoothing(self, value: float) -> float:
        """Экспоненциальное скользящее среднее (EMA)."""
        if self._smoothed_value is None:
            self._smoothed_value = value
        else:
            self._smoothed_value = self.alpha * value + (1 - self.alpha) * self._smoothed_value
        return self._smoothed_value

    def reset(self) -> None:
        """Сброс внутреннего состояния (вызывать при смене пользователя/сцены)."""
        self._smoothed_value = None

    @abstractmethod
    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float | bool]:
        """Реализация математики метрики. Возвращает сырое число/булево."""
        ...

    @abstractmethod
    def _map_to_state(self, raw: float | bool, smoothed: float) -> Any:
        """Преобразование значения в понятное состояние (0/1/2, bool, str и т.д.)."""
        ...