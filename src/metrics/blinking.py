import numpy as np
from typing import Optional
from .base_metric import BaseFatigueMetric

RIGHT_EAR_POINTS = [0, 3, 5, 8, 11, 13]
LEFT_EAR_POINTS = [16, 19, 21, 24, 27, 29]

class BlinkMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: dict, smoothing_alpha: float = 0.3):
        super().__init__(name, config, smoothing_alpha)
        self._is_blinking = False
        self._blink_cooldown = 0
        
    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        if face_kp is None or face_kp.size == 0: return None
        pts = face_kp[0] if face_kp.ndim == 3 else face_kp
        try:
            ear_l = self._ear(pts[LEFT_EAR_POINTS])
            ear_r = self._ear(pts[RIGHT_EAR_POINTS])
            if ear_l is None or ear_r is None: return None
            return (ear_l + ear_r) / 2.0
        except: return None

    def _map_to_state(self, raw: float, smoothed: float) -> int:
        # Используем более строгий порог для "закрыто" и гистерезис
        close_thresh = self.config.get("ear_threshold", 0.22)
        open_thresh = close_thresh + 0.05 # Чтобы не мерцало на границе
        
        if self._blink_cooldown > 0:
            self._blink_cooldown -= 1
            return 0

        if not self._is_blinking:
            # Ищем начало моргания
            if smoothed < close_thresh:
                self._is_blinking = True
                return 1 # Фиксируем событие моргания
        else:
            # Ждем пока глаз откроется, чтобы сбросить флаг
            if smoothed > open_thresh:
                self._is_blinking = False
                self._blink_cooldown = 5 # Защита от двойного счета (дребезга)
        
        return 0

    @staticmethod
    def _ear(eye: np.ndarray) -> Optional[float]:
        if eye.shape != (6, 2): return None
        p1, p2, p3, p4, p5, p6 = eye
        h = np.linalg.norm(p1 - p4)
        if h < 1e-6: return 0.0
        return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * h)