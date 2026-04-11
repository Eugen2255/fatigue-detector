"""Детекция наклона головы относительно линии плеч."""
import logging
import numpy as np
from typing import Optional, Dict, Any
from .base_metric import BaseFatigueMetric

logger = logging.getLogger(__name__)

# Индексы относительно ваших массивов key pointers
NOSE_IDX = 88               # Убедитесь, что соответствует точке носа в FACE
L_SHOULDER_IDX = 0          # MP 11 -> локальный 0 в pose_kp
R_SHOULDER_IDX = 1          # MP 12 -> локальный 1 в pose_kp


class HeadTiltMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: Dict[str, Any], smoothing_alpha: float = 0.3):
        super().__init__(name, config, smoothing_alpha)

    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        """Возвращает угол отклонения головы от вертикали в градусах."""
        if face_kp is None or pose_kp is None or face_kp.size == 0 or pose_kp.size == 0:
            return None

        try:
            face_pts = face_kp[0] if face_kp.ndim == 3 else face_kp
            pose_pts = pose_kp[0] if pose_kp.ndim == 3 else pose_kp

            # Проверка границ массивов
            if not (0 <= NOSE_IDX < len(face_pts)):
                return None
            if not (0 <= L_SHOULDER_IDX < len(pose_pts)) or not (0 <= R_SHOULDER_IDX < len(pose_pts)):
                return None

            nose = face_pts[NOSE_IDX]
            l_sh = pose_pts[L_SHOULDER_IDX]
            r_sh = pose_pts[R_SHOULDER_IDX]

            shoulder_center = (l_sh + r_sh) / 2.0
            vec = nose - shoulder_center

            if np.linalg.norm(vec) < 1e-6:
                return None

            # Угол относительно вертикали. В пикселях Y направлен вниз, поэтому -vec[1]
            angle_rad = np.arctan2(vec[0], -vec[1])
            return float(abs(np.degrees(angle_rad)))

        except Exception as e:
            logger.error(f"Ошибка вычисления угла в HeadTiltMetric: {e}", exc_info=True)
            return None

    def _map_to_state(self, raw: float, smoothed: float) -> int:
        """Классификация угла на 3 уровня строгости."""
        mild = self.config.get("mild_tilt_deg", 10.0)
        severe = self.config.get("severe_tilt_deg", 35.0)

        if smoothed < mild:
            return 0  # Норма
        elif smoothed < severe:
            return 1  # Умеренный наклон
        return 2       # Сильный наклон