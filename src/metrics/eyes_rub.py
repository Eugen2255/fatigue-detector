"""Детекция потирания глаз по минимальному расстоянию рук до точек лица."""
import logging
import numpy as np
from typing import Optional, Dict, Any
from detectors.key_pointers import FACE
from .base_metric import BaseFatigueMetric

logger = logging.getLogger(__name__)

# Глобальные индексы MediaPipe FaceMesh
MP_EYE_POINTS = [362, 386, 374, 263, 33, 133, 159, 145]  # Углы глаз + веки

# В pose_kp точки лица пропущены (сдвиг -11). 
# Локальные индексы 4-11 соответствуют MP 15-22 (запястья, кисти, большие пальцы)
POSE_HAND_POINTS_LOCAL = [4, 5, 6, 7, 8, 9, 10, 11]


class RubbingMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: Dict[str, Any], smoothing_alpha: float = 0.3):
        super().__init__(name, config, smoothing_alpha)
        
        # Предрасчёт маппинга для производительности (делается 1 раз при инициализации)
        self._face_index_map = {pt: idx for idx, pt in enumerate(FACE)}
        self._local_eye_indices = [
            self._face_index_map[pt] for pt in MP_EYE_POINTS if pt in self._face_index_map
        ]
        if len(self._local_eye_indices) != len(MP_EYE_POINTS):
            logger.warning("Не все точки глаз найдены в массиве FACE. Проверьте константы.")

    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        """Возвращает минимальное расстояние (в пикселях) между глазами и руками."""
        if face_kp is None or pose_kp is None or face_kp.size == 0 or pose_kp.size == 0:
            return None

        try:
            face_pts = face_kp[0] if face_kp.ndim == 3 else face_kp
            pose_pts = pose_kp[0] if pose_kp.ndim == 3 else pose_kp

            valid_hand_idx = [i for i in POSE_HAND_POINTS_LOCAL if 0 <= i < len(pose_pts)]
            if not valid_hand_idx or not self._local_eye_indices:
                return None

            eye_coords = face_pts[self._local_eye_indices]
            hand_coords = pose_pts[valid_hand_idx]

            # Матрица попарных расстояний: (N_eyes, N_hands)
            dist_matrix = np.linalg.norm(eye_coords[:, np.newaxis] - hand_coords, axis=2)
            return float(np.min(dist_matrix))

        except Exception as e:
            logger.error(f"Ошибка вычисления расстояния в RubbingMetric: {e}", exc_info=True)
            return None

    def _map_to_state(self, raw: float, smoothed: float) -> bool:
        """Пороговое сравнение сглаженного расстояния."""
        threshold = self.config.get("distance_threshold_px", 50.0)
        return smoothed < threshold