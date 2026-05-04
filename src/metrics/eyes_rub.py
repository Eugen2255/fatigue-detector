import logging
import numpy as np
from typing import Optional, Dict, Any
from detectors.key_pointers import FACE
from .base_metric import BaseFatigueMetric

logger = logging.getLogger(__name__)

MP_EYE_POINTS = [362, 386, 374, 263, 33, 133, 159, 145]
POSE_HAND_POINTS_LOCAL = [4, 5, 6, 7, 8, 9, 10, 11]

class RubbingMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: Dict[str, Any], smoothing_alpha: float = 0.3):
        super().__init__(name, config, smoothing_alpha)
        self._face_index_map = {pt: idx for idx, pt in enumerate(FACE)}
        self._local_eye_indices = [self._face_index_map[pt] for pt in MP_EYE_POINTS if pt in self._face_index_map]
        self._is_rubbing = False
        self._cooldown = 0

    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        if face_kp is None or pose_kp is None or face_kp.size == 0 or pose_kp.size == 0: return None
        try:
            face_pts = face_kp[0] if face_kp.ndim == 3 else face_kp
            pose_pts = pose_kp[0] if pose_kp.ndim == 3 else pose_kp
            valid_hand = [i for i in POSE_HAND_POINTS_LOCAL if 0 <= i < len(pose_pts)]
            if not valid_hand or not self._local_eye_indices: return None
            eye_coords = face_pts[self._local_eye_indices]
            hand_coords = pose_pts[valid_hand]
            return float(np.min(np.linalg.norm(eye_coords[:, np.newaxis] - hand_coords, axis=2)))
        except: return None

    def _map_to_state(self, raw: float, smoothed: float) -> bool:
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        
        threshold = self.config.get("distance_threshold_px", 45.0)
        is_close = smoothed < threshold
        
        # Фиксируем событие только при входе в зону
        if is_close and not self._is_rubbing:
            self._is_rubbing = True
            self._cooldown = 40  # ~1.3 сек защиты от повторного счета
            return True
        elif not is_close:
            self._is_rubbing = False
        return False