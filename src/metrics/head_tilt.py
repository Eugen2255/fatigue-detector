import logging
import numpy as np
from typing import Optional, Dict, Any
from .base_metric import BaseFatigueMetric

logger = logging.getLogger(__name__)

NOSE_IDX = 88
L_SHOULDER_IDX = 0
R_SHOULDER_IDX = 1

class HeadTiltMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: Dict[str, Any], smoothing_alpha: float = 0.1): # Более сильное сглаживание (0.1)
        super().__init__(name, config, smoothing_alpha)

    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        if face_kp is None or pose_kp is None or face_kp.size == 0 or pose_kp.size == 0:
            return None
        try:
            face_pts = face_kp[0] if face_kp.ndim == 3 else face_kp
            pose_pts = pose_kp[0] if pose_kp.ndim == 3 else pose_kp

            if not (0 <= NOSE_IDX < len(face_pts)): return None
            if not (0 <= L_SHOULDER_IDX < len(pose_pts)) or not (0 <= R_SHOULDER_IDX < len(pose_pts)): return None

            nose = face_pts[NOSE_IDX]
            l_sh = pose_pts[L_SHOULDER_IDX]
            r_sh = pose_pts[R_SHOULDER_IDX]

            shoulder_center = (l_sh + r_sh) / 2.0
            vec = nose - shoulder_center

            if np.linalg.norm(vec) < 1e-6: return None

            angle_rad = np.arctan2(vec[0], -vec[1])
            return float(abs(np.degrees(angle_rad)))
        except Exception as e:
            return None

    def _map_to_state(self, raw: float, smoothed: float) -> int:
        mild = self.config.get("mild_tilt_deg", 12.0)
        severe = self.config.get("severe_tilt_deg", 35.0)

        if smoothed < mild: return 0
        elif smoothed < severe: return 1
        return 2