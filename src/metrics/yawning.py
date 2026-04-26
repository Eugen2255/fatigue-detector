import numpy as np
from typing import Optional
from detectors.key_pointers import FACE
from .base_metric import BaseFatigueMetric

TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_CORNER = 78
RIGHT_CORNER = 308
CHIN = 152
FOREHEAD = 10

class YawnMetric(BaseFatigueMetric):
    def __init__(self, name: str, config: dict, smoothing_alpha: float = 0.3):
        super().__init__(name, config, smoothing_alpha)
        self._face_index_map = {pt: idx for idx, pt in enumerate(FACE)}
        self._open_frames = 0
        self._min_duration = config.get("min_duration_frames", 15)
        self._threshold = config.get("mar_threshold", 0.55)
        self._close_threshold = config.get("mar_close_threshold", self._threshold * 0.82)
        self._min_opening_px = config.get("min_opening_px", 10.0)
        self._face_height_ratio = config.get("min_face_height_ratio", 0.10)
        self._yawn_counted = False

    def _compute_raw(self, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> Optional[float]:
        if face_kp is None or face_kp.size == 0:
            return 0.0
        pts = face_kp[0] if face_kp.ndim == 3 else face_kp
        
        try:
            top_idx = self._face_index_map[TOP_LIP]
            bottom_idx = self._face_index_map[BOTTOM_LIP]
            left_idx = self._face_index_map[LEFT_CORNER]
            right_idx = self._face_index_map[RIGHT_CORNER]
            chin_idx = self._face_index_map[CHIN]
            forehead_idx = self._face_index_map[FOREHEAD]

            v = np.linalg.norm(pts[top_idx] - pts[bottom_idx])
            h = np.linalg.norm(pts[left_idx] - pts[right_idx])
            face_h = np.linalg.norm(pts[chin_idx] - pts[forehead_idx])

            if h < 1e-6 or face_h < 1e-6:
                return 0.0
            if v < self._min_opening_px or (v / face_h) < self._face_height_ratio:
                return 0.0

            return float(v / h)
        except Exception:
            return 0.0

    def _map_to_state(self, raw: float, smoothed: float) -> int:
        if smoothed >= self._threshold:
            self._open_frames += 1
        elif smoothed <= self._close_threshold:
            self._open_frames = 0
            
        if self._open_frames >= self._min_duration and not self._yawn_counted:
            self._yawn_counted = True
            return 1
            
        if smoothed <= self._close_threshold:
            self._yawn_counted = False
            
        return 0

    def reset(self):
        super().reset()
        self._open_frames = 0
        self._yawn_counted = False
