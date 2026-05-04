import logging
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from .key_pointers import FACE

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str, num_faces: int = 1):
        self._num_faces = num_faces
        self._landmarker = self._init_landmarker(model_path)

    def _init_landmarker(self, model_path: str):
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=self._num_faces
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def get_key_pointers(self, frame: np.ndarray, indices: Optional[tuple] = None) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0: return None
        target_indices = indices if indices is not None else FACE

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks: return None

            h, w, _ = frame.shape
            all_landmarks = [pt for face in result.face_landmarks for pt in face]
            landmarks = np.array([[pt.x, pt.y] for pt in all_landmarks], dtype=np.float32)
            landmarks = landmarks.reshape(len(result.face_landmarks), -1, 2)
            landmarks[:, :, 0] *= w
            landmarks[:, :, 1] *= h
            return np.round(landmarks[:, target_indices, :]).astype(np.int32)
        except Exception as e:
            logger.error(f"FaceDetector error: {e}", exc_info=True)
            return None

    def close(self):
        if hasattr(self._landmarker, "close"): self._landmarker.close()

    def __del__(self): self.close()