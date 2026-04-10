import logging
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from .key_pointers import FACE

logger = logging.getLogger(__name__)

class FaceDetector:
    """Детектор ключевых точек лица на базе MediaPipe Face Landmarker."""

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
        """Возвращает координаты ключевых точек лица, масштабированные под размер кадра.
        
        Args:
            frame: Входное изображение в формате BGR (OpenCV).
            indices: Опциональный список индексов для фильтрации точек. 
                     По умолчанию используется константа FACE.
        Returns:
            np.ndarray формы (num_faces, N, 2) или None, если лица не найдены.
        """
        if frame is None or frame.size == 0:
            return None

        # По умолчанию берём только нужные точки
        target_indices = indices if indices is not None else FACE

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                return None

            h, w, _ = frame.shape
            
            # Быстрое преобразование в numpy
            all_landmarks = [pt for face in result.face_landmarks for pt in face]
            landmarks = np.array([[pt.x, pt.y] for pt in all_landmarks], dtype=np.float32)
            landmarks = landmarks.reshape(len(result.face_landmarks), -1, 2)

            # Масштабирование
            landmarks[:, :, 0] *= w
            landmarks[:, :, 1] *= h

            # Фильтрация + округление до ближайшего пикселя
            return np.round(landmarks[:, target_indices, :]).astype(np.int32)

        except Exception as e:
            logger.error(f"Ошибка в FaceDetector: {e}", exc_info=True)
            return None

    def close(self):
        """Освобождает ресурсы MediaPipe."""
        if hasattr(self._landmarker, "close"):
            self._landmarker.close()

    def __del__(self):
        self.close()