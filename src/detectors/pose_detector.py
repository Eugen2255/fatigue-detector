import logging
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Sequence, Union

logger = logging.getLogger(__name__)

# По умолчанию пропускаем точки лица (0-10) и берем только тело (11-32)
DEFAULT_POSE_INDICES = slice(11, None)

class PoseDetector:
    """Детектор ключевых точек позы на базе MediaPipe Pose Landmarker."""

    def __init__(self, model_path: str, num_poses: int = 1):
        self._landmarker = self._init_landmarker(model_path, num_poses)

    def _init_landmarker(self, model_path: str, num_poses: int):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=num_poses
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def get_key_pointers(
        self,
        frame: np.ndarray,
        indices: Optional[Union[Sequence[int], slice]] = DEFAULT_POSE_INDICES
    ) -> Optional[np.ndarray]:
        """Возвращает координаты ключевых точек позы, масштабированные под размер кадра.

        Args:
            frame: Входное изображение в формате BGR (OpenCV).
            indices: Индексы или срез точек для возврата.
                     По умолчанию пропускает точки лица (0-10) и возвращает тело (11-32).

        Returns:
            np.ndarray формы (num_poses, N, 2) или None, если позы не найдены.
        """
        if frame is None or frame.size == 0:
            return None

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self._landmarker.detect(mp_image)

            if not result.pose_landmarks:
                return None

            h, w, _ = frame.shape
            num_poses = len(result.pose_landmarks)

            # Определяем нужные индексы (поддержка slice и list/tuple)
            if isinstance(indices, slice):
                idx_range = range(*indices.indices(33))
            else:
                idx_range = indices

            # Преобразование в numpy массив
            landmarks = np.array([
                [pose[i].x, pose[i].y] for pose in result.pose_landmarks for i in idx_range
            ], dtype=np.float32)

            num_points = len(idx_range)
            landmarks = landmarks.reshape(num_poses, num_points, 2)

            # Масштабирование к пикселям кадра
            landmarks[:, :, 0] *= w
            landmarks[:, :, 1] *= h

            return np.round(landmarks).astype(np.int32)

        except Exception as e:
            logger.error(f"Ошибка в PoseDetector: {e}", exc_info=True)
            return None

    def close(self):
        """Явно освобождает нативные ресурсы MediaPipe."""
        if hasattr(self._landmarker, "close"):
            self._landmarker.close()

    def __del__(self):
        self.close()