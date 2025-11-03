import cv2
import mediapipe as mp
import numpy as np

from .key_pointers import FACE


class FaceDetector:
    ''' Класс детектора ключевых точек ЛИЦА
    
        Args:
            model_path (str): Путь к модели .task файлу
            num_detect (int): Максимальное количество детектируемых лиц (default: 1)
    '''

    def __init__(self, model_path, num_faces=1):
        
        ''' Инициализируем единожды настройки '''

        self.BaseOptions           = mp.tasks.BaseOptions
        self.FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.FaceLandmarkerResult  = mp.tasks.vision.FaceLandmarkerResult
        self.VisionRunningMode     = mp.tasks.vision.RunningMode

        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_detect
        )

        # создаем детектор
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)


    def get_key_pointers(self, frame):

        ''' Находит ключевые точки ЛИЦА (468), масштабированные относительно переданного изображения (кадра?) '''

        if frame is None or frame.size == 0:
            return None
        
        try:     
            # MediaPipe требует RGB своего формата
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Пропускаем изображение через нейросеть
            result = self.landmarker.detect(mp_image)

            h, w, _ = frame.shape
            
            # result.face_landmarks содержит список лиц, каждое лицо - 468 точек
            if result.face_landmarks:

                # Преобразуем landmarks в numpy массив            
                landmarks = np.array([ [lm.x, lm.y] for face in result.face_landmarks for lm in face ])
                landmarks = landmarks.reshape(len(result.face_landmarks), -1, 2)

                # Масштабируем относительно размеров изображения
                landmarks[:,:,0] *= w
                landmarks[:,:,1] *= h

                # Формат возвращаемых данных:
                # [
                #     [  # Первое лицо
                #         [x11, y11],  # точка 0
                #         [x12, y12],  # точка 1
                #         ...          # все 468 точек
                #     ],
                #     [  # Второе лицо (если num_faces > 1)
                #         [x21, y21],  # точка 0
                #         [x22, y22],  # точка 1
                #         ...          # все 468 точек
                #     ]
                # ]

                return landmarks[:, FACE, :].astype(int)
            
            # если лица не найдены
            return None

        except Exception as e:
            print(f'Проблемы в face_detector: {e}')
            return None
