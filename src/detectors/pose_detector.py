import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:

    ''' Класс детектора ключевых точек ПОЗЫ '''

    def __init__(self, model_path, name_detect=1):

        ''' Инициализируем единожды настройки '''

        self.BaseOptions           = mp.tasks.BaseOptions
        self.PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode     = mp.tasks.vision.RunningMode

        self.options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=name_detect
        )

        self.landmarker = self.PoseLandmarker.create_from_options(self.options)


    def get_key_pointers(self, frame):

        ''' Находит ключевые точки ПОЗЫ (33), масштабированные относительно переданного изображения (кадра?) '''

        if frame is None or frame.size == 0:
            return None
        
        try:
            # MediaPipe требует RGB своего формата
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            result = self.landmarker.detect(mp_image)

            h,w,_ = frame.shape

            # 33 ключевые точки включают:
            # 0-10: лицо
            # 11-16: плечи, локти, запястья
            # 23-28: бёдра, колени, лодыжки
            # 29-32: ступни
            # result.pose_landmarks - возвращает список обнаруженных поз, где каждая поза набор 33 точек
            if result.pose_landmarks:

                # проходимя по каждой позе, по каждой точке позы
                landmarks = np.array( [ [lm.x, lm.y] for pose in result.pose_landmarks for lm in pose[11:]] )
                landmarks = landmarks.reshape(len(result.pose_landmarks), -1, 2)

                # масштабируем относительно размеров изображения
                landmarks[:,:,0] *= w
                landmarks[:,:,1] *= h

                # [
                #     [  # Первая поза (первый человек)
                #         [x11, y11],  # точка 0: nose
                #         [x12, y12],  # точка 1: left_eye_inner
                #         [x13, y13],  # точка 2: left_eye
                #         ...          # все 33 точки
                #     ],
                #     [  # Вторая поза (второй человек)  
                #         [x21, y21],  # точка 0: nose
                #         [x22, y22],  # точка 1: left_eye_inner
                #         [x23, y23],  # точка 2: left_eye
                #         ...          # все 33 точки
                #     ]
                # ]

                return landmarks.astype(int)
            
            # если позы не нашлись
            return None
        
        except Exception as e:
            print(f'Проблемы в pose_detector: {e}')
            return None