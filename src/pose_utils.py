import cv2
import mediapipe as mp
import numpy as np


# --- базовые классы MediaPipe Tasks ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


''' Для некоторых метрик, придется создавать отдельные точки (присущие только им) '''

def found_kp (img, model_path):

    ''' Находит ключевые точки масштабированные относительно переданного изображения (кадра?) '''

    # общие настройки
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # MediaPipe требует RGB своего формата
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # мы пропускаем изображение через нейросеть
    with PoseLandmarker.create_from_options(options) as landmarker:
        # и получаем объект с 33 ключевыми точками
        result = landmarker.detect(mp_image)
        # 33 ключевые точки включают:
        # 0-10: лицо
        # 11-16: плечи, локти, запястья
        # 23-28: бёдра, колени, лодыжки
        # 29-32: ступни

    h,w,_ = img.shape
    # result.pose_landmarks - возвращает список обнаруженных поз, где каждая поза набор 33 точек
    if result.pose_landmarks:

        # проходимя по каждой позе, по каждой точке позы
        landmarks = np.array( [ [lm.x, lm.y] for pose in result.pose_landmarks for lm in pose] )
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
