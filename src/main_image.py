import cv2
import numpy as np

from detectors import *


''' ФАЙЛ ДЛЯ ТЕСТОВ НА КОНКРЕТНОМ ИЗОБРАЖЕНИИ '''


# пути к модели и изображению (добавьте ваши пути для тестов)
pose_model_path = r'D:/Desktop/fatigue-detector/models/pose_landmarker_lite.task'
face_model_path = r'D:/Desktop/fatigue-detector/models/face_landmarker.task'

# путь к вашему изображению
img_path = r'D:/Desktop/fatigue-detector/images/img1.jpg'

cap = cv2.VideoCapture(0)


img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f'Файл по пути {img_path} не найден')


pose = PoseDetector(pose_model_path)
face = FaceDetector(face_model_path)

poses_kps = pose.get_key_pointers(img)
faces_kps = face.get_key_pointers(img)

if poses_kps is not None:
    for person in poses_kps:
        for x, y in person:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

if faces_kps is not None:
    for face_points in faces_kps:
        for x, y in face_points:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()