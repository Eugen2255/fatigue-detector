import cv2
import numpy as np

from detectors import *


''' ФАЙЛ ДЛЯ ТЕСТОВ НА ВЕБ-КАМЕРЕ '''


# пути к модели и изображению (добавьте ваши пути для тестов)
pose_model_path = r'D:/Desktop/fatigue-detector/models/pose_landmarker_lite.task'
face_model_path = r'D:/Desktop/fatigue-detector/models/face_landmarker.task'

cap = cv2.VideoCapture(0)

pose = PoseDetector(pose_model_path)
face = FaceDetector(face_model_path)


while True:

    suc, img = cap.read()
    if not suc: continue

    poses_kps = pose.get_key_pointers(img)
    faces_kps = face.get_key_pointers(img)

    all_kps = np.concatenate( [poses_kps, faces_kps], axis = 1 )

    for el in faces_kps:
        for x,y in el:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
    cv2.imshow("result image", img)

    key = cv2.waitKey(1)
    if key & 0xFF in (ord('q'), 27):
        break

cv2.destroyAllWindows()