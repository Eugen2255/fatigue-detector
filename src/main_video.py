import cv2
import numpy as np

from detectors import *
from exp_smoothing import exp_smoothing


''' ФАЙЛ ДЛЯ ТЕСТОВ НА ВЕБ-КАМЕРЕ '''


# пути к модели и изображению (добавьте ваши пути для тестов)
pose_model_path = r'D:/Desktop/fatigue-detector/models/pose_landmarker_lite.task'
face_model_path = r'D:/Desktop/fatigue-detector/models/face_landmarker.task'

cap = cv2.VideoCapture(0)



face = FaceDetector(face_model_path, num_faces=2)
# словарь предыдущих ключевых точек ЛИЦ (от 0 до num_faces-1) (каждый индекс соотвествует одному лицу)
pred_faces_kps = dict()
# сглаживать точки лица каждые SMOOTH_FACE_EXERY кадров
SMOOTH_FACE_EXERY = 5


pose = PoseDetector(pose_model_path, num_poses=2)
pred_poses_kps = dict()
SMOOTH_POSE_EVERY = 1


# кол-во кадров
num_frame = 0

while True:

    suc, img = cap.read()
    if not suc: continue
    else: num_frame += 1


    ''' Код ниже представлен для работы со сглаженными точками ЛИЦ '''

    # получаем все ключевые точки всех лиц
    faces_kps = face.get_key_pointers(img)

    # если есть лица, то пройдемся по ним
    if not faces_kps is None:
        # проходимся по каждому лицу
        for ind_one_face in range(len(faces_kps)):

            # если лицо детектилось раньше И нужно сгладить точки
            if ind_one_face in pred_faces_kps and (num_frame % SMOOTH_POSE_EVERY == 0):
                smooth_one_face = exp_smoothing( pred_faces_kps[ind_one_face], faces_kps[ind_one_face] )
            # иначе просто отображаем
            else:
                # smooth_one_face = exp_smoothing( faces_kps[ind_one_face], faces_kps[ind_one_face] )
                smooth_one_face = faces_kps[ind_one_face]

            # сохраняем последние актуальные ключевые точки лица
            pred_faces_kps[ind_one_face] = smooth_one_face

            # проходимся по каждой ключевой точке
            for x,y in smooth_one_face:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    
    
    ''' Код ниже представлен для работы со сглаженными точками ПОЗ '''

    poses_kps = pose.get_key_pointers(img)

    if not poses_kps is None:

        for ind_one_pose in range(len(poses_kps)):

            if ind_one_pose in pred_poses_kps and (num_frame % SMOOTH_POSE_EVERY == 0):
                smooth_one_pose = exp_smoothing( pred_poses_kps[ind_one_pose], poses_kps[ind_one_pose] )
            else:
                # smooth_one_pose = exp_smoothing( poses_kps[ind_one_pose], poses_kps[ind_one_pose] )
                smooth_one_pose = poses_kps[ind_one_pose]

            pred_poses_kps[ind_one_pose] = smooth_one_pose

            for x,y in smooth_one_pose:
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    

    cv2.imshow("result image", img)

    key = cv2.waitKey(1)
    if key & 0xFF in (ord('q'), 27): break

cv2.destroyAllWindows()