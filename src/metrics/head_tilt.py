import numpy as np

def detect_head_tilt(face_kp: np.ndarray, pose_kp: np.ndarray,
                     verbose: bool = False):
    """
    Определение наклона головы на основе расстояния между головой и плечами.
    Аргументы:
        face_kp: numpy array точек головы (shape: (1, N_face_points, 2))
        pose_kp: numpy array точек позы (shape: (1, N_pose_points, 2))
        verbose: печать отладочной информации
        
    Возвращает:
        tuple(angle_deg: float, state: int)
        где:
            state = 0 → наклон < 10°
            state = 1 → 10°–35°
            state = 2 → > 35°
        или None в случае ошибки
    """

    # Проверки входа
    if (face_kp is None or pose_kp is None 
        or len(face_kp) == 0 or len(pose_kp) == 0):
        if verbose:
            print("Пустой ввод ключевых точек, браток :)")
        return None
    
    try:
        nose_idx = 88
        l_shoulder_idx = 0
        r_shoulder_idx = 1

        nose = face_kp[nose_idx]
        l_sh = pose_kp[l_shoulder_idx]
        r_sh = pose_kp[r_shoulder_idx]

        # Центр плеч
        cx = (l_sh[0] + r_sh[0]) / 2
        cy = (l_sh[1] + r_sh[1]) / 2

        # Вектор от центра плеч к голове
        dx = nose[0] - cx
        dy = cy - nose[1]   # направление вверх

        # Защита от нулевого вектора
        if dx == 0 and dy == 0:
            if verbose:
                print("Нулевой вектор: голова и плечи сжаты")
            return None

        # Угол наклона головы
        angle_rad = np.arctan2(dx, dy)
        angle_deg = abs(np.degrees(angle_rad))

        # Состояние
        if angle_deg < 10:
            state = 0
        elif angle_deg < 35:
            state = 1
        else:
            state = 2

        if verbose:
            print(f"Angle: {angle_deg:.2f}°  State: {state}")

        return angle_deg, state

    except Exception as e:
        if verbose:
            print(f"Вышла ошибочка :) : {e}")
        return None

