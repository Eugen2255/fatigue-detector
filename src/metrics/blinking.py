import numpy as np
from numpy.linalg import norm


RIGHT_EAR_POINTS = [0, 3, 5, 8, 11, 13]
LEFT_EAR_POINTS = [16, 19, 21, 24, 27, 29]


def _compute_ear(eye, verbose=False):
    """
    Специально для данечки :) ну и евгения точно :)
    Вычисляем EAR (Eye Aspect Ratio) для одного глаза EAR крч — это коэффициент раскрытия глаза, определяемый как отношение сумм
    вертикальных расстояний между веками к горизонтальной ширине глаза:

    EAR = ((p2 - p6) + (p3 - p5)) / (2 * (p1 - p4))
    Где p1–p6 - заранее определённые ключевые точки глаза

    eye (np.ndarray): массив shape (6, 2) с координатами шести точек глаза
    verbose (bool): печать отладочных значений расстояний и EAR
    """
    try:
        p1, p2, p3, p4, p5, p6 = eye

        v1 = norm(p2 - p6) # первая вертикаль
        v2 = norm(p3 - p5) # вторая вертикаль
        h  = norm(p1 - p4) # горизонталь

        if h == 0:
            return None

        ear = (v1 + v2) / (2.0 * h)

        if verbose:
            print(f"v1={v1:.4f}, v2={v2:.4f}, h={h:.4f}, EAR={ear:.4f}")
        return ear
    
    except:
        return None

def detect_blink(face_kp: np.ndarray,
                 ear_threshold: float = 0.18,
                 verbose: bool = False):
    """
    Снова специально для данечки :) ну и евгения точно :)
    Детектор моргания на основе EAR

        1)Извлекаем точки левого и правого глаз
        2)Вычисляем EAR для каждого глаза
        3)Находим среднее EAR
        4)Сравниваем EAR со статическим порогом (обычно ~0.18(так надо)) Если EAR ниже порога, то ок, глаз считается закрытым

        face_kp (np.ndarray):
            Массив ключевых точек лица вида (1, N_points, 2),
            где N_points ≥ максимальный индекс используемых точек MediaPipe
        ear_threshold (float):
            Порог закрытого глаза, EAR < threshold, то омг глаз закрыт

            0 — глаза открыты  
            1 — глаза закрыты (моргание)  
    """

    if face_kp is None or len(face_kp) == 0:
        if verbose:
            print("Нет face_kп")
        return None

    try:
        left_eye = face_kp[LEFT_EAR_POINTS]
        right_eye = face_kp[RIGHT_EAR_POINTS]

        ear_left = _compute_ear(left_eye, verbose)
        ear_right = _compute_ear(right_eye, verbose)

        if ear_left is None or ear_right is None:
            if verbose:
                print("EAR невозможно вычислить")
            return None

        ear_avg = (ear_left + ear_right) / 2
        state = 1 if ear_avg < ear_threshold else 0

        if verbose:
            print(f"State={state}")

        return state

    except Exception as e:
        if verbose:
            print("Ошибка в detect_blink:", e)
        return None
