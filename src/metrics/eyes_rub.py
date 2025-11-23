import numpy as np
from detectors.key_pointers import FACE

def detect_rubbing(face_kp: np.ndarray, pose_kp: np.ndarray,
                   verbose: bool = False) -> bool:
    """
    Определяет потирает ли человек глаза. Считаем минимальное растояние от точек руки до точек глаз
    Args:
        face_kp: numpy array точек головы
        pose_kp: numpy array точек туловища(без точек головы)
        verbose: если True, печатаем дополнительную информацию (по умолчанию: False)
    
    Returns:
        bool: True если есть потирания, иначе False 
    """

    if (face_kp is None or pose_kp is None or 
        len(face_kp) == 0 or len(pose_kp) == 0):
        return False
    
    """
    функция для пересчёта индексов точек
    
    # Используем индексы из доступных точек FACE
    # Преобразуем глобальные индексы MediaPipe в индексы нашего подмножества FACE

    # можно сразу переписать, тогда они не будут пересчитываться каждый раз
    # Создаем словарь для быстрого поиска индексов
    face_index_map = {point: idx for idx, point in enumerate(FACE)}
    
    # Выбираем точки глаз из доступных в FACE
    eye_points = []
     
    # Правый глаз + левый глаз
    for point in [362, 386, 374, 263, 33, 133, 159, 145]:  
        if point in face_index_map:
            eye_points.append(face_index_map[point])
    """

    eye_points = [16, 20, 28, 24, 0, 8, 4, 12]

    # Точки рук из позы
    #hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]

    # -11, т.к. точки лица не детектятся
    hands = [4, 5, 6, 7, 8, 9, 10, 11] 

    if not eye_points or not hands:
        print(f"No eye points: {len(eye_points)}, no hand points: {len(hands)}")
        return False

    try:
        # Создаем массивы координат
        eye_coords = np.array([face_kp[0][i] for i in eye_points])
        hand_coords = np.array([pose_kp[0][j] for j in hands])
        
        # Вычисляем расстояния
        distances = np.linalg.norm(eye_coords[:, np.newaxis] - hand_coords, axis=2)
        min_distance = np.min(distances)
        
        if verbose:
            print(f"Min distance between eyes and hands: {min_distance:.2f}")
        
        # Пороговое значение, мб стоит как-то высчитывать
        rubbing_threshold = 50  # пикселей
        is_rubbing = min_distance < rubbing_threshold
        
        if is_rubbing:
            # для тестов, убрать ближе к релизу
            print("ALERT: Eye rubbing detected!")
            
        return is_rubbing
        
    except Exception as e:
        print(f"Error in detect_rubbing: {e}")
        return False
    
