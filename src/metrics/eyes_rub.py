import numpy as np

def detect_rubbing(face_kp: np.ndarray, pose_kp: np.ndarray) -> bool:
    if face_kp is None or pose_kp is None or len(face_kp) == 0 or len(pose_kp) == 0:
        return False

    # Используем индексы из доступных точек FACE
    # Преобразуем глобальные индексы MediaPipe в индексы нашего подмножества FACE
    from detectors.key_pointers import FACE
    
    # Создаем словарь для быстрого поиска индексов
    face_index_map = {point: idx for idx, point in enumerate(FACE)}
    
    # Выбираем точки глаз из доступных в FACE
    eye_points = []
    # Левое глаз (примерные точки - используем те, что есть в RIGHT_EYE/LEFT_EYE)
    for point in [33, 133, 159, 145]:  # левый глаз
        if point in face_index_map:
            eye_points.append(face_index_map[point])
    
    # Правый глаз
    for point in [362, 386, 374, 263]:  # правый глаз
        if point in face_index_map:
            eye_points.append(face_index_map[point])
        
    # Точки рук из позы
    #hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]

    # -11, т.к. точки лица не детектятся
    hand_indices = [4, 5, 6, 7, 8, 9, 10, 11] 

    hands = []
    for idx in hand_indices:
        hands.append(idx)
    
    if not eye_points or not hands:
        print(f"No eye points: {len(eye_points)}, no hand points: {len(hands)}")
        return False

    try:
        # Создаем массивы координат
        # !!!! тут крашимся
        print(face_kp[0])
        eye_coords = np.array([face_kp[0][i] for i in eye_points])
        hand_coords = np.array([pose_kp[0][j] for j in hands])
        
        # Вычисляем расстояния
        distances = np.linalg.norm(eye_coords[:, np.newaxis] - hand_coords, axis=2)
        min_distance = np.min(distances)

        print(f"Min distance between eyes and hands: {min_distance:.2f}")
        
        # Пороговое значение (настройте под ваш случай)
        rubbing_threshold = 80  # пикселей
        is_rubbing = min_distance < rubbing_threshold
        
        if is_rubbing:
            print("ALERT: Eye rubbing detected!")
            
        return is_rubbing
        
    except Exception as e:
        print(f"Error in detect_rubbing: {e}")
        return False