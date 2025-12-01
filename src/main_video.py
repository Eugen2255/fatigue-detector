import cv2
import numpy as np

from detectors import *
from exp_smoothing import exp_smoothing
from metrics import *
from config import pose_path, face_path
from collector import *

''' ФАЙЛ ДЛЯ ТЕСТОВ НА ВЕБ-КАМЕРЕ '''

# пути к модели и изображению
pose_model_path = pose_path
face_model_path = face_path

cap = cv2.VideoCapture(0)

# Инициализация детекторов
face = FaceDetector(face_model_path, num_faces=2)
pose = PoseDetector(pose_model_path, num_poses=2)

# Словари для хранения предыдущих точек
pred_faces_kps = dict()
pred_poses_kps = dict()

# Параметры сглаживания
SMOOTH_FACE_EVERY = 5
SMOOTH_POSE_EVERY = 1

# Счетчик кадров
num_frame = 0

# ============== ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ СБОРА МЕТРИК ==============
metrics_collector = MetricsCollector(save_interval=300, save_path="collected_metrics")
metrics_visualizer = MetricsVisualizer(position=(10, 30))

# Переменные для хранения последних результатов детекции
last_head_tilt = None
last_blink = 0
last_rubbing = False

def is_valid_kps(kps):
    """Проверяет, валидны ли ключевые точки"""
    return kps is not None and len(kps) > 0

while True:
    suc, img = cap.read()
    if not suc: 
        continue
    else: 
        num_frame += 1

    # Получаем ключевые точки лиц
    faces_kps = face.get_key_pointers(img)
    
    # Получаем ключевые точки поз
    poses_kps = pose.get_key_pointers(img)

    # Обработка лиц (сглаживание и отрисовка)
    if is_valid_kps(faces_kps):
        for ind_one_face in range(len(faces_kps)):
            if ind_one_face in pred_faces_kps and (num_frame % SMOOTH_POSE_EVERY == 0):
                smooth_one_face = exp_smoothing(pred_faces_kps[ind_one_face], faces_kps[ind_one_face])
            else:
                smooth_one_face = faces_kps[ind_one_face]

            pred_faces_kps[ind_one_face] = smooth_one_face

            # Отрисовка точек лица
            for x, y in smooth_one_face:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    
    # Обработка поз (сглаживание и отрисовка)
    if is_valid_kps(poses_kps):
        for ind_one_pose in range(len(poses_kps)):
            if ind_one_pose in pred_poses_kps and (num_frame % SMOOTH_POSE_EVERY == 0):
                smooth_one_pose = exp_smoothing(pred_poses_kps[ind_one_pose], poses_kps[ind_one_pose])
            else:
                smooth_one_pose = poses_kps[ind_one_pose]

            pred_poses_kps[ind_one_pose] = smooth_one_pose

            # Отрисовка точек позы
            for x, y in smooth_one_pose:
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    # ============== СБОР МЕТРИК ==============
    # Инициализация переменных для метрик
    last_head_tilt = None
    last_blink = 0
    last_rubbing = False
    
    # Детекция метрик (берем первое лицо, если есть)
    if is_valid_kps(faces_kps):
        face_kps_for_metrics = faces_kps[0] if len(faces_kps) > 0 else None
        
        # Для pose_kps проверяем и берем первый элемент если есть
        pose_kps_for_metrics = None
        if is_valid_kps(poses_kps) and len(poses_kps) > 0:
            pose_kps_for_metrics = poses_kps[0]
        
        # Детекция наклона головы
        try:
            last_head_tilt = detect_head_tilt(face_kps_for_metrics, pose_kps_for_metrics)
        except Exception as e:
            print(f"Ошибка в detect_head_tilt: {e}")
            last_head_tilt = None
        
        # Детекция моргания
        try:
            last_blink = detect_blink(face_kps_for_metrics, verbose=True)
        except Exception as e:
            print(f"Ошибка в detect_blink: {e}")
            last_blink = 0
        
        # Детекция потирания глаз (передаем все точки, а не только первого лица)
        try:
            last_rubbing = detect_rubbing(faces_kps, poses_kps)
        except Exception as e:
            print(f"Ошибка в detect_rubbing: {e}")
            last_rubbing = False
    else:
        # Нет лиц - нет метрик
        last_head_tilt = None
        last_blink = 0
        last_rubbing = False

    # Сбор метрик текущего кадра
    metrics_series = metrics_collector.collect_frame_metrics(
        frame_num=num_frame,
        head_tilt_result=last_head_tilt,
        blink_result=last_blink,
        rubbing_result=last_rubbing,
        face_kps=faces_kps if is_valid_kps(faces_kps) else None,
        pose_kps=poses_kps if is_valid_kps(poses_kps) else None
    )

    # ============== ВИЗУАЛИЗАЦИЯ МЕТРИК ==============
    # Получаем текущие метрики для отображения
    current_metrics = metrics_collector.get_current_metrics()
    
    # Отрисовываем метрики на кадре
    img = metrics_visualizer.draw_metrics(img, current_metrics)
    
    # Каждые 100 кадров отображаем сводку
    if num_frame % 100 == 0:
        summary = metrics_collector.get_summary()
        img = metrics_visualizer.draw_summary(img, summary, position=(400, 30))

    cv2.imshow("Detection", img)

    # Обработка клавиш
    key = cv2.waitKey(1)
    if key & 0xFF in (ord('q'), 27):  # 'q' или ESC для выхода
        break


# ============== ЗАВЕРШЕНИЕ РАБОТЫ ==============
# Сохраняем все данные перед выходом
metrics_collector.save_current_data()

# Генерируем финальный отчет
try:
    analyzer = MetricsAnalyzer(metrics_collector.metrics_history)
    report = analyzer.generate_report("final_report.json")
    print("Финальный отчет сгенерирован")
    
    # Вывод сводки в консоль
    summary = metrics_collector.get_summary()
    print("\n=== ФИНАЛЬНАЯ СВОДКА ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
        
except Exception as e:
    print(f"Ошибка при генерации отчета: {e}")

cap.release()
cv2.destroyAllWindows()