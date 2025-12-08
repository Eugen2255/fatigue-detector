import cv2
import numpy as np
import pandas as pd

from detectors import *
from exp_smoothing import exp_smoothing
from metrics import *
from config import pose_path, face_path
from collector import *
from model.fatigue_classifier import FatigueClassifier 

''' ФАЙЛ ДЛЯ ТЕСТОВ НА ВЕБ-КАМЕРЕ '''

# пути к модели и изображению
pose_model_path = pose_path
face_model_path = face_path

cap = cv2.VideoCapture(0)

# Инициализация детекторов
face = FaceDetector(face_model_path, num_faces=1)
pose = PoseDetector(pose_model_path, num_poses=1)

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

# ============== ИНИЦИАЛИЗАЦИЯ КЛАССИФИКАТОРА УСТАЛОСТИ ==============
fatigue_classifier = FatigueClassifier()
try:
    fatigue_classifier.load_model(r"C:\Users\Eugen\fatigue-detector\models\fatigue_classifier.pkl")
    print("Модель классификации усталости загружена успешно!")
    model_loaded = True

except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    print("Создаем и обучаем новую модель...")
    # Создаем и обучаем простую модель на синтетических данных
    from model.fatigue_classifier import train_simple_classifier
    fatigue_classifier, _ = train_simple_classifier()
    model_loaded = True

# Переменные для хранения последних результатов детекции
last_head_tilt = None
last_blink = 0
last_rubbing = False

# Окно для классификации усталости (30 кадров = 1 секунда при 30 FPS)
WINDOW_SIZE = 30
fatigue_history = []  # История предсказаний усталости
current_fatigue_level = 0  # Текущий уровень усталости
fatigue_confidence = 0.0  # Уверенность в предсказании

def is_valid_kps(kps):
    """Проверяет, валидны ли ключевые точки"""
    return kps is not None and len(kps) > 0

def draw_fatigue_indicator(img, fatigue_level, confidence, x=10, y=200):
    """
    Рисует индикатор усталости на кадре
    """
    # Определяем цвет и текст в зависимости от уровня усталости
    if fatigue_level == 0:
        color = (0, 255, 0)  # Зеленый
        text = "Norm"
   
    elif fatigue_level == 1:
        color = (0, 255, 255)  # Желтый
        text = "Medium"

    else:
        color = (0, 0, 255)  # Красный
        text = "Strong"
        
    
    # Фон для индикатора
    cv2.rectangle(img, (x, y), (x + 250, y + 100), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + 250, y + 100), color, 2)
    
    # Текст усталости
    cv2.putText(img, "Fatigue:", (x + 10, y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Уровень усталости
    cv2.putText(img, text, (x + 10, y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Уверенность
    confidence_text = f"Confidience: {confidence*100:.1f}%"
    cv2.putText(img, confidence_text, (x + 10, y + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Индикатор уровня
    bar_width = 200
    bar_height = 15
    bar_x = x + 25
    bar_y = y + 120
    
    # Фон индикатора
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (60, 60, 60), -1)
    
    # Заполненная часть
    fill_width = int(bar_width * (fatigue_level + 1) / 3)  # 0->1/3, 1->2/3, 2->3/3
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                  color, -1)
    
    # Деления
    for i in range(1, 3):
        line_x = bar_x + int(bar_width * i / 3)
        cv2.line(img, (line_x, bar_y - 5), (line_x, bar_y + bar_height + 5), 
                 (100, 100, 100), 1)
    
    # Подписи
    cv2.putText(img, "Norma", (bar_x - 10, bar_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "Srednya", (bar_x + bar_width//3 - 15, bar_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "Strong", (bar_x + 2*bar_width//3 - 15, bar_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return img

def update_fatigue_prediction():
    """
    Обновляет предсказание усталости на основе собранных метрик
    """
    global current_fatigue_level, fatigue_confidence
    
    if not model_loaded or len(metrics_collector.metrics_history) < WINDOW_SIZE:
        return
    
    # Получаем последние WINDOW_SIZE кадров метрик
    recent_metrics = pd.DataFrame(metrics_collector.metrics_history[-WINDOW_SIZE:])
    
    try:
        # Предсказание усталости
        prediction, probabilities = fatigue_classifier.predict_from_window_metrics(recent_metrics)
        
        # Обновляем текущие значения
        current_fatigue_level = prediction[0]
        fatigue_confidence = np.max(probabilities[0])
        
        # Сохраняем в историю
        fatigue_history.append({
            'frame': num_frame,
            'level': current_fatigue_level,
            'confidence': fatigue_confidence,
            'probabilities': probabilities[0].tolist()
        })
        
        # Ограничиваем размер истории
        if len(fatigue_history) > 100:
            fatigue_history.pop(0)
            
    except Exception as e:
        print(f"Ошибка при предсказании усталости: {e}")

print("=" * 60)
print("СИСТЕМА ДЕТЕКЦИИ УСТАЛОСТИ")
print("=" * 60)
print("Используемые модели:")
print(f"1. Детектор лиц: {face_model_path}")
print(f"2. Детектор поз: {pose_model_path}")
print(f"3. Классификатор усталости: {'Загружен' if model_loaded else 'Не загружен'}")
print("\nГорячие клавиши:")
print("  q / ESC - Выход")
print("  r - Сбросить счетчики")
print("  d - Показать отладочную информацию")
print("=" * 60)

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
    
    # Детекция метрик
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
            last_blink = detect_blink(face_kps_for_metrics)
        except Exception as e:
            print(f"Ошибка в detect_blink: {e}")
            last_blink = 0
        
        # Детекция потирания глаз 
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

    # ============== КЛАССИФИКАЦИЯ УСТАЛОСТИ ==============
    # Обновляем предсказание каждые 10 кадров
    if num_frame % 10 == 0 and model_loaded:
        update_fatigue_prediction()

    # ============== ВИЗУАЛИЗАЦИЯ МЕТРИК ==============
    # Получаем текущие метрики для отображения
    current_metrics = metrics_collector.get_current_metrics()
    
    # Отрисовываем метрики на кадре
    img = metrics_visualizer.draw_metrics(img, current_metrics)
    
    # Каждые 100 кадров отображаем сводку
    if num_frame % 100 == 0:
        summary = metrics_collector.get_summary()
        img = metrics_visualizer.draw_summary(img, summary, position=(400, 30))
    
    # ============== ВИЗУАЛИЗАЦИЯ УСТАЛОСТИ ==============
    if model_loaded:
        # Рисуем индикатор усталости
        img = draw_fatigue_indicator(img, current_fatigue_level, fatigue_confidence, x=10, y=200)
        
        # Отображаем историю предсказаний (последние 5)
        if len(fatigue_history) > 0:
            history_text = "History: "
            recent_history = fatigue_history[-5:]
            for item in recent_history:
                history_text += f"{item['level']} "
            cv2.putText(img, history_text, (10, img.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # ============== ОТОБРАЖЕНИЕ ДОПОЛНИТЕЛЬНОЙ ИНФОРМАЦИИ ==============
    # Отображение номера кадра
    cv2.putText(img, f"Frame: {num_frame}", (img.shape[1] - 100, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    

    cv2.imshow("Система детекции усталости", img)

    # Обработка клавиш
    key = cv2.waitKey(1)
    if key & 0xFF in (ord('q'), 27):  # 'q' или ESC для выхода
        break
    elif key == ord('d'):  # 'd' для отладки - вывод текущих метрик
        print(f"\n=== Отладочная информация (кадр {num_frame}) ===")
        print(f"Лиц обнаружено: {len(faces_kps) if is_valid_kps(faces_kps) else 0}")
        print(f"Поз обнаружено: {len(poses_kps) if is_valid_kps(poses_kps) else 0}")
        print(f"Head tilt: {last_head_tilt}")
        print(f"Blink: {last_blink}")
        print(f"Rubbing: {last_rubbing}")
        
        if current_metrics:
            print(f"Текущие метрики:")
            print(f"  Морганий всего: {current_metrics.get('blink_total', 0)}")
            print(f"  Потираний всего: {current_metrics.get('rubbing_total', 0)}")
            print(f"  Наклонов всего: {current_metrics.get('head_tilt_total', 0)}")
        
        if model_loaded and len(fatigue_history) > 0:
            print(f"Уровень усталости: {current_fatigue_level}")
            print(f"Уверенность: {fatigue_confidence:.2f}")
    
    elif key == ord('m'):  # 'm' для информации о модели
        if model_loaded:
            print(f"\n=== Информация о модели ===")
            print(f"Тип модели: {fatigue_classifier.model_type}")
            print(f"Количество признаков: {len(fatigue_classifier.feature_names)}")
            print(f"История усталости: {len(fatigue_history)} записей")
        else:
            print("Модель не загружена")

# ============== ЗАВЕРШЕНИЕ РАБОТЫ ==============
print("\nЗавершение работы...")

# Сохраняем все данные перед выходом
metrics_collector.save_current_data()

# Сохраняем историю усталости
if fatigue_history:
    fatigue_df = pd.DataFrame(fatigue_history)
    fatigue_df.to_csv("final_fatigue_history.csv", index=False)
    print(f"Финальная история усталости сохранена")

# Генерируем финальный отчет
try:
    analyzer = MetricsAnalyzer(metrics_collector.metrics_history)
    report = analyzer.generate_report("final_report.json")
    print("Финальный отчет сгенерирован")
    
    # Вывод сводки в консоль
    summary = metrics_collector.get_summary()
    print("\n=== ФИНАЛЬНАЯ СВОДКА МЕТРИК ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
        
    # Сводка по усталости
    if fatigue_history:
        fatigue_levels = [item['level'] for item in fatigue_history]
        if fatigue_levels:
            avg_fatigue = sum(fatigue_levels) / len(fatigue_levels)
            max_fatigue = max(fatigue_levels)
            fatigue_dist = {0: 0, 1: 0, 2: 0}
            for level in fatigue_levels:
                fatigue_dist[level] += 1
            
            print("\n=== СВОДКА ПО УСТАЛОСТИ ===")
            print(f"Средний уровень усталости: {avg_fatigue:.2f}")
            print(f"Максимальный уровень: {max_fatigue}")
            print(f"Распределение уровней:")
            print(f"  Норма: {fatigue_dist[0]} ({fatigue_dist[0]/len(fatigue_levels)*100:.1f}%)")
            print(f"  Средняя: {fatigue_dist[1]} ({fatigue_dist[1]/len(fatigue_levels)*100:.1f}%)")
            print(f"  Сильная: {fatigue_dist[2]} ({fatigue_dist[2]/len(fatigue_levels)*100:.1f}%)")
            
except Exception as e:
    print(f"Ошибка при генерации отчета: {e}")

cap.release()
cv2.destroyAllWindows()
