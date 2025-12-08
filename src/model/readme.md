Ключевые признаки для классификации усталости:
1. Основные признаки:
blink_count - количество морганий

blink_frequency - частота морганий

head_tilt_max_angle - максимальный угол наклона

head_tilt_heavy_percentage - % времени с сильным наклоном

rubbing_count - количество потираний

rubbing_percentage - % времени потирания глаз

2. Статистические признаки:
Интервалы между морганиями

Длительность потираний

Стандартное отклонение угла наклона

3. Комбинированные признаки:
Общий скор активности

Композитные признаки усталости

Временные паттерны

4. Контекстуальные признаки:
Время суток (синус/косинус)

Продолжительность сессии

пример данных датасета:
        'blink_count': 5,
        'blink_frequency': 4.5,
        'blink_max_interval': 15,
        'blink_avg_interval': 8.2,
        'head_tilt_max_angle': 25.3,
        'head_tilt_avg_angle': 12.1,
        'head_tilt_std_angle': 8.5,
        'head_tilt_light_percentage': 15.2,
        'head_tilt_heavy_percentage': 5.7,
        'head_tilt_total_events': 3,
        'rubbing_count': 2,
        'rubbing_duration_avg': 2.1,
        'rubbing_percentage': 6.3,
        'face_detected_percentage': 98.5,
        'pose_detected_percentage': 95.2,
        'frames_without_face': 1,
        'activity_score': 0.45,
        'fatigue_score_1': 4.2,
        'fatigue_score_2': 2.8,
        'fatigue_score_3': 1.9,
        'time_sin': -0.258,
        'time_cos': 0.966