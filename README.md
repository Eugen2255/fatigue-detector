

# Fatigue Detector

Fatigue Detector — это интеллектуальная система для определения уровня усталости пользователя. Проект может использоваться в задачах мониторинга состояния операторов, водителей, студентов или сотрудников с целью повышения безопасности и продуктивности.

---

## 🔍 Возможности

- Анализ видеопотока в реальном времени
- Оценка усталости на основе поведенческих или физиологических признаков (например, на основе моргания, положения головы и т.д.)
---

## 📂 Структура проекта
```
📂 fatigue-detector/
├── README.md
├── requirements.txt
├── .gitignore
├── 📂 src/
│   ├── config.py
│   ├── exp_smoothing.py
│   ├── main_video.py           # Видео обработка (тестирование)
│   ├── main_image.py           # Изображение обработка (тестирование)
│   ├── 📂 detectors/           # Детекторы
│   │   ├── face_detector.py
│   │   ├── key_pointers.py
│   │   ├── pose_detector.py
│   │   └── __init__.py
│   ├── 📂 metrics/           # Метрики усталости
│   │   ├── blinking.py
│   │   ├── eyes_rub.py
│   │   ├── head_tilt.py
│   │   └── __init__.py
│   ├── 📂 collector/         # Сбор и анализ данных
│   │   ├── analyzer.py
│   │   ├── collector.py
│   │   ├── visualizer.py
│   │   └── __init__.py
│   ├── 📂 model/             # Модели классификации
│   │   ├── fatigue_classifier.py
│   │   └── readme.md
│   └── 📂 dataset_builder/   # Сбор датасета
│       ├── dataset_builder.py
│       └── __init__.py
├── 📂 docs/                  # Документация
│   └── main_idea.md
└── 📂 images/               # Тестовые изображения
    └── img1.jpg
```
---

## 🛠 Требования

- Python 3.11
- NumPy, Pandas
- Дополнительные зависимости — см. requirements.txt

---

## 📦 Установка

1. Клонируйте репозиторий:
```
   git clone https://github.com/Eugen2255/fatigue-detector.git
   cd fatigue-detector
```
2. Установите зависимости:
```
   pip install -r requirements.txt
```
3. Загрузите предобученные модели, если они не включены в репозиторий.
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker?hl=ru#models
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ru#models
---

## ▶️ Запуск
```
python main.py
```
---

