"""
Web Demo для Fatigue Detector на Gradio.
Запускается локально или деплоится на Hugging Face Spaces.
"""
import os
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
import logging

# Настройка логов
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# === Импорт ваших модулей ===
# Если структура папок отличается — поправьте пути
from config import MetricConfig, face_path, pose_path
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from pipeline import FatiguePipeline
from model.fatigue_classifier import FatigueClassifier
from model.config import ModelConfig as ClassifierConfig

# === Глобальные переменные (инициализируются один раз) ===
detectors_initialized = False
face_det = None
pose_det = None
pipeline = None
classifier = None
clf_cfg = None
app_cfg = None

def init_detectors():
    """Ленивая инициализация детекторов (экономит память при старте)."""
    global detectors_initialized, face_det, pose_det, pipeline, classifier, clf_cfg, app_cfg
    
    if detectors_initialized:
        return True
    
    try:
        logger.info("🔄 Инициализация детекторов...")
        app_cfg = MetricConfig()
        clf_cfg = ClassifierConfig()
        
        face_det = FaceDetector(face_path, num_faces=1)
        pose_det = PoseDetector(pose_path, num_poses=1)
        pipeline = FatiguePipeline(app_cfg)
        
        classifier = FatigueClassifier(clf_cfg)
        classifier.load(str(clf_cfg.default_model_path))
        
        detectors_initialized = True
        logger.info("✅ Детекторы готовы")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        return False

def draw_overlay(frame: np.ndarray, metrics: dict, fatigue_level: int, fatigue_conf: float) -> np.ndarray:
    """Рисует метрики и индикатор усталости поверх кадра."""
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # === Панель метрик (левый верхний угол) ===
    y_offset = 30
    lines = [
        f"Blink: {'YES' if metrics.get('blink') else 'OK'}",
        f"Yawn: {'YES' if metrics.get('yawn') else 'OK'}",
        f"Rubbing: {'YES' if metrics.get('rubbing') else 'OK'}",
        f"Tilt: {['Normal', 'Light', 'Heavy'][metrics.get('tilt_state', 0)]}",
        f"PERCLOS: {metrics.get('perclos', 0):.1f}%"
    ]
    
    for line in lines:
        cv2.putText(frame, line, (15, y_offset), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 25
    
    # === Индикатор усталости (правый нижний угол) ===
    colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
    labels = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL"}
    
    color = colors.get(fatigue_level, (100, 100, 100))
    label = labels.get(fatigue_level, "UNKNOWN")
    
    # Фон
    cv2.rectangle(frame, (w - 220, h - 90), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (w - 220, h - 90), (w - 10, h - 10), color, 2)
    
    # Текст
    cv2.putText(frame, f"FATIGUE: {label}", (w - 210, h - 60), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {fatigue_conf*100:.1f}%", (w - 210, h - 35), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Прогресс-бар
    bar_w, bar_h = 180, 12
    bar_x, bar_y = w - 205, h - 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * (fatigue_level + 1) / 3)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    
    return frame

def process_frame(frame, history: list):
    """
    Обрабатывает один кадр: детекция → метрики → классификация → визуализация.
    history: список предыдущих метрик для оконного анализа.
    """
    if frame is None:
        return None, history, "Waiting for camera..."
    
    # Инициализация при первом кадре
    if not init_detectors():
        return frame, history, "Error loading models"
    
    try:
        # 1. Детекция ключевых точек
        face_kp = face_det.get_key_pointers(frame)
        pose_kp = pose_det.get_key_pointers(frame)
        
        # 2. Вычисление метрик
        metrics = pipeline.step(face_kp, pose_kp)
        
        # 3. Обновление истории для классификации
        history.append(metrics)
        if len(history) > 30:  # Окно ~1 секунда при 30 FPS
            history.pop(0)
        
        # 4. Классификация усталости (если накоплено достаточно данных)
        fatigue_level = 0
        fatigue_conf = 0.0
        
        if len(history) >= 20 and classifier.is_ready:
            try:
                # Простая агрегация для демо (в продакшене — полноценный DataFrame)
                recent = history[-20:]
                # Здесь можно вызвать classifier.predict_from_window(), но для демо упростим:
                blink_count = sum(m.get('blink', 0) for m in recent)
                tilt_heavy = sum(1 for m in recent if m.get('tilt_state') == 2)
                
                # Эвристика для демо (замените на реальный предикт при желании)
                if blink_count > 8 or tilt_heavy > 3:
                    fatigue_level = 2
                    fatigue_conf = 0.85
                elif blink_count > 4 or tilt_heavy > 1:
                    fatigue_level = 1
                    fatigue_conf = 0.70
                else:
                    fatigue_level = 0
                    fatigue_conf = 0.90
            except Exception as e:
                logger.debug(f"Classification skipped: {e}")
        
        # 5. Визуализация
        annotated = draw_overlay(frame.copy(), metrics, fatigue_level, fatigue_conf)
        
        # 6. Формирование текста метрик для отображения
        status_text = (
            f"Frame processed | "
            f"Blink:{metrics.get('blink',0)} | "
            f"Yawn:{metrics.get('yawn',0)} | "
            f"Rub:{metrics.get('rubbing',False)} | "
            f"Tilt:{metrics.get('tilt_state',0)}"
        )
        
        # Конвертация BGR → RGB для Gradio
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), history, status_text
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return frame, history, f"Error: {str(e)[:50]}"

def reset_session():
    """Сбрасывает историю метрик."""
    return []

# === Интерфейс Gradio ===
with gr.Blocks(title="Fatigue Detector Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 👁️ Fatigue Detection System")
    gr.Markdown("""
    Real-time analysis of fatigue indicators via webcam:
    - 👁️ Blinking & PERCLOS
    - 😮 Yawning detection  
    - 🤚 Eye rubbing
    - 📐 Head tilt & posture
    - 🧠 Fatigue classification
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            input_video = gr.Webcam(label="📹 Webcam Input", format="numpy", mirror_webcam=True)
            output_video = gr.Image(label="🔍 Annotated Output", format="numpy")
        
        with gr.Column(scale=1):
            status_text = gr.Textbox(label="📊 Live Metrics", value="Waiting for camera...", interactive=False)
            reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
            gr.Markdown("""
            ### ℹ️ Info
            - Works best with good lighting
            - Keep your face visible in frame
            - Processing runs locally in browser
            """)
    
    # Состояние: история метрик
    history_state = gr.State(value=[])
    
    # Обработчики
    input_video.stream(
        fn=process_frame,
        inputs=[input_video, history_state],
        outputs=[output_video, history_state, status_text],
        time_limit=5,  # Таймаут для HF Spaces
        show_progress="hidden"
    )
    
    reset_btn.click(fn=reset_session, inputs=[], outputs=[history_state])

# === Запуск ===
if __name__ == "__main__":
    # Для локального запуска
    demo.queue(max_size=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )