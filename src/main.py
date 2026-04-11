"""
Главный пайплайн детекции усталости.
Production-ready версия с модульной архитектурой.

Запуск:
    python main.py

Горячие клавиши:
    q / ESC — выход
    r — сброс счётчиков и буферов
    d — отладочная информация в консоль
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# ===== Локальные импорты проекта =====
from config import MetricConfig, pose_path, face_path
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from pipeline import FatiguePipeline
from collector.collector import MetricsCollector
from collector.analyzer import MetricsAnalyzer
from collector.visualizer import MetricsVisualizer
from model.fatigue_classifier import FatigueClassifier
from model.model_config import ModelConfig as ClassifierConfig

# ===== Настройка логирования =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ===== Вспомогательные функции визуализации =====
def draw_fatigue_indicator(
    img: np.ndarray,
    level: int,
    confidence: float,
    pos: tuple = (10, 200)
) -> np.ndarray:
    """Рисует индикатор уровня усталости на кадре."""
    x, y = pos
    colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
    labels = {0: "Norm", 1: "Medium", 2: "Strong"}
    
    color = colors.get(level, (100, 100, 100))
    text = labels.get(level, "Unknown")
    
    # Фон панели
    cv2.rectangle(img, (x, y), (x + 250, y + 100), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + 250, y + 100), color, 2)
    
    # Заголовки
    cv2.putText(img, "Fatigue:", (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img, text, (x + 10, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, f"Confidence: {confidence*100:.1f}%", (x + 10, y + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Прогресс-бар
    bar_w, bar_h = 200, 15
    bar_x, bar_y = x + 25, y + 120
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * (level + 1) / 3)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    
    # Подписи шкалы
    for i, label in enumerate(["Norm", "Med", "Strong"]):
        lx = bar_x + int(bar_w * (i + 1) / 3)
        cv2.line(img, (lx, bar_y - 5), (lx, bar_y + bar_h + 5), (100, 100, 100), 1)
        cv2.putText(img, label, (lx - 15, bar_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return img


def load_classifier(model_path: str, cfg: ClassifierConfig) -> Optional[FatigueClassifier]:
    """Загружает классификатор с откатом на заглушку при ошибке."""
    classifier = FatigueClassifier(cfg)
    try:
        if classifier.load(model_path):
            logger.info(f"Модель усталости загружена: {model_path}")
            return classifier
        else:
            logger.warning("Модель не загружена (load() вернул False)")
            return None
    except Exception as e:
        logger.warning(f"Ошибка загрузки модели: {e}. Классификация будет отключена.")
        return None


# ===== Главный пайплайн =====
def main():
    # ----- Конфигурация -----
    app_cfg = MetricConfig.from_yaml("metrics_config.yaml")  # или MetricConfig() для дефолта
    clf_cfg = ClassifierConfig()
    
    # ----- Инициализация детекторов -----
    logger.info("Инициализация детекторов...")
    face_detector = FaceDetector(face_path, num_faces=1)
    pose_detector = PoseDetector(pose_path, num_poses=1)
    
    # ----- Пайплайн метрик -----
    pipeline = FatiguePipeline(app_cfg)
    
    # ----- Сбор и анализ данных -----
    collector = MetricsCollector(fps=app_cfg.fps_target)
    analyzer = MetricsAnalyzer(fps=app_cfg.fps_target)
    visualizer = MetricsVisualizer()
    
    # ----- Классификатор усталости -----
    classifier = load_classifier(str(clf_cfg.default_model_path), clf_cfg)
    model_ready = classifier is not None and classifier.is_ready
    
    if not model_ready:
        logger.warning("⚠ Классификация усталости отключена (модель не загружена)")
    
    # ----- Состояние для классификации -----
    WINDOW_SIZE = 30
    fatigue_buffer: list[Dict[str, Any]] = []
    current_level, current_conf = 0, 0.0
    
    # ----- Захват видео -----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ Не удалось открыть камеру. Проверьте индекс устройства (cv2.VideoCapture(0)).")
        return
    
    # Попытка установить разрешение (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_idx = 0
    logger.info("✅ Система запущена. Нажмите 'q' или ESC для выхода.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Пропущен кадр от камеры")
                continue
            
            frame_idx += 1
            
            # ===== 1. Детекция ключевых точек =====
            face_kp = face_detector.get_key_pointers(frame)
            pose_kp = pose_detector.get_key_pointers(frame)
            
            # ===== 2. Вычисление метрик (единый интерфейс) =====
            metrics = pipeline.step(face_kp, pose_kp)
            # metrics = {
            #   'blink': 0|1,
            #   'rubbing': True|False,
            #   'head_tilt': (angle: float|None, state: 0|1|2)
            # }
            
            # ===== 3. Сбор данных (лёгкий, без pandas в цикле) =====
            tilt_angle = metrics["head_tilt"][0] if metrics["head_tilt"] else None
            tilt_state = metrics["head_tilt"][1] if metrics["head_tilt"] else 0
            
            collector.add_frame(
                frame_idx=frame_idx,
                blink=metrics["blink"],
                rubbing=metrics["rubbing"],
                head_tilt_state=tilt_state,
                head_tilt_angle=tilt_angle,
                face_kps=face_kp,
                pose_kps=pose_kp
            )
            
            # ===== 4. Классификация усталости (периодически) =====
            if model_ready and frame_idx % 10 == 0 and len(collector._history) >= WINDOW_SIZE:
                df_window = collector.to_dataframe().tail(WINDOW_SIZE)
                try:
                    pred, probs = classifier.predict_from_window(df_window)
                    current_level = int(pred[0])
                    current_conf = float(np.max(probs[0]))
                    
                    fatigue_buffer.append({
                        "frame": frame_idx,
                        "level": current_level,
                        "confidence": current_conf
                    })
                    if len(fatigue_buffer) > 100:
                        fatigue_buffer.pop(0)
                except Exception as e:
                    logger.debug(f"⚠ Ошибка классификации: {e}")
            
            # ===== 5. Визуализация =====
            current_metrics = {
                "frame": frame_idx,
                "blink_total": collector.total_blinks,
                "rubbing_current": metrics["rubbing"],
                "head_tilt_state": tilt_state,
                "head_tilt_total": collector.total_tilt_light + collector.total_tilt_heavy,
            }
            # Сводка обновляется реже, чтобы не перегружать UI
            summary = collector.get_summary() if frame_idx % 100 == 0 else None
            
            display_frame = visualizer.draw(frame.copy(), current_metrics, summary)
            
            # Индикатор усталости (если модель активна)
            if model_ready:
                display_frame = draw_fatigue_indicator(
                    display_frame, current_level, current_conf, pos=(10, 200)
                )
            
            # Номер кадра
            cv2.putText(display_frame, f"Frame: {frame_idx}",
                       (display_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Fatigue Detection", display_frame)
            
            # ===== 6. Обработка клавиш =====
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q или ESC
                logger.info("👋 Получен сигнал выхода")
                break
            elif key == ord('r'):  # сброс
                collector.reset()
                pipeline.reset()
                fatigue_buffer.clear()
                current_level, current_conf = 0, 0.0
                logger.info("🔄 Счётчики и буферы сброшены")
            elif key == ord('d'):  # отладка
                logger.info(
                    f"🔍 Frame {frame_idx}: "
                    f"blink={metrics['blink']}, "
                    f"rubbing={metrics['rubbing']}, "
                    f"tilt={tilt_state}, "
                    f"faces={len(face_kp) if face_kp is not None else 0}, "
                    f"poses={len(pose_kp) if pose_kp is not None else 0}"
                )
    
    except KeyboardInterrupt:
        logger.info("⚡ Прервано пользователем (Ctrl+C)")
    
    finally:
        # ===== Завершение и сохранение =====
        logger.info("💾 Завершение работы, сохранение данных...")
        
        # Освобождение ресурсов детекторов (важно для MediaPipe!)
        face_detector.close()
        pose_detector.close()
        
        # Сохранение собранных данных
        output_dir = Path("output") / f"session_{frame_idx}_{Path(face_path).stem}"
        collector.save(str(output_dir))
        logger.info(f"📁 Данные сохранены в {output_dir}")
        
        # Пост-анализ (если есть данные)
        df = collector.to_dataframe()
        if not df.empty:
            try:
                report = analyzer.analyze(df)
                logger.info(f"📊 Анализ завершён: {report}")
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе: {e}")
        
        # История предсказаний усталости
        if fatigue_buffer and model_ready:
            import pandas as pd
            fatigue_path = output_dir / "fatigue_history.csv"
            pd.DataFrame(fatigue_buffer).to_csv(fatigue_path, index=False)
            logger.info(f"📈 История усталости: {fatigue_path}")
        
        # Очистка
        cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Готово. Система остановлена.")


# ===== Точка входа =====
if __name__ == "__main__":
    main()