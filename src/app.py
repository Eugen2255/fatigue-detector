import sys
import cv2
import logging
from pathlib import Path
from typing import Optional
import os
import warnings

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QProgressBar, QWidget, QGroupBox
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

# Ваши существующие модули
from config import MetricConfig, pose_path, face_path
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from pipeline import FatiguePipeline
from collector.collector import MetricsCollector
from model.fatigue_classifier import FatigueClassifier
from model.model_config import ModelConfig as ClassifierConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CVWorker(QThread):
    """Фоновый поток для захвата видео, детекции и метрик."""
    frame_ready = Signal(QImage)
    metrics_ready = Signal(dict)
    log_message = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self._running = False
        self._cap = None
        self._frame_idx = 0
        
        # Инициализация
        self.app_cfg = MetricConfig()
        self.clf_cfg = ClassifierConfig()
        self.face_det = FaceDetector(face_path, num_faces=1)
        self.pose_det = PoseDetector(pose_path, num_poses=1)
        self.pipeline = FatiguePipeline(self.app_cfg)
        self.collector = MetricsCollector(fps=self.app_cfg.fps_target)
        
        self.classifier = FatigueClassifier(self.clf_cfg)
        self.classifier.load(str(self.clf_cfg.default_model_path))
        self.model_ready = self.classifier.is_ready
        
        self.WINDOW_SIZE = 30
        self.current_level = 0
        self.current_conf = 0.0

    def run(self):
        self._running = True
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.error_occurred.emit("Не удалось открыть камеру")
            return

        self.log_message.emit("Камера подключена. Обработка запущена.")
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            self._frame_idx += 1
            
            # 1. Детекция
            face_kp = self.face_det.get_key_pointers(frame)
            pose_kp = self.pose_det.get_key_pointers(frame)
            
            # 2. Метрики
            metrics = self.pipeline.step(face_kp, pose_kp)
            
            # 3. Сбор данных
            tilt = metrics.get("head_tilt")
            tilt_angle = tilt[0] if tilt else None
            tilt_state = tilt[1] if tilt else 0
            
            self.collector.add_frame(
                frame_idx=self._frame_idx,
                blink=metrics.get("blink", 0),
                rubbing=metrics.get("rubbing", False),
                head_tilt_state=tilt_state,
                head_tilt_angle=tilt_angle,
                face_kps=face_kp, pose_kps=pose_kp
            )
            
            # 4. Классификация (каждые 10 кадров)
            if self.model_ready and self._frame_idx % 10 == 0 and len(self.collector._history) >= self.WINDOW_SIZE:
                try:
                    df = self.collector.to_dataframe().tail(self.WINDOW_SIZE)
                    pred, probs = self.classifier.predict_from_window(df)
                    self.current_level = int(pred[0])
                    self.current_conf = float(probs[0].max())
                except Exception as e:
                    self.log_message.emit(f"Ошибка классификации: {e}")

            # 5. Эммит данных в UI
            h, w, ch = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.frame_ready.emit(q_img)
            
            self.metrics_ready.emit({
                "frame": self._frame_idx,
                "blink": metrics.get("blink", 0),
                "rubbing": metrics.get("rubbing", False),
                "tilt_state": tilt_state,
                "fatigue_level": self.current_level,
                "fatigue_conf": self.current_conf
            })

    def stop(self):
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self.wait()
        self.log_message.emit("Оптика и потоки остановлены.")


class FatigueApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Detection System")
        self.resize(960, 640)
        self.worker = None
        
        self._setup_ui()
        self._setup_worker()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # === Левая часть: Видео ===
        vid_group = QGroupBox("Live Feed")
        vid_layout = QVBoxLayout(vid_group)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #222; color: #888;")
        vid_layout.addWidget(self.video_label)
        
        # === Правая часть: Панель метрик ===
        panel = QWidget()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout(panel)
        
        # Кнопки
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        panel_layout.addLayout(btn_layout)
        
        # Метрики
        self.lbl_frame = QLabel("Frame: 0")
        self.lbl_blink = QLabel("Blinks: 0")
        self.lbl_rubbing = QLabel("Rubbing: No")
        self.lbl_tilt = QLabel("Tilt: Normal")
        for lbl in [self.lbl_frame, self.lbl_blink, self.lbl_rubbing, self.lbl_tilt]:
            lbl.setFont(QFont("Consolas", 10))
            panel_layout.addWidget(lbl)
            
        # Индикатор усталости
        panel_layout.addSpacing(15)
        panel_layout.addWidget(QLabel("Fatigue Level:"))
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 2)
        self.fatigue_bar.setTextVisible(True)
        self.fatigue_bar.setFormat("Level: %v")
        panel_layout.addWidget(self.fatigue_bar)
        
        self.lbl_conf = QLabel("Confidence: 0.0%")
        panel_layout.addWidget(self.lbl_conf)
        
        # Лог
        panel_layout.addSpacing(15)
        panel_layout.addWidget(QLabel("System Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        self.log_box.setStyleSheet("background: #111; color: #0f0; font-family: monospace;")
        panel_layout.addWidget(self.log_box)
        
        main_layout.addWidget(vid_group, 3)
        main_layout.addWidget(panel, 2)
        
        # Связи кнопок
        self.btn_start.clicked.connect(self._start_processing)
        self.btn_stop.clicked.connect(self._stop_processing)

    def _setup_worker(self):
        self.worker = CVWorker()
        self.worker.frame_ready.connect(self._update_video)
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.log_message.connect(self._log)
        self.worker.error_occurred.connect(lambda msg: self._log(f"⚠️ {msg}", True))

    def _start_processing(self):
        if not self.worker.isRunning():
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.worker.start()
            
    def _stop_processing(self):
        self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
    def _update_video(self, q_img: QImage):
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
    def _update_metrics(self, data: dict):
        self.lbl_frame.setText(f"Frame: {data['frame']}")
        self.lbl_blink.setText(f"Blinks: {data['blink']}")
        self.lbl_rubbing.setText(f"Rubbing: {'⚠️ YES' if data['rubbing'] else 'OK'}")
        tilt_txt = {0: "Normal", 1: "Light", 2: "⚠️ Heavy"}
        self.lbl_tilt.setText(f"Tilt: {tilt_txt.get(data['tilt_state'], 'N/A')}")
        
        self.fatigue_bar.setValue(data['fatigue_level'])
        self.lbl_conf.setText(f"Confidence: {data['fatigue_conf']*100:.1f}%")
        
        # Цвет прогресс-бара в зависимости от уровня
        colors = {0: "#00ff00", 1: "#ffcc00", 2: "#ff0000"}
        self.fatigue_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colors.get(data['fatigue_level'], '#888')}; }}")

    def _log(self, msg: str, error: bool = False):
        self.log_box.append(f"<span style='color:{'#f00' if error else '#0f0'}'>{msg}</span>")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self._stop_processing()
        self.face_det.close() if hasattr(self, 'face_det') else None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FatigueApp()
    window.show()
    sys.exit(app.exec())