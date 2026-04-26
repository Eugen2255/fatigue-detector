import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QFont, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from collector.collector import MetricsCollector
from config import MetricConfig, face_path, pose_path
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from model.fatigue_classifier import FatigueClassifier
from model.model_config import ModelConfig as ClassifierConfig
from pipeline import FatiguePipeline
from utils.input_monitor import InputActivityMonitor
from utils.recommender import Recommender
from utils.state_tracker import FatigueStateTracker

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*landmark_projection_calculator.*")


def draw_landmarks(frame: np.ndarray, face_kp: np.ndarray | None, pose_kp: np.ndarray | None) -> None:
    if face_kp is not None:
        for pt in face_kp[0]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 200, 0), -1)
    if pose_kp is not None:
        for pt in pose_kp[0]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (150, 0, 255), -1)


class FrameGrabber:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self._cap.isOpened():
            return False
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        while self._running.is_set():
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            while True:
                try:
                    self.queue.put_nowait(frame)
                    break
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break

    def get_latest(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                latest = self.queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self._thread = None


class CVWorker(QThread):
    frame_ready = Signal(QImage)
    metrics_ready = Signal(dict)
    log_message = Signal(str)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._frame_idx = 0
        self._show_landmarks = False
        self._show_debug = False
        self._session_started = int(time.time())

        self.app_cfg = MetricConfig()
        self.clf_cfg = ClassifierConfig()
        self.face_det = FaceDetector(face_path, num_faces=1)
        self.pose_det = PoseDetector(pose_path, num_poses=1)
        self.pipeline = FatiguePipeline(self.app_cfg)
        db_path = Path("src") / "output" / f"session_{self._session_started}" / "metrics_live.sqlite3"
        self.collector = MetricsCollector(fps=self.app_cfg.fps_target, db_path=db_path)
        self.input_monitor = InputActivityMonitor()
        self.recommender = Recommender(cooldown_frames=300)
        self.state_tracker = FatigueStateTracker(decay_rate=0.985)
        self.grabber = FrameGrabber(camera_index=0)

        self.classifier = FatigueClassifier(self.clf_cfg)
        self.classifier.load(str(self.clf_cfg.default_model_path))

        self.window_size = 30
        self.current_level = 0
        self.current_conf = 0.0

    def set_show_landmarks(self, value: bool) -> None:
        self._show_landmarks = value

    def set_show_debug(self, value: bool) -> None:
        self._show_debug = value

    def reset_state(self) -> None:
        self.collector.reset()
        self.pipeline.reset()
        self.input_monitor.reset()
        self.state_tracker.fatigue_score = 0.0
        self.state_tracker.level = 0
        self.recommender.last_advice_frame = -self.recommender.cooldown
        self._frame_idx = 0
        self.current_level = 0
        self.current_conf = 0.0
        self.log_message.emit("Session state reset.")

    def run(self):
        self._running = True
        if not self.grabber.start():
            self.error_occurred.emit("Failed to open camera")
            return

        self.log_message.emit("Camera connected. Processing started.")
        while self._running:
            frame = self.grabber.get_latest()
            if frame is None:
                self.msleep(5)
                continue

            input_snapshot = self.input_monitor.sample()
            self._frame_idx += 1

            face_kp = self.face_det.get_key_pointers(frame)
            pose_kp = self.pose_det.get_key_pointers(frame)
            metrics = self.pipeline.step(face_kp, pose_kp)

            tilt_data = metrics.get("head_tilt")
            tilt_angle = tilt_data[0] if isinstance(tilt_data, tuple) else None
            tilt_state = tilt_data[1] if isinstance(tilt_data, tuple) else 0

            tracker_input = {
                "blink": metrics.get("blink", 0),
                "yawn": metrics.get("yawn", 0),
                "rubbing": metrics.get("rubbing", False),
                "tilt_state": tilt_state,
            }
            self.state_tracker.update(tracker_input)
            fatigue_level, fatigue_color, fatigue_text = self.state_tracker.get_status()

            self.collector.add_frame(
                frame_idx=self._frame_idx,
                blink=metrics.get("blink", 0),
                perclos=metrics.get("perclos", 0),
                rubbing=metrics.get("rubbing", False),
                head_tilt_state=tilt_state,
                head_tilt_angle=tilt_angle,
                yawn=metrics.get("yawn", 0),
                input_metrics=input_snapshot.to_dict(),
                face_kps=face_kp,
                pose_kps=pose_kp,
            )

            if self._frame_idx % 10 == 0 and self.collector.frame_count >= self.window_size:
                try:
                    df = self.collector.recent_window_dataframe(self.window_size)
                    pred, probs = self.classifier.predict_from_window(df)
                    self.current_level = int(pred[0])
                    self.current_conf = float(probs[0].max())
                    self.classifier.update_online_from_window(df, self.current_level, self.current_conf)
                except Exception as exc:
                    self.log_message.emit(f"Classification error: {exc}")

            input_summary = self.collector.recent_input_summary(self.window_size)
            advice = self.recommender.get_advice(fatigue_level, self._frame_idx)

            display_frame = frame.copy()
            if self._show_landmarks:
                draw_landmarks(display_frame, face_kp, pose_kp)

            self._draw_status(display_frame, fatigue_text, fatigue_color, advice)

            if self._show_debug:
                self._draw_debug_overlay(display_frame, tilt_state, input_summary)

            h, w, ch = display_frame.shape
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888).copy()
            self.frame_ready.emit(q_img)

            self.metrics_ready.emit(
                {
                    "frame": self._frame_idx,
                    "blink": metrics.get("blink", 0),
                    "yawn": metrics.get("yawn", 0),
                    "perclos": self.collector.recent_perclos_pct(),
                    "rubbing": metrics.get("rubbing", False),
                    "tilt_state": tilt_state,
                    "key_rate": input_summary["key_rate"],
                    "mouse_click_rate": input_summary["mouse_click_rate"],
                    "idle_sec": input_summary["idle_sec"],
                    "total_blinks": self.collector.total_blinks,
                    "total_yawns": self.collector.total_yawn,
                    "fatigue_level": self.current_level,
                    "fatigue_conf": self.current_conf,
                    "fatigue_text": fatigue_text,
                    "advice": advice,
                    "online_updates": self.classifier.online_learner.update_count,
                    "storage_path": str(self.collector.db_path),
                    "queue_depth": self.grabber.queue.qsize(),
                    "show_debug": self._show_debug,
                    "show_landmarks": self._show_landmarks,
                }
            )

        self._shutdown()

    def _draw_debug_overlay(self, frame, tilt_state, input_summary):
        lines = [
            f"Frame: {self._frame_idx}",
            f"Blinks total: {self.collector.total_blinks}",
            f"Yawns total: {self.collector.total_yawn}",
            f"PERCLOS: {self.collector.recent_perclos_pct():.1f}%",
            f"Keys/min: {input_summary['key_rate']:.1f}",
            f"Clicks/min: {input_summary['mouse_click_rate']:.1f}",
            f"Idle: {input_summary['idle_sec']:.1f}s",
            f"Tilt state: {tilt_state}",
        ]
        x, y = 20, 35
        overlay = frame.copy()
        box_h = 24 * len(lines) + 20
        cv2.rectangle(overlay, (10, 10), (355, 10 + box_h), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        for idx, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + idx * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (232, 232, 232), 1, cv2.LINE_AA)

    def _draw_status(self, frame, fatigue_text, fatigue_color, advice):
        h, w, _ = frame.shape
        status_size = cv2.getTextSize(fatigue_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - status_size[0]) // 2
        cv2.rectangle(frame, (text_x - 12, h - 68), (text_x + status_size[0] + 12, h - 18), (0, 0, 0), -1)
        cv2.putText(frame, fatigue_text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, fatigue_color, 2, cv2.LINE_AA)
        if advice:
            cv2.putText(frame, advice, (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)

    def stop(self):
        self._running = False
        self.wait()

    def _shutdown(self):
        self.classifier.online_learner.save()
        self.grabber.stop()
        session_dir = Path("src") / "output" / f"session_{self._session_started}"
        self.collector.save(str(session_dir))
        self.collector.close()
        self.face_det.close()
        self.pose_det.close()
        self.log_message.emit("Processing stopped.")


class FatigueApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fatigue Detector")
        self.resize(1360, 860)
        self.worker: CVWorker | None = None
        self._apply_style()
        self._setup_ui()
        self._setup_worker()
        self._setup_hotkeys()
        self._setup_menu()
        self.statusBar().showMessage("Ready")

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f141b;
                color: #edf3fb;
                font-family: Segoe UI;
                font-size: 13px;
            }
            QMenuBar {
                background: #0f141b;
                color: #d9e4f2;
                border-bottom: 1px solid #263342;
                padding: 4px;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background: #243447;
            }
            QMenu {
                background: #151d27;
                color: #edf3fb;
                border: 1px solid #2e3e50;
            }
            QStatusBar {
                background: #0c1117;
                color: #8fa2b8;
                border-top: 1px solid #243140;
            }
            QGroupBox {
                background: #151d27;
                border: 1px solid #2a3a4d;
                border-radius: 10px;
                margin-top: 20px;
                padding: 16px 12px 12px 12px;
                font-weight: 650;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #9fb4cc;
            }
            QPushButton {
                background: #223044;
                color: #f1f6ff;
                border: 1px solid #3a4f68;
                border-radius: 8px;
                padding: 9px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2b3c54;
                border-color: #4b6684;
            }
            QPushButton:pressed {
                background: #1b2839;
            }
            QPushButton:disabled {
                background: #171d25;
                color: #647287;
                border-color: #263241;
            }
            QLabel#VideoPreview {
                background: #090d12;
                color: #7f91a7;
                border: 1px solid #253449;
                border-radius: 12px;
            }
            QLabel#StatusLabel {
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#HelpStrip, QLabel#SessionLabel {
                color: #8fa2b8;
            }
            QTextEdit {
                background: #0b1118;
                color: #c6f6d5;
                border: 1px solid #253449;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas;
            }
            QProgressBar {
                background: #0c1219;
                border: 1px solid #2a3a4d;
                border-radius: 7px;
                height: 18px;
                text-align: center;
                color: #dce8f7;
            }
            QProgressBar::chunk {
                border-radius: 6px;
            }
            """
        )

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        video_group = QGroupBox("Live Feed")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("Camera preview")
        self.video_label.setObjectName("VideoPreview")
        self.video_label.setMinimumSize(860, 620)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        help_strip = QLabel("Hotkeys: S start/stop | R reset | D debug | L landmarks | Q quit")
        help_strip.setObjectName("HelpStrip")
        left.addWidget(video_group)
        left.addWidget(help_strip)

        right = QVBoxLayout()
        right.setSpacing(12)
        right.setContentsMargins(0, 0, 0, 0)

        controls_group = QGroupBox("Controls")
        controls_layout = QGridLayout(controls_group)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset")
        self.btn_debug = QPushButton("Show Debug (D)")
        self.btn_landmarks = QPushButton("Show Landmarks")
        self.btn_export = QPushButton("Export Session")
        self.btn_stop.setEnabled(False)
        controls = [
            self.btn_start,
            self.btn_stop,
            self.btn_reset,
            self.btn_debug,
            self.btn_landmarks,
            self.btn_export,
        ]
        for btn in controls:
            btn.setMinimumHeight(38)
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        for btn, pos in zip(controls, positions):
            controls_layout.addWidget(btn, *pos)

        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QGridLayout(metrics_group)
        self.metric_labels: dict[str, QLabel] = {}
        metric_names = [
            ("Frame", "frame"),
            ("Blink Event", "blink"),
            ("Yawn Event", "yawn"),
            ("PERCLOS", "perclos"),
            ("Rubbing", "rubbing"),
            ("Tilt", "tilt"),
            ("Keys/min", "keys"),
            ("Clicks/min", "clicks"),
            ("Idle", "idle"),
            ("Blinks Total", "total_blinks"),
            ("Yawns Total", "total_yawns"),
            ("Online Updates", "online_updates"),
            ("Queue", "queue"),
        ]
        for row, (title, key) in enumerate(metric_names):
            name_lbl = QLabel(title)
            val_lbl = QLabel("0")
            name_lbl.setStyleSheet("color:#8fa2b8;")
            val_lbl.setStyleSheet("color:#f3f8ff; font-weight:700;")
            name_lbl.setFont(QFont("Consolas", 10))
            val_lbl.setFont(QFont("Consolas", 10))
            metrics_layout.addWidget(name_lbl, row, 0)
            metrics_layout.addWidget(val_lbl, row, 1)
            self.metric_labels[key] = val_lbl

        fatigue_group = QGroupBox("Fatigue State")
        fatigue_layout = QVBoxLayout(fatigue_group)
        self.lbl_status = QLabel("NORMAL")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setStyleSheet("color:#43c97a;")
        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setRange(0, 2)
        self.fatigue_bar.setFormat("Model Level: %v")
        self.lbl_conf = QLabel("Confidence: 0.0%")
        self.lbl_advice = QLabel("Advice: waiting for more data")
        self.lbl_advice.setWordWrap(True)
        fatigue_layout.addWidget(self.lbl_status)
        fatigue_layout.addWidget(self.fatigue_bar)
        fatigue_layout.addWidget(self.lbl_conf)
        fatigue_layout.addWidget(self.lbl_advice)

        session_group = QGroupBox("Session")
        session_layout = QVBoxLayout(session_group)
        self.lbl_storage = QLabel("DB: not started")
        self.lbl_storage.setObjectName("SessionLabel")
        self.lbl_storage.setWordWrap(True)
        self.lbl_modes = QLabel("Debug: off | Landmarks: off")
        self.lbl_modes.setObjectName("SessionLabel")
        session_layout.addWidget(self.lbl_storage)
        session_layout.addWidget(self.lbl_modes)

        logs_group = QGroupBox("Log")
        logs_layout = QVBoxLayout(logs_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(220)
        logs_layout.addWidget(self.log_box)

        right.addWidget(controls_group)
        right.addWidget(metrics_group)
        right.addWidget(fatigue_group)
        right.addWidget(session_group)
        right.addWidget(logs_group, 1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        self.btn_start.clicked.connect(self._start_processing)
        self.btn_stop.clicked.connect(self._stop_processing)
        self.btn_reset.clicked.connect(self._reset_processing)
        self.btn_debug.clicked.connect(self._toggle_debug)
        self.btn_landmarks.clicked.connect(self._toggle_landmarks)
        self.btn_export.clicked.connect(self._export_session)

    def _setup_menu(self):
        file_menu = self.menuBar().addMenu("File")
        export_action = QAction("Export Session", self)
        export_action.triggered.connect(self._export_session)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _setup_hotkeys(self):
        QShortcut(QKeySequence("S"), self, activated=self._toggle_start_stop)
        QShortcut(QKeySequence("R"), self, activated=self._reset_processing)
        QShortcut(QKeySequence("D"), self, activated=self._toggle_debug)
        QShortcut(QKeySequence("L"), self, activated=self._toggle_landmarks)
        QShortcut(QKeySequence("Q"), self, activated=self.close)

    def _setup_worker(self):
        self.worker = CVWorker()
        self.worker.frame_ready.connect(self._update_video)
        self.worker.metrics_ready.connect(self._update_metrics)
        self.worker.log_message.connect(self._log)
        self.worker.error_occurred.connect(lambda msg: self._log(msg, True))

    def _toggle_start_stop(self):
        if self.worker and self.worker.isRunning():
            self._stop_processing()
        else:
            self._start_processing()

    def _start_processing(self):
        if self.worker and not self.worker.isRunning():
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.worker.start()
            self.statusBar().showMessage("Processing started")

    def _stop_processing(self, recreate_worker: bool = True):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Processing stopped")
        if recreate_worker:
            self._setup_worker()

    def _reset_processing(self):
        if self.worker:
            self.worker.reset_state()
            self.statusBar().showMessage("Session reset")

    def _toggle_debug(self):
        if not self.worker:
            return
        new_value = not self.worker._show_debug
        self.worker.set_show_debug(new_value)
        self.btn_debug.setText("Hide Debug (D)" if new_value else "Show Debug (D)")

    def _toggle_landmarks(self):
        if not self.worker:
            return
        new_value = not self.worker._show_landmarks
        self.worker.set_show_landmarks(new_value)
        self.btn_landmarks.setText("Hide Landmarks" if new_value else "Show Landmarks")

    def _export_session(self):
        if not self.worker:
            return
        target = QFileDialog.getExistingDirectory(self, "Choose export directory")
        if not target:
            return
        self.worker.collector.save(str(Path(target) / f"session_export_{int(time.time())}"))
        self._log(f"Session exported to {target}")

    def _update_video(self, q_img: QImage):
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _update_metrics(self, data: dict):
        self.metric_labels["frame"].setText(str(data["frame"]))
        self.metric_labels["blink"].setText(str(data["blink"]))
        self.metric_labels["yawn"].setText(str(data["yawn"]))
        self.metric_labels["perclos"].setText(f"{data['perclos']:.1f}%")
        self.metric_labels["rubbing"].setText("YES" if data["rubbing"] else "OK")
        self.metric_labels["tilt"].setText({0: "Normal", 1: "Light", 2: "Heavy"}.get(data["tilt_state"], "N/A"))
        self.metric_labels["keys"].setText(f"{data['key_rate']:.1f}")
        self.metric_labels["clicks"].setText(f"{data['mouse_click_rate']:.1f}")
        self.metric_labels["idle"].setText(f"{data['idle_sec']:.1f}s")
        self.metric_labels["total_blinks"].setText(str(data["total_blinks"]))
        self.metric_labels["total_yawns"].setText(str(data["total_yawns"]))
        self.metric_labels["online_updates"].setText(str(data["online_updates"]))
        self.metric_labels["queue"].setText(str(data["queue_depth"]))

        level = int(data["fatigue_level"])
        colors = {0: "#43c97a", 1: "#ffcb45", 2: "#ff5a5f"}
        texts = {0: "NORMAL", 1: "WARNING: TIRED", 2: "CRITICAL FATIGUE"}
        self.lbl_status.setText(data.get("fatigue_text", texts.get(level, "UNKNOWN")))
        self.lbl_status.setStyleSheet(f"color:{colors.get(level, '#d9dde3')};")
        self.fatigue_bar.setValue(level)
        self.fatigue_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {colors.get(level, '#888')}; }}")
        self.lbl_conf.setText(f"Confidence: {data['fatigue_conf'] * 100:.1f}%")
        self.lbl_advice.setText(f"Advice: {data['advice'] or 'keep monitoring'}")
        self.lbl_storage.setText(f"DB: {data['storage_path']}")
        self.lbl_modes.setText(
            f"Debug: {'on' if data['show_debug'] else 'off'} | "
            f"Landmarks: {'on' if data['show_landmarks'] else 'off'}"
        )

    def _log(self, msg: str, error: bool = False):
        color = "#ff7b7b" if error else "#9ff7bf"
        self.log_box.append(f"<span style='color:{color}'>{msg}</span>")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def closeEvent(self, event):
        self._stop_processing(recreate_worker=False)
        if self.worker is not None:
            self.worker.classifier.online_learner.save()
            self.worker.face_det.close()
            self.worker.pose_det.close()
            self.worker.collector.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Fatigue Detector")
    window = FatigueApp()
    window.show()
    sys.exit(app.exec())
