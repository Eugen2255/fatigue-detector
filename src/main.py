"""Main realtime fatigue detection pipeline."""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from config import MetricConfig, pose_path, face_path
from detectors.face_detector import FaceDetector
from detectors.pose_detector import PoseDetector
from pipeline import FatiguePipeline
from collector.collector import MetricsCollector
from collector.visualizer import MetricsVisualizer
from model.fatigue_classifier import FatigueClassifier
from model.model_config import ModelConfig as ClassifierConfig
from utils.recommender import Recommender
from utils.state_tracker import FatigueStateTracker
from utils.input_monitor import InputActivityMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def draw_landmarks(frame: np.ndarray, face_kp: Optional[np.ndarray], pose_kp: Optional[np.ndarray]) -> None:
    if face_kp is not None:
        for pt in face_kp[0]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 200, 0), -1)
    if pose_kp is not None:
        for pt in pose_kp[0]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (150, 0, 255), -1)


def main():
    app_cfg = MetricConfig.from_yaml("metrics_config.yaml")
    clf_cfg = ClassifierConfig()

    logger.info("Initializing realtime pipeline...")
    face_det = FaceDetector(face_path, num_faces=1)
    pose_det = PoseDetector(pose_path, num_poses=1)
    pipeline = FatiguePipeline(app_cfg)
    collector = MetricsCollector(fps=app_cfg.fps_target)
    visualizer = MetricsVisualizer()
    recommender = Recommender(cooldown_frames=300)
    state_tracker = FatigueStateTracker(decay_rate=0.985)
    input_monitor = InputActivityMonitor()

    classifier = FatigueClassifier(clf_cfg)
    model_ready = classifier.load(str(clf_cfg.default_model_path))
    window_size = 30

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        logger.error("Camera is not available")
        return

    frame_idx = 0
    show_counters = True
    show_landmarks = False

    logger.info("Ready. [Q] quit, [R] reset, [D] counters, [L] landmarks")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_idx += 1

            input_snapshot = input_monitor.sample()
            face_kp = face_det.get_key_pointers(frame)
            pose_kp = pose_det.get_key_pointers(frame)
            metrics = pipeline.step(face_kp, pose_kp)

            tilt_data = metrics.get("head_tilt")
            tilt_state = tilt_data[1] if isinstance(tilt_data, tuple) else 0
            tilt_angle = tilt_data[0] if isinstance(tilt_data, tuple) else None

            tracker_input = {
                "blink": metrics["blink"],
                "yawn": metrics.get("yawn", 0),
                "rubbing": metrics["rubbing"],
                "tilt_state": tilt_state,
            }
            state_tracker.update(tracker_input)
            fatigue_level, fatigue_color, fatigue_text = state_tracker.get_status()

            collector.add_frame(
                frame_idx=frame_idx,
                blink=metrics["blink"],
                perclos=metrics.get("perclos", 0),
                rubbing=metrics["rubbing"],
                head_tilt_state=tilt_state,
                head_tilt_angle=tilt_angle,
                yawn=metrics.get("yawn", 0),
                input_metrics=input_snapshot.to_dict(),
                face_kps=face_kp,
                pose_kps=pose_kp,
            )

            advice = recommender.get_advice(fatigue_level, frame_idx)
            input_summary = collector.recent_input_summary(window_size)

            if len(collector._history) >= window_size and frame_idx % 10 == 0:
                window_df = collector.recent_window_dataframe(window_size)
                teacher_label = fatigue_level
                teacher_confidence = 0.70
                if model_ready or classifier.online_learner.is_ready:
                    try:
                        pred, probs = classifier.predict_from_window(window_df)
                        teacher_label = int(pred[0])
                        teacher_confidence = float(np.max(probs[0]))
                    except Exception:
                        pass
                classifier.update_online_from_window(window_df, teacher_label, teacher_confidence)

            current_metrics = {
                "frame": frame_idx,
                "blink_total": collector.total_blinks,
                "yawn_total": collector.total_yawn,
                "rubbing_total": collector.total_rubbing,
                "perclos_pct": collector.recent_perclos_pct(),
                "head_tilt_state": tilt_state,
                "head_tilt_total": collector.total_tilt_light + collector.total_tilt_heavy,
                "key_rate": input_summary["key_rate"],
                "mouse_click_rate": input_summary["mouse_click_rate"],
                "idle_sec": input_summary["idle_sec"],
            }

            display_frame = visualizer.draw(frame.copy(), current_metrics, show_counters=show_counters)
            if show_landmarks:
                draw_landmarks(display_frame, face_kp, pose_kp)

            h, w, _ = display_frame.shape
            status_font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(fatigue_text, status_font, 1.0, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 20
            cv2.rectangle(display_frame, (text_x - 10, text_y - 35), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(display_frame, fatigue_text, (text_x, text_y), status_font, 1.0, fatigue_color, 3, cv2.LINE_AA)

            if advice:
                rec_size = cv2.getTextSize(advice, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                rec_x = (w - rec_size[0]) // 2
                cv2.putText(display_frame, advice, (rec_x, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

            cv2.imshow("Fatigue Detection Pro", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                collector.reset()
                pipeline.reset()
                input_monitor.reset()
                state_tracker.fatigue_score = 0.0
                state_tracker.level = 0
                recommender.last_advice_frame = -recommender.cooldown
                logger.info("State reset")
            elif key == ord("d"):
                show_counters = not show_counters
            elif key == ord("l"):
                show_landmarks = not show_landmarks

    except KeyboardInterrupt:
        pass
    finally:
        classifier.online_learner.save()
        face_det.close()
        pose_det.close()
        collector.save(str(Path("output") / f"session_{frame_idx}"))
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
