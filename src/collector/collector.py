"""SQLite-backed metrics collector for long-running sessions."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _HistoryProxy:
    def __init__(self, collector: "MetricsCollector"):
        self._collector = collector

    def __len__(self) -> int:
        return self._collector.frame_count


class MetricsCollector:
    def __init__(self, fps: float = 30.0, db_path: Optional[str | Path] = None):
        self.fps = fps
        self.db_path = Path(db_path) if db_path else Path("src") / "output" / "live_metrics.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        self._history = _HistoryProxy(self)
        self._reset_counters()
        self._sync_counters_from_db()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_records (
                frame INTEGER PRIMARY KEY,
                blink INTEGER NOT NULL,
                perclos INTEGER NOT NULL,
                rubbing INTEGER NOT NULL,
                head_tilt_state INTEGER NOT NULL,
                head_tilt_angle REAL,
                yawn INTEGER NOT NULL,
                key_presses INTEGER NOT NULL,
                active_keys INTEGER NOT NULL,
                keyboard_burst INTEGER NOT NULL,
                mouse_clicks INTEGER NOT NULL,
                mouse_distance_px REAL NOT NULL,
                mouse_speed_px_s REAL NOT NULL,
                mouse_active INTEGER NOT NULL,
                idle_ms REAL NOT NULL,
                face_detected INTEGER NOT NULL,
                pose_detected INTEGER NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_frame_records_frame_desc ON frame_records(frame DESC)")
        self._conn.commit()

    def _reset_counters(self) -> None:
        self.frame_count = 0
        self.total_blinks = 0
        self.total_perclos_frames = 0
        self.total_rubbing = 0
        self.total_tilt_light = 0
        self.total_tilt_heavy = 0
        self.total_yawn = 0
        self.total_key_presses = 0
        self.total_mouse_clicks = 0
        self.total_mouse_distance_px = 0.0

    def _sync_counters_from_db(self) -> None:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS frame_count,
                COALESCE(SUM(blink), 0) AS total_blinks,
                COALESCE(SUM(perclos), 0) AS total_perclos_frames,
                COALESCE(SUM(rubbing), 0) AS total_rubbing,
                COALESCE(SUM(CASE WHEN head_tilt_state = 1 THEN 1 ELSE 0 END), 0) AS total_tilt_light,
                COALESCE(SUM(CASE WHEN head_tilt_state = 2 THEN 1 ELSE 0 END), 0) AS total_tilt_heavy,
                COALESCE(SUM(yawn), 0) AS total_yawn,
                COALESCE(SUM(key_presses), 0) AS total_key_presses,
                COALESCE(SUM(mouse_clicks), 0) AS total_mouse_clicks,
                COALESCE(SUM(mouse_distance_px), 0.0) AS total_mouse_distance_px
            FROM frame_records
            """
        ).fetchone()
        self.frame_count = int(row["frame_count"])
        self.total_blinks = int(row["total_blinks"])
        self.total_perclos_frames = int(row["total_perclos_frames"])
        self.total_rubbing = int(row["total_rubbing"])
        self.total_tilt_light = int(row["total_tilt_light"])
        self.total_tilt_heavy = int(row["total_tilt_heavy"])
        self.total_yawn = int(row["total_yawn"])
        self.total_key_presses = int(row["total_key_presses"])
        self.total_mouse_clicks = int(row["total_mouse_clicks"])
        self.total_mouse_distance_px = float(row["total_mouse_distance_px"])

    def add_frame(
        self,
        frame_idx: int,
        blink: Optional[int] = None,
        perclos: Optional[int] = None,
        rubbing: Optional[bool] = None,
        head_tilt_state: Optional[int] = None,
        head_tilt_angle: Optional[float] = None,
        yawn: Optional[int] = None,
        input_metrics: Optional[Dict[str, Any]] = None,
        face_kps: Optional[np.ndarray] = None,
        pose_kps: Optional[np.ndarray] = None,
    ) -> None:
        blink_val = 1 if blink == 1 else 0
        perclos_val = 1 if perclos == 1 else 0
        rubbing_val = 1 if rubbing is True else 0
        tilt_state_val = head_tilt_state if head_tilt_state in (1, 2) else 0
        yawn_val = yawn if yawn in (0, 1) else 0
        input_metrics = input_metrics or {}
        key_presses = int(input_metrics.get("key_presses", 0) or 0)
        active_keys = int(input_metrics.get("active_keys", 0) or 0)
        keyboard_burst = 1 if input_metrics.get("keyboard_burst", 0) else 0
        mouse_clicks = int(input_metrics.get("mouse_clicks", 0) or 0)
        mouse_distance_px = float(input_metrics.get("mouse_distance_px", 0.0) or 0.0)
        mouse_speed_px_s = float(input_metrics.get("mouse_speed_px_s", 0.0) or 0.0)
        mouse_active = 1 if input_metrics.get("mouse_active", 0) else 0
        idle_ms = float(input_metrics.get("idle_ms", 0.0) or 0.0)
        is_face = 1 if face_kps is not None and face_kps.size > 0 else 0
        is_pose = 1 if pose_kps is not None and pose_kps.size > 0 else 0

        self._conn.execute(
            """
            INSERT OR REPLACE INTO frame_records (
                frame, blink, perclos, rubbing, head_tilt_state, head_tilt_angle, yawn,
                key_presses, active_keys, keyboard_burst, mouse_clicks, mouse_distance_px,
                mouse_speed_px_s, mouse_active, idle_ms, face_detected, pose_detected
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_idx,
                blink_val,
                perclos_val,
                rubbing_val,
                tilt_state_val,
                head_tilt_angle,
                yawn_val,
                key_presses,
                active_keys,
                keyboard_burst,
                mouse_clicks,
                mouse_distance_px,
                mouse_speed_px_s,
                mouse_active,
                idle_ms,
                is_face,
                is_pose,
            ),
        )
        self._conn.commit()

        self.frame_count += 1
        self.total_blinks += blink_val
        self.total_perclos_frames += perclos_val
        self.total_rubbing += rubbing_val
        if tilt_state_val == 1:
            self.total_tilt_light += 1
        elif tilt_state_val == 2:
            self.total_tilt_heavy += 1
        self.total_yawn += yawn_val
        self.total_key_presses += key_presses
        self.total_mouse_clicks += mouse_clicks
        self.total_mouse_distance_px += mouse_distance_px

    def _fetch_recent_rows(self, window_size: int):
        rows = self._conn.execute(
            "SELECT * FROM frame_records ORDER BY frame DESC LIMIT ?",
            (max(1, int(window_size)),),
        ).fetchall()
        return list(reversed(rows))

    def recent_perclos_pct(self) -> float:
        row = self._conn.execute(
            "SELECT AVG(perclos) AS avg_perclos FROM (SELECT perclos FROM frame_records ORDER BY frame DESC LIMIT ?)",
            (max(1, int(round(self.fps))),),
        ).fetchone()
        avg_perclos = row["avg_perclos"]
        return float(avg_perclos * 100.0) if avg_perclos is not None else 0.0

    def recent_window_dataframe(self, window_size: int):
        import pandas as pd

        rows = self._fetch_recent_rows(window_size)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])

    def recent_input_summary(self, window_size: int) -> Dict[str, float]:
        rows = self._fetch_recent_rows(window_size)
        if not rows:
            return {"key_rate": 0.0, "mouse_click_rate": 0.0, "idle_sec": 0.0}
        frames = max(len(rows), 1)
        frame_scale = self.fps * 60.0 / frames
        return {
            "key_rate": sum(int(row["key_presses"]) for row in rows) * frame_scale,
            "mouse_click_rate": sum(int(row["mouse_clicks"]) for row in rows) * frame_scale,
            "idle_sec": float(rows[-1]["idle_ms"]) / 1000.0,
        }

    def to_dataframe(self):
        import pandas as pd

        rows = self._conn.execute("SELECT * FROM frame_records ORDER BY frame ASC").fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(row) for row in rows])

    def get_summary(self) -> Dict[str, Any]:
        total_frames = self.frame_count
        if total_frames == 0:
            return {}

        duration_sec = total_frames / self.fps
        return {
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "total_blinks": self.total_blinks,
            "blink_rate_per_min": (self.total_blinks / duration_sec) * 60 if duration_sec else 0.0,
            "perclos_pct": (self.total_perclos_frames / total_frames) * 100,
            "rubbing_frames": self.total_rubbing,
            "rubbing_pct": (self.total_rubbing / total_frames) * 100,
            "tilt_light": self.total_tilt_light,
            "tilt_heavy": self.total_tilt_heavy,
            "tilt_pct": ((self.total_tilt_light + self.total_tilt_heavy) / total_frames) * 100,
            "yawn_count": self.total_yawn,
            "yawn_pct": (self.total_yawn / total_frames) * 100,
            "key_press_total": self.total_key_presses,
            "key_press_rate_per_min": (self.total_key_presses / duration_sec) * 60 if duration_sec else 0.0,
            "mouse_click_total": self.total_mouse_clicks,
            "mouse_click_rate_per_min": (self.total_mouse_clicks / duration_sec) * 60 if duration_sec else 0.0,
            "mouse_distance_px_total": self.total_mouse_distance_px,
            "sqlite_db": str(self.db_path),
        }

    def save(self, base_path: str) -> None:
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No metrics to save")
            return

        os.makedirs(base_path, exist_ok=True)
        csv_path = os.path.join(base_path, "metrics.csv")
        df.to_csv(csv_path, index=False)

        summary_path = os.path.join(base_path, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(self.get_summary(), fh, indent=2, ensure_ascii=False)

        sqlite_copy = os.path.join(base_path, "metrics.sqlite3")
        self._conn.commit()
        backup_conn = sqlite3.connect(sqlite_copy)
        with backup_conn:
            self._conn.backup(backup_conn)
        backup_conn.close()
        logger.info(f"Metrics saved to {csv_path}, {summary_path}, {sqlite_copy}")

    def reset(self) -> None:
        self._conn.execute("DELETE FROM frame_records")
        self._conn.commit()
        self._reset_counters()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __len__(self) -> int:
        return self.frame_count
