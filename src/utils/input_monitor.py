"""Global keyboard and mouse activity sampling for Windows."""
from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@dataclass
class InputSnapshot:
    key_presses: int = 0
    active_keys: int = 0
    keyboard_burst: int = 0
    mouse_clicks: int = 0
    mouse_distance_px: float = 0.0
    mouse_speed_px_s: float = 0.0
    mouse_active: int = 0
    idle_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "key_presses": float(self.key_presses),
            "active_keys": float(self.active_keys),
            "keyboard_burst": float(self.keyboard_burst),
            "mouse_clicks": float(self.mouse_clicks),
            "mouse_distance_px": float(self.mouse_distance_px),
            "mouse_speed_px_s": float(self.mouse_speed_px_s),
            "mouse_active": float(self.mouse_active),
            "idle_ms": float(self.idle_ms),
        }


class InputActivityMonitor:
    """Polls global input state using WinAPI without external dependencies."""

    def __init__(self, burst_window_sec: float = 1.0, burst_threshold: int = 4):
        self._user32 = ctypes.windll.user32
        self._tracked_keys = tuple(self._build_tracked_keys())
        self._prev_down = {vk: False for vk in self._tracked_keys}
        self._prev_mouse_buttons = {vk: False for vk in (VK_LBUTTON, VK_RBUTTON, VK_MBUTTON)}
        self._last_position = self._get_cursor_pos()
        now = time.perf_counter()
        self._last_sample_ts = now
        self._last_activity_ts = now
        self._keypress_times = []
        self._burst_window_sec = burst_window_sec
        self._burst_threshold = burst_threshold

    @staticmethod
    def _build_tracked_keys() -> Iterable[int]:
        keys = {
            VK_BACK, VK_TAB, VK_RETURN, VK_SHIFT, VK_CONTROL, VK_MENU,
            VK_SPACE, VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN,
        }
        keys.update(range(0x30, 0x3A))
        keys.update(range(0x41, 0x5B))
        keys.update(range(0x60, 0x6A))
        keys.update(range(0x6A, 0x70))
        keys.update(range(0xBA, 0xC1))
        keys.update(range(0xDB, 0xDF))
        return sorted(keys)

    def _get_key_down(self, vk: int) -> bool:
        return bool(self._user32.GetAsyncKeyState(vk) & 0x8000)

    def _get_cursor_pos(self) -> Tuple[float, float]:
        point = POINT()
        self._user32.GetCursorPos(ctypes.byref(point))
        return float(point.x), float(point.y)

    def sample(self) -> InputSnapshot:
        now = time.perf_counter()
        dt = max(now - self._last_sample_ts, 1e-6)
        self._last_sample_ts = now

        key_presses = 0
        active_keys = 0
        for vk in self._tracked_keys:
            is_down = self._get_key_down(vk)
            if is_down:
                active_keys += 1
            if is_down and not self._prev_down[vk]:
                key_presses += 1
                self._keypress_times.append(now)
            self._prev_down[vk] = is_down

        cutoff = now - self._burst_window_sec
        while self._keypress_times and self._keypress_times[0] < cutoff:
            self._keypress_times.pop(0)
        keyboard_burst = 1 if len(self._keypress_times) >= self._burst_threshold else 0

        mouse_clicks = 0
        for vk in self._prev_mouse_buttons:
            is_down = self._get_key_down(vk)
            if is_down and not self._prev_mouse_buttons[vk]:
                mouse_clicks += 1
            self._prev_mouse_buttons[vk] = is_down

        current_pos = self._get_cursor_pos()
        dx = current_pos[0] - self._last_position[0]
        dy = current_pos[1] - self._last_position[1]
        self._last_position = current_pos
        mouse_distance_px = math.hypot(dx, dy)
        mouse_speed_px_s = mouse_distance_px / dt
        mouse_active = 1 if mouse_distance_px >= 1.0 else 0

        if key_presses > 0 or mouse_clicks > 0 or mouse_active:
            self._last_activity_ts = now
        idle_ms = (now - self._last_activity_ts) * 1000.0

        return InputSnapshot(
            key_presses=key_presses,
            active_keys=active_keys,
            keyboard_burst=keyboard_burst,
            mouse_clicks=mouse_clicks,
            mouse_distance_px=mouse_distance_px,
            mouse_speed_px_s=mouse_speed_px_s,
            mouse_active=mouse_active,
            idle_ms=idle_ms,
        )

    def reset(self) -> None:
        now = time.perf_counter()
        self._last_sample_ts = now
        self._last_activity_ts = now
        self._keypress_times.clear()
        self._last_position = self._get_cursor_pos()
        for vk in self._tracked_keys:
            self._prev_down[vk] = self._get_key_down(vk)
        for vk in self._prev_mouse_buttons:
            self._prev_mouse_buttons[vk] = self._get_key_down(vk)
