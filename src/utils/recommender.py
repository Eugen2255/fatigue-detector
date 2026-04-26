"""Система умных рекомендаций."""
import random
import logging

logger = logging.getLogger(__name__)

ADVICE_DB = {
    1: [
        "Сделай перерыв на 5 минут.",
        "Посмотри вдаль (правило 20-20-20).",
        "Выпей стакан воды.",
        "Разомни шею."
    ],
    2: [
        "❗ СРОЧНЫЙ ПЕРЕРЫВ! Ты очень устал.",
        "Выйди на свежий воздух.",
        "Умойся холодной водой.",
        "Сделай зарядку для глаз."
    ]
}

class Recommender:
    def __init__(self, cooldown_frames: int = 600):
        self.cooldown = cooldown_frames
        self.last_advice_frame = -self.cooldown  # Исправлено

    def get_advice(self, level: int, current_frame: int) -> str:
        if level == 0:
            return ""
        
        if current_frame - self.last_advice_frame < self.cooldown:
            return ""
        
        self.last_advice_frame = current_frame
        options = ADVICE_DB.get(level, [])
        return random.choice(options) if options else ""