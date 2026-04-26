"""Отслеживание общего состояния усталости на основе накопления метрик."""
import logging

logger = logging.getLogger(__name__)

class FatigueStateTracker:
    def __init__(self, decay_rate: float = 0.98):
        self.fatigue_score = 0.0
        self.decay_rate = decay_rate # Скорость "восстановления" (чем меньше, тем быстрее спадает усталость)
        
        # Пороги для переключения уровней (накопленные очки)
        self.threshold_warn = 15.0
        self.threshold_crit = 40.0
        
        self.level = 0 # 0: OK, 1: Warn, 2: Crit

    def update(self, metrics: dict):
        """
        Принимает текущие события кадра и обновляет счетчик усталости.
        metrics: { 'blink': 0/1, 'yawn': 0/1, 'rubbing': bool, 'tilt_state': 0/1/2 }
        """
        score_increment = 0.0
        
        # Моргание: +1 очко за каждое моргание
        if metrics.get('blink', 0) == 1:
            score_increment += 1.0
            
        # Зевота: +5 очков (сильный признак)
        if metrics.get('yawn', 0) == 1:
            score_increment += 5.0
            
        # Потирание глаз: +3 очка
        if metrics.get('rubbing', False):
            score_increment += 3.0
            
        # Наклон головы: +0.5 за легкий, +2.0 за сильный
        tilt = metrics.get('tilt_state', 0)
        if tilt == 1: score_increment += 0.5
        elif tilt == 2: score_increment += 2.0
        
        # Обновляем общий счет
        self.fatigue_score += score_increment
        
        # Применяем затухание (восстановление)
        self.fatigue_score *= self.decay_rate
        
        # Ограничиваем минимум нулем
        if self.fatigue_score < 0: self.fatigue_score = 0.0
        
        # Определяем уровень
        if self.fatigue_score > self.threshold_crit:
            self.level = 2
        elif self.fatigue_score > self.threshold_warn:
            self.level = 1
        else:
            self.level = 0

    def get_status(self) -> tuple:
        """Возвращает (level: int, color: tuple, text: str)"""
        if self.level == 2:
            return 2, (0, 0, 255), "CRITICAL FATIGUE"
        elif self.level == 1:
            return 1, (0, 255, 255), "WARNING: TIRED"
        else:
            return 0, (0, 255, 0), "NORMAL"