import cv2
import numpy as np
from typing import Dict, Any

class MetricsVisualizer:
    """Класс для визуализации метрик на видеокадре"""
    
    def __init__(self, position: tuple = (10, 30)):
        """
        Args:
            position: (x, y) координаты начала отображения метрик
        """
        self.position = position
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (0, 255, 0)  # Зеленый
        self.warning_color = (0, 0, 255)  # Красный
        self.info_color = (255, 255, 0)  # Голубой
        self.thickness = 1
        self.line_height = 25
        
    def draw_metrics(self, img: np.ndarray, metrics: Dict[str, Any]) -> np.ndarray:
        """
        Рисует метрики на кадре.
        
        Args:
            img: входное изображение
            metrics: словарь с метриками
            
        Returns:
            Изображение с нарисованными метриками
        """
        x, y = self.position
        
        # Основные метрики
        main_metrics = {
            'Frame': metrics.get('frame', 0),
            'Blinks': f"{metrics.get('blink_total', 0)} (current: {metrics.get('blink_current', 0)})",
            'Rubbing': f"{metrics.get('rubbing_total', 0)} (current: {metrics.get('rubbing_current', 0)})",
            'Head tilt': self._get_head_tilt_text(metrics),
        }
        
        # Отображение основных метрик
        for i, (key, value) in enumerate(main_metrics.items()):
            y_pos = y + i * self.line_height
            cv2.putText(img, f"{key}: {value}", (x, y_pos), 
                       self.font, self.font_scale, self.text_color, 
                       self.thickness, cv2.LINE_AA)
        
        # Отображение предупреждений
        warning_start_y = y + len(main_metrics) * self.line_height + 10
        warning_lines = self._get_warnings(metrics)
        
        for i, warning in enumerate(warning_lines):
            y_pos = warning_start_y + i * self.line_height
            cv2.putText(img, warning, (x, y_pos), 
                       self.font, self.font_scale, self.warning_color, 
                       self.thickness + 1, cv2.LINE_AA)
        
        # Отображение дополнительной информации
        info_start_y = warning_start_y + len(warning_lines) * self.line_height + 10
        if metrics.get('head_tilt_angle') is not None:
            angle_text = f"Angle: {metrics['head_tilt_angle']:.1f}°"
            cv2.putText(img, angle_text, (x, info_start_y), 
                       self.font, self.font_scale - 0.1, self.info_color, 
                       self.thickness, cv2.LINE_AA)
        
        return img
    
    def _get_head_tilt_text(self, metrics: Dict[str, Any]) -> str:
        """Форматирует текст для наклона головы"""
        state = metrics.get('head_tilt_state', 0)
        state_text = {0: "Normal", 1: "Light", 2: "Heavy"}.get(state, "Unknown")
        return f"{state_text} (total: {metrics.get('head_tilt_total', 0)})"
    
    def _get_warnings(self, metrics: Dict[str, Any]) -> list:
        """Возвращает список предупреждений"""
        warnings = []
        
        if metrics.get('head_tilt_state') == 2:
            warnings.append("WARNING: Heavy head tilt!")
        
        if metrics.get('rubbing_current') == 1:
            warnings.append("Rubbing detected!")
        
        if metrics.get('blink_current') == 1:
            warnings.append("Blink detected!")
        
        return warnings
    
    def draw_summary(self, img: np.ndarray, summary: Dict[str, Any], position: tuple = (400, 30)) -> np.ndarray:
        """
        Рисует сводку метрик на кадре.
        
        Args:
            img: входное изображение
            summary: словарь со сводкой
            position: (x, y) координаты
            
        Returns:
            Изображение со сводкой
        """
        if not summary:
            return img
        
        x, y = position
        
        # Заголовок
        cv2.putText(img, "SUMMARY:", (x, y), self.font, 
                   self.font_scale, self.text_color, self.thickness + 1, cv2.LINE_AA)
        
        # Ключевые метрики для сводки
        summary_metrics = {
            'Total frames': summary.get('total_frames', 0),
            'Blink rate/min': f"{summary.get('blink_rate_per_min', 0):.1f}",
            'Rubbing %': f"{summary.get('rubbing_percentage', 0):.1f}%",
            'Head tilt %': f"{summary.get('head_tilt_percentage', 0):.1f}%",
        }
        
        # Отображение сводки
        for i, (key, value) in enumerate(summary_metrics.items()):
            y_pos = y + (i + 1) * self.line_height
            cv2.putText(img, f"{key}: {value}", (x, y_pos), 
                       self.font, self.font_scale - 0.1, (200, 200, 0), 
                       self.thickness, cv2.LINE_AA)
        
        return img