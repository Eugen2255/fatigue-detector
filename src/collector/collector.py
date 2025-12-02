import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
import json
import os

class MetricsCollector:
    """Класс для сбора и хранения метрик"""
    
    def __init__(self, save_interval: int = 100, save_path: str = "metrics_data"):
        """
        Args:
            save_interval: интервал сохранения данных (в кадрах)
            save_path: путь для сохранения данных
        """
        self.save_interval = save_interval
        self.save_path = save_path
        
        # Инициализация счетчиков
        self.total_blinks = 0
        self.total_rubbing = 0
        self.total_head_tilt_light = 0
        self.total_head_tilt_heavy = 0
        
        # Хранение всех метрик
        self.metrics_history = []
        self.current_frame_metrics = {}
        
        # Создаем директорию для сохранения, если её нет
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def collect_frame_metrics(
        self,
        frame_num: int,
        head_tilt_result: Optional[Tuple[float, int]] = None,
        blink_result: int = 0,
        rubbing_result: bool = False,
        face_kps: Optional[list] = None,
        pose_kps: Optional[list] = None
    ) -> pd.Series:
        """
        Собирает метрики для текущего кадра.
        
        Returns:
            pd.Series: метрики текущего кадра
        """
        # Инициализация значений для текущего кадра
        blink_current = 1 if blink_result == 1 else 0
        rubbing_current = 1 if rubbing_result else 0
        head_tilt_state = 0
        head_tilt_angle = None
        
        # Обработка наклона головы
        if head_tilt_result is not None:
            angle_deg, state = head_tilt_result
            head_tilt_state = state
            head_tilt_angle = angle_deg
            
            # Обновление счетчиков
            if state == 1:
                self.total_head_tilt_light += 1
            elif state == 2:
                self.total_head_tilt_heavy += 1
        
        # Обновление счетчиков
        self.total_blinks += blink_current
        self.total_rubbing += rubbing_current
        
        # Создание Series с метриками
        metrics_dict = {
            'frame': frame_num,
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'blink_current': blink_current,
            'blink_total': self.total_blinks,
            'rubbing_current': rubbing_current,
            'rubbing_total': self.total_rubbing,
            'head_tilt_state': head_tilt_state,
            'head_tilt_angle': head_tilt_angle,
            'head_tilt_light_total': self.total_head_tilt_light,
            'head_tilt_heavy_total': self.total_head_tilt_heavy,
            'head_tilt_total': self.total_head_tilt_light + self.total_head_tilt_heavy,
            'face_detected': 0 if face_kps is None else len(face_kps),
            'pose_detected': 0 if pose_kps is None else len(pose_kps),
        }
        
        # Сохраняем метрики текущего кадра
        self.current_frame_metrics = metrics_dict.copy()
        
        # Добавляем в историю
        metrics_series = pd.Series(metrics_dict)
        self.metrics_history.append(metrics_series)
        
        return metrics_series
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики текущего кадра"""
        return self.current_frame_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по всем метрикам"""
        if not self.metrics_history:
            return {}
        
        # Создаем DataFrame из истории
        df = pd.DataFrame(self.metrics_history)
        
        summary = {
            'total_frames': len(df),
            'total_blinks': self.total_blinks,
            'blink_rate_per_min': self._calculate_blink_rate(df),
            'total_rubbing': self.total_rubbing,
            'rubbing_percentage': (self.total_rubbing / len(df) * 100) if len(df) > 0 else 0,
            'head_tilt_light': self.total_head_tilt_light,
            'head_tilt_heavy': self.total_head_tilt_heavy,
            'head_tilt_total': self.total_head_tilt_light + self.total_head_tilt_heavy,
            'head_tilt_percentage': ((self.total_head_tilt_light + self.total_head_tilt_heavy) / len(df) * 100) if len(df) > 0 else 0,
        }
        
        return summary
    
    def _calculate_blink_rate(self, df: pd.DataFrame) -> float:
        """Рассчитывает частоту морганий в минуту"""
        if len(df) < 2:
            return 0
        
        # Предполагаем 30 FPS для расчета времени
        total_seconds = len(df) / 30
        if total_seconds > 0:
            return (self.total_blinks / total_seconds) * 60
        return 0
    
    def save_current_data(self):
        """Сохраняет текущие данные в файл"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Сохраняем как CSV
        csv_path = os.path.join(self.save_path, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_path, index=False)
        
        # Сохраняем сводку как JSON
        summary = self.get_summary()
        json_path = os.path.join(self.save_path, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Данные сохранены: {csv_path}")
    
    def reset(self):
        """Сброс всех счетчиков"""
        self.total_blinks = 0
        self.total_rubbing = 0
        self.total_head_tilt_light = 0
        self.total_head_tilt_heavy = 0
        self.metrics_history = []
        self.current_frame_metrics = {}