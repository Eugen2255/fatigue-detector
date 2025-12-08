import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class MetricsAnalyzer:
    """Класс для анализа собранных метрик"""
    
    def __init__(self, metrics_history: List[pd.Series] = None):
        self.metrics_history = metrics_history if metrics_history else []
    
    def add_metrics(self, metrics: pd.Series):
        """Добавляет метрики в историю"""
        self.metrics_history.append(metrics)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Возвращает DataFrame со всеми метриками"""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_history)
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Анализирует паттерны в метриках"""
        df = self.get_dataframe()
        if df.empty:
            return {}
        
        analysis = {
            'blink_intervals': self._calculate_blink_intervals(df),
            'rubbing_patterns': self._analyze_rubbing_patterns(df),
            'head_tilt_patterns': self._analyze_head_tilt_patterns(df),
            'correlations': self._calculate_correlations(df),
        }
        
        return analysis
    
    def _calculate_blink_intervals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Рассчитывает интервалы между морганиями"""
        blink_frames = df[df['blink_current'] == 1]['frame'].values
        
        if len(blink_frames) < 2:
            return {'avg_interval_frames': 0, 'avg_interval_seconds': 0}
        
        intervals = np.diff(blink_frames)
        
        return {
            'avg_interval_frames': float(np.mean(intervals)),
            'avg_interval_seconds': float(np.mean(intervals) / 30),  # Предполагаем 30 FPS
            'min_interval': float(np.min(intervals)),
            'max_interval': float(np.max(intervals)),
            'total_blinks': len(blink_frames),
        }
    
    def _analyze_rubbing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализирует паттерны потирания глаз"""
        rubbing_events = df[df['rubbing_current'] == 1]
        
        if rubbing_events.empty:
            return {'total_events': 0, 'avg_duration': 0}
        
        # Группируем последовательные события потирания
        rubbing_groups = []
        current_group = []
        
        for idx, row in rubbing_events.iterrows():
            if not current_group or (row['frame'] - current_group[-1]['frame'] == 1):
                current_group.append(row)
            else:
                if current_group:
                    rubbing_groups.append(current_group)
                current_group = [row]
        
        if current_group:
            rubbing_groups.append(current_group)
        
        durations = [len(group) for group in rubbing_groups]
        
        return {
            'total_events': len(rubbing_events),
            'total_groups': len(rubbing_groups),
            'avg_duration_frames': float(np.mean(durations)) if durations else 0,
            'avg_duration_seconds': float(np.mean(durations) / 30) if durations else 0,
        }
    
    def _analyze_head_tilt_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализирует паттерны наклона головы"""
        tilt_events = df[df['head_tilt_state'] > 0]
        
        if tilt_events.empty:
            return {'total_events': 0}
        
        # Анализ по типам наклона
        light_tilt = tilt_events[tilt_events['head_tilt_state'] == 1]
        heavy_tilt = tilt_events[tilt_events['head_tilt_state'] == 2]
        
        # Анализ углов
        angles = tilt_events['head_tilt_angle'].dropna()
        
        return {
            'total_events': len(tilt_events),
            'light_tilt_events': len(light_tilt),
            'heavy_tilt_events': len(heavy_tilt),
            'avg_angle': float(angles.mean()) if not angles.empty else None,
            'max_angle': float(angles.max()) if not angles.empty else None,
        }
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Рассчитывает корреляции между метриками"""
        # Выбираем числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        # Рассчитываем корреляции
        corr_matrix = df[numeric_cols].corr()
        
        # Находим интересные корреляции
        interesting_corrs = {}
        for col1 in ['blink_current', 'rubbing_current', 'head_tilt_state']:
            if col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.3:
                        interesting_corrs[f"{col1}_{col2}"] = float(corr_matrix.loc[col1, col2])
        
        return interesting_corrs
    
    def generate_report(self, save_path: str = None):
        """Генерирует отчет по метрикам"""
        df = self.get_dataframe()
        if df.empty:
            print("Нет данных для отчета")
            return
        
        summary = self.analyze_patterns()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_frames': len(df),
            'total_duration_seconds': len(df) / 30,  # Предполагаем 30 FPS
            'summary': {
                'blinks': summary.get('blink_intervals', {}),
                'rubbing': summary.get('rubbing_patterns', {}),
                'head_tilt': summary.get('head_tilt_patterns', {}),
                'correlations': summary.get('correlations', {}),
            },
            'basic_stats': {
                'blink_total': int(df['blink_total'].iloc[-1]) if not df.empty else 0,
                'rubbing_total': int(df['rubbing_total'].iloc[-1]) if not df.empty else 0,
                'head_tilt_total': int(df['head_tilt_total'].iloc[-1]) if not df.empty else 0,
            }
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Отчет сохранен: {save_path}")
        
        return report