import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

class FatigueDatasetBuilder:
    """
    Класс для построения датасета классификации усталости
    на основе собранных метрик
    """
    
    def __init__(self, window_size: int = 30, overlap: float = 0.5):
        """
        Args:
            window_size: размер окна в кадрах для агрегации
            overlap: перекрытие окон (0-1)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.window_step = int(window_size * (1 - overlap))
        self.data = []
        self.labels = []
        self.feature_names = []
        
    def load_metrics_data(self, metrics_file: str) -> pd.DataFrame:
        """Загружает данные метрик из CSV файла"""
        df = pd.read_csv(metrics_file)
        
        # Конвертируем временные метки если нужно
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                pass
        
        return df
    
    def extract_features_from_window(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """
        Извлекает признаки из окна данных
        
        Возможные признаки:
        1. Статистики по морганиям
        2. Статистики по наклонам головы
        3. Статистики по потираниям глаз
        4. Комбинированные признаки
        """
        features = {}
        
        # 1. Признаки морганий
        blink_series = window_df['blink_current'].values
        blink_total = window_df['blink_total'].iloc[-1] - window_df['blink_total'].iloc[0]
        
        features.update({
            'blink_count': blink_total,
            'blink_frequency': blink_total / len(window_df) * 100,  # морганий на 100 кадров
            'blink_max_interval': self._max_interval(blink_series),
            'blink_avg_interval': self._avg_interval(blink_series),
        })
        
        # 2. Признаки наклона головы
        tilt_angles = window_df['head_tilt_angle'].dropna().values
        tilt_states = window_df['head_tilt_state'].values
        
        if len(tilt_angles) > 0:
            features.update({
                'head_tilt_max_angle': float(np.max(tilt_angles)),
                'head_tilt_avg_angle': float(np.mean(tilt_angles)),
                'head_tilt_std_angle': float(np.std(tilt_angles)),
                'head_tilt_light_percentage': np.mean(tilt_states == 1) * 100,
                'head_tilt_heavy_percentage': np.mean(tilt_states == 2) * 100,
                'head_tilt_total_events': np.sum((tilt_states > 0).astype(int)),
            })
        else:
            # Если нет данных об угле
            features.update({
                'head_tilt_max_angle': 0,
                'head_tilt_avg_angle': 0,
                'head_tilt_std_angle': 0,
                'head_tilt_light_percentage': 0,
                'head_tilt_heavy_percentage': 0,
                'head_tilt_total_events': 0,
            })
        
        # 3. Признаки потираний глаз
        rubbing_series = window_df['rubbing_current'].values
        rubbing_total = window_df['rubbing_total'].iloc[-1] - window_df['rubbing_total'].iloc[0]
        
        features.update({
            'rubbing_count': rubbing_total,
            'rubbing_duration_avg': self._avg_duration(rubbing_series),
            'rubbing_percentage': np.mean(rubbing_series) * 100,
        })
        
        # 4. Комбинированные и дополнительные признаки
        features.update({
            'face_detected_percentage': np.mean(window_df['face_detected'] > 0) * 100,
            'pose_detected_percentage': np.mean(window_df['pose_detected'] > 0) * 100,
            'frames_without_face': np.sum(window_df['face_detected'] == 0),
            
            # Признаки активности
            'activity_score': self._calculate_activity_score(
                blink_total, 
                features.get('head_tilt_total_events', 0),
                rubbing_total
            ),
            
            # Признаки усталости
            'fatigue_score_1': blink_total / (len(window_df) / 100),  # Моргания на 100 кадров
            'fatigue_score_2': features.get('head_tilt_heavy_percentage', 0) * 0.5,
            'fatigue_score_3': features.get('rubbing_percentage', 0) * 0.3,
        })
        
        # 5. Временные признаки
        if 'timestamp' in window_df.columns and hasattr(window_df['timestamp'].iloc[0], 'hour'):
            hour = window_df['timestamp'].iloc[-1].hour
            features['time_of_day'] = hour
            
            # Нормализация времени суток
            features['time_sin'] = np.sin(2 * np.pi * hour / 24)
            features['time_cos'] = np.cos(2 * np.pi * hour / 24)
        
        self.feature_names = list(features.keys())
        return features
    
    def _max_interval(self, series: np.ndarray) -> float:
        """Максимальный интервал между событиями"""
        indices = np.where(series == 1)[0]
        if len(indices) < 2:
            return 0
        return float(np.max(np.diff(indices)))
    
    def _avg_interval(self, series: np.ndarray) -> float:
        """Средний интервал между событиями"""
        indices = np.where(series == 1)[0]
        if len(indices) < 2:
            return 0
        return float(np.mean(np.diff(indices)))
    
    def _avg_duration(self, series: np.ndarray) -> float:
        """Средняя продолжительность события"""
        durations = []
        current_duration = 0
        
        for val in series:
            if val == 1:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return float(np.mean(durations)) if durations else 0
    
    def _calculate_activity_score(self, blink_count: int, tilt_count: int, rubbing_count: int) -> float:
        """Расчет общего скора активности"""
        # Веса можно настраивать
        weights = {
            'blink': 0.3,
            'tilt': 0.4,
            'rubbing': 0.3
        }
        
        # Нормализация (примерные нормальные значения)
        blink_norm = min(blink_count / 10, 1.0)  # 10 морганий на окно - максимум
        tilt_norm = min(tilt_count / 5, 1.0)     # 5 наклонов на окно - максимум
        rubbing_norm = min(rubbing_count / 3, 1.0)  # 3 потирания на окно - максимум
        
        score = (blink_norm * weights['blink'] + 
                tilt_norm * weights['tilt'] + 
                rubbing_norm * weights['rubbing'])
        
        return float(score)
    
    def build_features_from_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Строит признаки из метрик с использованием скользящего окна"""
        features_list = []
        start_indices = []
        
        n_frames = len(metrics_df)
        
        for start in range(0, n_frames - self.window_size + 1, self.window_step):
            end = start + self.window_size
            window_df = metrics_df.iloc[start:end]
            
            # Извлекаем признаки из окна
            features = self.extract_features_from_window(window_df)
            features['window_start'] = start
            features['window_end'] = end
            
            features_list.append(features)
            start_indices.append(start)
        
        features_df = pd.DataFrame(features_list)
        return features_df
    
    def add_label(self, window_indices: List[int], label: int, label_source: str = "manual"):
        """Добавляет метку для определенных окон"""
        for idx in window_indices:
            if 0 <= idx < len(self.data):
                self.labels.append({
                    'window_idx': idx,
                    'label': label,
                    'label_source': label_source,
                    'timestamp': datetime.now().isoformat()
                })
    
    def save_dataset(self, output_path: str, include_labels: bool = True):
        """Сохраняет датасет в CSV"""
        # Сохраняем признаки
        features_df = pd.DataFrame(self.data)
        
        if include_labels and self.labels:
            labels_df = pd.DataFrame(self.labels)
            
            # Сопоставляем метки с окнами
            if 'window_idx' in labels_df.columns and 'window_start' in features_df.columns:
                # Создаем DataFrame с метками для каждого окна
                window_labels = {row['window_idx']: row['label'] for _, row in labels_df.iterrows()}
                features_df['fatigue_label'] = features_df.index.map(lambda x: window_labels.get(x, -1))
            else:
                features_df['fatigue_label'] = labels_df['label'].tolist() if len(labels_df) == len(features_df) else -1
        
        # Сохраняем
        features_df.to_csv(output_path, index=False)
        print(f"Датасет сохранен: {output_path}")
        
        # Сохраняем описание признаков
        features_info = {
            'generated_at': datetime.now().isoformat(),
            'window_size': self.window_size,
            'overlap': self.overlap,
            'total_samples': len(features_df),
            'features': self.feature_names,
            'label_distribution': features_df['fatigue_label'].value_counts().to_dict() if 'fatigue_label' in features_df.columns else {}
        }
        
        info_path = output_path.replace('.csv', '_info.json')
        with open(info_path, 'w') as f:
            json.dump(features_info, f, indent=2)
        
        return features_df
    
    def generate_synthetic_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует синтетические метки на основе признаков
        (Для первоначального тестирования моделей)
        """
        labels = []
        
        for _, row in features_df.iterrows():
            # Эвристические правила для определения усталости
            score = 0
            
            # 1. Моргания (редкие или частые - признак усталости)
            blink_freq = row.get('blink_frequency', 0)
            if blink_freq < 1 or blink_freq > 5:  # Норма: 1-5 морганий на 100 кадров
                score += 1
            
            # 2. Наклон головы (тяжелый наклон)
            if row.get('head_tilt_heavy_percentage', 0) > 10:  # >10% времени тяжелый наклон
                score += 1
            
            # 3. Потирания глаз
            if row.get('rubbing_percentage', 0) > 5:  # >5% времени потирание
                score += 1
            
            # 4. Максимальный угол наклона
            if row.get('head_tilt_max_angle', 0) > 40:  # Сильный наклон
                score += 1
            
            # Определяем класс усталости
            if score >= 3:
                label = 2  # Сильная усталость
            elif score >= 2:
                label = 1  # Средняя усталость
            else:
                label = 0  # Легкая/нет усталости
            
            labels.append(label)
        
        features_df['fatigue_label_synthetic'] = labels
        return features_df
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Создает примерный датасет для тестирования"""
        np.random.seed(42)
        
        data = []
        
        for i in range(n_samples):
            # Генерация реалистичных признаков
            
            # Нормальное состояние
            if i < n_samples // 3:
                features = {
                    'blink_count': np.random.randint(1, 4),
                    'blink_frequency': np.random.uniform(1, 3),
                    'head_tilt_max_angle': np.random.uniform(0, 15),
                    'head_tilt_heavy_percentage': np.random.uniform(0, 5),
                    'rubbing_count': np.random.randint(0, 1),
                    'rubbing_percentage': np.random.uniform(0, 2),
                    'fatigue_label': 0
                }
            
            # Средняя усталость
            elif i < 2 * n_samples // 3:
                features = {
                    'blink_count': np.random.randint(3, 7),
                    'blink_frequency': np.random.uniform(3, 6),
                    'head_tilt_max_angle': np.random.uniform(15, 30),
                    'head_tilt_heavy_percentage': np.random.uniform(5, 20),
                    'rubbing_count': np.random.randint(1, 3),
                    'rubbing_percentage': np.random.uniform(2, 8),
                    'fatigue_label': 1
                }
            
            # Сильная усталость
            else:
                features = {
                    'blink_count': np.random.randint(6, 12),
                    'blink_frequency': np.random.uniform(6, 10),
                    'head_tilt_max_angle': np.random.uniform(30, 50),
                    'head_tilt_heavy_percentage': np.random.uniform(20, 50),
                    'rubbing_count': np.random.randint(3, 8),
                    'rubbing_percentage': np.random.uniform(8, 25),
                    'fatigue_label': 2
                }
            
            # Добавляем шум
            for key in features:
                if key != 'fatigue_label':
                    features[key] += np.random.normal(0, features[key] * 0.1)
                    features[key] = max(features[key], 0)
            
            data.append(features)
        
        df = pd.DataFrame(data)
        return df