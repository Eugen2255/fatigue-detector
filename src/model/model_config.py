"""Конфигурация модели: признаки, пути, гиперпараметры."""
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

def get_base_path() -> Path:
    """Возвращает корневую папку: для PyInstaller или разработки."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).resolve().parent.parent

@dataclass
class ModelConfig:
    """Конфигурация классификатора усталости."""
    
    # Ожидаемые признаки (строго в этом порядке!)
    expected_features: List[str] = field(default_factory=lambda: [
        'blink_count', 'blink_frequency', 'blink_max_interval', 'blink_avg_interval',
        'yawn_count', 'yawn_pct', 'perclos_pct',
        'head_tilt_max_angle', 'head_tilt_avg_angle', 'head_tilt_std_angle',
        'head_tilt_light_pct', 'head_tilt_heavy_pct', 'head_tilt_events',
        'rubbing_count', 'rubbing_duration_avg', 'rubbing_pct',
        'key_press_count', 'key_press_rate', 'active_keys_avg', 'keyboard_burst_pct',
        'mouse_click_count', 'mouse_click_rate', 'mouse_distance_total', 'mouse_speed_avg',
        'mouse_active_pct', 'idle_pct', 'idle_avg_sec',
        'face_detected_pct', 'pose_detected_pct', 'frames_no_face',
        'activity_score', 'input_activity_score', 'fatigue_score_1', 'fatigue_score_2', 'fatigue_score_3',
        'time_sin', 'time_cos'
    ])
    
    class_names: List[str] = field(default_factory=lambda: [
        'Norm/Light', 'Medium', 'Strong'
    ])
    
    model_dir: Path = field(init=False)
    default_model_path: Path = field(init=False)
    default_scaler_path: Path = field(init=False)
    default_meta_path: Path = field(init=False)
    
    model_type: str = "random_forest"
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42
    })
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42
    })
    
    def __post_init__(self):
        """Инициализация путей после создания экземпляра."""
        base = get_base_path()
        self.model_dir = base / "models"
        self.default_model_path = self.model_dir / "fatigue_classifier.pkl"
        self.default_scaler_path = self.model_dir / "scaler.pkl"
        self.default_meta_path = self.model_dir / "meta.json"