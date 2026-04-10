"""Конфигурация модели: признаки, пути, гиперпараметры."""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class ModelConfig:
    """Конфигурация классификатора усталости."""
    
    # Ожидаемые признаки (строго в этом порядке!)
    expected_features: List[str] = field(default_factory=lambda: [
        'blink_count', 'blink_frequency', 'blink_max_interval', 'blink_avg_interval',
        'head_tilt_max_angle', 'head_tilt_avg_angle', 'head_tilt_std_angle',
        'head_tilt_light_pct', 'head_tilt_heavy_pct', 'head_tilt_events',
        'rubbing_count', 'rubbing_duration_avg', 'rubbing_pct',
        'face_detected_pct', 'pose_detected_pct', 'frames_no_face',
        'activity_score', 'fatigue_score_1', 'fatigue_score_2', 'fatigue_score_3',
        'time_sin', 'time_cos'
    ])
    
    # Классы для предсказания
    class_names: List[str] = field(default_factory=lambda: [
        'Norm/Light', 'Medium', 'Strong'
    ])
    
    

    # Пути по умолчанию
    model_dir: Path = Path("models")
    default_model_path: Path = model_dir / "fatigue_classifier.pkl"
    default_scaler_path: Path = model_dir / "scaler.pkl"
    default_meta_path: Path = model_dir / "meta.json"
    
    # Гиперпараметры для обучения (используются в trainer.py)
    model_type: str = "random_forest"
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 10, "class_weight": "balanced", "random_state": 42
    })
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42
    })