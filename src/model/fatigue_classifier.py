"""Классификатор усталости: только загрузка и предсказание."""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import joblib

from .model_config import ModelConfig
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class FatigueClassifier:
    """
    Классификатор усталости.
    
    Отвечает ТОЛЬКО за:
    - Загрузку обученной модели и скалера
    - Валидацию входных признаков
    - Предсказание класса и вероятностей
    
    Обучение вынесено в trainer.py
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.model = None
        self.scaler = None
        self._is_loaded = False
        self.feature_extractor = FeatureExtractor(self.cfg)
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Загружает модель, скалер и метаданные.
        
        Returns:
            True если загрузка успешна, иначе False.
        """
        path = Path(model_path) if model_path else self.cfg.default_model_path
        
        try:
            # Загружаем модель и скалер раздельно для гибкости
            self.model = joblib.load(path)
            
            scaler_path = path.with_name("scaler.pkl")
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                # Обратная совместимость: скалер в том же файле
                self.scaler = joblib.load(path.with_name("fatigue_classifier_scaler.pkl"))
            
            # Метаданные (опционально)
            meta_path = path.with_name("meta.json")
            if meta_path.exists():
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    # Можно добавить валидацию версии модели и т.д.
            
            self._is_loaded = True
            logger.info(f"Модель загружена: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._is_loaded = False
            return False
    
    def predict_from_window(self, window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает уровень усталости по окну метрик.
        """
        if not self._is_loaded:
            raise RuntimeError("Модель не загружена. Вызовите load() перед predict.")
        
        # 1. Извлекаем признаки
        features = self.feature_extractor.extract(window)
        
        # 2. Формируем DataFrame с правильными именами колонок (ИСПРАВЛЕНИЕ)
        # Создаем DataFrame с одной строкой, где колонки называются так же, как при обучении
        X = pd.DataFrame([features], columns=self.cfg.expected_features)
        
        # 3. Масштабируем (теперь warning исчезнет, так как имена совпадают)
        X_scaled = self.scaler.transform(X)
        
        # 4. Предсказываем
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return prediction, probabilities
    def predict_from_dict(self, features: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает по готовому словарю признаков (для тестов или API).
        
        Args:
            features: dict с признаками из self.cfg.expected_features
        """
        if not self._is_loaded:
            raise RuntimeError("Модель не загружена.")
        
        # Валидация: все ожидаемые признаки должны присутствовать
        missing = set(self.cfg.expected_features) - set(features.keys())
        if missing:
            logger.warning(f"Отсутствуют признаки: {missing}. Заполняем нулями.")
            for f in missing:
                features[f] = 0.0
        
        X = np.array([[features[f] for f in self.cfg.expected_features]])
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        return self.model.predict(X_scaled), self.model.predict_proba(X_scaled)
    
    @property
    def is_ready(self) -> bool:
        """Готов ли классификатор к предсказаниям."""
        return self._is_loaded and self.model is not None
    
    def get_class_names(self) -> list[str]:
        """Возвращает названия классов для отображения."""
        return self.cfg.class_names