"""Классификатор усталости с поддержкой ONNX."""
import logging
import json
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .model_config import ModelConfig
from .feature_extractor import FeatureExtractor
from .online_learner import OnlineFatigueLearner

logger = logging.getLogger(__name__)

class FatigueClassifier:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.model = None
        self.onnx_session = None
        self.scaler = None
        self._is_loaded = False
        self.feature_extractor = FeatureExtractor(self.cfg)
        self.active_features = list(self.cfg.expected_features)
        self.online_learner = OnlineFatigueLearner(self.active_features, model_dir=self.cfg.model_dir)
    
    def load(self, model_path: Optional[str] = None) -> bool:
        path = Path(model_path) if model_path else self.cfg.default_model_path
        onnx_path = path.with_suffix(".onnx")
        meta_path = path.with_name("meta.json")
        
        try:
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                loaded_features = meta.get("features") or []
                if loaded_features:
                    self.active_features = loaded_features
                    self.online_learner = OnlineFatigueLearner(self.active_features, model_dir=self.cfg.model_dir)

            # Приоритет ONNX
            if ONNX_AVAILABLE and onnx_path.exists():
                self.onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                self.input_name = self.onnx_session.get_inputs()[0].name
                logger.info(f"Модель загружена через ONNX: {onnx_path}")
            elif path.exists():
                # Фоллбэк на Joblib (sklearn/xgb)
                self.model = joblib.load(path)
                logger.info(f"Модель загружена через Joblib: {path}")
            else:
                return False
            
            # Загрузка скалера
            scaler_path = path.with_name("scaler.pkl")
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            self._is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            self._is_loaded = False
            return False
    
    def predict_from_window(self, window: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = self.feature_extractor.extract(window)
        X = pd.DataFrame([features], columns=self.active_features)

        base_probs = None
        if self._is_loaded:
            X_scaled = self.scaler.transform(X) if self.scaler is not None else X.values
            if self.onnx_session:
                # ONNX inference
                outputs = self.onnx_session.run(None, {self.input_name: X_scaled.astype(np.float32)})
                base_probs = self._extract_probabilities(outputs)
            else:
                # Sklearn inference
                base_probs = self.model.predict_proba(X_scaled)

        online_result = self.online_learner.predict(features)
        online_probs = online_result[1] if online_result is not None else None
        probs = self.online_learner.blend(base_probs, online_probs)

        if probs is None:
            raise RuntimeError("Neither base model nor online learner is ready.")

        prediction = np.argmax(probs, axis=1)
        return prediction, probs

    def update_online_from_window(self, window: pd.DataFrame, teacher_label: int, teacher_confidence: float) -> bool:
        features = self.feature_extractor.extract(window)
        return self.online_learner.update(features, teacher_label, teacher_confidence)

    def _extract_probabilities(self, outputs) -> np.ndarray:
        for output in outputs:
            if isinstance(output, np.ndarray) and output.ndim == 2:
                return output
            if isinstance(output, list) and output and isinstance(output[0], dict):
                class_ids = sorted({int(key) for row in output for key in row.keys()})
                probs = np.zeros((len(output), len(class_ids)), dtype=np.float32)
                for i, row in enumerate(output):
                    for key, value in row.items():
                        probs[i, int(key)] = float(value)
                return probs
        return np.asarray(outputs[0], dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        return self._is_loaded or self.online_learner.is_ready
