"""Lightweight online adaptation layer for fatigue prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OnlineFatigueLearner:
    def __init__(
        self,
        feature_names: Sequence[str],
        model_dir: str | Path = "models",
        min_teacher_confidence: float = 0.80,
        blend_weight: float = 0.25,
    ):
        self.feature_names = list(feature_names)
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "online_fatigue_model.pkl"
        self.scaler_path = self.model_dir / "online_fatigue_scaler.pkl"
        self.min_teacher_confidence = min_teacher_confidence
        self.blend_weight = blend_weight
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
        self._is_fitted = False
        self._update_count = 0
        self._classes = np.array([0, 1, 2], dtype=np.int32)
        self._load()

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([[float(features.get(name, 0.0)) for name in self.feature_names]], dtype=np.float64)

    def _load(self) -> None:
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self._is_fitted = True
                logger.info("Online learner restored from disk")
        except Exception as exc:
            logger.warning(f"Failed to load online learner state: {exc}")

    def update(self, features: Dict[str, float], teacher_label: int, teacher_confidence: float) -> bool:
        if teacher_confidence < self.min_teacher_confidence:
            return False

        X = self._vectorize(features)
        y = np.array([teacher_label], dtype=np.int32)

        if not self._is_fitted:
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y, classes=self._classes)
            self._is_fitted = True
        else:
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.partial_fit(X_scaled, y)

        self._update_count += 1
        if self._update_count % 20 == 0:
            self.save()
        return True

    def predict(self, features: Dict[str, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._is_fitted:
            return None

        X = self._vectorize(features)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        pred = np.argmax(probs, axis=1)
        return pred, probs

    def blend(self, base_probs: Optional[np.ndarray], online_probs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if base_probs is None and online_probs is None:
            return None
        if base_probs is None:
            return online_probs
        if online_probs is None:
            return base_probs
        return (1.0 - self.blend_weight) * base_probs + self.blend_weight * online_probs

    def save(self) -> None:
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as exc:
            logger.warning(f"Failed to save online learner state: {exc}")

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    @property
    def update_count(self) -> int:
        return self._update_count
