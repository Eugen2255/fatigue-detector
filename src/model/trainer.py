"""Обучение и оценка моделей классификации усталости."""
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

from .model_config import ModelConfig
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class FatigueTrainer:
    """Пайплайн обучения классификатора усталости."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_extractor = FeatureExtractor(self.cfg)
    
    def load_dataset(self, path: str, label_col: str = "fatigue_label") -> Tuple[pd.DataFrame, pd.Series]:
        """Загружает датасет и разделяет на признаки/метки."""
        df = pd.read_csv(path)
        
        if label_col not in df.columns:
            raise ValueError(f"Колонка '{label_col}' не найдена в датасете")
        
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Оставляем только числовые признаки
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < len(X.columns):
            logger.warning(f"Удалены нечисловые колонки: {set(X.columns) - set(numeric)}")
            X = X[numeric]
        
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Разделяет и масштабирует данные."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_model(self, model_type: Optional[str] = None):
        """Создаёт модель согласно конфигурации."""
        mtype = model_type or self.cfg.model_type
        
        if mtype == "random_forest":
            self.model = RandomForestClassifier(**self.cfg.rf_params)
        elif mtype == "xgboost":
            self.model = xgb.XGBClassifier(**self.cfg.xgb_params, eval_metric="mlogloss")
        elif mtype == "svm":
            self.model = SVC(C=1.0, kernel="rbf", probability=True, 
                           class_weight="balanced", random_state=42)
        elif mtype == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {mtype}")
        
        logger.info(f"Создана модель: {mtype}")
    
    def train(self, X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
        """Обучает модель и возвращает метрики."""
        if self.model is None:
            self.create_model()
        
        self.model.fit(X_train, y_train)
        
        # Важность признаков (если доступна)
        importance = {}
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.cfg.expected_features, self.model.feature_importances_))
        
        return {"feature_importance": importance}
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """Оценивает модель на тестовых данных."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test) if hasattr(self.model, "predict_proba") else None
        
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        report = classification_report(y_test, y_pred, 
                                      target_names=self.cfg.class_names, 
                                      output_dict=True)
        
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        logger.info(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "probabilities_sample": y_proba[0].tolist() if y_proba is not None else None
        }
    
    def save(self, output_dir: str):
        """Сохраняет модель, скалер и метаданные."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Сохраняем модель и скалер через joblib
        joblib.dump(self.model, out_path / "fatigue_classifier.pkl")
        joblib.dump(self.scaler, out_path / "scaler.pkl")
        
        # 2. Конвертируем numpy-массив в стандартный Python list для JSON
        raw_importance = getattr(self.model, "feature_importances_", None)
        importance_list = raw_importance.tolist() if raw_importance is not None else None
        
        meta = {
            "model_type": self.cfg.model_type,
            "features": self.cfg.expected_features,
            "class_names": self.cfg.class_names,
            "feature_importance": importance_list
        }
        
        # 3. Записываем JSON
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            
        logger.info(f"✅ Модель сохранена в {out_path}")