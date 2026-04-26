"""Скрипт для конвертации модели sklearn/xgb в ONNX."""
import logging
import numpy as np
from pathlib import Path
import joblib
import json

from model.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_model(model_path: str, output_path: str):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.error("Установи: pip install skl2onnx onnx")
        return

    model = joblib.load(model_path)
    feature_count = len(ModelConfig().expected_features)
    meta_path = Path(model_path).with_name("meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        feature_count = len(meta.get("features") or ModelConfig().expected_features)
    
    # Определяем начальное состояние типов по актуальной схеме признаков
    initial_type = [('input', FloatTensorType([None, feature_count]))]
    
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"✅ Модель конвертирована в {output_path}")
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}. Возможно, модель не sklearn (например, XGBoost). Попробуй onnxmltools.")

if __name__ == "__main__":
    convert_model("models/fatigue_classifier.pkl", "models/fatigue_classifier.onnx")
