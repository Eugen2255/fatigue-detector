"""
Скрипт обучения классификатора усталости.
Поддерживает загрузку реального CSV или генерацию синтетических данных.
"""
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from model.model_config import ModelConfig
from model.trainer import FatigueTrainer
from model.synthetic_data import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def align_features(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Выравнивает колонки датасета под ожидаемую схему модели."""
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    
    if missing:
        logger.warning(f"⚠ Отсутствуют признаки: {missing}. Заполняем нулями.")
        for col in missing:
            df[col] = 0.0
            
    if extra:
        logger.info(f"ℹ Удаляем лишние признаки: {extra}")
        
    return df[expected]

def train_pipeline(
    dataset_path: str,
    label_col: str = "fatigue_label",
    model_type: str = "random_forest",
    output_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42
) -> dict:
    """Полный пайплайн: загрузка → подготовка → обучение → оценка → сохранение."""
    cfg = ModelConfig(model_type=model_type)
    trainer = FatigueTrainer(cfg)

    # 1. Загрузка данных
    logger.info(f"📥 Загрузка датасета: {dataset_path}")
    X, y = trainer.load_dataset(dataset_path, label_col=label_col)
    logger.info(f"✅ Загружено: {X.shape[0]} строк, {X.shape[1]} признаков")
    
    # 2. Выравнивание признаков (критично для совместимости!)
    X = align_features(X, cfg.expected_features)

    # 3. Разделение и масштабирование
    logger.info("🔪 Разделение на train/test и масштабирование...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=test_size)

    # 4. Создание и обучение модели
    logger.info(f"🧠 Создание модели: {model_type}")
    trainer.create_model()
    
    logger.info("⏳ Обучение...")
    train_info = trainer.train(X_train, y_train)
    
    # 5. Оценка
    logger.info("📊 Оценка на тестовой выборке...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # 6. Сохранение
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out_path))
    
    # 7. Вывод важности признаков
    importance = train_info.get("feature_importance", {})
    if importance:
        top10 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("🏆 Топ-10 важных признаков:")
        for feat, imp in top10:
            logger.info(f"   {feat}: {imp:.4f}")
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Обучение классификатора усталости")
    parser.add_argument("--data", type=str, default=None, 
                        help="Путь к CSV датасету. Если не указан, генерируется синтетический.")
    parser.add_argument("--label", type=str, default="fatigue_label", 
                        help="Имя колонки с метками (0, 1, 2)")
    parser.add_argument("--model", type=str, default="random_forest", 
                        choices=["random_forest", "xgboost", "svm", "gradient_boosting"])
    parser.add_argument("--output", type=str, default="models", 
                        help="Директория для сохранения модели")
    parser.add_argument("--synthetic", type=int, default=1000, 
                        help="Образцов на класс при генерации синтетических данных")
    parser.add_argument("--test-size", type=float, default=0.2, 
                        help="Доля данных для тестирования")
    
    args = parser.parse_args()
    
    # Если реальный датасет не указан → генерируем синтетический
    if args.data is None:
        logger.info("📊 Реальный датасет не указан. Генерация синтетических данных...")
        gen = SyntheticDataGenerator(ModelConfig())
        tmp_dir = Path("datasets")
        tmp_dir.mkdir(exist_ok=True)
        data_path = str(tmp_dir / "synthetic_train.csv")
        gen.generate_balanced(samples_per_class=args.synthetic, output_path=data_path, seed=42)
        args.data = data_path
        
    # Запуск пайплайна
    metrics = train_pipeline(
        dataset_path=args.data,
        label_col=args.label,
        model_type=args.model,
        output_dir=args.output,
        test_size=args.test_size
    )
    
    logger.info(f"\n✅ Обучение завершено! Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"📁 Файлы сохранены в: {args.output}/")

if __name__ == "__main__":
    main()