"""
Генерация синтетических данных для обучения классификатора усталости.

Использует ту же схему признаков, что и ModelConfig.expected_features,
что гарантирует совместимость с пайплайном обучения и предсказания.

Пример использования:
    >>> from model.synthetic_data import SyntheticDataGenerator, ModelConfig
    >>> cfg = ModelConfig()
    >>> generator = SyntheticDataGenerator(cfg)
    >>> df = generator.generate(n_samples=3000, seed=42)
    >>> df.to_csv("datasets/synthetic_fatigue.csv", index=False)
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from model.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class FatigueProfile:
    """Параметры генерации для одного класса усталости."""
    # Базовые диапазоны признаков
    blink_count: Tuple[int, int] = (1, 4)
    blink_interval: Tuple[float, float] = (8.0, 20.0)
    tilt_max_angle: Tuple[float, float] = (0.0, 15.0)
    tilt_avg_ratio: Tuple[float, float] = (0.3, 0.6)
    tilt_std_ratio: Tuple[float, float] = (0.1, 0.25)
    tilt_light_pct: Tuple[float, float] = (5.0, 20.0)
    tilt_heavy_pct: Tuple[float, float] = (0.0, 5.0)
    tilt_events: Tuple[int, int] = (0, 3)
    rubbing_count: Tuple[int, int] = (0, 1)
    rubbing_duration: Tuple[float, float] = (0.5, 2.0)
    rubbing_pct: Tuple[float, float] = (0.0, 3.0)
    
    # Качество детекции
    face_detected_pct: Tuple[float, float] = (95.0, 100.0)
    pose_detected_pct: Tuple[float, float] = (90.0, 98.0)
    
    # Агрегированные признаки (будут пересчитаны, но можно задать шум)
    activity_score: Tuple[float, float] = (0.1, 0.4)
    fatigue_score_base: Tuple[float, float] = (1.0, 3.0)
    
    # Шум (стандартное отклонение для нормального распределения)
    noise_scale: float = 0.15


class SyntheticDataGenerator:
    """Генератор синтетических данных для классификации усталости."""
    
    # Профили для трёх классов: 0=Norm, 1=Medium, 2=Strong
    PROFILES: Dict[int, FatigueProfile] = {
        0: FatigueProfile(
            blink_count=(1, 4),
            blink_interval=(8.0, 20.0),
            tilt_max_angle=(0.0, 15.0),
            tilt_light_pct=(5.0, 20.0),
            tilt_heavy_pct=(0.0, 3.0),
            rubbing_count=(0, 1),
            rubbing_pct=(0.0, 3.0),
            activity_score=(0.1, 0.4),
            fatigue_score_base=(1.0, 3.0),
            noise_scale=0.12
        ),
        1: FatigueProfile(
            blink_count=(4, 8),
            blink_interval=(4.0, 12.0),
            tilt_max_angle=(15.0, 30.0),
            tilt_light_pct=(15.0, 35.0),
            tilt_heavy_pct=(3.0, 12.0),
            rubbing_count=(1, 3),
            rubbing_pct=(3.0, 10.0),
            activity_score=(0.3, 0.7),
            fatigue_score_base=(3.0, 6.0),
            noise_scale=0.15
        ),
        2: FatigueProfile(
            blink_count=(7, 15),
            blink_interval=(2.0, 7.0),
            tilt_max_angle=(30.0, 55.0),
            tilt_light_pct=(25.0, 50.0),
            tilt_heavy_pct=(10.0, 30.0),
            rubbing_count=(3, 10),
            rubbing_pct=(8.0, 25.0),
            activity_score=(0.6, 1.0),
            fatigue_score_base=(5.0, 10.0),
            noise_scale=0.18
        ),
    }
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.cfg = config or ModelConfig()
        self._rng: Optional[np.random.Generator] = None
    
    def _get_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """Возвращает генератор случайных чисел с опциональным seed."""
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        return self._rng
    
    def _sample_range(self, rng: np.random.Generator, bounds: Tuple[float, float]) -> float:
        """Равномерная выборка из диапазона [low, high]."""
        return rng.uniform(bounds[0], bounds[1])
    
    def _sample_int_range(self, rng: np.random.Generator, bounds: Tuple[int, int]) -> int:
        """Равномерная выборка целого числа из [low, high] включительно."""
        return rng.integers(bounds[0], bounds[1] + 1)
    
    def _add_noise(self, rng: np.random.Generator, value: float, scale: float) -> float:
        """Добавляет гауссов шум с относительным масштабом."""
        noise = rng.normal(0, abs(value) * scale + 0.01)  # +0.01 чтобы не было 0 шума при value=0
        return value + noise
    
    def _generate_single_sample(self, rng: np.random.Generator, label: int) -> Dict[str, float]:
        """Генерирует одну строку данных для заданного класса."""
        profile = self.PROFILES[label]
        
        # === Базовые признаки ===
        blink_count = self._sample_int_range(rng, profile.blink_count)
        blink_freq = blink_count * self._sample_range(rng, (0.8, 1.2))  # нормализация к 30 FPS
        
        # Интервалы морганий (обратная зависимость от частоты)
        if blink_count >= 2:
            max_interval = self._sample_range(rng, profile.blink_interval)
            avg_interval = max_interval * self._sample_range(rng, (0.4, 0.8))
        else:
            max_interval = self._sample_range(rng, (15.0, 30.0))
            avg_interval = max_interval * 0.6
        
        # Наклон головы
        tilt_max = self._sample_range(rng, profile.tilt_max_angle)
        tilt_avg = tilt_max * self._sample_range(rng, profile.tilt_avg_ratio)
        tilt_std = tilt_max * self._sample_range(rng, profile.tilt_std_ratio)
        
        tilt_light = self._sample_range(rng, profile.tilt_light_pct)
        tilt_heavy = self._sample_range(rng, profile.tilt_heavy_pct)
        # Нормализуем чтобы сумма не превышала 100%
        total_tilt = tilt_light + tilt_heavy
        if total_tilt > 95:
            tilt_light = tilt_light / total_tilt * 90
            tilt_heavy = tilt_heavy / total_tilt * 90
        
        tilt_events = self._sample_int_range(rng, profile.tilt_events)
        
        # Потирание глаз
        rubbing_count = self._sample_int_range(rng, profile.rubbing_count)
        rubbing_dur = self._sample_range(rng, profile.rubbing_duration) if rubbing_count > 0 else 0.0
        rubbing_pct = self._sample_range(rng, profile.rubbing_pct)
        
        # Детекция
        face_pct = self._sample_range(rng, profile.face_detected_pct)
        pose_pct = self._sample_range(rng, profile.pose_detected_pct)
        frames_no_face = max(0, int((100 - face_pct) / 100 * 30))  # примерное кол-во кадров без лица за окно
        
        # === Агрегированные признаки (с корреляциями) ===
        # activity_score: комбинация частоты морганий, наклона и потирания
        activity = (
            min(blink_freq / 10, 1.0) * 0.4 +
            min(tilt_max / 50, 1.0) * 0.35 +
            min(rubbing_pct / 25, 1.0) * 0.25
        )
        
        # fatigue_score_*: производные признаки для модели
        fatigue_base = self._sample_range(rng, profile.fatigue_score_base)
        fatigue_1 = blink_freq * 0.5 + rng.normal(0, 0.3)
        fatigue_2 = tilt_heavy * 0.6 + rng.normal(0, 0.4)
        fatigue_3 = rubbing_pct * 0.4 + rng.normal(0, 0.2)
        
        # Временные признаки (синусоида от "времени" в окне)
        time_phase = rng.uniform(0, 2 * np.pi)
        time_sin = np.sin(time_phase)
        time_cos = np.cos(time_phase)
        
        # === Сбор признаков ===
        features = {
            'blink_count': blink_count,
            'blink_frequency': max(0.1, blink_freq),
            'blink_max_interval': max(1.0, max_interval),
            'blink_avg_interval': max(1.0, avg_interval),
            'head_tilt_max_angle': max(0.0, tilt_max),
            'head_tilt_avg_angle': max(0.0, tilt_avg),
            'head_tilt_std_angle': max(0.0, tilt_std),
            'head_tilt_light_pct': max(0.0, tilt_light),
            'head_tilt_heavy_pct': max(0.0, tilt_heavy),
            'head_tilt_events': tilt_events,
            'rubbing_count': rubbing_count,
            'rubbing_duration_avg': max(0.0, rubbing_dur),
            'rubbing_pct': max(0.0, rubbing_pct),
            'face_detected_pct': min(100.0, max(0.0, face_pct)),
            'pose_detected_pct': min(100.0, max(0.0, pose_pct)),
            'frames_no_face': frames_no_face,
            'activity_score': min(1.0, max(0.0, activity)),
            'fatigue_score_1': max(0.0, fatigue_1),
            'fatigue_score_2': max(0.0, fatigue_2),
            'fatigue_score_3': max(0.0, fatigue_3),
            'time_sin': time_sin,
            'time_cos': time_cos,
            'fatigue_label': label  # целевая переменная
        }
        
        # === Добавление шума ко всем числовым признакам ===
        noise_scale = profile.noise_scale
        for key in features:
            if key != 'fatigue_label' and isinstance(features[key], (int, float)):
                noisy = self._add_noise(rng, float(features[key]), noise_scale)
                # Ограничиваем физически возможные значения
                if 'pct' in key or 'score' in key:
                    features[key] = min(100.0, max(0.0, noisy))
                elif 'angle' in key:
                    features[key] = max(0.0, noisy)
                elif 'count' in key or 'events' in key or 'frames' in key:
                    features[key] = max(0, int(round(noisy)))
                else:
                    features[key] = max(0.0, noisy)
        
        return features
    
    def generate(
        self,
        n_samples: int = 3000,
        class_distribution: Optional[Dict[int, float]] = None,
        seed: Optional[int] = 42,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Генерирует синтетический датасет.
        
        Args:
            n_samples: Общее количество строк.
            class_distribution: dict {label: proportion}, сумма должна быть 1.0.
                               По умолчанию: равномерное распределение [0.33, 0.33, 0.34].
            seed: Seed для воспроизводимости.
            output_path: Если указан, сохраняет DataFrame в CSV.
        
        Returns:
            pd.DataFrame с признаками и меткой 'fatigue_label'.
        """
        rng = self._get_rng(seed)
        
        # Распределение классов
        if class_distribution is None:
            class_distribution = {0: 0.33, 1: 0.33, 2: 0.34}
        
        # Валидация
        total = sum(class_distribution.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Сумма class_distribution должна быть 1.0, получено {total}")
        
        # Генерация
        samples: List[Dict[str, float]] = []
        for label, proportion in class_distribution.items():
            n_class = int(round(n_samples * proportion))
            logger.info(f"Генерация класса {label} ({self.cfg.class_names[label]}): {n_class} образцов")
            
            for _ in range(n_class):
                sample = self._generate_single_sample(rng, label)
                samples.append(sample)
        
        # Создание DataFrame
        df = pd.DataFrame(samples)
        
        # Валидация схемы
        expected = set(self.cfg.expected_features) | {'fatigue_label'}
        actual = set(df.columns)
        if expected != actual:
            missing = expected - actual
            extra = actual - expected
            if missing:
                logger.warning(f"⚠ Отсутствуют признаки: {missing}")
            if extra:
                logger.warning(f"⚠ Лишние признаки: {extra}")
        
        # Сохранение
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"💾 Датасет сохранён: {path} ({len(df)} строк, {len(df.columns)} признаков)")
        
        return df
    
    def generate_balanced(
        self,
        samples_per_class: int = 1000,
        seed: Optional[int] = 42,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Удобный метод для генерации сбалансированного датасета."""
        return self.generate(
            n_samples=samples_per_class * 3,
            class_distribution={0: 1/3, 1: 1/3, 2: 1/3},
            seed=seed,
            output_path=output_path
        )


# ===== Утилиты для быстрого тестирования =====
def create_quick_dataset(
    n_per_class: int = 500,
    output_dir: str = "datasets",
    seed: int = 42
) -> Path:
    """
    Создаёт сбалансированный синтетический датасет "в один вызов".
    
    Returns:
        Путь к сохранённому файлу.
    """
    cfg = ModelConfig()
    generator = SyntheticDataGenerator(cfg)
    
    output_path = Path(output_dir) / "synthetic_fatigue_balanced.csv"
    df = generator.generate_balanced(
        samples_per_class=n_per_class,
        seed=seed,
        output_path=str(output_path)
    )
    
    # Краткая статистика
    logger.info(f"\n📊 Статистика датасета:")
    logger.info(f"   Размер: {len(df)} × {len(df.columns)}")
    logger.info(f"   Распределение меток:\n{df['fatigue_label'].value_counts().sort_index()}")
    logger.info(f"   Пример признаков: {list(df.columns)[:5]}...")
    
    return output_path


def preview_feature_correlations(df: pd.DataFrame, top_n: int = 10):
    """
    Выводит топ корреляций признаков с целевой переменной.
    
    Args:
        df: DataFrame с колонкой 'fatigue_label'.
        top_n: Сколько признаков показать.
    """
    if 'fatigue_label' not in df.columns:
        logger.warning("Нет колонки 'fatigue_label' для анализа корреляций")
        return
    
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['fatigue_label'])
    correlations = numeric.corrwith(df['fatigue_label']).abs().sort_values(ascending=False)
    
    logger.info(f"\n🔗 Топ-{top_n} корреляций с fatigue_label:")
    for feat, corr in correlations.head(top_n).items():
        logger.info(f"   {feat}: {corr:.3f}")


# ===== CLI для запуска из терминала =====
def main():
    """Запуск генерации через `python -m model.synthetic_data`."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Генерация синтетических данных для fatigue classifier")
    parser.add_argument("--samples", type=int, default=1000, help="Образцов на класс (по умолчанию: 1000)")
    parser.add_argument("--output", type=str, default="datasets/synthetic_fatigue.csv", help="Путь для сохранения")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--preview", action="store_true", help="Показать статистику и корреляции после генерации")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Генерация
    path = create_quick_dataset(
        n_per_class=args.samples,
        output_dir=Path(args.output).parent,
        seed=args.seed
    )
    
    # Предпросмотр
    if args.preview:
        df = pd.read_csv(path)
        preview_feature_correlations(df)
    
    print(f"\n✅ Готово: {path}")


if __name__ == "__main__":
    main()