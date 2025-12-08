import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import json
import os
import sys
# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import
from dataset_builder import *


class FatigueClassifier:
    """Классификатор усталости на основе метрик"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Args:
            model_type: 'random_forest', 'xgboost', 'svm', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.class_names = ['Нет/легкая усталость', 'Средняя усталость', 'Сильная усталость']
        self.training_features = []
        
    def load_data(self, dataset_path: str, label_column: str = 'fatigue_label'):
        """Загружает датасет"""
        df = pd.read_csv(dataset_path)
        
        # Определяем признаки и метки
        if label_column in df.columns:
            X = df.drop(columns=[label_column])
            y = df[label_column]
        else:
            # Если нет меток, используем синтетические
            if 'fatigue_label_synthetic' in df.columns:
                X = df.drop(columns=['fatigue_label_synthetic'])
                y = df['fatigue_label_synthetic']
            else:
                raise ValueError(f"Не найдена колонка с метками в датасете")
        
        # Сохраняем имена признаков, которые использовались при обучении
        self.training_features = X.columns.tolist()
        
        # Убираем нечисловые колонки если они есть
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(X.columns):
            print(f"Предупреждение: удалены нечисловые колонки: {set(X.columns) - set(numeric_cols)}")
            X = X[numeric_cols]
            self.training_features = numeric_cols
        
        return X, y
    
    def prepare_features(self, X_train, X_test):
        """Подготавливает признаки (масштабирование)"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train):
        """Обучает модель"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        self.model.fit(X_train, y_train)
    
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Сохраняем имена признаков
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
            self.training_features = X_train.columns.tolist()
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            self.training_features = X_train.columns.tolist()
        elif isinstance(X_train, np.ndarray) and hasattr(self, 'training_features'):
            # Если X_train - numpy array, используем сохраненные имена
            self.feature_names = self.training_features
        else:
            # Создаем имена признаков по умолчанию
            n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
            self.training_features = self.feature_names.copy()
            print(f"Внимание: созданы имена признаков по умолчанию: {self.feature_names[:5]}...")
        
        # Оценка важности признаков
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
        else:
            self.feature_importance = {}
    
    def evaluate(self, X_test, y_test):
        """Оценивает модель"""
        y_pred = self.model.predict(X_test)
        
        print("=== Результаты классификации ===")
        print(f"Точность: {accuracy_score(y_test, y_pred):.3f}")
        print("\nОтчет по классификации:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        print("\nМатрица ошибок:")
        print(confusion_matrix(y_test, y_pred))
        
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"\nКросс-валидация (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'cv_score': cv_scores.mean()
        }
    
    def predict(self, features: pd.DataFrame):
        """Предсказывает класс усталости для новых данных"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        # Проверяем, что все нужные признаки присутствуют
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            print(f"Предупреждение: отсутствуют признаки: {missing_features}")
            # Добавляем недостающие признаки с нулевыми значениями
            for feature in missing_features:
                features[feature] = 0
        
        # Убираем лишние признаки, если они есть
        extra_features = set(features.columns) - set(self.feature_names)
        if extra_features:
            print(f"Предупреждение: удалены лишние признаки: {extra_features}")
            features = features[self.feature_names]
        
        # Упорядочиваем признаки так же, как при обучении
        features = features[self.feature_names]
        
        # Масштабирование признаков
        features_scaled = self.scaler.transform(features)
        
        # Предсказание
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return prediction, probabilities
    
    def predict_realtime(self, metrics_dict: dict):
        """Предсказание в реальном времени на основе метрик"""
        # Преобразуем метрики в DataFrame с одной строкой
        features_df = pd.DataFrame([metrics_dict])
        
        # Обрабатываем признаки
        return self.predict(features_df)
    
    def predict_from_window_metrics(self, window_metrics: pd.DataFrame):
        """
        Предсказание на основе метрик окна
        
        Args:
            window_metrics: DataFrame с метриками за окно
        """
    
        # Извлекаем признаки из окна
        builder = FatigueDatasetBuilder(window_size=len(window_metrics), overlap=0)
        features_dict = builder.extract_features_from_window(window_metrics)
        
        # Удаляем технические поля
        features_dict.pop('window_start', None)
        features_dict.pop('window_end', None)
        
        return self.predict_realtime(features_dict)
    
    def save_model(self, path: str):
        """Сохраняет модель"""
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'class_names': self.class_names,
            'training_features': self.training_features
        }, path)
        
        print(f"Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """Загружает модель"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.feature_importance = data.get('feature_importance', {})
        self.class_names = data.get('class_names', ['Нет/легкая усталость', 'Средняя усталость', 'Сильная усталость'])
        self.training_features = data.get('training_features', [])
        
        print(f"Модель загружена: {path}")

def get_example_features():
    """Возвращает пример признаков, соответствующих реальному датасету"""
    return {
        'blink_count': 5,
        'blink_frequency': 4.5,
        'blink_max_interval': 15,
        'blink_avg_interval': 8.2,
        'head_tilt_max_angle': 25.3,
        'head_tilt_avg_angle': 12.1,
        'head_tilt_std_angle': 8.5,
        'head_tilt_light_percentage': 15.2,
        'head_tilt_heavy_percentage': 5.7,
        'head_tilt_total_events': 3,
        'rubbing_count': 2,
        'rubbing_duration_avg': 2.1,
        'rubbing_percentage': 6.3,
        'face_detected_percentage': 98.5,
        'pose_detected_percentage': 95.2,
        'frames_without_face': 1,
        'activity_score': 0.45,
        'fatigue_score_1': 4.2,
        'fatigue_score_2': 2.8,
        'fatigue_score_3': 1.9,
        'time_sin': -0.258,
        'time_cos': 0.966
    }

def create_simple_synthetic_dataset(n_samples=1000):
    """Создает простой синтетический датасет для тестирования"""
    np.random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Определяем класс усталости
        if i < n_samples // 3:  # Класс 0: Нет/легкая усталость
            label = 0
            blink_count = np.random.randint(1, 4)
            max_tilt = np.random.uniform(0, 15)
            rubbing_count = np.random.randint(0, 1)
            
        elif i < 2 * n_samples // 3:  # Класс 1: Средняя усталость
            label = 1
            blink_count = np.random.randint(3, 7)
            max_tilt = np.random.uniform(15, 30)
            rubbing_count = np.random.randint(1, 3)
            
        else:  # Класс 2: Сильная усталость
            label = 2
            blink_count = np.random.randint(6, 12)
            max_tilt = np.random.uniform(30, 50)
            rubbing_count = np.random.randint(3, 8)
        
        # Генерация остальных признаков с проверкой на ноль
        blink_freq = blink_count * np.random.uniform(0.8, 1.2)
        rubbing_pct = rubbing_count * np.random.uniform(0.5, 1.5)
        heavy_tilt_pct = max_tilt * np.random.uniform(0.2, 0.5)
        
        # Убеждаемся, что значения положительные
        blink_freq = max(blink_freq, 0.1)
        rubbing_pct = max(rubbing_pct, 0.1)
        heavy_tilt_pct = max(heavy_tilt_pct, 0.1)
        
        features = {
            'blink_count': blink_count,
            'blink_frequency': blink_freq,
            'blink_max_interval': np.random.uniform(10, 30) if blink_count > 0 else 5,
            'blink_avg_interval': np.random.uniform(5, 15) if blink_count > 0 else 10,
            'head_tilt_max_angle': max_tilt,
            'head_tilt_avg_angle': max_tilt * np.random.uniform(0.3, 0.7),
            'head_tilt_std_angle': max_tilt * np.random.uniform(0.1, 0.3),
            'head_tilt_light_percentage': heavy_tilt_pct * np.random.uniform(1.5, 3),
            'head_tilt_heavy_percentage': heavy_tilt_pct,
            'head_tilt_total_events': int(blink_count * np.random.uniform(0.5, 1.5)),
            'rubbing_count': rubbing_count,
            'rubbing_duration_avg': np.random.uniform(1, 4) if rubbing_count > 0 else 0,
            'rubbing_percentage': rubbing_pct,
            'face_detected_percentage': np.random.uniform(95, 100),
            'pose_detected_percentage': np.random.uniform(90, 98),
            'frames_without_face': np.random.randint(0, 3),
            'activity_score': np.random.uniform(0.1, 0.8),
            'fatigue_score_1': blink_freq * np.random.uniform(0.8, 1.2),
            'fatigue_score_2': heavy_tilt_pct * 0.5,
            'fatigue_score_3': rubbing_pct * 0.3,
            'fatigue_label': label
        }
        
        # Добавляем небольшой шум, избегая нулевых масштабов
        for key in features:
            if key != 'fatigue_label':
                value = features[key]
                # Минимальный масштаб 0.1 чтобы избежать scale=0
                scale = max(value * 0.1, 0.1)
                features[key] = value + np.random.normal(0, scale)
                features[key] = max(features[key], 0)  # Не отрицательные значения
        
        data.append(features)
    
    df = pd.DataFrame(data)
    
    # Создаем директорию если её нет
    os.makedirs('datasets', exist_ok=True)
    
    # Сохраняем датасет
    df.to_csv("datasets/simple_fatigue_dataset.csv", index=False)
    print(f"Синтетический датасет сохранен: datasets/simple_fatigue_dataset.csv")
    print(f"Размер: {len(df)} samples, Признаков: {len(df.columns)}")
    
    return df

def train_simple_classifier():
    """Пример обучения классификатора на синтетических данных"""
    print("=== Создание синтетического датасета ===")
    
    # Создаем простой датасет
    synthetic_df = create_simple_synthetic_dataset(n_samples=1000)
    
    # Проверяем, какие признаки создались
    print(f"\nПризнаки в датасете: {list(synthetic_df.columns)}")
    print(f"Распределение меток:\n{synthetic_df['fatigue_label'].value_counts()}")
    
    # Инициализируем классификатор
    classifier = FatigueClassifier(model_type='random_forest')
    
    # Разделяем данные
    X = synthetic_df.drop(columns=['fatigue_label'])
    y = synthetic_df['fatigue_label']
    
    print(f"\nРазмерность данных: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Подготавливаем признаки (теперь сохраняем масштабированные версии)
    X_train_scaled, X_test_scaled = classifier.prepare_features(X_train, X_test)
    
    # ИСПРАВЛЕНИЕ: Сохраняем имена признаков ДО масштабирования
    classifier.feature_names = X_train.columns.tolist()
    classifier.training_features = X_train.columns.tolist()
    
    # Обучаем модель на масштабированных данных
    print("\n=== Обучение модели ===")
    classifier.train(X_train_scaled, y_train)
    
    # Оцениваем
    print("\n=== Оценка модели ===")
    results = classifier.evaluate(X_test_scaled, y_test)
    
    # Сохраняем модель
    classifier.save_model("models/fatigue_classifier.pkl")
    
    # Выводим важность признаков
    if classifier.feature_importance:
        print("\n=== Топ-10 важных признаков ===")
        for feature, importance in sorted(
            classifier.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            print(f"{feature}: {importance:.4f}")
    
    return classifier, X.columns.tolist()


def main():
    """Основная функция"""
    print("=" * 60)
    print("СИСТЕМА КЛАССИФИКАЦИИ УСТАЛОСТИ")
    print("=" * 60)
    
    # 1. Создаем и обучаем модель
    print("\n[1] Создание и обучение модели")
    classifier, feature_names = train_simple_classifier()

    print("\n" + "=" * 60)
    print("Модель готова к использованию!")
    print("Файлы созданы:")
    print("  - models/fatigue_classifier.pkl (обученная модель)")
    print("  - datasets/simple_fatigue_dataset.csv (синтетические данные)")
    print("=" * 60)
    
    return classifier

if __name__ == "__main__":
    # Запускаем основную функцию
    classifier = main()