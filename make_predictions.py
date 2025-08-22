import pandas as pd
import joblib
from pathlib import Path

# 1. Загрузка модели и данных
model = joblib.load("model.pkl")
test_data = pd.read_csv('heart_test.csv')

# 2. Проверка колонок
required_columns = model.feature_names_in_  # Колонки, которые ожидает модель
missing_columns = set(required_columns) - set(test_data.columns)

if missing_columns:
    raise ValueError(f"В тестовых данных отсутствуют колонки: {missing_columns}")

# 3. Создание предсказаний (0 или 1)
probabilities = model.predict_proba(test_data[required_columns])[:, 1]
predictions = (probabilities >= 0.5).astype(int)  # если >=0.5 -> 1, иначе 0

# 4. Сохранение результатов
output = pd.DataFrame({
    "id": test_data["id"],  # Используем id из тестовых данных
    "prediction": predictions  # Чёткие предсказания 0 или 1
})

output.to_csv("student_predictions.csv", index=False)
print("Предсказания сохранены в student_predictions.csv")