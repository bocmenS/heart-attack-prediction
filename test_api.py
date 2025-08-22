import requests
import pandas as pd
import io

def test_predict():
    url = "http://localhost:8000/upload/"

    # Загружаем CSV и удаляем строку
    data_test = pd.read_csv('heart_test.csv')
    data_test = data_test.drop('Unnamed: 0', axis=1)

    binary_cols = ['Diabetes', 'Family History', 'Smoking', 'Obesity',
                   'Alcohol Consumption', 'Previous Heart Problems', 'Medication Use']
    data_test[binary_cols] = data_test[binary_cols].fillna(data_test[binary_cols].mode().iloc[0])
    num_cols = ['Stress Level', 'Physical Activity Days Per Week']
    data_test[num_cols] = data_test[num_cols].fillna(data_test[num_cols].median())

    data_test[binary_cols] = data_test[binary_cols].fillna(0).astype(int)

    # Конвертируем обратно в CSV для отправки
    csv_bytes = io.BytesIO()
    data_test.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    files = {"file": open('heart_test.csv', "rb")}

    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("Успешно! Результат:")
        print(response.json())
    else:
        print("Ошибка:", response.text)


if __name__ == "__main__":
    test_predict()
