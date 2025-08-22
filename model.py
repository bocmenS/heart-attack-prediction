import joblib
import pandas as pd
import os

class Model:
    def __init__(self, prob: float = 0.5):
        super().__init__()
        # загружаем модель из файла
        self.model = joblib.load("model.pkl")
        self.prob = prob

    def __call__(self, csv_path: str):
        # читаем данные
        df = pd.read_csv(csv_path)

        # получаем вероятности
        preds_proba = self.model.predict_proba(df)[:, 1]

        # преобразуем в 0/1 по порогу
        preds = (preds_proba >= self.prob).astype(int)

        # сохраняем результат
        os.makedirs("tmp", exist_ok=True)
        result_path = "tmp/submission.csv"
        df_out = pd.DataFrame({
            "id": df.index,
            "prediction": preds
        })
        df_out.to_csv(result_path, index=False)

        return "OK", result_path
