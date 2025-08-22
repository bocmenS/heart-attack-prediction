from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import joblib
import pandas as pd
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

# --- Инициализация приложения ---
app = FastAPI(title="Heart Attack Risk Predictor")

# Создаем временную директорию для загрузок
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Инициализация шаблонов
templates = Jinja2Templates(directory="templates")

# --- Загрузка модели с обработкой ошибок ---
MODEL_PATH = "model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

predictor = load_model()

# --- Вспомогательные функции ---
def save_upload_file(upload_file: UploadFile, destination: Path):
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def cleanup_file(file_path: Path):
    if file_path.exists():
        file_path.unlink()

# --- Маршруты ---
@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Проверка типа файла
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, detail="Only CSV files are supported")

    try:
        # Создаем временный файл
        temp_file = UPLOAD_DIR / f"temp_{file.filename}"
        save_upload_file(file, temp_file)

        # Чтение и проверка данных
        try:
            df = pd.read_csv(temp_file)
            if df.empty:
                raise HTTPException(400, detail="Uploaded file is empty")
        except Exception as e:
            raise HTTPException(400, detail=f"Error reading CSV: {str(e)}")

        # Предсказание
        try:
            proba = predictor.predict_proba(df)[:, 1]
            predictions = (proba >= 0.5).astype(int)  # бинаризация
            df['prediction'] = predictions
        except Exception as e:
            raise HTTPException(400, detail=f"Prediction error: {str(e)}")

        # Сохранение результата
        result_file = UPLOAD_DIR / f"result_{file.filename}"
        df.to_csv(result_file, index=False)

        return {
            "status": "success",
            "filename": file.filename,
            "result_file": str(result_file),
            "download_url": f"/download/{result_file.name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Internal server error: {str(e)}")
    finally:
        cleanup_file(temp_file)

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, detail="File not found")
    return FileResponse(file_path, filename=filename)


@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Запуск сервера ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



