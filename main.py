from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import subprocess
import io
from typing import Dict, Any, List
import logging
import pickle
from models import ModelManager
import dvc.api
import uvicorn

import os
from dvc.repo import Repo

app = FastAPI()
model_manager = ModelManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelManager")


from minio import Minio
from minio.error import S3Error

# Инициализация клиента MinIO
minio_client = Minio(
    "minio:9000",  # Имя контейнера MinIO и порт
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

# Название бакета в MinIO
BUCKET_NAME = "datasets"

# Проверяем наличие бакета и создаём его при необходимости
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

# Путь к директории DVC проекта (локально)
DVC_REPO_PATH = "/app/dvc_repo"

os.makedirs(DVC_REPO_PATH, exist_ok=True)

# Настраиваем имя пользователя и email для Git (чтобы избежать ошибки)
subprocess.run(["git", "config", "--global", "user.name", "FastAPI DVC"])
subprocess.run(["git", "config", "--global", "user.email", "fastapi-dvc@example.com"])


# Инициализация DVC (если не инициализировано)
if not os.path.exists(os.path.join(DVC_REPO_PATH, ".dvc")):
    subprocess.run(["git", "init"], cwd=DVC_REPO_PATH)
    subprocess.run(["dvc", "init"], cwd=DVC_REPO_PATH)

    # Настраиваем MinIO как удалённое хранилище для DVC
    subprocess.run(
        [
            "dvc", "remote", "add", "-d", "minio",
            f"s3://{BUCKET_NAME}",
        ],
        cwd=DVC_REPO_PATH,
    )
    
    # Установка параметров доступа к MinIO
    subprocess.run(
        [
            "dvc", "remote", "modify", "minio", "access_key_id", "minioadmin"
        ],
        cwd=DVC_REPO_PATH,
    )
    subprocess.run(
        [
            "dvc", "remote", "modify", "minio", "secret_access_key", "minioadmin"
        ],
        cwd=DVC_REPO_PATH,
    )
    subprocess.run(
        [
            "dvc", "remote", "modify", "minio", "endpointurl", "http://minio:9000"
        ],
        cwd=DVC_REPO_PATH,
    )

    # Добавляем удаленное хранилище как default
    subprocess.run(
        [
            "dvc", "remote", "default", "minio"
        ],
        cwd=DVC_REPO_PATH,
    )


def upload_to_minio_and_dvc(file_name: str, data: bytes):
    """
    Загружает файл в MinIO и версионирует его через DVC.
    """
    try:
        data_stream = io.BytesIO(data)
        minio_client.put_object(
            BUCKET_NAME,
            file_name,
            data_stream,
            length=len(data),
        )


        # Сохраняем файл локально для добавления в DVC
        local_file_path = os.path.join(DVC_REPO_PATH, file_name)
        with open(local_file_path, "wb") as f:
            f.write(data)

        # Добавляем файл в DVC
        subprocess.run(["dvc", "add", file_name], cwd=DVC_REPO_PATH, check=True)

        # Сохраняем изменения в DVC (коммит)
        subprocess.run(
            ["git", "add", f"{file_name}.dvc"], cwd=DVC_REPO_PATH, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"Add {file_name}"], cwd=DVC_REPO_PATH, check=True
        )

        # Отправляем данные в удалённое хранилище MinIO через DVC
        subprocess.run(["dvc", "push"], cwd=DVC_REPO_PATH, check=True)



        # Удаляем локальный файл после успешного пуша
        os.remove(local_file_path)

        return f"File {file_name} uploaded to MinIO and versioned with DVC"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


def download_from_minio(file_name: str):
    """
    Загружает файл из MinIO.
    """
    try:
        response = minio_client.get_object(BUCKET_NAME, file_name)
        return response.read()
    except S3Error as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")


@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Загружает датасет через API и сохраняет его в MinIO с версионированием через DVC.
    """
    data = await file.read()
    result = upload_to_minio_and_dvc(file.filename, data)
    return {"detail": result}


@app.get("/download_dataset/{file_name}")
async def download_dataset(file_name: str):
    """
    Загружает датасет из MinIO.
    """
    data = download_from_minio(file_name)
    return StreamingResponse(io.BytesIO(data), media_type="application/octet-stream")




class TrainRequest(BaseModel):
    """
    Модель запроса для обучения модели.
    
    Attributes:
        model_type (str): Тип модели для обучения.
        hyperparameters (Dict[str, Any]): Гиперпараметры для модели.
        training_data (List[List[float]]): Обучающие данные.
        target_data (List[float]): Целевые значения для обучения.
        name (str): Имя модели (необязательно).
    """
    model_type: str
    hyperparameters: Dict[str, Any]
    training_data: List[List[float]]
    target_data: List[float]
    name: str = None

class PredictRequest(BaseModel):
    """
    Модель запроса для предсказания.
    
    Attributes:
        model_id (str): ID модели для предсказания.
        data (List[List[float]]): Данные для предсказания.
        filename (str): Имя файла для сохранения предсказаний.
    """
    model_id: str
    data: List[List[float]]
    filename: str

class RetrainRequest(BaseModel):
    """
    Модель запроса для переобучения модели.
    
    Attributes:
        model_id (str): ID модели для переобучения.
        data (List[List[float]]): Данные для переобучения.
        target (List[float]): Целевые значения для переобучения.
        filename (str): Имя файла для сохранения предсказаний.
    """
    model_id: str
    data: List[List[float]]
    target: List[float]
    filename: str

@app.get("/status")
def get_status():
    """
    Проверяет статус сервиса.
    
    Returns:
        dict: Словарь с информацией о статусе сервиса.
    """
    logger.info("Проверка статуса сервиса")
    return {"status": "Сервис работает"}

@app.get("/models")
def get_available_models():
    """
    Получает список доступных типов моделей.
    
    Returns:
        list: Список доступных типов моделей.
    """
    logger.info("Получение доступных типов моделей")
    return model_manager.get_available_model_types()

@app.get("/models/info")
def get_models_info():
    """
    Получает информацию обо всех обученных моделях.
    
    Returns:
        dict: Словарь с информацией об обученных моделях.
    """
    logger.info("Получение информации обо всех обученных моделях")
    return model_manager.get_models_info()

@app.post("/train")
def train_model(request: TrainRequest):
    """
    Обучает модель на основе предоставленных данных и гиперпараметров.
    
    Args:
        request (TrainRequest): Запрос на обучение модели.

    Returns:
        dict: Словарь с ID обученной модели.

    Raises:
        HTTPException: Если произошла ошибка при обучении модели.
    """
    logger.info(f"Обучение модели типа {request.model_type} с гиперпараметрами {request.hyperparameters} и именем {request.name}")
    try:

        model_id = model_manager.train_model(request.model_type, request.hyperparameters, request.training_data, request.target_data, request.name)
        return {"model_id": model_id}
    except ValueError as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Делает предсказание с использованием указанной модели и сохраняет результаты в файл.
    
    Args:
        request (PredictRequest): Запрос на предсказание.

    Returns:
        dict: Словарь с информацией о сохранении предсказаний.

    Raises:
        HTTPException: Если произошла ошибка при предсказании.
    """
    logger.info(f"Предсказание с моделью {request.model_id}")
    try:
        predictions = model_manager.predict(request.model_id, request.data)
        with open(request.filename, 'wb') as f:
            pickle.dump(predictions, f)


        # model_manager.upload_to_minio(request.filename, request.filename)
        # model_manager.save_dataset_with_dvc(request.filename)


        return {"detail": "Предсказания сохранены", "filename": request.filename}
    except ValueError as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    """
    Удаляет модель по её ID.
    
    Args:
        model_id (str): ID модели для удаления.

    Returns:
        dict: Словарь с информацией об удалении модели.

    Raises:
        HTTPException: Если произошла ошибка при удалении модели.
    """
    logger.info(f"Удаление модели {model_id}")
    try:
        model_manager.delete_model(model_id)
        return {"detail": "Модель удалена"}
    except ValueError as e:
        logger.error(f"Ошибка при удалении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain")
def retrain_model(request: RetrainRequest):
    """
    Переобучает существующую модель на новых данных и сохраняет результаты в файл.
    
    Args:
        request (RetrainRequest): Запрос на переобучение модели.

    Returns:
        dict: Словарь с информацией о переобучении модели и сохранении предсказаний.

    Raises:
        HTTPException: Если произошла ошибка при переобучении модели.
    """
    logger.info(f"Переобучение модели {request.model_id}")
    try:
        predictions = model_manager.retrain_model(request.model_id, request.data, request.target)
        with open(request.filename, 'wb') as f:
            pickle.dump(predictions, f)

        # model_manager.upload_to_minio(request.filename, request.filename)
        # model_manager.save_dataset_with_dvc(request.filename)
        

        return {"detail": "Модель переобучена и предсказания сохранены", "filename": request.filename}
    except ValueError as e:
        logger.error(f"Ошибка при переобучении модели: {e}")
        raise HTTPException(status_code=400, detail=str(e))



