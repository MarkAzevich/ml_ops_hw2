import boto3
import os
import uuid
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dvc.repo import Repo
import dvc.api


class ModelManager:
    def __init__(self):
        """
        Инициализация менеджера моделей.
        Создает словари для хранения моделей и информации о них,
        а также доступные модели для обучения.
        """
        self.models = {}
        self.model_info = {}
        self.available_models = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier
        }


    def save_dataset_version(self, dataset_path):
        try:
            os.system(f"dvc add {dataset_path}")
            os.system("git add .")
            os.system(f"git commit -m 'Added dataset version for {dataset_path}'")
            os.system(f"dvc push")
            print(f"Dataset {dataset_path} saved and pushed to remote storage.")
        except Exception as e:
            print(f"Error saving dataset version: {e}")




    def get_available_model_types(self):
        """
        Получение списка доступных типов моделей.

        Returns:
            list: Список строк с названиями доступных типов моделей.
        """
        return list(self.available_models.keys())
    
    def train_model(self, model_type: str, hyperparameters: dict, X_train, y_train, name: str = None):
        """
        Обучение модели указанного типа с заданными гиперпараметрами.

        Args:
            model_type (str): Тип модели для обучения (например, "logistic_regression").
            hyperparameters (dict): Гиперпараметры для модели.
            X_train: Данные для обучения (признаки).
            y_train: Целевые данные для обучения.
            name (str, optional): Имя модели. Если не указано, будет сгенерировано автоматически.

        Returns:
            str: Уникальный идентификатор обученной модели.

        Raises:
            ValueError: Если указанный тип модели не поддерживается.
        """
        if model_type not in self.available_models:
            raise ValueError(f"Model type {model_type} not supported")
        
        ModelClass = self.available_models[model_type]
        model = ModelClass(**hyperparameters)
        model.fit(X_train, y_train)
        
        model_id = str(uuid.uuid4())
        self.models[model_id] = model
        self.model_info[model_id] = {
            "type": model_type,
            "name": name or model_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": hyperparameters
        }
        
        return model_id

    def get_models_info(self):
        """
        Получение информации обо всех обученных моделях.

        Returns:
            dict: Словарь с информацией о моделях, где ключи - идентификаторы моделей.
        """
        return self.model_info

    def predict(self, model_id: str, data: list):
        """
        Получение предсказаний от обученной модели.

        Args:
            model_id (str): Уникальный идентификатор модели.
            data (list): Данные для предсказания.

        Returns:
            list: Список предсказанных значений.

        Raises:
            ValueError: Если модель с указанным идентификатором не найдена.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with id {model_id} not found")
        
        model = self.models[model_id]
        predictions = model.predict(data)
        
        return predictions.tolist()

    def delete_model(self, model_id: str):
        """
        Удаление обученной модели по ее идентификатору.

        Args:
            model_id (str): Уникальный идентификатор модели.

        Raises:
            ValueError: Если модель с указанным идентификатором не найдена.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with id {model_id} not found")
        
        del self.models[model_id]
        del self.model_info[model_id]

    def retrain_model(self, model_id: str, X_train, y_train):
        """
        Повторное обучение существующей модели.

        Args:
            model_id (str): Уникальный идентификатор модели.
            X_train: Данные для повторного обучения (признаки).
            y_train: Целевые данные для повторного обучения.

        Returns:
            list: Список предсказанных значений после повторного обучения.

        Raises:
            ValueError: Если модель с указанным идентификатором не найдена.
        """
        if model_id not in self.models:
            raise ValueError(f"Model with id {model_id} not found")
        
        model_type = self.model_info[model_id]["type"]
        hyperparameters = self.model_info[model_id]["hyperparameters"]
        ModelClass = self.available_models[model_type]
        
        # Создаем новую модель и обучаем её
        model = ModelClass(**hyperparameters)
        
        model.fit(X_train, y_train)
        
        # Обновляем информацию о модели
        self.models[model_id] = model
        
        # Получаем предсказания после повторного обучения
        predictions = model.predict(X_train)
        
        return predictions.tolist()
