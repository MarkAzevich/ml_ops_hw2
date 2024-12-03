# Базовый образ для Python приложений
FROM python:3.9-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git
     

# Копирование приложения в контейнер
COPY . /app

# Установка переменных окружения
ENV API_URL=http://localhost:8000

# Команда для запуска приложения (FastAPI и Streamlit)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0"]
