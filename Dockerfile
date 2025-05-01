FROM python:3.12.7

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv
# Установка UV

# Копирование файлов проекта
COPY requirements.txt .
COPY setup.py .
COPY src/ src/

# Установка Python-зависимостей с использованием UV
RUN uv pip install -r requirements.txt
RUN uv pip install -e .

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.ocr_bank_docs.api:app", "--host", "0.0.0.0", "--port", "8000"] 
