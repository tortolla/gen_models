FROM python:3.11-slim

# 1. Базовые утилиты
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория внутри контейнера
WORKDIR /app

# 3. Устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем код и ноутбук внутрь контейнера
COPY src ./src
COPY inference.ipynb .

# 5. Откроем порт для Jupyter
EXPOSE 8888

# 6. При старте контейнера запускаем Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token="]
