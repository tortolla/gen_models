FROM python:3.11-slim

# 1. Базовые утилиты
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория
WORKDIR /app

# 3. Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем весь проект
COPY . .

# 5. Открываем порт для Jupyter
EXPOSE 8888

# 6. Запуск Jupyter Notebook без токена
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", 
"--NotebookApp.token="]

