FROM python:3.11-slim

# 1. Базовые утилиты (только ставим, ничего не удаляем с хоста, это ТОЛЬКО внутри образа)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Рабочая директория внутри контейнера
WORKDIR /app

# 3. Ставим обычные питон-зависимости (БЕЗ torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Ставим CUDA-версию PyTorch (умеет и на CPU, и на GPU)
# cu118 можно заменить на нужную, если у тебя другая CUDA-линейка на серверах
RUN pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch torchvision torchaudio

# 5. Копируем код и ноутбук внутрь контейнера
COPY src ./src
COPY inference.ipynb .

# 6. Откроем порт для Jupyter
EXPOSE 8888

# 7. При старте контейнера запускаем Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token="]
