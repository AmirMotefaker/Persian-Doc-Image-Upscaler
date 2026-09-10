FROM python:3.12.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install --no-deps -e .

EXPOSE 7860
CMD ["python", "app.py"]
