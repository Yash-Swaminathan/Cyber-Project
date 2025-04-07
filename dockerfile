# Use Python 3.11 slim image
FROM python:3.11-slim AS build

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install poetry && poetry install --no-dev --no-root

EXPOSE 8000

CMD ["uvicorn", "detection_api:app", "--host", "0.0.0.0", "--port", "8000"]