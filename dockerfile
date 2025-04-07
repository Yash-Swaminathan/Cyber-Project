FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install poetry && poetry install --only main --no-root


EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "Backend.detection_api:app", "--host", "0.0.0.0", "--port", "8000"]
