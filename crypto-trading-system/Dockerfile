FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY config /app/config
COPY docs /app/docs

RUN pip install --no-cache-dir .

CMD ["python", "-m", "src.main", "scheduler"]
