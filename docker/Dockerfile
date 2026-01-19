# Multi-arch base image (works on amd64 and arm64)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build essentials help when wheels are not available on ARM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE requirements.txt /app/
COPY src/ /app/src/

RUN pip install -U pip \
    && pip install -r /app/requirements.txt \
    && pip install .

# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Default model path can be mounted as a volume
ENV SHNET_MODEL_PATH=/app/models/model.joblib

CMD ["uvicorn", "shnet_ml.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
