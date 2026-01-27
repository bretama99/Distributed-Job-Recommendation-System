FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps (Spark needs Java)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    openjdk-21-jre \
 && rm -rf /var/lib/apt/lists/*

# Python deps (more robust)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --default-timeout=300 --retries 10 --no-cache-dir -r /app/requirements.txt

# App code
COPY src /app/src

RUN mkdir -p /app/data /app/logs
ENV PYTHONPATH=/app

EXPOSE 7860
EXPOSE 4040

CMD ["python", "-m", "src.ui"]
