FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

# Your code creates needed folders too, but keep it safe for container startup
RUN mkdir -p /app/data /app/logs

ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "-m", "src.ui"]
