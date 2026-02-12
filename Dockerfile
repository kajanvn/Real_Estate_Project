FROM python:3.13-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY model/ ./model/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]