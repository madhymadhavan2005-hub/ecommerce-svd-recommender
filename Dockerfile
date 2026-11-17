# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install build essentials for C extensions (required by scikit-surprise on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, data, and dashboard
COPY . .

# Generate initial dataset & train initial models
RUN python src/generate_mock_data.py && python src/preprocess.py && python src/recommender.py

# Expose default port
EXPOSE 8000

# Run FastAPI app with dynamic PORT binding
CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port ${PORT}"]
