FROM python:3.10-slim

WORKDIR /app

# Install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY main.py .

# Copy model artifacts.
# Default paths expect artifacts in /app/models/artifacts/
RUN mkdir -p /app/models/artifacts

# Environment variables with defaults.
ENV MODEL_PATH=/app/models/artifacts/logistic_regression.pkl
ENV SCALER_PATH=/app/models/artifacts/scaler.pkl
ENV FEATURE_COLS_PATH=/app/models/artifacts/feature_columns.json
ENV API_VERSION=1.0.0
ENV MODEL_VERSION=1.0.0

# Expose port.
EXPOSE 8000

# Run the service.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
