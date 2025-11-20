FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts.
COPY scripts/ ./scripts/

# Create data directories.
RUN mkdir -p data/raw

# Default command
CMD ["python", "scripts/ws_ingest.py", "--pair", "BTC-USD", "--minutes", "15", "--save-raw"]