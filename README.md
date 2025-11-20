# Crypto Volatility Spike Prediction

Real-time pipeline for predicting 60-second forward volatility spikes in cryptocurrency markets using streaming data from Coinbase.

## Overview

This project builds a complete ML pipeline that:
1. Ingests real-time market data via Coinbase WebSocket API
2. Streams data through Kafka for processing
3. Computes windowed features across multiple timeframes
4. Trains models to predict volatility spikes
5. Tracks experiments with MLflow
6. Monitors data drift with Evidently

## Quick Start

### 1. Start Infrastructure
```bash
docker compose -f docker/compose.yaml up -d
```

### 2. Ingest Data
```bash
python scripts/ws_ingest.py --minutes 15 --save-raw
```

### 3. Generate Features
```bash
python scripts/featurizer_l2_optimized.py --raw data/raw --out data/processed/features.parquet
```

### 4. Add Labels and Split
```bash
python scripts/add_labels_perproduct.py --features data/processed/features.parquet
python scripts/split_data.py --features data/processed/features_labeled_perproduct.parquet
```

### 5. Train Models
```bash
python scripts/train_new.py --train data/processed/train.parquet \
                            --val data/processed/val.parquet \
                            --test data/processed/test.parquet
```

### 6. View Results
```bash
mlflow ui
# Open http://localhost:5000
```

## Installation

### Prerequisites
- Python 3.10+
- Docker and Docker Compose

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration
Copy and edit the configuration file:
```bash
cp config.yaml.example config.yaml
```

## Project Structure

```
├── data/
│   ├── raw/                    # Raw NDJSON tick data
│   └── processed/              # Feature parquet files
├── docker/
│   ├── compose.yaml            # Kafka + MLflow services
│   └── Dockerfile.ingestor     # Ingestor container
├── docs/
│   ├── scoping_brief.pdf       # Project scope document
│   ├── feature_spec.md         # Feature documentation
│   ├── model_card_v1.md        # Model documentation
│   └── genai_appendix.md       # GenAI usage log
├── models/
│   └── artifacts/              # Saved models and scalers
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
├── reports/
│   ├── evidently/              # Drift reports
│   └── model_eval.pdf          # Evaluation report
├── scripts/
│   ├── ws_ingest.py            # WebSocket data ingestion
│   ├── kafka_consume_check.py  # Stream validation
│   ├── replay_l2_optimized.py  # Feature generation from raw data
│   ├── add_labels_perproduct.py # Target label creation
│   ├── split_data.py           # Train/val/test splitting
│   ├── split_stratified.py     # Stratified splitting by session
│   ├── train_new.py            # Model training
│   ├── tune.py                 # Hyperparameter tuning
│   └── utilities.py            # Shared utilities
├── config.yaml                 # Configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Components

### Data Ingestion (`scripts/ws_ingest.py`)

Connects to Coinbase WebSocket API and streams data to Kafka.

```bash
# Basic usage
python scripts/ws_ingest.py --minutes 15

# With raw file backup
python scripts/ws_ingest.py --minutes 30 --save-raw

# Specific trading pairs
python scripts/ws_ingest.py --pair "BTC-USD,ETH-USD" --minutes 15
```

**Channels subscribed**:
- `ticker`: Real-time price updates
- `matches`: Trade executions
- `level2_batch`: Order book snapshots
- `heartbeat`: Connection health

### Stream Validation (`scripts/kafka_consume_check.py`)

Validates data quality in Kafka topics.

```bash
# Check minimum message count
python scripts/kafka_consume_check.py --topic ticks.raw --min 100

# Extended validation
python scripts/kafka_consume_check.py --topic ticks.raw --duration 60 --check-staleness
```

### Feature Engineering (`scripts/replay_l2_optimized.py`)

Computes windowed features from raw data.

```bash
# Standard replay
python scripts/replay_l2_optimized.py --raw data/raw --out data/processed/features.parquet

# Faster processing (compute every 5th tick)
python scripts/replay_l2_optimized.py --raw data/raw --out data/processed/features.parquet --compute-every-n 5
```

**Feature windows**: 30s, 60s, 300s (5min), 900s (15min)

**Feature categories**:
- Price dynamics (mean, std, range, momentum, trend)
- Return statistics (mean, std, skew, kurtosis)
- Spread indicators (mean, std)
- Trade intensity (count, volume, imbalance)
- Microstructure timing (tick intervals, rate)
- Order book (depth, imbalance, microprice)
- Temporal (hour, day of week)

### Target Labeling (`scripts/add_labels_perproduct.py`)

Creates binary spike labels with per-product thresholds.

```bash
# Default 95th percentile threshold
python scripts/add_labels_perproduct.py --features data/processed/features.parquet

# Custom threshold
python scripts/add_labels_perproduct.py --features data/processed/features.parquet --threshold-percentile 90
```

### Data Splitting

**Chronological split** (`scripts/split_data.py`):
```bash
python scripts/split_data.py --features data/processed/features_labeled_perproduct.parquet \
                             --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

**Stratified by session** (`scripts/split_stratified.py`):
```bash
python scripts/split_stratified.py --input-dir data/processed/labeled/split \
                                   --sessions "1,2,3,4,5" \
                                   --exclude-products "USDT-USD"
```

### Model Training (`scripts/train_new.py`)

Trains baseline and ML models with MLflow tracking.

```bash
# Full feature set
python scripts/train_new.py --train data/processed/train.parquet \
                            --val data/processed/val.parquet \
                            --test data/processed/test.parquet

# Top features only
python scripts/train_new.py --train data/processed/train.parquet \
                            --val data/processed/val.parquet \
                            --test data/processed/test.parquet \
                            --use-top-features

# Exclude products
python scripts/train_new.py --train data/processed/train.parquet \
                            --val data/processed/val.parquet \
                            --test data/processed/test.parquet \
                            --exclude-products "USDT-USD"
```

**Models trained**:
- Baseline: Z-score threshold on 60s volatility
- Logistic Regression: Balanced class weights
- XGBoost: With scale_pos_weight for imbalance

### Hyperparameter Tuning (`scripts/tune.py`)

Grid search with time-series cross-validation.

```bash
# Tune both models
python scripts/tune.py --train data/processed/train.parquet \
                       --val data/processed/val.parquet \
                       --test data/processed/test.parquet

# Tune specific model
python scripts/tune.py --model lr --n-splits 5
```

## Docker Services

### Kafka (KRaft mode)
```yaml
# In docker/compose.yaml
kafka:
  image: bitnami/kafka:latest
  ports:
    - "9092:9092"
```

### MLflow
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  ports:
    - "5000:5000"
  command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlflow.db
```

### Build Ingestor Container
```bash
docker build -f docker/Dockerfile.ingestor -t crypto-ingestor .
docker run --network host crypto-ingestor --minutes 15
```

## MLflow Experiment Tracking

All training runs are now logged to MLflow with:
- Parameters (model type, hyperparameters, feature count)
- Metrics (PR-AUC, ROC-AUC, F1, precision, recall)
- Artifacts (model files, scalers, feature lists)
- Session tags for grouping related runs

Access the UI:
```bash
mlflow ui
# Navigate to http://localhost:5000
```
