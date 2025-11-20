# Model Card: Cryptocurrency Volatility Spike Predictor

## Model Details

- **Model Type**: XGBoost Classifier
- **Input**: 111 features (float64)
- **Output**: Binary classification (int32)
- **Model Size**: ~466 KB
- **Framework Versions**:
  - Python: 3.13.7
  - scikit-learn: 1.7.1
  - MLflow: 3.6.0
- **Training Date**: 2025-11-19
- **MLflow Run ID**: 3b858571574e416c934cc5635accdce8

## Intended Use
- **Primary use**: Predict 60-second forward volatility spikes in cryptocurrency markets
- **Users**: Trading systems, risk monitoring
- **Out of scope**: Not for actual trading decisions without human oversight

## Training Data
- **Source**: Coinbase WebSocket API (ticker, matches, level2_batch)
- **Products**: BTC-USD, ETH-USD, SOL-USD
- **Time period**: November 14-18, 2024
- **Size**: 198,045 rows

## Evaluation Metrics
- **Primary metric**: PR-AUC (target ≥ 0.65)
- **Test PR-AUC**: 0.50

## Inference Performance:
- Batch size: 198,045 samples
- Total inference time: 0.52s
- Average latency: 0.0026ms per sample
- Throughput: 382,322 samples/s
- Real-time capability: ~38,000x faster than real-time ticker rate

## Features
- 69 windowed features (30s, 60s, 300s, 900s windows)
- 111 total features
- Top predictors: spread_std, ob_microprice, return_std

## Limitations
- Trained in short windows during varying market periods
- Performance varies across time of day/week
- USDT-USD excluded (stablecoin, different behavior)

## Ethical Considerations
- Model predictions should not be sole basis for trading.
- Performance may vary in different market regimes.