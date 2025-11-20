# Scoping Brief: Real-Time Cryptocurrency Volatility Spike Detection | Kaitlin Moore

## Use Case

This project develops a real-time system to predict short-term volatility spikes in cryptocurrency markets, enabling traders and risk management systems to anticipate and respond to rapid price movements. The system ingests live market data from Coinbase's Advanced Trade WebSocket API, processes it through a streaming pipeline, and generates predictions about imminent volatility events.

Cryptocurrency markets are open 24/7, move quickly, and often experience sudden bursts of volatility that create risk for traders and automated systems. This project aims to build a real-time analytics pipeline that can detect and predict short-term volatility spikes for major cryptocurrency trading pairs, using publicly available market data from Coinbase's Advanced Trade WebSocket API.

The system will continuously ingest live ticker messages, store them in Kafka for streaming processing, and generate features that characterize short-term price dynamics. The ultimate goal is to produce a 60-second-look-ahead volatility-spike prediction to anticipate rapid price swings.

## Prediction Goal

**Primary Objective**: Detect volatility spikes 60 seconds before they occur in major cryptocurrency trading pairs.

**Technical Definition**:
- **Target Variable**: Binary classification of whether the 60-second forward volatility exceeds a defined threshold
- **Prediction Horizon**: 60 seconds ahead
- **Volatility Measure**: Rolling standard deviation of midprice returns over the forward 60-second window
- **Spike Threshold**: Percentile of historical volatility per product (ensuring a particular spike percentage represented across products)

**Monitored Products**: 
- Primary targets: BTC-USD, ETH-USD, SOL-USD
- Control/auxiliary: USDT-USD (stablecoin for potential anomaly detection)

## Success Metrics

1. **Model Performance**:
   - **Primary Metric**: PR-AUC (prioritizing precision-recall due to class imbalance)
   - **Secondary Metrics**: 
     - F1 Score
     - Precision at 50% recall
   - **Latency**: Inference time

2. **System Reliability**:
   - Data ingestion uptime
   - Feature computation lag
   - Message loss from WebSocket to Kafka

## Risk Assumptions

### Data Quality Risks
- **WebSocket Stability**: Assumes Coinbase WebSocket maintains consistent message flow; implemented reconnection logic to handle disconnections
- **Market Hours Coverage**: System designed for 24/7 crypto markets but may see reduced effectiveness due to limited collection windows

### Model Risks
- **Non-stationarity**: Cryptocurrency markets exhibit regime changes, and drift is a real potential issue.
- **Extreme Events**: Extreme events (>99th percentile) may not be predictable with 60-second lead time

### Technical Risks
- **Infrastructure**: Single Kafka broker setup acceptable for development but would need clustering to scale to production
- **Resource Constraints**: Feature computation for multiple products may strain resources during high-volume periods

## Constraints & Assumptions

- **No Trading Execution**: System provides signals only. There is no automated trading.
- **Public Data Only**: Using only publicly available ticker and trade data.
- **60-Second Horizon**: Fixed prediction window aligned with high-frequency trading needs but not ultra-low latency
- **Local Development**: Docker-based setup suitable for development; production would likely scale to a cloud deployment
