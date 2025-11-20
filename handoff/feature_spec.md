# Feature Specification: Crypto Volatility Spike Prediction

## Overview

This document specifies the feature engineering pipeline for predicting 60-second forward volatility spikes in cryptocurrency markets. The system processes real-time WebSocket data from Coinbase Web Socket API, computes windowed features across multiple timeframes, and generates binary labels for classification.

## Target Definition

### Prediction Horizon
- **Horizon**: 60 seconds
- **Task**: Binary classification of volatility spikes

### Volatility Proxy
- **Metric**: Rolling standard deviation of midprice returns over the forward horizon
- **Computation**: For each timestamp t, compute the standard deviation of price returns from t to t+60s

### Label Definition
```
target_spike = 1 if σ_future >= τ_product else 0
```

Where:
- `σ_future`: Forward-looking 60-second rolling volatility (std of returns)
- `τ_product`: Product-specific threshold at 95th percentile

### Threshold Calibration

Thresholds are computed **per-product** to account for different baseline volatility levels across cryptocurrencies:

| Product | Threshold (τ) | Rationale |
|---------|---------------|-----------|
| BTC-USD | Computed at p90 | Lower baseline volatility, established market |
| ETH-USD | Computed at p80 | Moderate volatility, high liquidity |
| SOL-USD | Computed at p80 | Higher baseline volatility, smaller cap |

**Note**: I tested multiple thresholds, most at 95th before moving the threshold down. I think thresholds in the range of 80 - 90 are worth consideration.

## Data Sources

### Products Tracked
- **Prediction Targets**: BTC-USD, ETH-USD, SOL-USD
- **Auxiliary Data**: USDT-USD (stablecoin reference) NOTE: I did not implement this, but captured it for a market anomaly signal (canary in the coalmine) and also a sanity check.

### WebSocket Channels
- `ticker`: Real-time price updates with best bid/ask
- `matches`: Individual trade executions
- `level2_batch`: Order book snapshots (top-of-book liquidity)
- `heartbeat`: Connection health monitoring

## Feature Windows

Features are computed across multiple time windows to capture dynamics at different scales:

| Window | Duration | Purpose |
|--------|----------|---------|
| w30 | 30 seconds | Immediate microstructure, noise detection |
| w60 | 60 seconds | Primary prediction horizon alignment |
| w300 | 5 minutes | Short-term trend context |
| w900 | 15 minutes | Regime identification, longer-term patterns |

## Feature Categories

### 1. Price Dynamics (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `price_mean` | Average price in window | `mean(prices)` |
| `price_std` | Price volatility | `std(prices)` |
| `price_min` | Minimum price | `min(prices)` |
| `price_max` | Maximum price | `max(prices)` |
| `price_range` | Price range | `max - min` |
| `price_momentum` | Price change over window | `(p_last - p_first) / p_first` |
| `price_trend` | Linear regression slope | Normalized slope of price vs time |

### 2. Return Statistics (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `return_mean` | Average return | `mean(returns)` |
| `return_std` | Return volatility | `std(returns)` |
| `return_skew` | Return distribution skewness | `skew(returns)` |
| `return_kurt` | Return distribution kurtosis | `kurtosis(returns)` |
| `volatility_lag1` | Previous window's volatility | Lagged `return_std` |

### 3. Spread Indicators (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `spread_mean` | Average bid-ask spread | `mean((ask - bid) / midprice)` |
| `spread_std` | Spread volatility | `std(spreads)` |

### 4. Trade Intensity (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `trade_count` | Number of trades | Count of match messages |
| `trade_volume` | Total traded volume | `sum(trade_sizes)` |
| `avg_trade_size` | Average trade size | `volume / count` |
| `trade_imbalance` | Buy/sell pressure | `(buy_vol - sell_vol) / total_vol` |

### 5. Microstructure Timing (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `avg_tick_interval` | Mean time between ticks | `mean(time_diffs)` |
| `tick_interval_std` | Tick timing variability | `std(time_diffs)` |
| `tick_rate` | Ticks per second | `n_ticks / window_seconds` |
| `n_observations` | Data points in window | Count of valid observations |

### 6. Order Book / L2 Features (per window)

| Feature | Description | Formula |
|---------|-------------|---------|
| `ob_bid_depth_L1_mean` | Avg top-of-book bid size | `mean(bid_sizes)` |
| `ob_ask_depth_L1_mean` | Avg top-of-book ask size | `mean(ask_sizes)` |
| `ob_depth_imbalance_L1` | Liquidity imbalance | `(bid - ask) / (bid + ask)` |
| `ob_microprice_mean` | Liquidity-weighted midprice | See formula below |
| `ob_microprice_std` | Microprice volatility | `std(microprices)` |

**Microprice Formula**:
```
microprice = (best_ask * bid_size + best_bid * ask_size) / (bid_size + ask_size)
```

### 7. Temporal Features

| Feature | Description | Values |
|---------|-------------|--------|
| `hour_sin` | Cyclical hour encoding | `sin(2π * hour / 24)` |
| `hour_cos` | Cyclical hour encoding | `cos(2π * hour / 24)` |
| `day_of_week` | Day indicator | 0 (Monday) to 6 (Sunday) |

## Total Feature Count

- **Per-window features**: ~25 features × 4 windows = 100 base features
- **Time features**: 3 features
- **Total**: ~103 features before selection

## Feature Selection

Based on feature importance analysis, the top 30 features ranked by predictive power:

```python
top_features = [
    'w60_spread_std',           # Spread volatility - strongest predictor
    'w300_spread_std',          
    'w30_spread_std',           
    'w60_spread_mean',          
    'w30_spread_mean',
    'w60_ob_microprice_mean',   # L2 features important
    'w900_spread_std',          
    'w60_ob_depth_imbalance_L1',
    'w300_trade_imbalance',     
    'w30_ob_microprice_mean',   
    'w30_ob_depth_imbalance_L1',
    'w60_return_std',           # Direct volatility measures
    'w60_volatility_lag1',      
    'w300_spread_mean',         
    'w900_price_momentum',      
    'w60_trade_count',          # Activity measures
    'w60_tick_rate',            
    'w60_n_observations',       
    'w300_price_momentum',      
    'w60_trade_imbalance',      
    'w30_volatility_lag1',      
    'w30_return_std',           
    'w900_trade_count',         
    'w900_tick_rate',           
    'w900_n_observations',      
    'w30_price_std',            
    'w300_ob_microprice_mean',  
    'w30_price_range',          
    'w60_price_std',            
    'w300_trade_count'
]
```

### Key Insights from Feature Importance

1. **Spread features dominate**: Bid-ask spread volatility across all windows is the strongest predictor, indicating that market maker behavior signals upcoming volatility.

2. **60-second window most predictive**: Features aligned with the prediction horizon provide the strongest signal, as expected.

3. **L2 orderbook features add value**: Microprice and depth imbalance capture information beyond simple price/spread.

4. **Longer windows provide context**: 300s and 900s features help distinguish noise from regime changes.

5. **Lagged volatility important**: `volatility_lag1` confirms autocorrelation in volatility (volatility clustering).

## Data Quality Handling

### Missing Values
- NaN values filled with 0 (conservative assumption)
- Infinity values replaced with 0

### Minimum Observations
- Features require at least 2 price observations per window
- Windows with insufficient data return `None`

### Deduplication
- Duplicate timestamps handled by taking latest value
- Message ordering preserved via Kafka partitioning

## Reproducibility

### Live vs Replay Consistency
The feature pipeline produces identical outputs for:
- Real-time Kafka consumption
- Replay from saved NDJSON files

This is ensured by:
- Deterministic window management using deques
- Consistent timestamp handling (UTC)
- Same computation logic in both paths

### Session Tracking
- Each training run generates a unique session ID
- Artifacts are saved with session timestamps
- MLflow experiments tagged with session ID

# NOTE: My earliest experiments were not logged, but I re-optimized my scripts a a few times, and there are still multiple experiments across my datasets.

## Performance Optimizations

Several optimizations were implemented to handle high message volume:

1. **Deque-based windows**: O(1) append/pop vs O(n) list operations
2. **Manual slope calculation**: Replaced `np.polyfit` with direct formula (10x speedup) | NOTE: Claude helped optimize this, and it really helped.
3. **Batch saves**: Features saved in batches of 10,000 to reduce I/O | NOTE: THis was a HUGE perforamce gain on my replay script.
4. **Selective computation**: `compute_every_n` parameter to reduce computation frequency | NOTE: I implemented this as an option, but generally still ran with n=1.

## Usage

### Generate Features from Live Stream
```bash
python features/featurizer_l2_optiomized.py --topic-in ticks.raw --topic-out ticks.features
```

### Replay Raw Data to Generate Features
```bash
python scripts/replay_l2_optimized.py --raw data/raw --out data/processed/features.parquet
```

### Add Target Labels
```bash
python scripts/add_labels_perproduct.py --features data/processed/features.parquet
```

### Split Data
```bash
python scripts/split_data.py --features data/processed/features_labeled_perproduct.parquet
```
