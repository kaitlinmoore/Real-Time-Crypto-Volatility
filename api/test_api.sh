# Test commands for RT Crypto Volatility API
# Use double quotes.

API_URL="${API_URL:-http://localhost:8000}"

echo "Testing Volatility Prediction API"
echo "API URL: $API_URL"
echo ""

# Health check
echo "Health Check (GET /health)"
echo "curl -s $API_URL/health | jq"
curl -s "$API_URL/health" | jq
echo ""

# Version info
echo "Version Info (GET /version)"
echo "curl -s $API_URL/version | jq"
curl -s "$API_URL/version" | jq
echo ""

# Get expected features.
echo "Expected Features (GET /features)"
echo "curl -s $API_URL/features | jq '.feature_count'"
curl -s "$API_URL/features" | jq '.feature_count'
echo ""

# Single prediction
echo "Single Prediction (POST /predict)"
echo 'curl -s -X POST $API_URL/predict -H "Content-Type: application/json" -d {...}'
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "BTC-USD",
    "features": {
      "w60_spread_std": 0.00015,
      "w60_return_std": 0.0003,
      "w300_spread_std": 0.00012,
      "w60_spread_mean": 0.0001,
      "w30_spread_std": 0.00018,
      "w30_spread_mean": 0.00012,
      "w60_ob_microprice_mean": 45000.0,
      "w900_spread_std": 0.0001,
      "w60_volatility_lag1": 0.00025,
      "w60_trade_count": 150,
      "w60_tick_rate": 2.5,
      "w60_n_observations": 150
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }' | jq
echo ""

# Metrics
echo "Prometheus Metrics (GET /metrics)"
echo "curl -s $API_URL/metrics"
curl -s "$API_URL/metrics"
echo ""

echo "Testing Complete
