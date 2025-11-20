# Model Evaluation Report: Crypto Volatility Spike Prediction

**Session ID:** 20251120_085507
**Experiment ID:** 17

---

## Executive Summary

This report evaluates models trained to predict 60-second forward volatility spikes in cryptocurrency markets. The primary evaluation metric is PR-AUC (Precision-Recall Area Under Curve).

**Key Results:**
- Best performing model: **logisticlogistic_regression_20251120_085507**
- Best PR-AUC achieved: **[0.5693]**

## Experimental Setup

### Products

- **Included:** BTC-USD, ETH-USD, SOL-USD
- **Excluded:** USDT-USD (stablecoin, minimal volatility)

### Target Definition

- **Prediction horizon:** 60 seconds
- **Volatility proxy:** Rolling standard deviation of midprice returns
- **Threshold:** 95th percentile (computed per-product)
- **Label:** 1 if forward volatility ≥ threshold, else 0

### Features

- **Total features:** [111]
- **Feature windows:** 30s, 60s, 300s, 900s
- **Feature categories:** Price dynamics, returns, spreads, trade intensity, microstructure, order book (L2), temporal

---

## Model Comparison

### Test Set Performance

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Training Time (s) |
|-------|--------|---------|-----------|--------|-----|-------------------|
| Baseline (Z-score) | [0.0443] | [0.8332] | [0.0432] | [0.73] | [0.0815] | N/A |
| Logistic Regression | [0.5693] | [0.9632] | [0.1536] | [0.7933] | [0.2574] | [5.3min] |
| XGBoost | [0.0895] | [0.8556] | [0] | [0] | [0] | [8.1s] |

## Visualizations

### 4.1 Precision-Recall Curves

![alt text](pr_curve_logistic_regression.png)


### Confusion Matrix

![alt text](confusion_matrix_logistic_regression-1.png)