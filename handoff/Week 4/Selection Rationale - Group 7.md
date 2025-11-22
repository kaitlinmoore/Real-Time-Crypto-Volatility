# Model Selection Rationale - Group 7

## Summary

After evaluating each team member's individual models, we selected a logistic regression model as our base model for the real-time prediction service. We have two logistic regression models, the one detailed below that achieved 0.6202** PR-AUC at the 85th percentile threshold. We also have a Logistic Regression model that achieved 0.70 PR-AUC. We intend to consolidate features and continue to experiment. pivot to a different model if we have success with further collaboartion and experimentation.

#### Selected Model: Logistic Regression

**Strengths:**
- Highest PR-AUC among candidates
- Lowlatency
- Simple deployment with sklearn pickle format
- Interpretable coefficients for debugging
- Well-documented feature engineering pipeline

**Performance Metrics:**
- Test PR-AUC: 0.6202
- Test ROC-AUC: 0.8117
- Precision @ threshold: [0.4167]
- Recall @ threshold: [0.9938]
- F1 Score: [0.5872]

**Feature Engineering:**
- 30 engineered features, evaluated from 111 candidatefeatures
- Window sizes: 30s, 60s, 300s, 900s
- Top predictors: spread volatility features (w60_spread_std correlation: 0.20-0.24)
- Per-product threshold calibration at 85th percentile

**Infrastructure:**
- Docker containerization ready
- MLflow experiment tracking configured

#### Runner-up: XGBoost

**Strengths:**
- Captures non-linear feature interactions
- Feature importance readily available

**Limitations:**
- Higher inference latency than Logistic Regression
- More complex hyperparameter tuning required
- Larger model artifact size

## Feature Pipeline Alignment

The selected model uses the following feature categories:

| Category | Example Features | Importance |
|----------|------------------|------------|
| Spread Volatility | w60_spread_std, w300_spread_std | High |
| Price Microstructure | ob_microprice_mean, ob_depth_imbalance | Medium |
| Trade Flow | trade_imbalance, trade_count | Medium |
| Return Statistics | return_std, return_skew | Medium |
| Temporal | hour_sin, hour_cos, day_of_week | Low |

### Integration Plan

```
Week 4: Deploy base model in FastAPI container
Week 5: Add Prometheus/Grafana monitoring
Week 6: Evaluate performance, consider XGBoost upgrade if needed
Week 7: Final tuning and documentation
```