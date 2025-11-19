import pickle
import pandas as pd
import json

# Load model and feature names
with open('models/artifacts/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/artifacts/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

# Get importances
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(20))
print(f'\nFeatures with importance > 0.01: {(importances["importance"] > 0.01).sum()}')