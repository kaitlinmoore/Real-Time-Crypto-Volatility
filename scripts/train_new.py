import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri('http://localhost:5000')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print('Warning: XGBoost not installed, skipping XGBoost model')


def load_data(train_path, val_path, test_path, exclude_products=None):
    '''Load train/val/test splits.'''
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    # Exclude specified products.
    if exclude_products:
        for product in exclude_products:
            train_df = train_df[train_df['product_id'] != product]
            val_df = val_df[val_df['product_id'] != product]
            test_df = test_df[test_df['product_id'] != product]
        print(f'Excluded products: {exclude_products}')
    
    print(f'Train: {len(train_df)} rows')
    print(f'Val: {len(val_df)} rows')
    print(f'Test: {len(test_df)} rows')
    
    # Show products included.
    products = train_df['product_id'].unique()
    print(f'Products: {list(products)}')
    
    return train_df, val_df, test_df


def get_feature_columns(df):
    '''Get list of feature columns (windowed features).'''
    
    # Select windowed features.
    feature_cols = [col for col in df.columns if col.startswith('w')]
    
    # Add time features.
    time_features = ['hour_sin', 'hour_cos', 'day_of_week']
    feature_cols.extend([f for f in time_features if f in df.columns])
    
    return feature_cols


def prepare_features(df, feature_cols, scaler=None):
    '''Prepare feature matrix, handling NaN values.'''
    
    X = df[feature_cols].copy()
    
    # Fill NaN with 0.
    X = X.fillna(0)
    
    # Replace inf with large values.
    X = X.replace([np.inf, -np.inf], 0)
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def compute_metrics(y_true, y_pred, y_prob):
    '''Compute evaluation metrics.'''
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    return metrics


def train_baseline(train_df, val_df, feature_cols):
    '''Train baseline threshold model using volatility z-score.'''
    
    print('Training Baseline Model (Volatility Z-Score Threshold)')
    
    # Use recent volatility as predictor.
    vol_col = 'w60_return_std'
    
    if vol_col not in train_df.columns:
        print(f'Error: {vol_col} not found in features')
        return None
    
    # Compute threshold on training data.
    train_vol = train_df[vol_col].fillna(0)
    vol_mean = train_vol.mean()
    vol_std = train_vol.std()
    
    # Try different z-score thresholds.
    best_threshold = 1.5
    best_pr_auc = 0
    
    for z_threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        threshold = vol_mean + z_threshold * vol_std
        
        # Predict on validation.
        val_vol = val_df[vol_col].fillna(0)
        y_pred = (val_vol >= threshold).astype(int)
        y_prob = (val_vol - vol_mean) / vol_std  # z-score as probability proxy
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # Normalize to 0-1.
        
        y_true = val_df['target_spike']
        pr_auc = average_precision_score(y_true, y_prob)
        
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_threshold = z_threshold
    
    print(f'Best z-score threshold: {best_threshold}')
    
    # Final threshold
    threshold = vol_mean + best_threshold * vol_std
    
    # Model parameters to save
    model_params = {
        'vol_mean': vol_mean,
        'vol_std': vol_std,
        'z_threshold': best_threshold,
        'threshold': threshold,
        'feature_col': vol_col
    }
    
    return model_params


def predict_baseline(df, model_params):
    '''Make predictions with baseline model.'''
    
    vol_col = model_params['feature_col']
    vol = df[vol_col].fillna(0)
    
    # Binary prediction
    y_pred = (vol >= model_params['threshold']).astype(int)
    
    # Probability (normalized z-score)
    y_prob = (vol - model_params['vol_mean']) / model_params['vol_std']
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-10)
    
    return y_pred, y_prob


def train_logistic_regression(X_train, y_train, X_val, y_val):
    '''Train logistic regression model.'''
    
    print('Training Logistic Regression')
    
    # Train with class weight balancing.
    model = LogisticRegression(
        max_iter=10000,
        class_weight='balanced',
        solver='saga',
        random_state=23
    )
    
    model.fit(X_train, y_train)
    
    # Validate.
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    print(f'Validation PR-AUC: {metrics["pr_auc"]:.4f}')
    
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    '''Train XGBoost model.'''
    
    if not HAS_XGBOOST:
        return None
    
    print('Training XGBoost')
    
    # Calculate scale_pos_weight for imbalanced data.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,  # Reduce overfit by lowering or underfit by raising.
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=23,
        eval_metric='aucpr'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Validate.
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    print(f'Validation PR-AUC: {metrics["pr_auc"]:.4f}')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train volatility prediction models')
    parser.add_argument('--train', type=str, default='data/processed/train.parquet',
                       help='Path to training data')
    parser.add_argument('--val', type=str, default='data/processed/val.parquet',
                       help='Path to validation data')
    parser.add_argument('--test', type=str, default='data/processed/test.parquet',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='models/artifacts',
                       help='Directory for model artifacts')
    parser.add_argument('--experiment-name', type=str, default='volatility-prediction',
                       help='MLflow experiment name')
    parser.add_argument('--use-top-features', action='store_true',
                       help='Use only top 30 features by importance.')
    parser.add_argument('--exclude-products', type=str, default=None,
                       help='Comma-separated list of products to exclude (e.g., "USDT-USD,SOL-USD")')
    
    args = parser.parse_args()
    
    # Create session ID for this training run.
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory with session ID.
    output_dir = Path(args.output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Session ID: {session_id}')
    print(f'Output directory: {output_dir}')
    
    # Build exclusion list.
    exclude_products = []
    if args.exclude_products:
        exclude_products.extend([p.strip() for p in args.exclude_products.split(',')])
    
    # Load data.
    print('Loading data...')
    train_df, val_df, test_df = load_data(
        args.train, args.val, args.test,
        exclude_products=exclude_products if exclude_products else None
    )

    # Target feature selection based on feature importance check. 
    top_features = ['w60_spread_std', 'w300_spread_std', 'w30_spread_std', 'w60_spread_mean', 'w30_spread_mean',
                    'w60_ob_microprice_mean', 'w900_spread_std', 'w60_ob_depth_imbalance_L1', 'w300_trade_imbalance',
                    'w30_ob_microprice_mean', 'w30_ob_depth_imbalance_L1', 'w60_return_std', 'w60_volatility_lag1',
                    'w300_spread_mean', 'w900_price_momentum', 'w60_trade_count', 'w60_tick_rate', 'w60_n_observations',
                    'w300_price_momentum', 'w60_trade_imbalance', 'w30_volatility_lag1', 'w30_return_std', 'w900_trade_count',
                    'w900_tick_rate', 'w900_n_observations', 'w30_price_std', 'w300_ob_microprice_mean', 'w30_price_range',
                    'w60_price_std', 'w300_trade_count'
                    ]
    
    # Get feature columns.
    if args.use_top_features:
        feature_cols = top_features
    else:
        feature_cols = get_feature_columns(train_df)
    print(f'\nUsing {len(feature_cols)} features')
    
    # Prepare targets.
    y_train = train_df['target_spike'].values
    y_val = val_df['target_spike'].values
    y_test = test_df['target_spike'].values
    
    # Prepare features.
    print('Preparing features...')
    X_train, scaler = prepare_features(train_df, feature_cols)
    X_val, _ = prepare_features(val_df, feature_cols, scaler)
    X_test, _ = prepare_features(test_df, feature_cols, scaler)
    
    # Save scaler.
    scaler_path = output_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns.
    features_path = output_dir / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)
    
    # Setup MLflow.
    mlflow.set_experiment(args.experiment_name)
    
    # Results storage.
    results = {}

    # Train and evaluate baseline.
    with mlflow.start_run(run_name=f'baseline_zscore_{session_id}') as run:
        mlflow.set_tag('session_id', session_id)
        run_id = run.info.run_id

        baseline_params = train_baseline(train_df, val_df, feature_cols)
        
        if baseline_params:
            # Evaluate on test.
            y_pred, y_prob = predict_baseline(test_df, baseline_params)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            
            print(f'\nBaseline Test Results:')
            print(f'  PR-AUC: {metrics["pr_auc"]:.4f}')
            print(f'  ROC-AUC: {metrics["roc_auc"]:.4f}')
            print(f'  F1: {metrics["f1"]:.4f}')
            
            # Log to MLflow.
            mlflow.log_params(baseline_params)
            mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
            
            # Save model.
            baseline_path = output_dir / 'baseline_model.json'
            with open(baseline_path, 'w') as f:
                json.dump(baseline_params, f)
            mlflow.log_artifact(baseline_path)
            
            results['baseline'] = metrics
    
    # Train and evaluate Logistic Regression.
    with mlflow.start_run(run_name=f'logistic_regression_{session_id}') as run:
        mlflow.set_tag('session_id', session_id)
        run_id = run.info.run_id

        lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Evaluate on test.
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        y_pred = lr_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        
        print(f'\nLogistic Regression Test Results:')
        print(f'  PR-AUC: {metrics["pr_auc"]:.4f}')
        print(f'  ROC-AUC: {metrics["roc_auc"]:.4f}')
        print(f'  F1: {metrics["f1"]:.4f}')
        
        # Log to MLflow.
        mlflow.log_params({
            'model_type': 'logistic_regression',
            'class_weight': 'balanced',
            'n_features': len(feature_cols)
        })
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        mlflow.sklearn.log_model(lr_model, name='model', input_example=X_train[:5])
        
        # Save model.
        lr_path = output_dir / 'logistic_regression.pkl'
        with open(lr_path, 'wb') as f:
            pickle.dump(lr_model, f)
        
        results['logistic_regression'] = metrics
    
    # Train and evaluate XGBoost.
    if HAS_XGBOOST:
        with mlflow.start_run(run_name=f'xgboost_{session_id}') as run:
            mlflow.set_tag('session_id', session_id)
            run_id = run.info.run_id

            xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
            
            if xgb_model:
                # Evaluate on test.
                y_prob = xgb_model.predict_proba(X_test)[:, 1]
                y_pred = xgb_model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred, y_prob)
                
                print(f'\nXGBoost Test Results:')
                print(f'  PR-AUC: {metrics["pr_auc"]:.4f}')
                print(f'  ROC-AUC: {metrics["roc_auc"]:.4f}')
                print(f'  F1: {metrics["f1"]:.4f}')
                
                # Log to MLflow.
                mlflow.log_params({
                    'model_type': 'xgboost',
                    'n_estimators': 100,
                    'max_depth': 4, # started at 6, down to 3, up to 4 - Trying grid search approach.
                    'learning_rate': 0.1,
                    'n_features': len(feature_cols)
                })
                mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
                mlflow.sklearn.log_model(xgb_model, name='model', input_example=X_train[:5])
                
                # Save model.
                xgb_path = output_dir / 'xgboost.pkl'
                with open(xgb_path, 'wb') as f:
                    pickle.dump(xgb_model, f)
                
                results['xgboost'] = metrics
    
    # Summary
    print('\n' + '='*60)
    print('Model Comparison (Test Set)')
    print('='*60)
    print(f'{"Model":<25} {"PR-AUC":<10} {"ROC-AUC":<10} {"F1":<10}')
    print('-'*60)
    for model_name, metrics in results.items():
        print(f'{model_name:<25} {metrics["pr_auc"]:<10.4f} {metrics["roc_auc"]:<10.4f} {metrics["f1"]:<10.4f}')
    
    # Identify best model.
    best_model = max(results.items(), key=lambda x: x[1]['pr_auc'])
    print(f'\nBest model by PR-AUC: {best_model[0]} ({best_model[1]["pr_auc"]:.4f})')
    
    print(f'\nArtifacts saved to: {output_dir}')
    print(f'Session ID: {session_id}')
    print('MLflow experiments logged. View with: mlflow ui')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
