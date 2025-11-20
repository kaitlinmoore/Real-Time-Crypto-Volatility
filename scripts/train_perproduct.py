import argparse
import json
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print('Warning: XGBoost not installed, skipping XGBoost model')


def load_data(train_path, val_path, test_path):
    '''Load train/val/test splits.'''
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    return train_df, val_df, test_df


def get_feature_columns(df, use_top=False):
    '''Get list of feature columns.'''
    
    if use_top:
    # Target feature selection based on feature importance check. 
        top_features = [
            'w60_spread_std', 'w300_spread_std', 'w30_spread_std', 'w60_spread_mean', 'w30_spread_mean',
            'w60_ob_microprice_mean', 'w900_spread_std', 'w60_ob_depth_imbalance_L1', 'w300_trade_imbalance',
            'w30_ob_microprice_mean', 'w30_ob_depth_imbalance_L1', 'w60_return_std', 'w60_volatility_lag1',
            'w300_spread_mean', 'w900_price_momentum', 'w60_trade_count', 'w60_tick_rate', 'w60_n_observations',
            'w300_price_momentum', 'w60_trade_imbalance', 'w30_volatility_lag1', 'w30_return_std', 'w900_trade_count',
            'w900_tick_rate', 'w900_n_observations', 'w30_price_std', 'w300_ob_microprice_mean', 'w30_price_range',
            'w60_price_std', 'w300_trade_count'
        ]
    
    # Select windowed features
    feature_cols = [col for col in df.columns if col.startswith('w')]
    
    # Add time features
    time_features = ['hour_sin', 'hour_cos', 'day_of_week']
    feature_cols.extend([f for f in time_features if f in df.columns])
    
    return feature_cols


def prepare_features(df, feature_cols, scaler=None):
    '''Prepare feature matrix, handling NaN values.'''
    
    X = df[feature_cols].copy()
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def compute_metrics(y_true, y_pred, y_prob):
    '''Compute evaluation metrics.'''
    
    return {
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def find_optimal_threshold(y_true, y_prob):
    '''Find threshold that maximizes F1 score.'''
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    f1_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0)
    
    if not f1_scores:
        return 0.5
    
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def train_product_model(train_df, val_df, test_df, product_id, feature_cols, output_dir):
    '''Train models for a single product.'''
    
    print(f'\n{"="*60}')
    print(f'Training models for {product_id}')
    print(f'{"="*60}')
    
    # Filter data for this product
    train = train_df[train_df['product_id'] == product_id]
    val = val_df[val_df['product_id'] == product_id]
    test = test_df[test_df['product_id'] == product_id]
    
    print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
    print(f'Spike rates - Train: {train["target_spike"].mean():.2%}, Val: {val["target_spike"].mean():.2%}, Test: {test["target_spike"].mean():.2%}')
    
    # Prepare features
    X_train, scaler = prepare_features(train, feature_cols)
    X_val, _ = prepare_features(val, feature_cols, scaler)
    X_test, _ = prepare_features(test, feature_cols, scaler)
    
    y_train = train['target_spike'].values
    y_val = val['target_spike'].values
    y_test = test['target_spike'].values
    
    results = {}
    
    # Train Logistic Regression
    print(f'\nLogistic Regression:')
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Find optimal threshold
    y_val_prob = lr_model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, y_val_prob)
    
    # Evaluate on test
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    print(f'  PR-AUC: {metrics["pr_auc"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}, F1: {metrics["f1"]:.4f}')
    results['logistic_regression'] = metrics
    
    # Save LR model
    lr_path = output_dir / f'lr_{product_id.replace("-", "_").lower()}.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump({'model': lr_model, 'scaler': scaler, 'threshold': threshold}, f)
    
    # Train XGBoost
    if HAS_XGBOOST:
        print(f'\nXGBoost:')
        
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='aucpr'
        )
        
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Find optimal threshold
        y_val_prob = xgb_model.predict_proba(X_val)[:, 1]
        threshold = find_optimal_threshold(y_val, y_val_prob)
        
        # Evaluate on test
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        
        print(f'  PR-AUC: {metrics["pr_auc"]:.4f}, ROC-AUC: {metrics["roc_auc"]:.4f}, F1: {metrics["f1"]:.4f}')
        results['xgboost'] = metrics
        
        # Save XGBoost model
        xgb_path = output_dir / f'xgb_{product_id.replace("-", "_").lower()}.pkl'
        with open(xgb_path, 'wb') as f:
            pickle.dump({'model': xgb_model, 'scaler': scaler, 'threshold': threshold}, f)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train per-product volatility prediction models')
    parser.add_argument('--train', type=str, default='data/processed/train_stratified.parquet',
                       help='Path to training data')
    parser.add_argument('--val', type=str, default='data/processed/val_stratified.parquet',
                       help='Path to validation data')
    parser.add_argument('--test', type=str, default='data/processed/test_stratified.parquet',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='models/artifacts/per_product',
                       help='Directory for model artifacts')
    parser.add_argument('--use-top-features', action='store_true',
                       help='Use only top 20 features by importance')
    parser.add_argument('--experiment-name', type=str, default='volatility-per-product',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print('Loading data...')
    train_df, val_df, test_df = load_data(args.train, args.val, args.test)
    
    print(f'Train: {len(train_df)} rows')
    print(f'Val: {len(val_df)} rows')
    print(f'Test: {len(test_df)} rows')
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df, use_top=args.use_top_features)
    print(f'\nUsing {len(feature_cols)} features')
    
    # Get products
    products = sorted(train_df['product_id'].unique())
    print(f'Products: {products}')
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Train per-product models
    all_results = {}
    
    for product in products:
        with mlflow.start_run(run_name=f'{product}'):
            results = train_product_model(
                train_df, val_df, test_df, product, feature_cols, output_dir
            )
            all_results[product] = results
            
            # Log to MLflow
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f'{model_name}_{metric_name}', value)
    
    # Summary
    print('\n' + '='*60)
    print('Per-Product Model Comparison (Test Set)')
    print('='*60)
    
    print(f'\n{"Product":<12} {"Model":<20} {"PR-AUC":<10} {"ROC-AUC":<10} {"F1":<10}')
    print('-'*62)
    
    for product in products:
        for model_name, metrics in all_results[product].items():
            print(f'{product:<12} {model_name:<20} {metrics["pr_auc"]:<10.4f} {metrics["roc_auc"]:<10.4f} {metrics["f1"]:<10.4f}')
    
    # Find best per product
    print('\n' + '='*60)
    print('Best Model per Product')
    print('='*60)
    
    total_pr_auc = 0
    for product in products:
        best = max(all_results[product].items(), key=lambda x: x[1]['pr_auc'])
        print(f'{product}: {best[0]} (PR-AUC: {best[1]["pr_auc"]:.4f})')
        total_pr_auc += best[1]['pr_auc']
    
    avg_pr_auc = total_pr_auc / len(products)
    print(f'\nAverage best PR-AUC across products: {avg_pr_auc:.4f}')
    
    print(f'\nArtifacts saved to: {output_dir}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
