import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print('Warning: LightGBM not installed, skipping LightGBM model')


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

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
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


def log_data_info(train_df, val_df, test_df, y_train, y_val, y_test, feature_cols):
    '''Log dataset information to MLflow.'''
    
    mlflow.log_params({
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'n_features': len(feature_cols),
        'products': ','.join(sorted(train_df['product_id'].unique())),
        'n_products': len(train_df['product_id'].unique()),
    })
    
    mlflow.log_metrics({
        'train_spike_rate': float(y_train.mean()),
        'val_spike_rate': float(y_val.mean()),
        'test_spike_rate': float(y_test.mean()),
        'train_spikes': int(y_train.sum()),
        'val_spikes': int(y_val.sum()),
        'test_spikes': int(y_test.sum()),
    })
    
    # Log time ranges if timestamp column exists.
    if 'timestamp' in train_df.columns:
        mlflow.log_params({
            'train_start': str(train_df['timestamp'].min()),
            'train_end': str(train_df['timestamp'].max()),
            'test_start': str(test_df['timestamp'].min()),
            'test_end': str(test_df['timestamp'].max()),
        })


def log_confusion_metrics(y_true, y_pred):
    '''Log confusion matrix values to MLflow.'''
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    mlflow.log_metrics({
        'test_true_positives': int(tp),
        'test_false_positives': int(fp),
        'test_true_negatives': int(tn),
        'test_false_negatives': int(fn),
        'test_specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    })


def log_pr_curve(y_true, y_prob, output_dir, model_name):
    '''Log precision-recall curve to MLflow.'''
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add PR-AUC annotation.
    pr_auc = average_precision_score(y_true, y_prob)
    ax.annotate(f'PR-AUC: {pr_auc:.4f}', xy=(0.6, 0.9), fontsize=11)
    
    filepath = output_dir / f'pr_curve_{model_name}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    mlflow.log_artifact(filepath)


def log_confusion_matrix(y_true, y_pred, output_dir, model_name):
    '''Log confusion matrix visualization to MLflow.'''
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Spike', 'Spike'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    
    filepath = output_dir / f'confusion_matrix_{model_name}.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    mlflow.log_artifact(filepath)


def log_feature_importance(model, feature_cols, output_dir, model_name):
    '''Log feature importance for tree-based models.'''
    
    if not hasattr(model, 'feature_importances_'):
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save to CSV.
    filepath = output_dir / f'feature_importance_{model_name}.csv'
    importance_df.to_csv(filepath, index=False)
    mlflow.log_artifact(filepath)
    
    # Log top 5 features as params.
    for i, row in importance_df.head(5).iterrows():
        mlflow.log_param(f'top_feature_{importance_df.index.get_loc(i)+1}', row['feature'])
    
    # Create visualization.
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20)
    ax.barh(top_20['feature'], top_20['importance'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top 20 Feature Importances - {model_name}', fontsize=14)
    ax.invert_yaxis()
    
    fig_path = output_dir / f'feature_importance_{model_name}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    mlflow.log_artifact(fig_path)


def train_baseline(train_df, val_df, feature_cols):
    '''Train baseline threshold model using volatility z-score.'''
    
    print('Training Baseline Model (Volatility Z-Score Threshold)')
    
    # Use recent volatility as predictor.
    vol_col = 'w60_return_std'
    
    if vol_col not in train_df.columns:
        print(f'Error: {vol_col} not found in features')
        return None, None
    
    # Compute threshold on training data.
    train_vol = train_df[vol_col].fillna(0)
    vol_mean = train_vol.mean()
    vol_std = train_vol.std()
    
    # Try different z-score thresholds.
    best_threshold = 1.5
    best_pr_auc = 0
    best_val_metrics = None
    
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
            best_val_metrics = compute_metrics(y_true, y_pred, y_prob)
    
    print(f'Best z-score threshold: {best_threshold}')
    
    # Final threshold.
    threshold = vol_mean + best_threshold * vol_std
    
    # Model parameters to save.
    model_params = {
        'vol_mean': float(vol_mean),
        'vol_std': float(vol_std),
        'z_threshold': float(best_threshold),
        'threshold': float(threshold),
        'feature_col': vol_col
    }
    
    return model_params, best_val_metrics


def predict_baseline(df, model_params):
    '''Make predictions with baseline model.'''
    
    vol_col = model_params['feature_col']
    vol = df[vol_col].fillna(0)
    
    # Binary prediction.
    y_pred = (vol >= model_params['threshold']).astype(int)
    
    # Probability (normalized z-score).
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
    
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Validate.
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    print(f'Validation PR-AUC: {metrics["pr_auc"]:.4f}')
    
    return model, metrics, train_time


def train_xgboost(X_train, y_train, X_val, y_val):
    '''Train XGBoost model.'''
    
    if not HAS_XGBOOST:
        return None, None, 0
    
    print('Training XGBoost')
    
    # Calculate scale_pos_weight for imbalanced data.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=23,
        eval_metric='aucpr'
    )
    
    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - train_start
    
    # Validate.
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    print(f'Validation PR-AUC: {metrics["pr_auc"]:.4f}')
    
    return model, metrics, train_time


def train_lightgbm(X_train, y_train, X_val, y_val):
    '''Train LightGBM model.'''
    
    if not HAS_LIGHTGBM:
        return None, None, 0
    
    print('Training LightGBM')
    
    # Calculate scale_pos_weight for imbalanced data.
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=23,
        verbose=-1
    )
    
    train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    train_time = time.time() - train_start
    
    # Validate.
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    metrics = compute_metrics(y_val, y_pred, y_prob)
    print(f'Validation PR-AUC: {metrics["pr_auc"]:.4f}')
    
    return model, metrics, train_time


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
    top_features = [
        'w60_spread_std', 'w300_spread_std', 'w30_spread_std', 'w60_spread_mean', 'w30_spread_mean',
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
        mlflow.set_tag('model_type', 'baseline')
        
        # Log data info.
        log_data_info(train_df, val_df, test_df, y_train, y_val, y_test, feature_cols)

        baseline_params, val_metrics = train_baseline(train_df, val_df, feature_cols)
        
        if baseline_params:
            # Log validation metrics.
            if val_metrics:
                mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            
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
            
            # Log confusion matrix metrics.
            log_confusion_metrics(y_test, y_pred)
            
            # Log visualizations.
            log_pr_curve(y_test, y_prob, output_dir, 'baseline')
            log_confusion_matrix(y_test, y_pred, output_dir, 'baseline')
            
            # Save and log model.
            baseline_path = output_dir / 'baseline_model.json'
            with open(baseline_path, 'w') as f:
                json.dump(baseline_params, f)
            mlflow.log_artifact(baseline_path)
            mlflow.log_artifact(scaler_path)
            mlflow.log_artifact(features_path)
            
            results['baseline'] = metrics
    
    # Train and evaluate Logistic Regression.
    with mlflow.start_run(run_name=f'logistic_regression_{session_id}') as run:
        mlflow.set_tag('session_id', session_id)
        mlflow.set_tag('model_type', 'logistic_regression')
        
        # Log data info.
        log_data_info(train_df, val_df, test_df, y_train, y_val, y_test, feature_cols)

        lr_model, val_metrics, train_time = train_logistic_regression(X_train, y_train, X_val, y_val)
        
        # Log validation metrics.
        mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
        mlflow.log_metric('training_time_sec', train_time)
        
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
            'solver': 'saga',
            'max_iter': 10000,
            'random_state': 23,
        })
        mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
        
        # Log confusion matrix metrics.
        log_confusion_metrics(y_test, y_pred)
        
        # Log visualizations.
        log_pr_curve(y_test, y_prob, output_dir, 'logistic_regression')
        log_confusion_matrix(y_test, y_pred, output_dir, 'logistic_regression')
        
        # Log model and artifacts.
        mlflow.sklearn.log_model(lr_model, name='model', input_example=X_train[:5].astype('float64'))
        mlflow.log_artifact(scaler_path)
        mlflow.log_artifact(features_path)
        
        # Save model locally.
        lr_path = output_dir / 'logistic_regression.pkl'
        with open(lr_path, 'wb') as f:
            pickle.dump(lr_model, f)
        
        results['logistic_regression'] = metrics
    
    # Train and evaluate XGBoost.
    if HAS_XGBOOST:
        with mlflow.start_run(run_name=f'xgboost_{session_id}') as run:
            mlflow.set_tag('session_id', session_id)
            mlflow.set_tag('model_type', 'xgboost')
            
            # Log data info.
            log_data_info(train_df, val_df, test_df, y_train, y_val, y_test, feature_cols)

            xgb_model, val_metrics, train_time = train_xgboost(X_train, y_train, X_val, y_val)
            
            if xgb_model:
                # Log validation metrics.
                mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
                mlflow.log_metric('training_time_sec', train_time)
                
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
                    'n_estimators': 500, # Was at 200
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 23,
                    'colsample_bytree': 0.8,
                    'subsample': 0.8
                })
                mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
                
                # Log confusion matrix metrics.
                log_confusion_metrics(y_test, y_pred)
                
                # Log visualizations.
                log_pr_curve(y_test, y_prob, output_dir, 'xgboost')
                log_confusion_matrix(y_test, y_pred, output_dir, 'xgboost')
                
                # Log feature importance.
                log_feature_importance(xgb_model, feature_cols, output_dir, 'xgboost')
                
                # Log model and artifacts.
                mlflow.sklearn.log_model(xgb_model, name='model', input_example=X_train[:5].astype('float64'))
                mlflow.log_artifact(scaler_path)
                mlflow.log_artifact(features_path)
                
                # Save model locally.
                xgb_path = output_dir / 'xgboost.pkl'
                with open(xgb_path, 'wb') as f:
                    pickle.dump(xgb_model, f)
                
                results['xgboost'] = metrics
    
    # Train and evaluate LightGBM.
    if HAS_LIGHTGBM:
        with mlflow.start_run(run_name=f'lightgbm_{session_id}') as run:
            mlflow.set_tag('session_id', session_id)
            mlflow.set_tag('model_type', 'lightgbm')
            
            # Log data info.
            log_data_info(train_df, val_df, test_df, y_train, y_val, y_test, feature_cols)

            lgb_model, val_metrics, train_time = train_lightgbm(X_train, y_train, X_val, y_val)
            
            if lgb_model:
                # Log validation metrics.
                mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
                mlflow.log_metric('training_time_sec', train_time)
                
                # Evaluate on test.
                y_prob = lgb_model.predict_proba(X_test)[:, 1]
                y_pred = lgb_model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred, y_prob)
                
                print(f'\nLightGBM Test Results:')
                print(f'  PR-AUC: {metrics["pr_auc"]:.4f}')
                print(f'  ROC-AUC: {metrics["roc_auc"]:.4f}')
                print(f'  F1: {metrics["f1"]:.4f}')
                
                # Log to MLflow.
                mlflow.log_params({
                    'model_type': 'lightgbm',
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 23,
                })
                mlflow.log_metrics({f'test_{k}': v for k, v in metrics.items()})
                
                # Log confusion matrix metrics.
                log_confusion_metrics(y_test, y_pred)
                
                # Log visualizations.
                log_pr_curve(y_test, y_prob, output_dir, 'lightgbm')
                log_confusion_matrix(y_test, y_pred, output_dir, 'lightgbm')
                
                # Log feature importance.
                log_feature_importance(lgb_model, feature_cols, output_dir, 'lightgbm')
                
                # Log model and artifacts.
                mlflow.sklearn.log_model(lgb_model, name='model', input_example=X_train[:5].astype('float64')) 
                mlflow.log_artifact(scaler_path)
                mlflow.log_artifact(features_path)
                
                # Save model locally.
                lgb_path = output_dir / 'lightgbm.pkl'
                with open(lgb_path, 'wb') as f:
                    pickle.dump(lgb_model, f)
                
                results['lightgbm'] = metrics
    
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
