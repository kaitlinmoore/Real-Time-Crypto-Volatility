import argparse
import json
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

mlflow.set_tracking_uri('http://localhost:5000')

# Target feature selection based on feature importance check. 
top_features = ['w60_spread_std', 'w300_spread_std', 'w30_spread_std', 'w60_spread_mean', 'w30_spread_mean',
                'w60_ob_microprice_mean', 'w900_spread_std', 'w60_ob_depth_imbalance_L1', 'w300_trade_imbalance',
                'w30_ob_microprice_mean', 'w30_ob_depth_imbalance_L1', 'w60_return_std', 'w60_volatility_lag1',
                'w300_spread_mean', 'w900_price_momentum', 'w60_trade_count', 'w60_tick_rate', 'w60_n_observations',
                'w300_price_momentum', 'w60_trade_imbalance', 'w30_volatility_lag1', 'w30_return_std', 'w900_trade_count',
                'w900_tick_rate', 'w900_n_observations', 'w30_price_std', 'w300_ob_microprice_mean', 'w30_price_range',
                'w60_price_std', 'w300_trade_count'
                ]


def load_data(train_path, val_path, test_path, exclude_products=None):
    '''Load and combine train/val for tuning, keep test separate.'''
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    if exclude_products:
        for product in exclude_products:
            train_df = train_df[train_df['product_id'] != product]
            val_df = val_df[val_df['product_id'] != product]
            test_df = test_df[test_df['product_id'] != product]
        print(f'Excluded: {exclude_products}')
    
    # Combine train and val for cross-validation tuning.
    tune_df = pd.concat([train_df, val_df], ignore_index=True)
    tune_df = tune_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f'Tuning set: {len(tune_df)} rows')
    print(f'Test set: {len(test_df)} rows')
    print(f'Products: {list(tune_df["product_id"].unique())}')
    
    return tune_df, test_df


def get_feature_columns(df):
    '''Get feature columns.'''

    feature_cols = [col for col in df.columns if col.startswith('w')]
    time_features = ['hour_sin', 'hour_cos', 'day_of_week']
    feature_cols.extend([f for f in time_features if f in df.columns])
    return feature_cols


def prepare_features(df, feature_cols, scaler=None):
    '''Prepare feature matrix.'''

    X = df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler


def tune_logistic_regression(X_tune, y_tune, X_test, y_test, n_splits=3):
    '''Tune Logistic Regression with grid search.'''
    
    print('Tuning Logistic Regression')
    
    # Parameter grid.
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],  # Supports both L1 and L2.
        'class_weight': ['balanced'],
        'max_iter': [10000], # Try 10000 potentially?
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # PR-AUC scorer
    pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
    
    # Grid search
    model = LogisticRegression(random_state=23)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    
    grid_search.fit(X_tune, y_tune)
    
    # Results
    print(f'\nBest parameters: {grid_search.best_params_}')
    print(f'Best CV PR-AUC: {grid_search.best_score_:.4f}')
    
    # Evaluate on test.
    best_model = grid_search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, y_prob)
    
    print(f'Test PR-AUC: {test_pr_auc:.4f}')
    
    # Log all runs to MLflow.
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    for i, row in results_df.iterrows():
        with mlflow.start_run(run_name=f'lr_tune_{i}'):
            mlflow.log_params(row['params'])
            mlflow.log_metric('cv_pr_auc_mean', row['mean_test_score'])
            mlflow.log_metric('cv_pr_auc_std', row['std_test_score'])
    
    # Log best model.
    with mlflow.start_run(run_name='lr_best'):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('cv_pr_auc', grid_search.best_score_)
        mlflow.log_metric('test_pr_auc', test_pr_auc)
        mlflow.sklearn.log_model(best_model, artifact_path='model', input_example=X_tune[:5])
    
    return best_model, grid_search.best_params_, test_pr_auc


def tune_xgboost(X_tune, y_tune, X_test, y_test, n_splits=3):
    '''Tune XGBoost with grid search.'''
    
    if not HAS_XGBOOST:
        print('XGBoost not installed')
        return None, None, None
    
    print('Tuning XGBoost')
    
    # Calculate scale_pos_weight.
    neg_count = (y_tune == 0).sum()
    pos_count = (y_tune == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    # Parameter grid.
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    
    # Smaller grid for faster tuning - uncomment for quick test.
    # param_grid = {
    #     'n_estimators': [100],
    #     'max_depth': [2, 3, 4],
    #     'learning_rate': [0.05, 0.1],
    #     'min_child_weight': [1, 3],
    #     'subsample': [0.8],
    #     'colsample_bytree': [0.8],
    # }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # PR-AUC scorer
    pr_auc_scorer = make_scorer(average_precision_score, response_method='predict_proba')
    
    # Grid search
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=23,
        eval_metric='aucpr',
    )
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=tscv,
        scoring=pr_auc_scorer,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    
    grid_search.fit(X_tune, y_tune)
    
    # Results
    print(f'\nBest parameters: {grid_search.best_params_}')
    print(f'Best CV PR-AUC: {grid_search.best_score_:.4f}')
    
    # Evaluate on test.
    best_model = grid_search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, y_prob)
    
    print(f'Test PR-AUC: {test_pr_auc:.4f}')
    
    # Log best model to MLflow.
    with mlflow.start_run(run_name='xgb_best'):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('cv_pr_auc', grid_search.best_score_)
        mlflow.log_metric('test_pr_auc', test_pr_auc)
        mlflow.sklearn.log_model(best_model, name='model', input_example=X_tune[:5])
    
    return best_model, grid_search.best_params_, test_pr_auc


def main():
    parser = argparse.ArgumentParser(description='Tune volatility prediction models')
    parser.add_argument('--train', type=str, default='data/processed/train.parquet')
    parser.add_argument('--val', type=str, default='data/processed/val.parquet')
    parser.add_argument('--test', type=str, default='data/processed/test.parquet')
    parser.add_argument('--output-dir', type=str, default='models/artifacts')
    parser.add_argument('--experiment-name', type=str, default='volatility-tuning')
    parser.add_argument('--exclude-usdt', default=True)
    parser.add_argument('--model', type=str, default='both',
                       choices=['lr', 'xgb', 'both'],
                       help='Which model to tune')
    parser.add_argument('--n-splits', type=int, default=3,
                       help='Number of CV splits')
    parser.add_argument('--quick', action='store_true',
                       help='Use smaller parameter grid for faster tuning')
    parser.add_argument('--exclude-products', type=str, default=None,
                    help='Comma-separated list of products to exclude (e.g., "USDT-USD,SOL-USD")')
    parser.add_argument('--use-top-features', action='store_true',
                    help='Use only top 30 features by importance.')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exclude = ['USDT-USD'] if args.exclude_usdt else None
    
    # Load data.
    print('Loading data...')
    tune_df, test_df = load_data(args.train, args.val, args.test, exclude)
    
    # Prepare features.
    feature_cols = get_feature_columns(tune_df)
    print(f'Using {len(feature_cols)} features')
    
    X_tune, scaler = prepare_features(tune_df, feature_cols)
    X_test, _ = prepare_features(test_df, feature_cols, scaler)
    
    y_tune = tune_df['target_spike'].values
    y_test = test_df['target_spike'].values
    
    # Save scaler and features.
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Setup MLflow.
    mlflow.set_experiment(args.experiment_name)
    
    results = {}
    
    # Tune models.
    if args.model in ['lr', 'both']:
        lr_model, lr_params, lr_score = tune_logistic_regression(
            X_tune, y_tune, X_test, y_test, args.n_splits
        )
        results['logistic_regression'] = {
            'params': lr_params,
            'test_pr_auc': lr_score
        }
        
        with open(output_dir / 'logistic_regression_tuned.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
    
    if args.model in ['xgb', 'both']:
        xgb_model, xgb_params, xgb_score = tune_xgboost(
            X_tune, y_tune, X_test, y_test, args.n_splits
        )
        if xgb_model:
            results['xgboost'] = {
                'params': xgb_params,
                'test_pr_auc': xgb_score
            }
            
            with open(output_dir / 'xgboost_tuned.pkl', 'wb') as f:
                pickle.dump(xgb_model, f)
    
    # Summary
    print('\n' + '='*60)
    print('Tuning Results Summary')
    print('='*60)
    
    for model_name, result in results.items():
        print(f'\n{model_name}:')
        print(f'  Test PR-AUC: {result["test_pr_auc"]:.4f}')
        print(f'  Best params: {result["params"]}')
    
    if results:
        best = max(results.items(), key=lambda x: x[1]['test_pr_auc'])
        print(f'\nBest model: {best[0]} (PR-AUC: {best[1]["test_pr_auc"]:.4f})')
    
    print(f'\nArtifacts saved to: {output_dir}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
