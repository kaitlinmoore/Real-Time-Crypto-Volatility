import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utilities import setup_logger


def load_feature_columns(path='models/artifacts/feature_columns.json'):
    '''Load feature column names.'''
    
    with open(path, 'r') as f:
        return json.load(f)


def generate_drift_report(reference_df, current_df, feature_cols, output_path, 
                          target_col='target_spike'):
    '''Generate data drift report comparing reference and current data.'''
    
    logger = setup_logger('EvidentlyReport')
    
    # Subset to feature columns + target + metadata.
    cols_to_use = feature_cols.copy()
    if target_col and target_col in reference_df.columns:
        cols_to_use.append(target_col)
    if 'product_id' in reference_df.columns:
        cols_to_use.append('product_id')
    
    # Filter columns that exist in both datasets.
    cols_to_use = [c for c in cols_to_use if c in reference_df.columns and c in current_df.columns]
    
    reference = reference_df[cols_to_use].copy()
    current = current_df[cols_to_use].copy()
    
    logger.info(f'Reference data: {len(reference):,} rows')
    logger.info(f'Current data: {len(current):,} rows')
    logger.info(f'Features: {len(cols_to_use)}')
    
    # Build report - pass DataFrames directly for drift detection.
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    logger.info('Running Evidently analysis...')
    my_eval = report.run(
        reference_data=reference,
        current_data=current,
    )
    
    # Save report.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    my_eval.save_html(str(output_path))
    logger.info(f'Report saved to {output_path}')
    
    # Also save JSON version.
    json_path = output_path.with_suffix('.json')
    my_eval.save_json(str(json_path))
    logger.info(f'JSON report saved to {json_path}')
    
    return my_eval


def generate_train_test_report(train_path, test_path, feature_cols_path, output_dir):
    '''Generate drift report comparing training and test data.'''
    
    logger = setup_logger('TrainTestReport')
    
    # Load data.
    logger.info(f'Loading training data from {train_path}')
    train_df = pd.read_parquet(train_path)
    
    logger.info(f'Loading test data from {test_path}')
    test_df = pd.read_parquet(test_path)
    
    # Load feature columns.
    feature_cols = load_feature_columns(feature_cols_path)
    
    # Generate report.
    output_path = Path(output_dir) / 'train_test_drift.html'
    generate_drift_report(train_df, test_df, feature_cols, output_path)
    
    return output_path


def generate_temporal_report(features_path, feature_cols_path, output_dir, split_ratio=0.5):
    '''Generate drift report comparing early and late portions of data.'''
    
    logger = setup_logger('TemporalReport')
    
    # Load data.
    logger.info(f'Loading features from {features_path}')
    df = pd.read_parquet(features_path)
    
    # Sort by timestamp.
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split into early and late.
    split_idx = int(len(df) * split_ratio)
    early_df = df.iloc[:split_idx]
    late_df = df.iloc[split_idx:]
    
    logger.info(f'Early period: {len(early_df):,} rows')
    logger.info(f'Late period: {len(late_df):,} rows')
    
    # Load feature columns.
    feature_cols = load_feature_columns(feature_cols_path)
    
    # Generate report.
    output_path = Path(output_dir) / 'temporal_drift.html'
    generate_drift_report(early_df, late_df, feature_cols, output_path)
    
    return output_path


def generate_product_report(features_path, feature_cols_path, output_dir, 
                            reference_product='BTC-USD', current_product='ETH-USD'):
    '''Generate drift report comparing two products.'''
    
    logger = setup_logger('ProductReport')
    
    # Load data.
    logger.info(f'Loading features from {features_path}')
    df = pd.read_parquet(features_path)
    
    # Split by product.
    reference_df = df[df['product_id'] == reference_product]
    current_df = df[df['product_id'] == current_product]
    
    logger.info(f'{reference_product}: {len(reference_df):,} rows')
    logger.info(f'{current_product}: {len(current_df):,} rows')
    
    # Load feature columns.
    feature_cols = load_feature_columns(feature_cols_path)
    
    # Generate report.
    output_path = Path(output_dir) / f'product_drift_{reference_product}_vs_{current_product}.html'
    generate_drift_report(reference_df, current_df, feature_cols, output_path)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate Evidently drift reports')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train-test', 'temporal', 'product', 'custom'],
                       help='Report type to generate')
    parser.add_argument('--train', type=str, default='data/processed/train.parquet',
                       help='Path to training data')
    parser.add_argument('--test', type=str, default='data/processed/test.parquet',
                       help='Path to test data')
    parser.add_argument('--features', type=str, default='data/processed/features_labeled_perproduct.parquet',
                       help='Path to features file')
    parser.add_argument('--reference', type=str, default=None,
                       help='Path to reference data (custom mode)')
    parser.add_argument('--current', type=str, default=None,
                       help='Path to current data (custom mode)')
    parser.add_argument('--feature-cols', type=str, default='models/artifacts/feature_columns.json',
                       help='Path to feature columns JSON')
    parser.add_argument('--output-dir', type=str, default='reports/evidently',
                       help='Output directory for reports')
    parser.add_argument('--split-ratio', type=float, default=0.5,
                       help='Split ratio for temporal report')
    parser.add_argument('--ref-product', type=str, default='BTC-USD',
                       help='Reference product for product comparison')
    parser.add_argument('--cur-product', type=str, default='ETH-USD',
                       help='Current product for product comparison')
    
    args = parser.parse_args()
    
    # Create output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'train-test':
        output_path = generate_train_test_report(
            args.train, args.test, args.feature_cols, args.output_dir
        )
        print(f'\nTrain vs Test drift report: {output_path}')
    
    elif args.mode == 'temporal':
        output_path = generate_temporal_report(
            args.features, args.feature_cols, args.output_dir, args.split_ratio
        )
        print(f'\nTemporal drift report: {output_path}')
    
    elif args.mode == 'product':
        output_path = generate_product_report(
            args.features, args.feature_cols, args.output_dir,
            args.ref_product, args.cur_product
        )
        print(f'\nProduct comparison report: {output_path}')
    
    elif args.mode == 'custom':
        if not args.reference or not args.current:
            print('Error: --reference and --current required for custom mode')
            return 1
        
        feature_cols = load_feature_columns(args.feature_cols)
        reference_df = pd.read_parquet(args.reference)
        current_df = pd.read_parquet(args.current)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(args.output_dir) / f'custom_drift_{timestamp}.html'
        
        generate_drift_report(reference_df, current_df, feature_cols, output_path)
        print(f'\nCustom drift report: {output_path}')
    
    print(f'\nAll reports saved to: {args.output_dir}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
