import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from utilities import setup_logger


def compute_forward_volatility(df, product_id, n_periods=6):
    '''Compute std of the next n_periods returns starting from t+1.'''
    
    # Filter to specific product and sort by timestamp.
    product_df = df[df['product_id'] == product_id].copy()
    product_df = product_df.sort_values('timestamp').reset_index(drop=True)
    
    # Get price series from 60s mean price.
    prices = product_df['w60_price_mean'].values
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, np.nan)
    
    # Calculate target.
    forward_vols = []
    for i in range(len(returns)):
        future_returns = returns[i+1 : i+1+n_periods]
        if len(future_returns) > 0:
            forward_vols.append(np.nanstd(future_returns))
        else:
            forward_vols.append(np.nan)
    
    return pd.Series(forward_vols, index=product_df['timestamp'])


def add_target_labels(df, threshold_percentile=95, horizon_seconds=60):
    '''Add target labels to feature DataFrame using PER-PRODUCT thresholds.'''

    logger = setup_logger('TargetLabeler')
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize columns.
    df['target_volatility'] = np.nan
    df['target_spike'] = 0
    df['target_threshold'] = np.nan
    
    # Compute forward volatility for each product.
    products = df['product_id'].unique()
    
    # Compute volatility for all products.
    for product in products:
        logger.info(f'Computing forward volatility for {product}')
        
        # Compute forward volatility (returns Series indexed by timestamp).
        forward_vol = compute_forward_volatility(df, product, horizon_seconds=horizon_seconds)
        
        # Convert to dictionary for easy lookup.
        vol_dict = forward_vol.to_dict()
        
        # Map values back to original dataframe.
        product_mask = df['product_id'] == product
        df.loc[product_mask, 'target_volatility'] = df.loc[product_mask, 'timestamp'].map(vol_dict)
    
    # Compute threshold and labels separately for each product.
    logger.info(f'\nComputing per-product thresholds at p{threshold_percentile}:')
    
    for product in products:
        product_mask = df['product_id'] == product
        product_vols = df.loc[product_mask, 'target_volatility'].dropna()
        
        if len(product_vols) == 0:
            logger.warning(f'No valid volatility values for {product}')
            continue
        
        # Compute threshold for this product.
        product_threshold = product_vols.quantile(threshold_percentile / 100)
        
        # Create binary label for this product.
        df.loc[product_mask, 'target_spike'] = (
            df.loc[product_mask, 'target_volatility'] >= product_threshold
        ).astype(int)
        
        # Store threshold for this product.
        df.loc[product_mask, 'target_threshold'] = product_threshold
        
        # Log stats for this product.
        spike_count = df.loc[product_mask, 'target_spike'].sum()
        total_count = product_mask.sum()
        spike_rate = spike_count / total_count if total_count > 0 else 0
        
        logger.info(f'  {product}:')
        logger.info(f'    Threshold: {product_threshold:.8f}')
        logger.info(f'    Spikes: {spike_count}/{total_count} ({spike_rate:.2%})')
        logger.info(f'    Vol mean: {product_vols.mean():.8f}')
        logger.info(f'    Vol std: {product_vols.std():.8f}')
    
    # Add metadata.
    df['target_horizon_sec'] = horizon_seconds
    
    # Overall statistics
    logger.info(f'\nOverall Statistics:')
    logger.info(f'  Total samples: {len(df)}')
    logger.info(f'  Total spikes: {df["target_spike"].sum()}')
    logger.info(f'  Overall spike rate: {df["target_spike"].mean():.2%}')
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Add target labels with per-product thresholds'
    )
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features Parquet file')
    parser.add_argument('--out', type=str, default=None,
                       help='Output path (default: same as input with _labeled suffix)')
    parser.add_argument('--threshold-percentile', type=float, default=95,
                       help='Percentile for spike threshold (default: 95, applied per product)')
    parser.add_argument('--horizon', type=int, default=60,
                       help='Prediction horizon in seconds (default: 60)')
    
    args = parser.parse_args()
    
    logger = setup_logger('TargetLabeler')
    
    # Load features.
    logger.info(f'Loading features from {args.features}')
    df = pd.read_parquet(args.features)
    
    logger.info(f'Loaded {len(df)} feature rows')
    logger.info(f'Products: {df["product_id"].unique()}')
    
    # Add labels.
    df_labeled = add_target_labels(
        df, 
        threshold_percentile=args.threshold_percentile,
        horizon_seconds=args.horizon
    )
    
    # Determine output path.
    if args.out is None:
        input_path = Path(args.features)
        output_path = input_path.parent / f'{input_path.stem}_labeled_perproduct{input_path.suffix}'
    else:
        output_path = Path(args.out)
    
    # Save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_parquet(output_path, index=False)
    
    logger.info(f'\nSaved labeled features to {output_path}')
    
    # Show distribution by product.

    print('### Label Distribution by Product ###')

    for product in df_labeled['product_id'].unique():
        product_df = df_labeled[df_labeled['product_id'] == product]
        spike_rate = product_df['target_spike'].mean()
        threshold = product_df['target_threshold'].iloc[0]
        
        print(f'\n{product}:')
        print(f'  Total samples: {len(product_df)}')
        print(f'  Spikes: {product_df["target_spike"].sum()} ({spike_rate:.2%})')
        print(f'  Threshold: {threshold:.8f}')
        
        if 'target_volatility' in product_df.columns:
            vol_stats = product_df['target_volatility'].describe()
            print(f'  Volatility mean: {vol_stats["mean"]:.8f}')
            print(f'  Volatility std: {vol_stats["std"]:.8f}')
            print(f'  Volatility min: {vol_stats["min"]:.8f}')
            print(f'  Volatility max: {vol_stats["max"]:.8f}')
    
    print('\n' + '='*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
