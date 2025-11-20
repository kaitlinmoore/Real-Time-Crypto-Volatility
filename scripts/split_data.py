import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utilities import setup_logger


def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    '''Split data chronologically by timestamp.    '''
    logger = setup_logger('DataSplitter')
    
    # Sort by timestamp.
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f'Total samples: {n}')
    logger.info(f'Train: {len(train_df)} ({len(train_df)/n:.1%})')
    logger.info(f'Val: {len(val_df)} ({len(val_df)/n:.1%})')
    logger.info(f'Test: {len(test_df)} ({len(test_df)/n:.1%})')
    
    # Show time ranges.
    logger.info(f'\nTime ranges:')
    logger.info(f'  Train: {train_df["timestamp"].min()} to {train_df["timestamp"].max()}')
    logger.info(f'  Val: {val_df["timestamp"].min()} to {val_df["timestamp"].max()}')
    logger.info(f'  Test: {test_df["timestamp"].min()} to {test_df["timestamp"].max()}')
    
    # Show spike rates per split.
    if 'target_spike' in df.columns:
        logger.info(f'\nSpike rates:')
        logger.info(f'  Train: {train_df["target_spike"].mean():.2%}')
        logger.info(f'  Val: {val_df["target_spike"].mean():.2%}')
        logger.info(f'  Test: {test_df["target_spike"].mean():.2%}')
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Split labeled data into train/val/test')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to labeled features Parquet file')
    parser.add_argument('--out-dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--suffix', type=str, default=None,
                       help='File suffix to append for sanity')
    
    args = parser.parse_args()
    
    # Validate ratios.
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f'Error: Ratios must sum to 1.0 (got {total_ratio})')
        return 1
    
    logger = setup_logger('DataSplitter')
    
    # Load data.
    logger.info(f'Loading {args.features}')
    df = pd.read_parquet(args.features)
    
    # Split data.
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Determine output directory.
    input_path = Path(args.features)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = input_path.parent
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits.
    train_path = out_dir / f'train{args.suffix}.parquet'
    val_path = out_dir / f'val{args.suffix}.parquet'
    test_path = out_dir / f'test{args.suffix}.parquet'
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    logger.info(f'\nSaved splits to {out_dir}:')
    logger.info(f'  {train_path.name}: {len(train_df)} rows')
    logger.info(f'  {val_path.name}: {len(val_df)} rows')
    logger.info(f'  {test_path.name}: {len(test_df)} rows')
    
    # Show per-product breakdown.
    print('\n=== Per-Product Split Summary ===')
    for product in df['product_id'].unique():
        train_count = len(train_df[train_df['product_id'] == product])
        val_count = len(val_df[val_df['product_id'] == product])
        test_count = len(test_df[test_df['product_id'] == product])
        
        print(f'\n{product}:')
        print(f'  Train: {train_count}')
        print(f'  Val: {val_count}')
        print(f'  Test: {test_count}')
        
        if 'target_spike' in df.columns:
            train_spikes = train_df[train_df['product_id'] == product]['target_spike'].sum()
            val_spikes = val_df[val_df['product_id'] == product]['target_spike'].sum()
            test_spikes = test_df[test_df['product_id'] == product]['target_spike'].sum()
            print(f'  Spikes: {train_spikes}/{val_spikes}/{test_spikes}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
