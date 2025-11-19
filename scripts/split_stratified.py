import argparse
import sys
from pathlib import Path

import pandas as pd


def split_session_temporally(df, train_ratio=0.7, val_ratio=0.15):
    '''Split a single session by time into train/val/test.'''
    
    df = df.sort_values('timestamp')
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def load_and_split_session(filepath, session_num, train_ratio, val_ratio, exclude_products=None):
    '''Load a session file and split it temporally.'''
    
    if not filepath.exists():
        print(f'Warning: {filepath} not found, skipping')
        return None, None, None
    
    df = pd.read_parquet(filepath)
    df['session'] = session_num
    
    # Filter out excluded products.
    if exclude_products:
        before_len = len(df)
        df = df[~df['product_id'].isin(exclude_products)]
        excluded = before_len - len(df)
    else:
        excluded = 0
    
    # Split temporally.
    train, val, test = split_session_temporally(df, train_ratio, val_ratio)
    
    print(f'  Session {session_num}: {len(df)} rows (excluded {excluded})')
    print(f'    Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')
    
    return train, val, test


def report_split_stats(df, split_name):
    '''Report statistics for a data split.'''
    
    if df is None or len(df) == 0:
        print(f'{split_name}: No data')
        return
    
    print(f'\n{split_name}:')
    print(f'  Total rows: {len(df)}')
    print(f'  Sessions: {sorted(df["session"].unique())}')
    print(f'  Spike rate: {df["target_spike"].mean():.2%}')
    print(f'  Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    # Per-product breakdown
    for product in sorted(df['product_id'].unique()):
        product_df = df[df['product_id'] == product]
        spike_rate = product_df['target_spike'].mean()
        print(f'  {product}: {len(product_df)} rows, {spike_rate:.2%} spikes')


def main():
    parser = argparse.ArgumentParser(
        description='Split each session temporally, then combine across sessions'
    )
    parser.add_argument('--input-dir', type=str, default='data/processed/labeled/split',
                       help='Directory containing session files')
    parser.add_argument('--output-dir', type=str, default='data/processed/sesion_split',
                       help='Directory for output files')
    parser.add_argument('--sessions', type=str, default='1,2,3,4,5',
                       help='Comma-separated session numbers to include')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Fraction of each session for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Fraction of each session for validation (default: 0.15)')
    parser.add_argument('--pattern', type=str, default='session_{}_labeled_perproduct.parquet',
                       help='Filename pattern with {} placeholder for session number')
    parser.add_argument('--exclude-products', type=str, default=None,
                       help='Comma-separated product IDs to exclude (e.g., USDT-USD)')
    parser.add_argument('--output-suffix', type=str, default='_stratified',
                       help='Suffix for output files (default: _stratified)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse arguments.
    sessions = [int(x.strip()) for x in args.sessions.split(',')]
    
    exclude_products = None
    if args.exclude_products:
        exclude_products = [x.strip() for x in args.exclude_products.split(',')]
        print(f'Excluding products: {exclude_products}')
    
    print(f'Sessions: {sessions}')
    print(f'Split ratios: {args.train_ratio:.0%} train, {args.val_ratio:.0%} val, {1-args.train_ratio-args.val_ratio:.0%} test')
    
    # Collect splits from each session.
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    print('\nSplitting sessions...')
    for session_num in sessions:
        filepath = input_dir / args.pattern.format(session_num)
        
        train, val, test = load_and_split_session(
            filepath, session_num, args.train_ratio, args.val_ratio, exclude_products
        )
        
        if train is not None:
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
    
    # Combine across sessions.
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else None
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else None
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else None
    
    # Report statistics.
    print('\n' + '='*60)
    print('Split Statistics (Stratified by Session)')
    print('='*60)
    
    report_split_stats(train_df, 'Train')
    report_split_stats(val_df, 'Validation')
    report_split_stats(test_df, 'Test')
    
    # Save splits.
    print('\n' + '='*60)
    print('Saving splits')
    print('='*60)
    
    suffix = args.output_suffix
    
    if train_df is not None:
        train_path = output_dir / f'train{suffix}.parquet'
        train_df.to_parquet(train_path, index=False)
        print(f'\nSaved {train_path}')
    
    if val_df is not None:
        val_path = output_dir / f'val{suffix}.parquet'
        val_df.to_parquet(val_path, index=False)
        print(f'Saved {val_path}')
    
    if test_df is not None:
        test_path = output_dir / f'test{suffix}.parquet'
        test_df.to_parquet(test_path, index=False)
        print(f'Saved {test_path}')
    
    print('\nDone.')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
