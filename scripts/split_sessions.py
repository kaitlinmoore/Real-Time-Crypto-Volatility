import argparse
import sys
from pathlib import Path

import pandas as pd


def load_and_concatenate(input_dir, session_numbers, pattern='session_{}_labeled_perproduct.parquet', exclude_products=None):
    '''Load and concatenate multiple session files.'''

    dfs = []
    for session_num in session_numbers:
        filepath = input_dir / pattern.format(session_num)
        if not filepath.exists():
            print(f'Warning: {filepath} not found, skipping')
            continue
        
        df = pd.read_parquet(filepath)
        df['session'] = session_num
        
        # Filter out excluded products.
        if exclude_products:
            before_len = len(df)
            df = df[~df['product_id'].isin(exclude_products)]
            print(f'  Loaded session {session_num}: {len(df)} rows (excluded {before_len - len(df)})')
        else:
            print(f'  Loaded session {session_num}: {len(df)} rows')
        
        dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


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
    
    # Per-product breakdown.
    for product in sorted(df['product_id'].unique()):
        product_df = df[df['product_id'] == product]
        spike_rate = product_df['target_spike'].mean()
        print(f'  {product}: {len(product_df)} rows, {spike_rate:.2%} spikes')


def main():
    parser = argparse.ArgumentParser(description='Split session files into train/val/test')
    parser.add_argument('--input-dir', type=str, default='data/processed',
                       help='Directory containing session files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory for output files')
    parser.add_argument('--train-sessions', type=str, default='1,2,3',
                       help='Comma-separated session numbers for training')
    parser.add_argument('--val-sessions', type=str, default='4',
                       help='Comma-separated session numbers for validation')
    parser.add_argument('--test-sessions', type=str, default='5',
                       help='Comma-separated session numbers for testing')
    parser.add_argument('--pattern', type=str, default='session_{}_labeled_perproduct.parquet',
                       help='Filename pattern with {} placeholder for session number')
    parser.add_argument('--exclude-products', type=str, default=None,
                       help='Comma-separated product IDs to exclude (e.g., USDT-USD)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse session numbers.
    train_sessions = [int(x.strip()) for x in args.train_sessions.split(',')]
    val_sessions = [int(x.strip()) for x in args.val_sessions.split(',')]
    test_sessions = [int(x.strip()) for x in args.test_sessions.split(',')]
    
    # Parse excluded products.
    exclude_products = None
    if args.exclude_products:
        exclude_products = [x.strip() for x in args.exclude_products.split(',')]
        print(f'Excluding products: {exclude_products}')
    
    print(f'Train sessions: {train_sessions}')
    print(f'Val sessions: {val_sessions}')
    print(f'Test sessions: {test_sessions}')
    
    # Load and concatenate each split.
    print('\nLoading training data...')
    train_df = load_and_concatenate(input_dir, train_sessions, args.pattern, exclude_products)
    
    print('\nLoading validation data...')
    val_df = load_and_concatenate(input_dir, val_sessions, args.pattern, exclude_products)
    
    print('\nLoading test data...')
    test_df = load_and_concatenate(input_dir, test_sessions, args.pattern, exclude_products)
    
    # Report statistics.
    print('\n' + '='*60)
    print('Split Statistics')
    print('='*60)
    
    report_split_stats(train_df, 'Train')
    report_split_stats(val_df, 'Validation')
    report_split_stats(test_df, 'Test')
    
    # Save splits.
    print('\n' + '='*60)
    print('Saving splits')
    print('='*60)
    
    if train_df is not None:
        train_path = output_dir / 'train.parquet'
        train_df.to_parquet(train_path, index=False)
        print(f'\nSaved {train_path}')
    
    if val_df is not None:
        val_path = output_dir / 'val.parquet'
        val_df.to_parquet(val_path, index=False)
        print(f'Saved {val_path}')
    
    if test_df is not None:
        test_path = output_dir / 'test.parquet'
        test_df.to_parquet(test_path, index=False)
        print(f'Saved {test_path}')
    
    print('\nDone.')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
