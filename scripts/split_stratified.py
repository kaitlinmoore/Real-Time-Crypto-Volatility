import argparse
import pandas as pd
import sys

from pathlib import Path




def split_session_temporally(df, train_ratio=0.7, val_ratio=0.15):
    '''Split a single session by time into train/val/test.'''
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def load_and_split_session(input_dir, pattern, session_num, train_ratio, val_ratio, exclude_products=None):
    '''Load session file(s) matching glob pattern and split temporally.'''
    
    # Format the pattern with session number, then glob for matches.
    glob_pattern = pattern.format(session_num)
    matching_files = sorted(input_dir.glob(glob_pattern))
    
    if not matching_files:
        print(f'Warning: No files matching {glob_pattern} in {input_dir}, skipping')
        return None, None, None
    
    # Load and concatenate all matching files.
    dfs = []
    for filepath in matching_files:
        dfs.append(pd.read_parquet(filepath))
        print(f'    Loaded: {filepath.name}')
    
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    df['session'] = session_num
    
    # Filter out excluded products.
    if exclude_products:
        before_len = len(df)
        df = df[~df['product_id'].isin(exclude_products)]
        excluded = before_len - len(df)
    else:
        excluded = 0
    
    # Check if dataframe is empty after filtering.
    if len(df) == 0:
        print(f'  Session {session_num}: No data after filtering')
        return None, None, None
    
    # Split temporally.
    train, val, test = split_session_temporally(df, train_ratio, val_ratio)
    
    print(f'  Session {session_num}: {len(df)} rows from {len(matching_files)} file(s) (excluded {excluded})')
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
    
    # Check if target_spike column exists before accessing it.
    if 'target_spike' in df.columns:
        print(f'  Spike rate: {df["target_spike"].mean():.2%}')
    
    print(f'  Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    
    # Per-product breakdown.
    for product in sorted(df['product_id'].unique()):
        product_df = df[df['product_id'] == product]
        if 'target_spike' in df.columns:
            spike_rate = product_df['target_spike'].mean()
            print(f'  {product}: {len(product_df)} rows, {spike_rate:.2%} spikes')
        else:
            print(f'  {product}: {len(product_df)} rows')


def main():
    parser = argparse.ArgumentParser(
        description='Split each session temporally, then combine across sessions'
    )
    parser.add_argument('--input-dir', type=str, default='data/processed/labeled/split',
                       help='Directory containing session files')
    parser.add_argument('--output-dir', type=str, default='data/processed/session_split',
                       help='Directory for output files')
    parser.add_argument('--sessions', type=str, default='1,2,3,4,5',
                       help='Comma-separated session numbers to include')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Fraction of each session for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Fraction of each session for validation (default: 0.15)')
    parser.add_argument('--pattern', type=str, default='session_{}_labeled_perproduct.parquet',
                       help='Filename pattern with {} for session number (supports glob wildcards like *)')
    parser.add_argument('--exclude-products', type=str, default=None,
                       help='Comma-separated product IDs to exclude (e.g., USDT-USD)')
    parser.add_argument('--output-suffix', type=str, default='_stratified',
                       help='Suffix for output files (default: _stratified)')
    
    args = parser.parse_args()
    
    # Validate ratios.
    total_ratio = args.train_ratio + args.val_ratio
    if total_ratio > 1.0:
        print(f'Error: train_ratio + val_ratio must be <= 1.0 (got {total_ratio})')
        return 1
    
    if args.train_ratio < 0 or args.val_ratio < 0:
        print('Error: Ratios must be non-negative')
        return 1
    
    test_ratio = 1.0 - total_ratio
    if test_ratio < 0:
        print(f'Error: No room left for test set (test_ratio = {test_ratio})')
        return 1
    
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
    print(f'Pattern: {args.pattern}')
    print(f'Split ratios: {args.train_ratio:.0%} train, {args.val_ratio:.0%} val, {test_ratio:.0%} test')
    
    # Collect splits from each session.
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    print('\nSplitting sessions...')
    for session_num in sessions:
        train, val, test = load_and_split_session(
            input_dir, args.pattern, session_num, 
            args.train_ratio, args.val_ratio, exclude_products
        )
        
        if train is not None:
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
    
    # Check if any data was loaded.
    if not train_dfs:
        print('\nError: No session data was loaded. Check input directory and file pattern.')
        return 1
    
    # Combine across sessions.
    train_df = pd.concat(train_dfs, ignore_index=True)
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
    
    if train_df is not None and len(train_df) > 0:
        train_path = output_dir / f'train{suffix}.parquet'
        train_df.to_parquet(train_path, index=False)
        print(f'\nSaved {train_path}')
    
    if val_df is not None and len(val_df) > 0:
        val_path = output_dir / f'val{suffix}.parquet'
        val_df.to_parquet(val_path, index=False)
        print(f'Saved {val_path}')
    
    if test_df is not None and len(test_df) > 0:
        test_path = output_dir / f'test{suffix}.parquet'
        test_df.to_parquet(test_path, index=False)
        print(f'Saved {test_path}')
    
    print('\nDone.')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
