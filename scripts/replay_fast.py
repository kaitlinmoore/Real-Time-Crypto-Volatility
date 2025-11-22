import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports.
sys.path.append(str(Path(__file__).parent.parent))
from feature_utils import FeatureComputer, DEFAULT_WINDOW_SIZES
from utilities import load_config, setup_logger


class ReplayProcessor:
    '''Processes saved raw data files to regenerate features. Uses temp file batching to handle large datasets efficiently.'''
    
    def __init__(self, products=None, window_sizes=None, 
                 batch_size=50000, compute_every_n=1):
        
        config = load_config()
        
        if products is None:
            products = (
                config['data']['product_ids']['prediction_targets'] + 
                config['data']['product_ids']['auxiliary_data']
            )
        
        self.products = products
        self.feature_computer = FeatureComputer(
            products=products, 
            window_sizes=window_sizes,
            compute_every_n=compute_every_n
        )
        self.logger = setup_logger('ReplayProcessor')
        
        # Temp file batching for memory efficiency.
        self.features = []
        self.batch_size = batch_size
        self.batch_files = []
        self.temp_dir = None
        self.total_features = 0
        self.run_id = f'{int(time.time() * 1000)}_{id(self)}'
        
    def process_file(self, filepath):
        '''Process a single NDJSON file.'''

        self.logger.info(f'Processing {filepath}')
        
        message_count = 0
        feature_count = 0
        start_time = time.time()
        
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    features = self.feature_computer.process_message(msg)
                    
                    if features:
                        self.features.append(features)
                        feature_count += 1
                    
                    message_count += 1
                    
                    # Save batch to temp file.
                    if len(self.features) >= self.batch_size:
                        self._save_batch()
                    
                    if message_count % 100000 == 0:
                        elapsed = time.time() - start_time
                        rate = message_count / elapsed
                        self.logger.info(
                            f'Processed {message_count:,} messages '
                            f'({rate:,.0f} msg/s), '
                            f'features: {self.total_features + len(self.features):,}'
                        )
                
                except json.JSONDecodeError:
                    continue
        
        elapsed = time.time() - start_time
        rate = message_count / elapsed if elapsed > 0 else 0
        self.logger.info(
            f'Completed {filepath}: {message_count:,} messages -> '
            f'{feature_count:,} features ({rate:,.0f} msg/s)'
        )
        
        return message_count, feature_count
    
    def process_directory(self, directory, pattern='ticks_*.ndjson'):
        '''Process all matching files in a directory.'''

        dir_path = Path(directory)
        files = sorted(dir_path.glob(pattern))
        
        self.logger.info(f'Found {len(files)} files matching {pattern}')
        
        total_messages = 0
        total_features = 0
        
        for filepath in files:
            msg_count, feat_count = self.process_file(filepath)
            total_messages += msg_count
            total_features += feat_count
        
        self.logger.info(f'Total: {total_messages:,} messages -> {total_features:,} features')
        
        return total_messages, total_features
    
    def _save_batch(self):
        '''Save current batch to temporary file.'''

        if not self.features:
            return
        
        if self.temp_dir is None:
            self.temp_dir = Path(f'data/processed/.temp_batches_{self.run_id}')
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        batch_file = self.temp_dir / f'batch_{len(self.batch_files):04d}.parquet'
        df = pd.DataFrame(self.features)
        df.to_parquet(batch_file, index=False)
        
        self.batch_files.append(batch_file)
        self.total_features += len(self.features)
        self.features = []
    
    def save_features(self, output_path):
        '''Combine all batches and save final output.'''

        # Save any remaining features.
        if self.features:
            self._save_batch()
        
        if not self.batch_files:
            self.logger.warning('No features to save')
            return None
        
        # Combine all batch files.
        self.logger.info(f'Combining {len(self.batch_files)} batch files...')
        
        dfs = []
        for batch_file in self.batch_files:
            dfs.append(pd.read_parquet(batch_file))
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Save final output.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        # Cleanup temp files.
        for batch_file in self.batch_files:
            try:
                batch_file.unlink()
            except Exception:
                pass
        
        if self.temp_dir and self.temp_dir.exists():
            try:
                self.temp_dir.rmdir()
            except OSError:
                pass
        
        self.logger.info(f'Saved {len(df):,} total feature rows to {output_path}')
        self.logger.info(f'Feature columns: {len(df.columns)}')
        self.logger.info(f'Products: {list(df["product_id"].unique())}')
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Replay raw data to regenerate features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Show syntax of args as reminder.
        epilog='''
Examples:
    # Process all files in directory
    python replay.py --raw data/raw --out data/processed/features.parquet
    
    # Process single file
    python replay.py --raw data/raw/ticks_20241115.ndjson --out features.parquet
    
    # Faster processing (compute every 10th ticker)
    python replay.py --raw data/raw --compute-every-n 10
    
    # Custom window sizes
    python replay.py --raw data/raw --windows 30,60,300
        '''
    )
    parser.add_argument('--raw', type=str, default='data/raw',
                       help='Directory or file containing raw NDJSON files')
    parser.add_argument('--out', type=str, default='data/processed/features_replay.parquet',
                       help='Output Parquet file path')
    parser.add_argument('--pattern', type=str, default='ticks_*.ndjson',
                       help='File pattern to match (for directory processing)')
    parser.add_argument('--compute-every-n', type=int, default=1,
                       help='Compute features every N ticker messages (default: 1)')
    parser.add_argument('--windows', type=str, default=None,
                       help=f'Comma-separated window sizes in seconds (default: {DEFAULT_WINDOW_SIZES})')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='Batch size for temp file saves (default: 50000)')
    
    args = parser.parse_args()
    
    # Parse window sizes.
    window_sizes = None
    if args.windows:
        window_sizes = [int(w) for w in args.windows.split(',')]
    
    print(f'\n=== Replay Processing ===')
    print(f'Window sizes: {window_sizes or DEFAULT_WINDOW_SIZES}')
    print(f'Computing every {args.compute_every_n} tickers')
    if args.compute_every_n > 1:
        print(f'Expected speedup: ~{args.compute_every_n}x\n')
    
    start_time = time.time()
    
    # Initialize processor.
    processor = ReplayProcessor(
        window_sizes=window_sizes,
        compute_every_n=args.compute_every_n,
        batch_size=args.batch_size
    )
    
    # Process raw data.
    raw_path = Path(args.raw)
    
    if raw_path.is_file():
        processor.process_file(raw_path)
    elif raw_path.is_dir():
        processor.process_directory(raw_path, pattern=args.pattern)
    else:
        print(f'Error: {args.raw} is not a valid file or directory')
        return 1
    
    # Save features.
    df = processor.save_features(args.out)
    
    elapsed = time.time() - start_time
    
    if df is not None:
        print(f'\n=== Replay Complete ===')
        print(f'Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
        print(f'Features generated: {len(df):,}')
        print(f'Products: {", ".join(df["product_id"].unique())}')
        
        for product in df['product_id'].unique():
            product_df = df[df['product_id'] == product]
            print(f'\n{product}: {len(product_df):,} observations')
            if 'w60_return_std' in product_df.columns:
                print(f'  60s volatility mean: {product_df["w60_return_std"].mean():.6f}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
