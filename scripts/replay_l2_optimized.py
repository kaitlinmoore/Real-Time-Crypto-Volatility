import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from optimized featurizer - update this path as needed.
from featurizer_l2_optimized import FeatureComputer
from scripts.utilities import load_config, setup_logger


class ReplayProcessor:
    '''Processes saved raw data files to regenerate features.'''
    
    def __init__(self, products=None, window_sizes=[60, 300, 900], 
                 batch_size=10000, compute_every_n=1):
        
        config = load_config()
        
        if products is None:
            products = (config['data']['product_ids']['prediction_targets'] + 
                       config['data']['product_ids']['auxiliary_data'])
        
        self.products = products
        self.feature_computer = FeatureComputer(
            products=products, 
            window_sizes=window_sizes,
            compute_every_n=compute_every_n
        )
        self.logger = setup_logger('ReplayProcessor')
        
        self.features = []
        self.batch_size = batch_size
        self.output_path = None
        self.total_saved = 0
        self.compute_every_n = compute_every_n
        
    def process_file(self, filepath):
        '''Process a single NDJSON file.'''

        self.logger.info(f'Processing {filepath}')
        
        message_count = 0
        feature_count = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    
                    # Process message.
                    features = self.feature_computer.process_message(msg)
                    
                    if features:
                        self.features.append(features)
                        feature_count += 1
                    
                    message_count += 1
                    
                    # Save periodically to avoid memory bloat.
                    if len(self.features) >= self.batch_size:
                        self._save_batch()
                    
                    if message_count % 50000 == 0:
                        self.logger.info(
                            f'Processed {message_count} messages, '
                            f'generated {feature_count} features, '
                            f'saved {self.total_saved}'
                        )
                
                except json.JSONDecodeError:
                    continue
        
        self.logger.info(f'Completed {filepath}: {message_count} messages -> {feature_count} features')
        
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
        
        self.logger.info(f'Total: {total_messages} messages -> {total_features} features')
        
        return total_messages, total_features
    
    def _save_batch(self):
        '''Save current batch and clear buffer.'''
        
        if not self.features or not self.output_path:
            return
        
        df = pd.DataFrame(self.features)
        batch_count = len(df)
        
        # Append to existing or create new.
        output_file = Path(self.output_path)
        if output_file.exists():
            existing = pd.read_parquet(output_file)
            df = pd.concat([existing, df], ignore_index=True)
        
        df.to_parquet(output_file, index=False)
        
        self.total_saved += batch_count
        self.logger.info(f'Saved batch of {batch_count} features (total: {self.total_saved})')
        
        # Clear buffer.
        self.features = []
    
    def save_features(self, output_path):
        '''Save any remaining features to parquet file.'''

        self.output_path = output_path
        
        # Ensure output directory exists.
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save any remaining features in buffer.
        if self.features:
            self._save_batch()
        
        # Load and return final dataframe.
        if not output_path.exists():
            self.logger.warning('No features to save')
            return None
        
        df = pd.read_parquet(output_path)
        
        self.logger.info(f'Saved {len(df)} total feature rows to {output_path}')
        self.logger.info(f'Feature columns: {list(df.columns)}')
        self.logger.info(f'Products: {df["product_id"].unique()}')
        self.logger.info(f'Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Replay raw data to regenerate features')
    parser.add_argument('--raw', type=str, default='data/raw',
                       help='Directory containing raw NDJSON files or path to specific file')
    parser.add_argument('--out', type=str, default='data/processed/features_replay.parquet',
                       help='Output Parquet file path')
    parser.add_argument('--pattern', type=str, default='ticks_*.ndjson',
                       help='File pattern to match (if --raw is a directory)')
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Save features every N rows to avoid memory bloat')
    parser.add_argument('--compute-every-n', type=int, default=1,
                       help='Compute features every N ticker messages (higher = faster)')
    
    args = parser.parse_args()
    
    print(f'\n=== Optimized Replay (L2 Featurizer) ===')
    if args.compute_every_n > 1:
        print(f'Computing features every {args.compute_every_n} ticker messages')
        print(f'This reduces computation by ~{args.compute_every_n}x\n')
    
    # Remove existing output file to start fresh.
    output_path = Path(args.out)
    if output_path.exists():
        output_path.unlink()
        print(f'Removed existing {args.out}')
    
    # Initialize processor.
    processor = ReplayProcessor(
        batch_size=args.batch_size,
        compute_every_n=args.compute_every_n
    )
    processor.output_path = args.out
    
    # Process raw data.
    raw_path = Path(args.raw)
    
    if raw_path.is_file():
        processor.process_file(raw_path)
    elif raw_path.is_dir():
        processor.process_directory(raw_path, pattern=args.pattern)
    else:
        print(f'Error: {args.raw} is not a valid file or directory')
        return 1
    
    # Save any remaining features.
    df = processor.save_features(args.out)
    
    if df is not None:
        print('\n=== Replay Summary ===')
        print(f'Features generated: {len(df)}')
        print(f'Products: {", ".join(df["product_id"].unique())}')
        print(f'Feature columns: {len(df.columns)}')
        print(f'\nSample feature statistics:')
        
        for product in df['product_id'].unique():
            product_df = df[df['product_id'] == product]
            print(f'\n{product}:')
            print(f'  Observations: {len(product_df)}')
            if 'w60_return_std' in product_df.columns:
                print(f'  60s volatility mean: {product_df["w60_return_std"].mean():.6f}')
                print(f'  60s volatility std: {product_df["w60_return_std"].std():.6f}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
