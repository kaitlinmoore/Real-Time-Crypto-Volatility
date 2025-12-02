
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Add parent directory to path for imports.
sys.path.append(str(Path(__file__).parent.parent))
from feature_utils import FeatureComputer
from utilities import load_config, setup_logger


class Featurizer:
    '''Main featurizer that consumes from Kafka and produces features.'''
    
    def __init__(self, topic_in='ticks.raw', topic_out='ticks.features', 
                 products=None, save_parquet=True, 
                 parquet_path='data/processed/features.parquet',
                 window_sizes=None, compute_every_n=1):
        
        config = load_config()
        
        if products is None:
            products = (
                config['data']['product_ids']['prediction_targets'] + 
                config['data']['product_ids']['auxiliary_data']
            )
        
        self.products = products
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.save_parquet = save_parquet
        self.parquet_path = Path(parquet_path)
        
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger('Featurizer')
        
        # Kafka setup
        bootstrap_servers = config['kafka']['bootstrap_servers']
        self.consumer = KafkaConsumer(
            topic_in,
            bootstrap_servers=[bootstrap_servers],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='featurizer-group'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
        )
        
        # Feature computation (uses shared module)
        self.feature_computer = FeatureComputer(
            products=products,
            window_sizes=window_sizes,
            compute_every_n=compute_every_n
        )
        
        # Temp file batching
        self.feature_buffer = []
        self.buffer_size = 5000
        self.batch_files = []
        self.run_id = f'{int(time.time() * 1000)}_{id(self)}'
        self.temp_dir = None
        
        self.logger.info(f'Featurizer initialized for products: {products}')
        self.logger.info(f'Consuming from: {topic_in}')
        self.logger.info(f'Publishing to: {topic_out}')
        self.logger.info(f'Window sizes: {self.feature_computer.window_sizes}')
        if save_parquet:
            self.logger.info(f'Saving to: {parquet_path}')
    
    def run(self, duration_minutes=None):
        '''Run the featurizer.'''
        start_time = time.time()
        message_count = 0
        feature_count = 0
        
        try:
            self.logger.info('Starting featurizer...')
            
            while True:
                if duration_minutes:
                    if (time.time() - start_time) / 60 >= duration_minutes:
                        self.logger.info(f'Duration limit reached ({duration_minutes} min)')
                        break
                
                msg_batch = self.consumer.poll(timeout_ms=1000)
                
                if not msg_batch:
                    continue
                
                for topic_partition, messages in msg_batch.items():
                    for message in messages:
                        msg = message.value
                        
                        features = self.feature_computer.process_message(msg)
                        
                        if features:
                            self.producer.send(self.topic_out, value=features)
                            
                            self.feature_buffer.append(features)
                            feature_count += 1
                            
                            if len(self.feature_buffer) >= self.buffer_size:
                                self._save_batch()
                        
                        message_count += 1
                        
                        if message_count % 5000 == 0:
                            elapsed = time.time() - start_time
                            rate = message_count / elapsed
                            self.logger.info(
                                f'Processed {message_count} msgs, '
                                f'generated {feature_count} features ({rate:.1f} msg/s)'
                            )
        
        except KeyboardInterrupt:
            self.logger.info('Interrupted by user')
        
        finally:
            # Save remaining buffer and combine all batches.
            self._finalize_output()
            self.cleanup()
            
            elapsed = time.time() - start_time
            self.logger.info(f'Featurizer stopped after {elapsed:.1f}s')
            self.logger.info(f'Processed {message_count} messages')
            self.logger.info(f'Generated {feature_count} feature rows')
    
    def _save_batch(self):
        '''Save current batch to a temp file (O(1) per batch).'''
        if not self.save_parquet or not self.feature_buffer:
            return
        
        # Create temp directory on first batch.
        if self.temp_dir is None:
            self.temp_dir = self.parquet_path.parent / f'.temp_batches_{self.run_id}'
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save batch to numbered temp file.
        batch_file = self.temp_dir / f'batch_{len(self.batch_files):04d}.parquet'
        df = pd.DataFrame(self.feature_buffer)
        df.to_parquet(batch_file, index=False)
        
        self.batch_files.append(batch_file)
        self.logger.debug(f'Saved batch {len(self.batch_files)} ({len(self.feature_buffer)} rows)')
        self.feature_buffer = []
    
    def _finalize_output(self):
        '''Combine all temp batches into final output file.'''
        if not self.save_parquet:
            return
        
        # Save any remaining buffer.
        if self.feature_buffer:
            self._save_batch()
        
        if not self.batch_files:
            self.logger.warning('No features to save')
            return
        
        # Combine all batch files.
        self.logger.info(f'Combining {len(self.batch_files)} batch files...')
        
        dfs = []
        for batch_file in self.batch_files:
            dfs.append(pd.read_parquet(batch_file))
        
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(self.parquet_path, index=False)
        
        self.logger.info(f'Saved {len(df)} features to {self.parquet_path}')
        
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
    
    def cleanup(self):
        '''Cleanup resources.'''
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()


def main():
    parser = argparse.ArgumentParser(description='Compute features from raw Kafka stream.')
    parser.add_argument('--topic-in', type=str, default='ticks.raw',
                       help='Input Kafka topic')
    parser.add_argument('--topic-out', type=str, default='ticks.features',
                       help='Output Kafka topic')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in minutes (None for continuous)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to Parquet')
    parser.add_argument('--compute-every-n', type=int, default=1,
                       help='Compute features every N ticker messages')
    parser.add_argument('--parquet-path', type=str, default='data/processed/features.parquet',
                       help='Output Parquet file path')
    
    args = parser.parse_args()
    
    featurizer = Featurizer(
        topic_in=args.topic_in,
        topic_out=args.topic_out,
        save_parquet=not args.no_save,
        compute_every_n=args.compute_every_n,
        parquet_path=args.parquet_path
    )
    
    featurizer.run(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
