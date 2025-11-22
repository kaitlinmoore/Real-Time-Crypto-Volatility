import argparse
import json
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

# Add parent directory to path for utilities import.
sys.path.append(str(Path(__file__).parent.parent))
from utilities import load_config, setup_logger


class FeatureWindow:
    '''Manages time-windowed data for feature computation'''
    
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        
        # Price data (synced together)
        self.timestamps = deque()
        self.prices = deque()
        self.volumes = deque()
        
        # Spread data (separate timestamps to avoid desync issues)
        self.spread_timestamps = deque()
        self.spreads = deque()
        
        # Trade data (durther optimized for O(1) cleanup)
        self.trade_timestamps = deque()
        self.trade_sizes = deque()
        self.trade_sides = deque()
        
        # Running totals for O(1) trade feature computation
        self.total_volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        
        # Tick timing (separate for microstructure features)
        self.tick_times = deque()
        self.tick_diffs = deque()  # Incremental storage of time diffs
        
        # Lagged volatility for autoregressive feature
        self.last_volatility = None

        # Orderbook / L2 data (separate timestamps)
        self.ob_timestamps = deque()
        self.bid_depth_L1 = deque()
        self.ask_depth_L1 = deque()
        self.microprice_vals = deque()
        
    def add_ticker(self, msg):
        '''Add ticker message to window.'''

        try:
            timestamp = pd.Timestamp(msg['time'])
            price = float(msg['price'])
            best_bid = float(msg.get('best_bid', 0))
            best_ask = float(msg.get('best_ask', 0))
            volume_24h = float(msg.get('volume_24h', 0))
            
            # Store price data (always synced).
            self.timestamps.append(timestamp)
            self.prices.append(price)
            self.volumes.append(volume_24h)
            
            # Store tick time and compute diff incrementally.
            if self.tick_times:
                diff = (timestamp - self.tick_times[-1]).total_seconds()
                self.tick_diffs.append(diff)
            self.tick_times.append(timestamp)
            
            # Store spread separately (only when valid).
            if best_bid > 0 and best_ask > 0:
                spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
                self.spread_timestamps.append(timestamp)
                self.spreads.append(spread)
            
            self._cleanup(timestamp)
            
        except (KeyError, ValueError, TypeError):
            pass
    
    def add_trade(self, msg):
        '''Add trade/match message to window.'''

        try:
            timestamp = pd.Timestamp(msg['time'])
            size = float(msg.get('size', 0))
            side = msg.get('side', 'unknown')
            
            self.trade_timestamps.append(timestamp)
            self.trade_sizes.append(size)
            self.trade_sides.append(side)
            
            # Update running totals.
            self.total_volume += size
            if side == 'buy':
                self.buy_volume += size
            elif side == 'sell':
                self.sell_volume += size
            
            self._cleanup(timestamp)
            
        except (KeyError, ValueError, TypeError):
            pass

    # Improve use of l2-batch data and order book.
    # Try both message formats.
    def add_orderbook(self, msg):
        '''Add level2/level2_batch snapshot to window (L2 features).'''

        try:
            timestamp = pd.Timestamp(msg['time'])
            
            # Try snapshot format first.
            bids = msg.get('bids', [])
            asks = msg.get('asks', [])
            
            # Handle Coinbase l2update 'changes' format.
            if not bids and not asks and 'changes' in msg:
                changes = msg['changes']
                # Extract best bid/ask from changes.
                # Note: This is approximate - changes are deltas, not full book. Not sure if streaming live l2 is feasible.
                buy_changes = [c for c in changes if c[0] == 'buy']
                sell_changes = [c for c in changes if c[0] == 'sell']
                
                if buy_changes:
                    # Take highest buy price as best bid.
                    best_buy = max(buy_changes, key=lambda x: float(x[1]))
                    bids = [[best_buy[1], best_buy[2]]]
                if sell_changes:
                    # Take lowest sell price as best ask.
                    best_sell = min(sell_changes, key=lambda x: float(x[1]))
                    asks = [[best_sell[1], best_sell[2]]]

            if not bids or not asks:
                return

            best_bid_price = float(bids[0][0])
            best_bid_size = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_size = float(asks[0][1])

            # Skip if sizes are zero.
            if best_bid_size <= 0 or best_ask_size <= 0:
                return

            bid_depth_L1 = best_bid_size
            ask_depth_L1 = best_ask_size

            # Compute microprice (liquidity-weighted mid).
            denom = bid_depth_L1 + ask_depth_L1
            if denom > 0:
                microprice = (
                    best_ask_price * bid_depth_L1 +
                    best_bid_price * ask_depth_L1
                ) / denom
            else:
                microprice = (best_bid_price + best_ask_price) / 2.0

            self.ob_timestamps.append(timestamp)
            self.bid_depth_L1.append(bid_depth_L1)
            self.ask_depth_L1.append(ask_depth_L1)
            self.microprice_vals.append(microprice)

            self._cleanup(timestamp)

        except Exception:
            pass
    
    def _cleanup(self, current_time):
        '''Remove data outside the window.'''

        cutoff = current_time - pd.Timedelta(seconds=self.window_seconds)
        
        # Clean price data (timestamps, prices, volumes are synced).
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            self.prices.popleft()
            self.volumes.popleft()
        
        # Clean spread data (separate tracking).
        while self.spread_timestamps and self.spread_timestamps[0] < cutoff:
            self.spread_timestamps.popleft()
            self.spreads.popleft()
        
        # Clean tick times and diffs.
        while self.tick_times and self.tick_times[0] < cutoff:
            self.tick_times.popleft()
            if self.tick_diffs:
                self.tick_diffs.popleft()
        
        # Clean trades and update running totals.
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()
            size = self.trade_sizes.popleft()
            side = self.trade_sides.popleft()
            
            # Subtract from running totals.
            self.total_volume -= size
            if side == 'buy':
                self.buy_volume -= size
            elif side == 'sell':
                self.sell_volume -= size

        # Clean orderbook data.
        while self.ob_timestamps and self.ob_timestamps[0] < cutoff:
            self.ob_timestamps.popleft()
            self.bid_depth_L1.popleft()
            self.ask_depth_L1.popleft()
            self.microprice_vals.popleft()
    
    def compute_features(self):
        '''Compute features from windowed data.'''

        features = {}
        
        if len(self.prices) < 2:
            return None
        
        prices = np.array(self.prices)
        
        # Price-based features
        features['price_mean'] = np.mean(prices)
        features['price_std'] = np.std(prices)
        features['price_min'] = np.min(prices)
        features['price_max'] = np.max(prices)
        features['price_range'] = features['price_max'] - features['price_min']
        
        # Momentum features
        features['price_momentum'] = (
            (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        )
        
        # Trend features - polyfit was too expensive computationally and unnecessary.
        if len(prices) > 2:
            n = len(prices)
            x = np.arange(n)
            x_mean = (n - 1) / 2.0
            y_mean = features['price_mean']
            
            numerator = np.sum((x - x_mean) * (prices - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                features['price_trend'] = slope / y_mean if y_mean != 0 else 0
            else:
                features['price_trend'] = 0
        else:
            features['price_trend'] = 0
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        if len(returns) > 0:
            features['return_mean'] = np.mean(returns)
            features['return_std'] = np.std(returns)
            # Skip skew/kurt for performance (can add back if needed).
            features['return_skew'] = 0
            features['return_kurt'] = 0
        else:
            features['return_mean'] = 0
            features['return_std'] = 0
            features['return_skew'] = 0
            features['return_kurt'] = 0
        
        # Lagged volatility
        if self.last_volatility is not None:
            features['volatility_lag1'] = self.last_volatility
        else:
            features['volatility_lag1'] = features['return_std']
        self.last_volatility = features['return_std']
        
        # Spread features (using separate tracking)
        if self.spreads:
            spreads = np.array(self.spreads)
            features['spread_mean'] = np.mean(spreads)
            features['spread_std'] = np.std(spreads)
        else:
            features['spread_mean'] = 0
            features['spread_std'] = 0
        
        # Trade intensity features - optimized
        trade_count = len(self.trade_timestamps)
        features['trade_count'] = trade_count
        features['trade_volume'] = self.total_volume
        features['avg_trade_size'] = (
            self.total_volume / trade_count if trade_count > 0 else 0
        )
        
        total_directional = self.buy_volume + self.sell_volume
        features['trade_imbalance'] = (
            (self.buy_volume - self.sell_volume) / total_directional 
            if total_directional > 0 else 0
        )
        
        # Microstructure timing features - optimized
        if self.tick_diffs:
            diffs = np.array(self.tick_diffs)
            features['avg_tick_interval'] = np.mean(diffs)
            features['tick_interval_std'] = np.std(diffs)
        else:
            features['avg_tick_interval'] = 0
            features['tick_interval_std'] = 0
        
        features['tick_rate'] = len(self.tick_times) / self.window_seconds
        features['n_observations'] = len(self.prices)

        # Orderbook / L2 features
        if self.bid_depth_L1 and self.ask_depth_L1:
            bid_depth = np.array(self.bid_depth_L1)
            ask_depth = np.array(self.ask_depth_L1)

            bid_mean = bid_depth.mean()
            ask_mean = ask_depth.mean()
            denom = bid_mean + ask_mean

            features['ob_bid_depth_L1_mean'] = bid_mean
            features['ob_ask_depth_L1_mean'] = ask_mean
            features['ob_depth_imbalance_L1'] = (
                (bid_mean - ask_mean) / denom if denom > 0 else 0
            )
        else:
            features['ob_bid_depth_L1_mean'] = 0
            features['ob_ask_depth_L1_mean'] = 0
            features['ob_depth_imbalance_L1'] = 0

        if self.microprice_vals:
            micro = np.array(self.microprice_vals)
            features['ob_microprice_mean'] = micro.mean()
            features['ob_microprice_std'] = micro.std()
        else:
            features['ob_microprice_mean'] = 0
            features['ob_microprice_std'] = 0
        
        return features


class FeatureComputer:
    '''Manages feature computation across products and windows'''
    
    def __init__(self, products, window_sizes=None, compute_every_n=1):
        self.products = products
        # Default: include 30s window for short-term signal.
        self.window_sizes = window_sizes or [30, 60, 300, 900]
        self.compute_every_n = compute_every_n
        
        # Create windows for each product and window size.
        self.windows = defaultdict(dict)
        for product in products:
            for window_size in self.window_sizes:
                self.windows[product][window_size] = FeatureWindow(window_size)
        
        # Counter for batched computation
        self.ticker_counts = defaultdict(int)
        
        self.logger = setup_logger('FeatureComputer')
        
    def process_message(self, msg):
        '''Process a single message and update windows.'''

        msg_type = msg.get('type')
        product_id = msg.get('product_id')
        
        if product_id not in self.products:
            return None
        
        # Update appropriate windows.
        if msg_type == 'ticker':
            for window_size in self.window_sizes:
                self.windows[product_id][window_size].add_ticker(msg)
        
        elif msg_type in ['match', 'last_match']:
            for window_size in self.window_sizes:
                self.windows[product_id][window_size].add_trade(msg)

        elif msg_type in ['level2', 'l2update', 'snapshot', 'level2_batch']:
            for window_size in self.window_sizes:
                self.windows[product_id][window_size].add_orderbook(msg)
        
        # Compute features on ticker updates (with optional batching).
        if msg_type == 'ticker':
            self.ticker_counts[product_id] += 1
            
            if self.ticker_counts[product_id] % self.compute_every_n == 0:
                return self._compute_features(product_id, msg.get('time'))
        
        return None
    
    def _compute_features(self, product_id, timestamp):
        '''Compute features for all windows.'''

        feature_row = {
            'product_id': product_id,
            'timestamp': timestamp,
            'feature_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add time-of-day features.
        timestamp_dt = pd.to_datetime(timestamp)
        feature_row['hour_of_day'] = timestamp_dt.hour
        feature_row['minute_of_hour'] = timestamp_dt.minute
        feature_row['hour_sin'] = np.sin(2 * np.pi * timestamp_dt.hour / 24)
        feature_row['hour_cos'] = np.cos(2 * np.pi * timestamp_dt.hour / 24)
        feature_row['day_of_week'] = timestamp_dt.dayofweek
        
        # Compute features for each window size.
        for window_size in self.window_sizes:
            window = self.windows[product_id][window_size]
            features = window.compute_features()
            
            if features:
                prefix = f'w{window_size}_'
                for key, value in features.items():
                    feature_row[prefix + key] = value
        
        # Only return if we have at least one valid window.
        if len(feature_row) > 8:
            return feature_row
        
        return None


class Featurizer:
    '''Main featurizer that consumes from Kafka and produces features. Uses temp file batching to improve performance.'''
    
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
        
        # Feature computation
        self.feature_computer = FeatureComputer(
            products=products,
            window_sizes=window_sizes,
            compute_every_n=compute_every_n
        )
        
        # Temp file batching - optimized
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
        '''Save current batch to a temp file.'''

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
    
    args = parser.parse_args()
    
    featurizer = Featurizer(
        topic_in=args.topic_in,
        topic_out=args.topic_out,
        save_parquet=not args.no_save,
        compute_every_n=args.compute_every_n
    )
    
    featurizer.run(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
