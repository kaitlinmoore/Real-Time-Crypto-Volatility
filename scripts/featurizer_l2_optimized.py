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
    '''Manages time-windowed data for feature computation.'''
    
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.prices = deque()
        self.timestamps = deque()
        self.volumes = deque()
        self.spreads = deque()
        self.trades = deque()
        self.trade_timestamps = deque()  # Separate deque for efficient cleanup.
        self.tick_times = deque()
        self.last_volatility = None

        # Orderbook / L2-related state.
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
            
            self.timestamps.append(timestamp)
            self.tick_times.append(timestamp)
            self.prices.append(price)
            self.volumes.append(volume_24h)
            
            # Compute spread.
            if best_bid > 0 and best_ask > 0:
                spread = (best_ask - best_bid) / ((best_ask + best_bid) / 2)
                self.spreads.append(spread)
            
            self._cleanup(timestamp)
            
        except (KeyError, ValueError, TypeError):
            pass
    
    def add_trade(self, msg):
        '''Add trade/match message to window.'''

        try:
            timestamp = pd.Timestamp(msg['time'])
            size = float(msg.get('size', 0))
            
            self.trade_timestamps.append(timestamp)
            self.trades.append({
                'timestamp': timestamp,
                'size': size,
                'side': msg.get('side', 'unknown')
            })
            
            self._cleanup(timestamp)
            
        except (KeyError, ValueError, TypeError):
            pass

    def add_orderbook(self, msg):
        '''Add level2/level2_batch snapshot to window (L2 features).'''

        try:
            timestamp = pd.Timestamp(msg['time'])
            bids = msg.get('bids', [])
            asks = msg.get('asks', [])

            if not bids or not asks:
                return

            best_bid_price = float(bids[0][0])
            best_bid_size = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_size = float(asks[0][1])

            bid_depth_L1 = best_bid_size
            ask_depth_L1 = best_ask_size

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
        '''Remove data outside the window - OPTIMIZED with to O(1).'''

        cutoff = current_time - pd.Timedelta(seconds=self.window_seconds)
        
        # Clean prices and timestamps.
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
            if self.prices:
                self.prices.popleft()
            if self.spreads:
                self.spreads.popleft()
            if self.volumes:
                self.volumes.popleft()
        
        # Clean tick times.
        while self.tick_times and self.tick_times[0] < cutoff:
            self.tick_times.popleft()
        
        # Clean trades - OPTIMIZED: use parallel timestamp deque.
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()
            if self.trades:
                self.trades.popleft()

        # Clean orderbook snapshots.
        while self.ob_timestamps and self.ob_timestamps[0] < cutoff:
            self.ob_timestamps.popleft()
            if self.bid_depth_L1:
                self.bid_depth_L1.popleft()
            if self.ask_depth_L1:
                self.ask_depth_L1.popleft()
            if self.microprice_vals:
                self.microprice_vals.popleft()
    
    def compute_features(self):
        '''Compute features from windowed data.'''

        features = {}
        
        if len(self.prices) < 2:
            return None
        
        prices = np.array(list(self.prices))
        
        # Price-based features.
        features['price_mean'] = np.mean(prices)
        features['price_std'] = np.std(prices)
        features['price_min'] = np.min(prices)
        features['price_max'] = np.max(prices)
        features['price_range'] = features['price_max'] - features['price_min']
        
        # Momentum features.
        features['price_momentum'] = (
            (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        )
        
        # Trend features - OPTIMIZED: manual slope calculation instead of polyfit.
        if len(prices) > 2:
            n = len(prices)
            x = np.arange(n)
            x_mean = (n - 1) / 2.0
            y_mean = np.mean(prices)
            
            # slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
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
            features['return_skew'] = (
                pd.Series(returns).skew() if len(returns) > 2 else 0
            )
            features['return_kurt'] = (
                pd.Series(returns).kurt() if len(returns) > 3 else 0
            )
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
        
        # Spread features
        if self.spreads:
            spreads = np.array(list(self.spreads))
            features['spread_mean'] = np.mean(spreads)
            features['spread_std'] = np.std(spreads)
        else:
            features['spread_mean'] = 0
            features['spread_std'] = 0
        
        # Trade intensity features
        features['trade_count'] = len(self.trades)
        if self.trades:
            total_volume = sum(t['size'] for t in self.trades)
            features['trade_volume'] = total_volume
            features['avg_trade_size'] = (
                total_volume / len(self.trades) if len(self.trades) > 0 else 0
            )
            
            buy_volume = sum(t['size'] for t in self.trades if t['side'] == 'buy')
            sell_volume = sum(t['size'] for t in self.trades if t['side'] == 'sell')
            total = buy_volume + sell_volume
            features['trade_imbalance'] = (
                (buy_volume - sell_volume) / total if total > 0 else 0
            )
        else:
            features['trade_volume'] = 0
            features['avg_trade_size'] = 0
            features['trade_imbalance'] = 0
        
        # Microstructure timing features
        if len(self.tick_times) > 1:
            time_diffs = [
                (self.tick_times[i+1] - self.tick_times[i]).total_seconds()
                for i in range(len(self.tick_times)-1)
            ]
            features['avg_tick_interval'] = np.mean(time_diffs)
            features['tick_interval_std'] = np.std(time_diffs)
            features['tick_rate'] = len(self.tick_times) / self.window_seconds
        else:
            features['avg_tick_interval'] = 0
            features['tick_interval_std'] = 0
            features['tick_rate'] = 0
        
        features['n_observations'] = len(self.prices)

        # Orderbook / L2-based features
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
    '''Manages feature computation across products and windows.'''
    
    def __init__(self, products, window_sizes=[60, 300, 900], compute_every_n=1):
        self.products = products
        self.window_sizes = window_sizes
        self.compute_every_n = compute_every_n
        
        # Create windows for each product and window size.
        self.windows = defaultdict(dict)
        for product in products:
            for window_size in window_sizes:
                self.windows[product][window_size] = FeatureWindow(window_size)
        
        # Counter for batched computation
        self.ticker_counts = defaultdict(int)
        
        self.logger = setup_logger('FeatureComputer')
        self.feature_buffer = []
        
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
            
            # Only compute every N tickers.
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
        
        if len(feature_row) > 8:
            return feature_row
        
        return None


class Featurizer:
    '''Main featurizer that consumes from Kafka and produces features.'''
    
    def __init__(self, topic_in='ticks.raw', topic_out='ticks.features', 
                 products=None, save_parquet=True, parquet_path='data/processed/features.parquet'):
        
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
        
        self.feature_computer = FeatureComputer(products=products)
        
        self.feature_buffer = []
        self.buffer_size = 1000
        
        self.logger.info(f'Featurizer initialized for products: {products}')
        self.logger.info(f'Consuming from: {topic_in}')
        self.logger.info(f'Publishing to: {topic_out}')
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
                                self._save_buffer()
                        
                        message_count += 1
                        
                        if message_count % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = message_count / elapsed
                            self.logger.info(
                                f'Processed {message_count} msgs, '
                                f'generated {feature_count} features ({rate:.1f} msg/s)'
                            )
        
        except KeyboardInterrupt:
            self.logger.info('Interrupted by user')
        
        finally:
            if self.feature_buffer:
                self._save_buffer()
            
            self.cleanup()
            
            elapsed = time.time() - start_time
            self.logger.info(f'Featurizer stopped after {elapsed:.1f}s')
            self.logger.info(f'Processed {message_count} messages')
            self.logger.info(f'Generated {feature_count} feature rows')
    
    def _save_buffer(self):
        '''Save buffered features to Parquet.'''
        if not self.save_parquet or not self.feature_buffer:
            return
        
        df = pd.DataFrame(self.feature_buffer)
        
        if self.parquet_path.exists():
            existing_df = pd.read_parquet(self.parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(self.parquet_path, index=False)
        
        self.logger.info(f'Saved {len(self.feature_buffer)} features to {self.parquet_path}')
        self.feature_buffer = []
    
    def cleanup(self):
        '''Cleanup resources.'''
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()


def main():
    parser = argparse.ArgumentParser(description='Compute features from raw Kafka stream')
    parser.add_argument('--topic-in', type=str, default='ticks.raw',
                       help='Input Kafka topic')
    parser.add_argument('--topic-out', type=str, default='ticks.features',
                       help='Output Kafka topic')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in minutes (None for continuous)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save to Parquet')
    
    args = parser.parse_args()
    
    featurizer = Featurizer(
        topic_in=args.topic_in,
        topic_out=args.topic_out,
        save_parquet=not args.no_save
    )
    
    featurizer.run(duration_minutes=args.duration)


if __name__ == '__main__':
    main()
