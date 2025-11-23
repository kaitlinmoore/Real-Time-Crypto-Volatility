from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class FeatureWindow:
    '''Manages time-windowed data for feature computation. Handles ticker, trade, and orderbook messages, computing windowed
    features like price statistics, spread, trade intensity, and orderbook imbalance.
    '''
    
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        
        # Price data (synced together)
        self.timestamps = deque()
        self.prices = deque()
        self.volumes = deque()
        
        # Spread data (separate timestamps to avoid desync issue)
        self.spread_timestamps = deque()
        self.spreads = deque()
        
        # Trade data - optimized
        self.trade_timestamps = deque()
        self.trade_sizes = deque()
        self.trade_sides = deque()
        
        # Running totals - optimized
        self.total_volume = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        
        # Tick timing (separate for microstructure features)
        self.tick_times = deque()
        self.tick_diffs = deque()  # Incremental storage of time diffs
        
        # Lagged volatility
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

    # Try to improve  orderbook implementation. Don't know if level2 stream is feasible, but hopefully batch works.
    #Can handle either  message format from Coinbase if we do move to level 2 channel.
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
        '''Remove data outside the window - optimized'''

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
        
        # Trend features - Polyfit was way too expensive and unnecessary. 
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
            
            # Added skew/kurt to see if helpful
            if len(returns) > 2:
                features['return_skew'] = float(skew(returns))
                features['return_kurt'] = float(kurtosis(returns))
            else:
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


# Default window sizes for consistency across scripts.
DEFAULT_WINDOW_SIZES = [30, 60, 300, 900]


class FeatureComputer:
    '''Manages feature computation across products and windows. Maintains separate FeatureWindow instances
    for each product/window combination and handles message routing.'''
    
    def __init__(self, products, window_sizes=None, compute_every_n=1):
        self.products = products
        self.window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
        self.compute_every_n = compute_every_n
        
        # Create windows for each product and window size.
        self.windows = defaultdict(dict)
        for product in products:
            for window_size in self.window_sizes:
                self.windows[product][window_size] = FeatureWindow(window_size)
        
        # Counter for batched computation.
        self.ticker_counts = defaultdict(int)
        
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
