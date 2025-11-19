import argparse # CLI 
import json
import time
import websocket # Synchronous - Consider switching to asynchronous if adding level2 channel for this or group project.
# import yaml

from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
from pathlib import Path

from utilities import *

# Get config file.
# with open('config.yaml', 'r') as f:
#     config = yaml.safe_load(f)
config = load_config()


# CONSTANTS
BOOTSTRAP_SERVERS = config['kafka']['bootstrap_servers']
DELAY = config['ingestion']['reconnect_delay']
PING = config['ingestion']['heartbeat_interval']
PONG = config['ingestion']['hb_response_timeout']
PRODUCT_IDS = get_product_ids(config)
            # (config['data']['product_ids']['prediction_targets'] + 
            # config['data']['product_ids']['auxiliary_data'])
RAW = config['kafka']['topics']['raw']
WS_URL = config['data']['url']

# SOURCE: CoinBase API info: https://docs.cdp.coinbase.com/exchange/websocket-feed/overview
# SOURCE: Coinbase WebSocket Best Practices: https://docs.cdp.coinbase.com/exchange/websocket-feed/best-practices
# RESOURCE: Coinbase WebSocket Rate Limits: https://docs.cdp.coinbase.com/exchange/websocket-feed/rate-limits

class CoinbaseWebSocketClient:
    '''Client for Coinbase Advanced Trade WebSocket API.'''
    
    def __init__(self, product_ids, kafka_topic=RAW, save_raw=False):
        '''Initialize the WebSocket client.'''

        self.kafka_topic = kafka_topic
        self.message_counts = {}
        self.perf_monitor = PerformanceMonitor()
        self.product_ids = product_ids # CB terminology for trade pairs
        self.running = False
        self.save_raw = save_raw # boolean for whether to save raw data
        self.url = WS_URL # web socket
        
        # Initialize Kafka producer.
        self.producer = KafkaProducer(
            acks='all',  # Wait for replica (1 in this case) to respond. Use for now. Consider adjustment if latency is an issue.
            bootstrap_servers=[BOOTSTRAP_SERVERS],
            retries=5, # Honestly have no idea what is reasonable. Trying 5. Maybe lower to 0 if debugging or issues. Raise if network trouble.
            value_serializer=lambda v: json.dumps(v).encode('utf-8') # Convert python data dict -> JSON -> UT8 encoded bytes for kafka
        )
        
        # Setup raw data file if needed.
        if self.save_raw:
            self.raw_file = self._setup_raw_file()
        
        # Progress Notes
        print(f'Initialized client for Coinbase products (trade pairs): {product_ids}')
        print(f'Publishing to Kafka topic: {kafka_topic}')

    # Save as NDJSON for readability and quick appends of the real-time data.    
    def _setup_raw_file(self):
        '''Create a timestamped NDJSON file for raw data.'''

        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # Convert to string.
        filename = raw_dir / f'ticks_{timestamp}.ndjson' # Add timestamp to filename.
        return open(filename, 'w')
    
    # Subscribe & Resubscribe
    def _on_open(self, ws):
        '''Callback when WebSocket connection opens or reconnects.'''

        # Progress Note
        print('WebSocket connection opened')
        
        # Subscribe to ticker channel.
        subscribe_message = {
            'type': 'subscribe',
            'product_ids': self.product_ids,
            'channels': [
                'ticker',
                'heartbeat',
                'matches',
                'level2_batch'
            ]
        }
        
        ws.send(json.dumps(subscribe_message))
        print(f"Subscribed to {subscribe_message['channels']} channels for: {self.product_ids}") # Progress Note
    
    def _on_message(self, ws, message):
        '''Callback when a message is received.'''

        try:
            data = json.loads(message)
            
            # Add timestamp.
            data['received_at'] = datetime.now(timezone.utc).isoformat()
            
            # Publish to Kafka.
            future = self.producer.send(self.kafka_topic, value=data)

            # Save to raw file if enabled.
            if self.save_raw and self.raw_file:
                self.raw_file.write(json.dumps(data) + '\n')
                self.raw_file.flush()

            # Update monitoring.
            self.perf_monitor.increment()
            
            # Log ticker updates.
            if data.get('type') == 'ticker':
                product = data.get('product_id')
                price = data.get('price')
                print(f'[{product}] Price: {price}')
            
            # Log trades.
            elif data.get('type') == 'match':
                product = data.get('product_id')
                size = data.get('size')
                print(f'[TRADE] {product}: {size}')
            
            # Log heartbeats.
            elif data.get('type') == 'heartbeat':
                print('[HB] Heartbeat received')
            
        # Error Messages    
        except json.JSONDecodeError as e:
            print(f'Error decoding message: {e}')
            self.perf_monitor.increment(has_error=True)
        except KafkaError as e:
            print(f'Kafka error: {e}')
            self.perf_monitor.increment(has_error=True)
        except Exception as e:
            print(f'Unexpected error: {e}')
            self.perf_monitor.increment(has_error=True)

    def _on_send_success(self, record_metadata):
        # Message delivered successfully
        pass

    def _on_send_error(self, exception):
        print(f'Kafka send failed: {exception}')
    
    def _on_error(self, ws, error):
        '''Callback when an error occurs.'''

        print(f'WebSocket error: {error}') # Error Message
    
    def _on_close(self, ws, close_status_code, close_msg):
        '''Callback when WebSocket connection closes.'''

        print(f'WebSocket closed: {close_status_code} - {close_msg}') # Progress Note
        
        # Attempt to reconnect after a delay.
        if self.running:
            print(f'Attempting to reconnect in {DELAY} seconds...') # Progress Note
            time.sleep(DELAY)
            self.connect()
    
    def connect(self):
        '''Establish WebSocket connection.'''

        self.running = True
        
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Network connection heartbeat signal
        self.ws.run_forever(
            ping_interval=PING,  # Start with 30. Adjust if needed.
            ping_timeout=PONG    # Start with 10. Adjust if needed.
        )
    
    def get_quality_stats(self):
        '''Stats for ingestion run'''

        total = self.perf_monitor.message_count
        errors = self.perf_monitor.errors
        elapsed = time.time() - self.perf_monitor.start_time
        rate = total / elapsed if elapsed > 0 else 0
        
        print(f'\n*** Ingestion Quality Report: ***')
        print(f'Duration: {elapsed:.1f}s')
        print(f'Messages received: {total:,}')
        print(f'Message rate: {rate:.1f}/s')
        print(f'Errors: {errors:,}')
        print(f'\nMessage types:')
        
        for msg_type, count in sorted(self.message_counts.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f'  {msg_type}: {count:,} ({pct:.1f}%)')
    
    def stop(self):
        '''Stop the WebSocket connection gracefully.'''

        print('Stopping WebSocket client...') # Progress Note
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        if self.save_raw and self.raw_file:
            self.raw_file.close()
        
        print('Client stopped') # Progress Note


def main():
    '''Ingestion main workflow'''

    parser = argparse.ArgumentParser(description='Ingest Coinbase ticker data.')
    parser.add_argument(
        '--pair',
        type=str,
        default=PRODUCT_IDS,
        help='Trading pair(s) (\'BTC-USD\' or \'BTC-USD, ETH-USD\')'
    )
    parser.add_argument(
        '--minutes',
        type=int,
        default=15,
        help='Number (int) of minutes to run'
    )
    parser.add_argument(
        '--save-raw',
        action='store_true',
        help='Save raw data to data/raw/.'
    )
    
    args = parser.parse_args()
    
    # Support multiple pairs separated by comma.
    if isinstance(args.pair, list):
        product_ids = args.pair
    else:
        product_ids = [p.strip() for p in args.pair.split(',')]
    
    # Create and start client.
    client = CoinbaseWebSocketClient(
        product_ids=product_ids,
        save_raw=args.save_raw
    )
    
    try:
        print(f'Starting ingestion for {args.minutes} minutes...') # Progress Note
        print('Press Ctrl+C to stop early.')
        
        # Start in a separate thread so I can time it.
        import threading
        thread = threading.Thread(target=client.connect)
        thread.daemon = True
        thread.start()
        
        # Run for specified duration.
        time.sleep(args.minutes * 60)
        
        print(f'\n{args.minutes} minutes elapsed. Stopping...\n') # Progress Note
        client.get_quality_stats()
        client.stop()

    # Error Messages    
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        client.stop()
    except Exception as e:
        print(f'Error: {e}')
        client.stop()


if __name__ == '__main__':
    main()