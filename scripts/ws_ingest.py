import argparse
import json
import threading
import time
import websocket

from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
from pathlib import Path

from utilities import load_config, get_product_ids, PerformanceMonitor

# Get config file.
config = load_config()

# CONSTANTS
BOOTSTRAP_SERVERS = config['kafka']['bootstrap_servers']
DELAY = config['ingestion']['reconnect_delay']
PING = config['ingestion']['heartbeat_interval']
PONG = config['ingestion']['hb_response_timeout']
PRODUCT_IDS = get_product_ids(config)
RAW = config['kafka']['topics']['raw']
WS_URL = config['data']['url']


class CoinbaseWebSocketClient:
    '''Client for Coinbase Advanced Trade WebSocket API.'''
    
    def __init__(self, product_ids, kafka_topic=RAW, save_raw=False):
        '''Initialize the WebSocket client.'''

        self.kafka_topic = kafka_topic
        self.message_counts = {}  # Track message types.
        self.perf_monitor = PerformanceMonitor()
        self.product_ids = product_ids
        self.running = False
        self.save_raw = save_raw
        self.url = WS_URL
        self.ws = None  # Initialize ws attribute.
        self.raw_file = None  # Initialize raw_file attribute.
        
        # Initialize Kafka producer.
        self.producer = KafkaProducer(
            acks='all',
            bootstrap_servers=[BOOTSTRAP_SERVERS],
            retries=5,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Setup raw data file if needed.
        if self.save_raw:
            self.raw_file = self._setup_raw_file()
        
        print(f'Initialized client for Coinbase products (trade pairs): {product_ids}')
        print(f'Publishing to Kafka topic: {kafka_topic}')

    def _setup_raw_file(self):
        '''Create a timestamped NDJSON file for raw data.'''

        raw_dir = Path('data/raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = raw_dir / f'ticks_{timestamp}.ndjson'
        return open(filename, 'w')
    
    def _on_open(self, ws):
        '''Callback when WebSocket connection opens or reconnects.'''

        print('WebSocket connection opened')
        
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
        print(f"Subscribed to {subscribe_message['channels']} channels for: {self.product_ids}")
    
    def _on_message(self, ws, message):
        '''Callback when a message is received.'''

        try:
            data = json.loads(message)
            
            # Add timestamp.
            data['received_at'] = datetime.now(timezone.utc).isoformat()
            
            # Track message types.
            msg_type = data.get('type', 'unknown')
            self.message_counts[msg_type] = self.message_counts.get(msg_type, 0) + 1
            
            # Publish to Kafka with callbacks.
            future = self.producer.send(self.kafka_topic, value=data)
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)

            # Save to raw file if enabled.
            if self.save_raw and self.raw_file:
                self.raw_file.write(json.dumps(data) + '\n')
                self.raw_file.flush()

            # Update monitoring.
            self.perf_monitor.increment()
            
            # Log ticker updates.
            if msg_type == 'ticker':
                product = data.get('product_id')
                price = data.get('price')
                print(f'[{product}] Price: {price}')
            
            # Log trades.
            elif msg_type == 'match':
                product = data.get('product_id')
                size = data.get('size')
                print(f'[TRADE] {product}: {size}')
            
            # Log heartbeats.
            elif msg_type == 'heartbeat':
                print('[HB] Heartbeat received')
            
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
        '''Callback when Kafka message is delivered successfully.'''
        pass

    def _on_send_error(self, exception):
        '''Callback when Kafka message delivery fails.'''
        print(f'Kafka send failed: {exception}')
        self.perf_monitor.increment(has_error=True)
    
    def _on_error(self, ws, error):
        '''Callback when an error occurs.'''

        print(f'WebSocket error: {error}')
    
    def _on_close(self, ws, close_status_code, close_msg):
        '''Callback when WebSocket connection closes.'''

        print(f'WebSocket closed: {close_status_code} - {close_msg}')
        
        # Attempt to reconnect after a delay (non-recursive via thread).
        if self.running:
            print(f'Attempting to reconnect in {DELAY} seconds...')
            time.sleep(DELAY)
            # Use a new thread to avoid recursion depth issues.
            reconnect_thread = threading.Thread(target=self.connect)
            reconnect_thread.daemon = True
            reconnect_thread.start()
    
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
        
        self.ws.run_forever(
            ping_interval=PING,
            ping_timeout=PONG
        )
    
    def get_quality_stats(self):
        '''Stats for ingestion run.'''

        total = self.perf_monitor.message_count
        errors = self.perf_monitor.errors
        elapsed = time.time() - self.perf_monitor.start_time
        rate = total / elapsed if elapsed > 0 else 0
        
        print(f'\n*** Ingestion Quality Report: ***')
        print(f'Duration: {elapsed:.1f}s')
        print(f'Messages received: {total:,}')
        print(f'Message rate: {rate:.1f}/s')
        print(f'Errors: {errors:,}')
        
        if self.message_counts:
            print(f'\nMessage types:')
            for msg_type, count in sorted(self.message_counts.items()):
                pct = 100 * count / total if total > 0 else 0
                print(f'  {msg_type}: {count:,} ({pct:.1f}%)')
        else:
            print('\nNo messages received.')
    
    def stop(self):
        '''Stop the WebSocket connection gracefully.'''

        print('Stopping WebSocket client...')
        self.running = False
        
        if self.ws:
            self.ws.close()
        
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        if self.raw_file:
            self.raw_file.close()
        
        print('Client stopped')


def main():
    '''Ingestion main workflow.'''

    parser = argparse.ArgumentParser(description='Ingest Coinbase ticker data.')
    parser.add_argument(
        '--pair',
        type=str,
        default=PRODUCT_IDS,
        help="Trading pair(s) ('BTC-USD' or 'BTC-USD, ETH-USD')"
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
        print(f'Starting ingestion for {args.minutes} minutes...')
        print('Press Ctrl+C to stop early.')
        
        # Start in a separate thread so we can time it.
        thread = threading.Thread(target=client.connect)
        thread.daemon = True
        thread.start()
        
        # Run for specified duration.
        time.sleep(args.minutes * 60)
        
        print(f'\n{args.minutes} minutes elapsed. Stopping...\n')
        client.get_quality_stats()
        client.stop()

    except KeyboardInterrupt:
        print('\nInterrupted by user')
        client.stop()
    except Exception as e:
        print(f'Error: {e}')
        client.stop()


if __name__ == '__main__':
    main()