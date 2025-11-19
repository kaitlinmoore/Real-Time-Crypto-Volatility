import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from kafka import KafkaConsumer

from utilities import get_product_ids, load_config


def validate_ticker(data):
    '''Check if ticker message has all required fields.'''

    required = ['product_id', 'price', 'best_bid', 'best_ask', 'time']
    missing = [f for f in required if f not in data or data[f] is None]
    return len(missing) == 0, missing


def validate_match(data):
    '''Check if match (trade) message has required fields.'''

    required = ['product_id', 'size', 'price', 'time']
    missing = [f for f in required if f not in data or data[f] is None]
    return len(missing) == 0, missing


def check_data_staleness(timestamps):
    '''Check if data is stale based on most recent timestamp.'''

    if not timestamps:
        return None
    
    latest = max(timestamps)
    try:
        # Handle both formats: ISO with Z or without.
        if latest.endswith('Z'):
            latest_dt = datetime.fromisoformat(latest.replace('Z', '+00:00'))
        else:
            latest_dt = datetime.fromisoformat(latest)
        
        age_seconds = (datetime.now(timezone.utc) - latest_dt).total_seconds()
        return age_seconds
    except Exception:
        return None


def main():

    # CLI
    parser = argparse.ArgumentParser(description='Validate Kafka topic data quality.')
    parser.add_argument(
        '--topic',
        type=str,
        default='ticks.raw',
        help='Kafka topic to consume from'
    )
    parser.add_argument(
        '--min',
        type=int,
        default=100,
        help='Minimum number of messages expected'
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        default=10,
        help='Maximum messages to display'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Read for specified seconds (overrides message count)'
    )
    parser.add_argument(
        '--check-staleness',
        action='store_true',
        help='Check if data is stale (use for live monitoring only)'
    )
    
    args = parser.parse_args()
    
    print(f'Connecting to Kafka and reading from topic: {args.topic}')
    
    # Load expected products from config.
    config = load_config()
    expected_products = set(get_product_ids(config))
    expected_channels = {'ticker', 'heartbeat', 'match', 'l2update'}
    
    # Create consumer.
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    # Tracking Variables
    message_count = 0
    displayed_count = 0
    message_types = Counter()
    products = Counter()
    validation_errors = 0
    timestamps = []
    message_times = []
    start_time = time.time()
    
    try:
        if args.duration:
            print(f'Reading messages for {args.duration} seconds...')
        else:
            print(f'Reading messages (will display first {args.max_messages})...')
        
        for message in consumer:
            current_time = time.time()
            message_times.append(current_time)
            message_count += 1
            data = message.value
            
            # Track message types and products.
            msg_type = data.get('type')
            product = data.get('product_id')
            message_types[msg_type] += 1
            if product:
                products[product] += 1
            
            # Collect timestamps for staleness check.
            if 'time' in data:
                timestamps.append(data['time'])
            elif 'received_at' in data:
                timestamps.append(data['received_at'])
            
            # Display first few messages.
            if displayed_count < args.max_messages:
                print(f'\nMessage {message_count}:')
                print(f'  Type: {msg_type}')
                print(f'  Product: {product}')
                print(f'  Partition: {message.partition}, Offset: {message.offset}')
                if msg_type == 'ticker':
                    print(f'  Price: {data.get("price")}, Bid: {data.get("best_bid")}, Ask: {data.get("best_ask")}')
                elif msg_type == 'match':
                    print(f'  Size: {data.get("size")}, Price: {data.get("price")}')
                displayed_count += 1
            
            # Validate required fields.
            if msg_type == 'ticker':
                valid, missing = validate_ticker(data)
                if not valid:
                    validation_errors += 1
                    if validation_errors <= 3:
                        print(f'Invalid ticker at message {message_count}: missing {missing}')
            elif msg_type == 'match':
                valid, missing = validate_match(data)
                if not valid:
                    validation_errors += 1
                    if validation_errors <= 3:
                        print(f'Invalid match at message {message_count}: missing {missing}')
            
            # Check for large gaps in message arrival.
            if len(message_times) > 1:
                gap = message_times[-1] - message_times[-2]
                if gap > 5.0:
                    print(f'Large time gap: {gap:.1f}s at message {message_count}')
            
            # Stop Conditions
            if args.duration:
                if (current_time - start_time) >= args.duration:
                    break
            else:
                if message_count >= args.min * 2:
                    break
        
        # Calculate summary statistics.
        elapsed = time.time() - start_time
        message_rate = message_count / elapsed if elapsed > 0 else 0
        observed_products = set(products.keys())
        observed_channels = set(message_types.keys())
        
        # Check data staleness.
        staleness = check_data_staleness(timestamps)
        
        # Print comprehensive validation report.
        print(f'\n{"="*60}')
        print(f'STREAM VALIDATION REPORT')
        print(f'{"="*60}')
        
        print(f'\n--- Volume Metrics ---')
        print(f'Total messages: {message_count:,}')
        print(f'Duration: {elapsed:.1f}s')
        print(f'Message rate: {message_rate:.1f} msg/s')
        
        print(f'\n--- Message Distribution ---')
        for msg_type, count in sorted(message_types.items()):
            pct = 100 * count / message_count if message_count > 0 else 0
            print(f'  {msg_type:15s}: {count:6,} ({pct:5.1f}%)')
        
        print(f'\n--- Product Coverage ---')
        print(f'Expected products: {sorted(expected_products)}')
        print(f'Observed products: {sorted(observed_products)}')
        for product in sorted(observed_products):
            count = products[product]
            pct = 100 * count / message_count if message_count > 0 else 0
            print(f'  {product:10s}: {count:6,} ({pct:5.1f}%)')
        
        missing_products = expected_products - observed_products
        if missing_products:
            print(f'Missing products: {sorted(missing_products)}')
        
        print(f'\n--- Channel Coverage ---')
        print(f'Expected channels: {sorted(expected_channels)}')
        print(f'Observed channels: {sorted(observed_channels)}')
        
        missing_channels = expected_channels - observed_channels
        if missing_channels:
            print(f'Missing channels: {sorted(missing_channels)}')
        
        if 'heartbeat' in observed_channels:
            hb_count = message_types['heartbeat']
            hb_interval = elapsed / hb_count if hb_count > 0 else 0
            print(f'Heartbeats detected: {hb_count} (avg interval: {hb_interval:.1f}s)')
        
        print(f'\n--- Data Quality ---')
        if validation_errors > 0:
            error_rate = 100 * validation_errors / message_count
            print(f'Validation errors: {validation_errors} ({error_rate:.2f}%)')
        else:
            print(f'No validation errors detected')
        
        if args.check_staleness:
            if staleness is not None:
                if staleness > 300:  # 5 minutes
                    print(f'Data is stale: latest message is {staleness:.0f}s old')
                else:
                    print(f'Data is fresh: latest message is {staleness:.0f}s old')
        
        # Overall Validation Result
        print(f'VALIDATION RESULT')
        
        success_conditions = [
            (message_count >= args.min, f'Message count: {message_count:,} >= {args.min}'),
            (len(missing_products) == 0, f'Product coverage: {len(observed_products)}/{len(expected_products)}'),
            (validation_errors / message_count < 0.05 if message_count > 0 else False, f'Error rate: {100*validation_errors/message_count:.2f}% < 5%'),
            (message_rate > 1.0, f'Message rate: {message_rate:.1f} msg/s > 1.0')
        ]
        
        passed = sum(1 for condition, _ in success_conditions if condition)
        total = len(success_conditions)
        
        for condition, description in success_conditions:
            status = '✓' if condition else '✗'
            print(f'{status} {description}')
        
        print(f'\nOverall: {passed}/{total} checks passed')
        
        if passed == total:
            print(f'\nSUCCESS: Stream validation passed all checks.')
        else:
            print(f'\nWARNING: Some validation checks failed.')
            print(f'\nPossible issues:')
            if message_count < args.min:
                print(f'  - Ingestion script didn\'t run long enough')
            if missing_products:
                print(f'  - Some products not sending data')
            if validation_errors / message_count >= 0.05:
                print(f'  - High rate of malformed messages')
        
    except KeyboardInterrupt:
        print('\n\nStopped by user')
    finally:
        consumer.close()


if __name__ == '__main__':
    main()