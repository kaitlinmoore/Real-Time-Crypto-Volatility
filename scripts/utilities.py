## Scripts Folder Utilities ##
'''Shared utilities to potentially use for ingestion and other scripts'''

import json
import logging
import os
import time
import yaml

from datetime import datetime
from typing import Dict, List, Any, Optional

# Since this may be used downstream in team project, use type hints and comments for clarity.

### Configuration ###

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    '''Load configuration from YAML file.'''

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Extract ALL product IDs to track from config.yaml file.
def get_product_ids(config: Optional[Dict] = None) -> List[str]: # Can load alternate config if needed
    '''Get all product IDs to track.'''

    if config is None:
        config = load_config()
    return (config['data']['product_ids']['prediction_targets'] + 
            config['data']['product_ids']['auxiliary_data']) # Tracking USDT as a control.

### Logging Setup ###

def setup_logger(name: str, level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    '''Create a configured logger instance.'''

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    return logger

## Kafka Helpers ##

def get_kafka_config(config: Optional[Dict] = None) -> Dict[str, Any]:
    '''Get Kafka configuration.'''

    if config is None:
        config = load_config()
    return {
        'bootstrap_servers': config['kafka']['bootstrap_servers'],
        'topics': config['kafka']['topics']
    }

def create_kafka_producer(config: Optional[Dict] = None):
    '''Create a configured Kafka producer.'''

    from kafka import KafkaProducer
    kafka_config = get_kafka_config(config)
    
    return KafkaProducer(
        bootstrap_servers=kafka_config['bootstrap_servers'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        compression_type='gzip',  # Compress for efficiency
        max_request_size=1048576  # 1MB max message size
    )

def create_kafka_consumer(topic: str, config: Optional[Dict] = None, **kwargs):
    '''Create a configured Kafka consumer.'''

    from kafka import KafkaConsumer
    kafka_config = get_kafka_config(config)
    
    return KafkaConsumer(
        topic,
        bootstrap_servers=kafka_config['bootstrap_servers'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        **kwargs  # Allow override of defaults
    )

## Data Path Helpers ##

def get_data_path(data_type: str = 'raw', create: bool = True) -> str:
    '''Get path to data directory, create if needed.'''

    path = f'data/{data_type}'
    if create and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_timestamped_filename(prefix: str = 'ticks', extension: str = 'ndjson', product_id: Optional[str] = None) -> str:
    '''Generate timestamped filename.'''

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    if product_id:
        return f'{prefix}_{product_id}_{timestamp}.{extension}'
    return f'{prefix}_{timestamp}.{extension}'

## Validation Helpers ##
def validate_ticker_message(msg: Dict[str, Any]) -> bool:
    '''Validate Coinbase ticker message has required fields.'''
    required_fields = [
        'type', 'product_id', 'price', 
        'best_bid', 'best_ask', 'time'
    ]
    
    if msg.get('type') != 'ticker':
        return False
    
    return all(field in msg for field in required_fields)

def validate_connection_health(last_message_time: float, max_silence_seconds: int = 60) -> bool:
    '''Check if connection seems healthy based on message frequency.'''

    return (time.time() - last_message_time) < max_silence_seconds

## File I/O Helpers ##

def save_to_ndjson(data: List[Dict], filepath: str, mode: str = 'a'):
    '''Save list of dictionaries to NDJSON file.'''

    with open(filepath, mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_from_ndjson(filepath: str) -> List[Dict]:
    '''Load NDJSON file into list of dictionaries.'''

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

#@ Performance Monitoring ##
class PerformanceMonitor:
    '''Simple performance monitoring for ingestion.'''

    def __init__(self, log_interval: int = 1000):
        self.message_count = 0
        self.start_time = time.time()
        self.log_interval = log_interval
        self.gaps = 0
        self.errors = 0
        self.last_message_time = time.time()
        self.message_times = []  # Track timing

    def check_for_silence(self, max_silence_seconds: int = 10):
        '''Detect if messages stopped arriving'''

        silence = time.time() - self.last_message_time
        if silence > max_silence_seconds:
            print(f'***** No messages for {silence:.1f}s - connection issue? *****')
            return True
        return False
        
    def increment(self, has_gap: bool = False, has_error: bool = False):
        '''Increment counters and log if interval reached.'''

        self.message_count += 1
        current_time = time.time()

        # Detect abnormal gaps between messages.
        if self.last_message_time:
            gap = current_time - self.last_message_time
            if gap > 5.0:  # 5 second gap is suspicious
                print(f'Large time gap: {gap:.1f}s between messages')
                
        self.last_message_time = current_time

        if has_gap:
            self.gaps += 1
        if has_error:
            self.errors += 1
            
        if self.message_count % self.log_interval == 0:
            self.log_stats()
    
    def log_stats(self):
        '''Log current statistics.'''

        elapsed = time.time() - self.start_time
        rate = self.message_count / elapsed if elapsed > 0 else 0
        print(f'Messages: {self.message_count} | '
              f'Rate: {rate:.1f}/s | '
              f'Gaps: {self.gaps} | '
              f'Errors: {self.errors}')