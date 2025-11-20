import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from sklearn.metrics import average_precision_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utilities import load_config, setup_logger


class VolatilityPredictor:
    '''Real-time volatility spike predictor.'''
    
    def __init__(self, model_path, scaler_path, feature_cols_path):
        self.logger = setup_logger('Predictor')
        
        # Load model.
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.logger.info(f'Loaded model from {model_path}')
        
        # Load scaler.
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature columns.
        with open(feature_cols_path, 'r') as f:
            self.feature_cols = json.load(f)
        self.logger.info(f'Using {len(self.feature_cols)} features')
    
    def predict(self, features_dict):
        '''Make prediction from feature dictionary.'''

        # Extract features in correct order.
        X = np.array([[features_dict.get(col, 0) for col in self.feature_cols]])
        
        # Handle NaN/inf.
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale.
        X_scaled = self.scaler.transform(X)
        
        # Predict.
        prob = self.model.predict_proba(X_scaled)[0, 1]
        pred = int(prob >= 0.5)
        
        return pred, prob
    
    def predict_batch(self, df):
        '''Make predictions on DataFrame.'''

        X = df[self.feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        return preds, probs


def run_realtime(predictor, topic_in='ticks.features', topic_out='ticks.predictions', 
                 duration_minutes=None):
    '''Run real-time inference from Kafka.'''
    
    config = load_config()
    bootstrap_servers = config['kafka']['bootstrap_servers']
    logger = setup_logger('RealtimeInference')
    
    # Setup consumer.
    consumer = KafkaConsumer(
        topic_in,
        bootstrap_servers=[bootstrap_servers],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id='inference-group'
    )
    
    # Setup producer.
    producer = KafkaProducer(
        bootstrap_servers=[bootstrap_servers],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    logger.info(f'Consuming from: {topic_in}')
    logger.info(f'Publishing to: {topic_out}')
    
    start_time = time.time()
    message_count = 0
    prediction_count = 0
    total_latency = 0
    
    try:
        while True:
            # Check duration.
            if duration_minutes:
                if (time.time() - start_time) / 60 >= duration_minutes:
                    break
            
            # Poll for messages.
            msg_batch = consumer.poll(timeout_ms=1000)
            
            if not msg_batch:
                continue
            
            for topic_partition, messages in msg_batch.items():
                for message in messages:
                    msg = message.value
                    message_count += 1
                    
                    # Time the prediction.
                    pred_start = time.time()
                    pred, prob = predictor.predict(msg)
                    pred_latency = time.time() - pred_start
                    total_latency += pred_latency
                    
                    # Create output message.
                    output = {
                        'product_id': msg.get('product_id'),
                        'timestamp': msg.get('timestamp'),
                        'predicted_spike': pred,
                        'spike_probability': float(prob),
                        'prediction_latency_ms': pred_latency * 1000
                    }
                    
                    # Publish prediction.
                    producer.send(topic_out, value=output)
                    prediction_count += 1
                    
                    # Log high probability predictions.
                    if prob > 0.7:
                        logger.info(
                            f'[{msg.get("product_id")}] HIGH SPIKE PROBABILITY: {prob:.3f}'
                        )
                    
                    # Log progress.
                    if message_count % 1000 == 0:
                        avg_latency = (total_latency / message_count) * 1000
                        logger.info(
                            f'Processed {message_count} messages, '
                            f'avg latency: {avg_latency:.2f}ms'
                        )
    
    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    
    finally:
        consumer.close()
        producer.flush()
        producer.close()
        
        elapsed = time.time() - start_time
        avg_latency = (total_latency / message_count * 1000) if message_count > 0 else 0
        
        logger.info(f'\nInference Summary:')
        logger.info(f'  Duration: {elapsed:.1f}s')
        logger.info(f'  Messages processed: {message_count}')
        logger.info(f'  Predictions made: {prediction_count}')
        logger.info(f'  Avg latency: {avg_latency:.2f}ms')


def run_batch(predictor, features_path, output_path=None):
    '''Run batch inference on parquet file.'''
    
    logger = setup_logger('BatchInference')
    
    # Load data.
    logger.info(f'Loading features from {features_path}')
    df = pd.read_parquet(features_path)
    logger.info(f'Loaded {len(df):,} rows')
    
    # Time the predictions.
    start_time = time.time()
    preds, probs = predictor.predict_batch(df)
    elapsed = time.time() - start_time
    
    # Add predictions to dataframe.
    df['predicted_spike'] = preds
    df['spike_probability'] = probs
    
    # Calculate metrics.
    avg_latency = (elapsed / len(df)) * 1000
    throughput = len(df) / elapsed
    
    logger.info(f'\nBatch Inference Results:')
    logger.info(f'  Total time: {elapsed:.2f}s')
    logger.info(f'  Avg latency: {avg_latency:.4f}ms per sample')
    logger.info(f'  Throughput: {throughput:.0f} samples/s')
    
    # Evaluate if labels exist.
    if 'target_spike' in df.columns:
        y_true = df['target_spike'].values
        pr_auc = average_precision_score(y_true, probs)
        f1 = f1_score(y_true, preds)
        
        logger.info(f'\nEvaluation:')
        logger.info(f'  PR-AUC: {pr_auc:.4f}')
        logger.info(f'  F1: {f1:.4f}')
    
    # Save predictions.
    if output_path:
        output_cols = ['product_id', 'timestamp', 'predicted_spike', 'spike_probability']
        if 'target_spike' in df.columns:
            output_cols.append('target_spike')
        
        df[output_cols].to_parquet(output_path, index=False)
        logger.info(f'\nPredictions saved to {output_path}')
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run volatility spike inference')
    parser.add_argument('--model', type=str, default='models/artifacts/xgboost_tuned.pkl',
                       help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='models/artifacts/scaler.pkl',
                       help='Path to fitted scaler')
    parser.add_argument('--features-json', type=str, default='models/artifacts/feature_columns.json',
                       help='Path to feature columns JSON')
    parser.add_argument('--mode', type=str, choices=['realtime', 'batch'], default='batch',
                       help='Inference mode')
    parser.add_argument('--features', type=str, default=None,
                       help='Path to features parquet (batch mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for predictions')
    parser.add_argument('--topic-in', type=str, default='ticks.features',
                       help='Input Kafka topic (realtime mode)')
    parser.add_argument('--topic-out', type=str, default='ticks.predictions',
                       help='Output Kafka topic (realtime mode)')
    parser.add_argument('--duration', type=int, default=None,
                       help='Run duration in minutes (realtime mode)')
    
    args = parser.parse_args()
    
    # Load predictor.
    predictor = VolatilityPredictor(
        model_path=args.model,
        scaler_path=args.scaler,
        feature_cols_path=args.features_json
    )
    
    if args.mode == 'realtime':
        run_realtime(
            predictor,
            topic_in=args.topic_in,
            topic_out=args.topic_out,
            duration_minutes=args.duration
        )
    
    elif args.mode == 'batch':
        if not args.features:
            print('Error: --features required for batch mode')
            return 1
        
        run_batch(predictor, args.features, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())