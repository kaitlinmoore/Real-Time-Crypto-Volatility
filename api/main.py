import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Configuration via environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'models/artifacts/logistic_regression.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'models/artifacts/scaler.pkl')
FEATURE_COLS_PATH = os.getenv('FEATURE_COLS_PATH', 'models/artifacts/feature_columns.json')
API_VERSION = os.getenv('API_VERSION', '1.0.0')
MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')


# Global state for model and metrics
class AppState:
    model = None
    scaler = None
    feature_columns = None
    model_name = None
    load_time = None
    
    # Metrics counters.
    prediction_count = 0
    prediction_latency_sum = 0.0
    prediction_errors = 0
    last_prediction_time = None


state = AppState()


# Pydantic models for request/response validation
class FeatureInput(BaseModel):
    '''Input features for prediction'''
    
    product_id: str = Field(..., description='Trading pair (e.g., BTC-USD)')
    features: Dict[str, float] = Field(
        ..., 
        description='Feature name to value mapping'
    )
    timestamp: Optional[str] = Field(
        None, 
        description='Optional timestamp for the prediction request'
    )

    class Config:
        json_schema_extra = {
            'example': {
                'product_id': 'BTC-USD',
                'features': {
                    'w60_spread_std': 0.00015,
                    'w60_return_std': 0.0003,
                    'w300_spread_std': 0.00012,
                    'w60_spread_mean': 0.0001,
                },
                'timestamp': '2024-01-15T10:30:00Z'
            }
        }


class PredictionResponse(BaseModel):
    '''Prediction response'''
    
    product_id: str
    prediction: int = Field(..., description='0 = no spike, 1 = spike predicted')
    probability: float = Field(..., description='Probability of volatility spike')
    timestamp: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    '''Health check response'''
    
    status: str
    model_loaded: bool
    uptime_seconds: float
    last_prediction: Optional[str]


class VersionResponse(BaseModel):
    '''Version information response'''
    
    api_version: str
    model_version: str
    model_name: str
    feature_count: int
    load_time: str


class BatchFeatureInput(BaseModel):
    '''Batch input for multiple predictions'''
    
    items: List[FeatureInput]


class BatchPredictionResponse(BaseModel):
    '''Batch prediction response'''
    
    predictions: List[PredictionResponse]
    batch_size: int
    total_latency_ms: float


def load_model_artifacts():
    '''Load model, scaler, and feature columns from disk.'''
    
    # Load model.
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')
    
    with open(model_path, 'rb') as f:
        state.model = pickle.load(f)
    
    state.model_name = model_path.stem
    
    # Load scaler.
    scaler_path = Path(SCALER_PATH)
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            state.scaler = pickle.load(f)
    else:
        print(f'Warning: Scaler not found at {SCALER_PATH}, predictions will use raw features')
        state.scaler = None
    
    # Load feature columns.
    feature_cols_path = Path(FEATURE_COLS_PATH)
    if not feature_cols_path.exists():
        raise FileNotFoundError(f'Feature columns not found: {FEATURE_COLS_PATH}')
    
    with open(feature_cols_path, 'r') as f:
        state.feature_columns = json.load(f)
    
    state.load_time = datetime.now(timezone.utc)
    
    print(f'Loaded model: {state.model_name}')
    print(f'Feature columns: {len(state.feature_columns)}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Load model on startup, cleanup on shutdown.'''
    
    print('Starting prediction service...')
    try:
        load_model_artifacts()
        print('Model loaded successfully')
    except Exception as e:
        print(f'Failed to load model: {e}')
        raise
    
    yield
    
    print('Shutting down prediction service...')


# Create FastAPI app.
app = FastAPI(
    title='Volatility Prediction API',
    description='Real-time cryptocurrency volatility spike prediction service',
    version=API_VERSION,
    lifespan=lifespan
)


@app.get('/health', response_model=HealthResponse)
def health_check():
    '''Service health check endpoint.'''
    
    uptime = 0.0
    if state.load_time:
        uptime = (datetime.now(timezone.utc) - state.load_time).total_seconds()
    
    last_pred = None
    if state.last_prediction_time:
        last_pred = state.last_prediction_time.isoformat()
    
    return HealthResponse(
        status='healthy' if state.model is not None else 'unhealthy',
        model_loaded=state.model is not None,
        uptime_seconds=uptime,
        last_prediction=last_pred
    )


@app.get('/version', response_model=VersionResponse)
def get_version():
    '''Get API and model version information.'''
    
    if state.model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    return VersionResponse(
        api_version=API_VERSION,
        model_version=MODEL_VERSION,
        model_name=state.model_name or 'unknown',
        feature_count=len(state.feature_columns) if state.feature_columns else 0,
        load_time=state.load_time.isoformat() if state.load_time else 'unknown'
    )


@app.post('/predict', response_model=PredictionResponse)
def predict(input_data: FeatureInput):
    '''Generate volatility spike prediction for given features.'''
    
    start_time = time.perf_counter()
    
    if state.model is None:
        state.prediction_errors += 1
        raise HTTPException(status_code=503, detail='Model not loaded')
    
    try:
        # Build feature vector in correct order.
        feature_vector = []
        missing_features = []
        
        for col in state.feature_columns:
            if col in input_data.features:
                feature_vector.append(input_data.features[col])
            else:
                # Use 0 for missing features (matches training behavior).
                feature_vector.append(0.0)
                missing_features.append(col)
        
        if missing_features and len(missing_features) > len(state.feature_columns) * 0.5:
            # Warn if more than half the features are missing.
            print(f'Warning: {len(missing_features)} features missing for {input_data.product_id}')
        
        # Convert to numpy array.
        X = np.array(feature_vector).reshape(1, -1)
        
        # Scale features if scaler is available.
        if state.scaler is not None:
            X = state.scaler.transform(X)
        
        # Handle NaN/inf values.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get prediction and probability.
        prediction = int(state.model.predict(X)[0])
        probability = float(state.model.predict_proba(X)[0][1])
        
        # Calculate latency.
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Update metrics.
        state.prediction_count += 1
        state.prediction_latency_sum += latency_ms
        state.last_prediction_time = datetime.now(timezone.utc)
        
        # Determine timestamp.
        timestamp = input_data.timestamp or datetime.now(timezone.utc).isoformat()
        
        return PredictionResponse(
            product_id=input_data.product_id,
            prediction=prediction,
            probability=probability,
            timestamp=timestamp,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 3)
        )
    
    except Exception as e:
        state.prediction_errors += 1
        raise HTTPException(status_code=500, detail=f'Prediction failed: {str(e)}')


@app.post('/predict/batch', response_model=BatchPredictionResponse)
def predict_batch(input_data: BatchFeatureInput):
    '''Generate predictions for multiple feature sets.'''
    
    start_time = time.perf_counter()
    
    predictions = []
    for item in input_data.items:
        pred = predict(item)
        predictions.append(pred)
    
    total_latency = (time.perf_counter() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_size=len(predictions),
        total_latency_ms=round(total_latency, 3)
    )


@app.get('/metrics', response_class=PlainTextResponse)
def metrics():
    '''Prometheus-format metrics endpoint'''
    
    avg_latency = 0.0
    if state.prediction_count > 0:
        avg_latency = state.prediction_latency_sum / state.prediction_count
    
    uptime = 0.0
    if state.load_time:
        uptime = (datetime.now(timezone.utc) - state.load_time).total_seconds()
    
    lines = [
        '# HELP predictions_total Total number of predictions made',
        '# TYPE predictions_total counter',
        f'predictions_total {state.prediction_count}',
        '',
        '# HELP prediction_errors_total Total number of prediction errors',
        '# TYPE prediction_errors_total counter',
        f'prediction_errors_total {state.prediction_errors}',
        '',
        '# HELP prediction_latency_avg_ms Average prediction latency in milliseconds',
        '# TYPE prediction_latency_avg_ms gauge',
        f'prediction_latency_avg_ms {avg_latency:.3f}',
        '',
        '# HELP model_loaded Whether the model is loaded (1) or not (0)',
        '# TYPE model_loaded gauge',
        f'model_loaded {1 if state.model is not None else 0}',
        '',
        '# HELP uptime_seconds Service uptime in seconds',
        '# TYPE uptime_seconds gauge',
        f'uptime_seconds {uptime:.1f}',
        '',
    ]
    
    return '\n'.join(lines)


@app.get('/features')
def get_features():
    '''Return list of expected feature names.'''
    
    if state.feature_columns is None:
        raise HTTPException(status_code=503, detail='Feature columns not loaded')
    
    return {
        'feature_count': len(state.feature_columns),
        'features': state.feature_columns
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
