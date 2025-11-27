import argparse
import json
import sys
import time

import requests


def test_health(base_url):
    '''Test health endpoint.'''
    
    print('Testing GET /health...')
    response = requests.get(f'{base_url}/health')
    
    print(f'  Status: {response.status_code}')
    print(f'  Response: {response.json()}')
    
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print('  PASSED')
    return True


def test_version(base_url):
    '''Test version endpoint.'''
    
    print('Testing GET /version...')
    response = requests.get(f'{base_url}/version')
    
    print(f'  Status: {response.status_code}')
    print(f'  Response: {response.json()}')
    
    assert response.status_code == 200
    assert 'api_version' in response.json()
    assert 'model_version' in response.json()
    print('  PASSED')
    return True


def test_features(base_url):
    '''Test features endpoint.'''
    
    print('Testing GET /features...')
    response = requests.get(f'{base_url}/features')
    
    print(f'  Status: {response.status_code}')
    data = response.json()
    print(f'  Feature count: {data["feature_count"]}')
    
    assert response.status_code == 200
    assert data['feature_count'] > 0
    print('  PASSED')
    return data['features']


def test_predict(base_url, features):
    '''Test predict endpoint.'''
    
    print('Testing POST /predict...')
    
    # Build feature dict with sample values.
    feature_values = {f: 0.001 for f in features[:20]}  # Use first 20 features
    
    # Add some realistic values for key features.
    feature_values.update({
        'w60_spread_std': 0.00015,
        'w60_return_std': 0.0003,
        'w300_spread_std': 0.00012,
        'w60_spread_mean': 0.0001,
    })
    
    payload = {
        'product_id': 'BTC-USD',
        'features': feature_values,
        'timestamp': '2024-01-15T10:30:00Z'
    }
    
    response = requests.post(
        f'{base_url}/predict',
        json=payload
    )
    
    print(f'  Status: {response.status_code}')
    print(f'  Response: {response.json()}')
    
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'probability' in data
    assert data['prediction'] in [0, 1]
    assert 0 <= data['probability'] <= 1
    print('  PASSED')
    return True


def test_predict_batch(base_url, features):
    '''Test batch predict endpoint.'''
    
    print('Testing POST /predict/batch...')
    
    # Build feature dict with sample values.
    feature_values = {f: 0.001 for f in features[:20]}
    
    items = [
        {'product_id': 'BTC-USD', 'features': feature_values},
        {'product_id': 'ETH-USD', 'features': feature_values},
        {'product_id': 'SOL-USD', 'features': feature_values},
    ]
    
    response = requests.post(
        f'{base_url}/predict/batch',
        json={'items': items}
    )
    
    print(f'  Status: {response.status_code}')
    data = response.json()
    print(f'  Batch size: {data["batch_size"]}')
    print(f'  Total latency: {data["total_latency_ms"]:.2f}ms')
    
    assert response.status_code == 200
    assert data['batch_size'] == 3
    print('  PASSED')
    return True


def test_metrics(base_url):
    '''Test metrics endpoint.'''
    
    print('Testing GET /metrics...')
    response = requests.get(f'{base_url}/metrics')
    
    print(f'  Status: {response.status_code}')
    print(f'  Sample output:')
    for line in response.text.split('\n')[:6]:
        print(f'    {line}')
    
    assert response.status_code == 200
    assert 'predictions_total' in response.text
    print('  PASSED')
    return True


def test_latency(base_url, features, n_requests=100):
    '''Test prediction latency.'''
    
    print(f'Testing latency ({n_requests} requests)...')
    
    feature_values = {f: 0.001 for f in features[:20]}
    payload = {
        'product_id': 'BTC-USD',
        'features': feature_values,
    }
    
    latencies = []
    
    for i in range(n_requests):
        start = time.perf_counter()
        response = requests.post(f'{base_url}/predict', json=payload)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        
        if response.status_code != 200:
            print(f'  Error on request {i}: {response.status_code}')
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
    
    print(f'  Requests: {n_requests}')
    print(f'  Avg latency: {avg_latency:.2f}ms')
    print(f'  Min latency: {min_latency:.2f}ms')
    print(f'  Max latency: {max_latency:.2f}ms')
    print(f'  p99 latency: {p99_latency:.2f}ms')
    
    # Check if we meet real-time requirements (< 100ms avg).
    if avg_latency < 100:
        print('  PASSED (meets real-time requirement)')
    else:
        print('  WARNING: latency exceeds 100ms')
    
    return avg_latency


def main():
    parser = argparse.ArgumentParser(description='Test the Volatility Prediction API')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--latency-test', action='store_true',
                       help='Run latency test')
    parser.add_argument('--n-requests', type=int, default=100,
                       help='Number of requests for latency test')
    
    args = parser.parse_args()
    
    print(f'Testing API at: {args.url}')
    print('=' * 50)
    
    try:
        # Run tests.
        test_health(args.url)
        print()
        
        test_version(args.url)
        print()
        
        features = test_features(args.url)
        print()
        
        test_predict(args.url, features)
        print()
        
        test_predict_batch(args.url, features)
        print()
        
        test_metrics(args.url)
        print()
        
        if args.latency_test:
            test_latency(args.url, features, args.n_requests)
            print()
        
        print('=' * 50)
        print('All tests passed.')
        return 0
    
    except requests.exceptions.ConnectionError:
        print(f'ERROR: Could not connect to {args.url}')
        print('Make sure the API server is running.')
        return 1
    
    except AssertionError as e:
        print(f'TEST FAILED: {e}')
        return 1
    
    except Exception as e:
        print(f'ERROR: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
