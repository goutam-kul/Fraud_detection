from prometheus_client import Counter, Histogram, Gauge
import numpy as np
from typing import Dict, List

# Essential Business Metrics
FRAUD_COUNTER = Counter(
    'fraud_detection_total',
    'Total number of fraud detections',
    ['result']  # 'fraud' or 'legitimate'
)

TRANSACTION_AMOUNT = Histogram(
    'transaction_amount_distribution',
    'Distribution of transaction amounts',
    buckets=[10, 50, 100, 500, 1000, 5000, 10000]
)

# API Performance Metrics
HTTP_REQUESTS = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['endpoint', 'status_code']
)

RESPONSE_TIME = Histogram(
    'http_response_time_seconds',
    'HTTP request latency',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_TIME = Histogram(
    'prediction_time_seconds',
    'Time taken for model prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

# Model Drift Metrics
PREDICTION_DISTRIBUTION = Histogram(
    'prediction_distribution',
    'Distribution of model predictions',
    buckets=np.linspace(0, 1, 11).tolist()   # 10 buckets from 0 to 1
)

FEATURE_DRIFT = Gauge(
    'feature_drift',
    'Feature drift score for each feature',
    ['feture_name']
)

MODEL_DRIFT_SCORE = Gauge(
    'model_drift_score',
    'Overall model drift score'
)

PSI_SCORE = Gauge(
    'population_stability_index',
    'PSI score of detecting distribution shifts',
    ['feature_name']
)

# Reference distribution for drift detection
REFERENCE_DISTRIBUTIONS: Dict[str, List[float]] = {}  # { 'V1': [....], 'V2':[...] ... 'amount':[...], 'day_part':[...] }

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    
    Guidelines:
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    """
    # Input validation
    if len(expected) < 2 or len(actual) < 2:
        return np.nan
        
    try:
        # Create histograms with same bins for both distributions
        hist_range = (min(expected.min(), actual.min()), max(expected.max(), actual.max()))
        expected_hist, bin_edges = np.histogram(expected, bins=bins, range=hist_range, density=True)
        actual_hist, _ = np.histogram(actual, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        expected_hist = expected_hist + epsilon
        actual_hist = actual_hist + epsilon
        
        # Calculate PSI
        psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
        
        return float(psi)
    except Exception as e:
        print(f"PSI calculation error: {str(e)}")
        return np.nan
    
def calculate_model_drift_score(
    current_predictions: List[float],
    historical_predictions: List[float],
    feature_psi_score: Dict[str, float]
):
    """Calculate overall model drift score combining multiple singals"""
    # Calculate prediction distribution drift using PSI
    pred_psi = calculate_psi(
        np.array(historical_predictions),
        np.array(current_predictions)
    )

    avg_feature_psi = np.mean(list(feature_psi_score.values()))
    # weigts the components 
    weights = {
        'prediction_drift': 0.6,  # Prediction distribution changes
        'feature_drift': 0.4       # Feature distribution changes
    }

    drift_score = (
        weights['prediction_drift'] * pred_psi +
        weights['feature_drift'] * avg_feature_psi
    )

    return min(1.0, drift_score)  # Cap at 1.0
    

def update_drift_metrics(
    features: Dict[str, float],
    prediction: float,
    reference_window_size: int = 1000
):
    """Updated version with model drift calculation"""
    
    # Store predictions for drift calculation
    if 'predictions' not in REFERENCE_DISTRIBUTIONS:
        REFERENCE_DISTRIBUTIONS['predictions'] = []
    
    REFERENCE_DISTRIBUTIONS['predictions'].append(prediction)
    
    # Maintain prediction history window
    if len(REFERENCE_DISTRIBUTIONS['predictions']) > reference_window_size:
        REFERENCE_DISTRIBUTIONS['predictions'].pop(0)

    # Track model prediction distribution
    PREDICTION_DISTRIBUTION.observe(prediction)
    
    # Store PSI scores for features
    current_psi_scores = {}

    # Update feature distribution and calculate drift
    for feature_name, value in features.items():
        if feature_name not in REFERENCE_DISTRIBUTIONS:
            REFERENCE_DISTRIBUTIONS[feature_name] = []
        
        REFERENCE_DISTRIBUTIONS[feature_name].append(float(value))
        
        if len(REFERENCE_DISTRIBUTIONS[feature_name]) > reference_window_size:
            REFERENCE_DISTRIBUTIONS[feature_name].pop(0)
            
        # Calculate PSI if enough data
        if len(REFERENCE_DISTRIBUTIONS[feature_name]) >= reference_window_size:
            try:
                all_data = REFERENCE_DISTRIBUTIONS[feature_name]
                historical_data = np.array(all_data[:-100])
                recent_data = np.array(all_data[-100:])
                
                psi_score = calculate_psi(historical_data, recent_data)
                
                if not np.isnan(psi_score):
                    current_psi_scores[feature_name] = psi_score
                    PSI_SCORE.labels(feature_name=feature_name).set(psi_score)
                    FEATURE_DRIFT.labels(feature_name=feature_name).set(psi_score)
                    
            except Exception as e:
                print(f"Error calculating PSI for {feature_name}: {str(e)}")
    
    # Calculate overall model drift if we have enough data
    if (len(REFERENCE_DISTRIBUTIONS['predictions']) >= reference_window_size and 
            current_psi_scores):  # Only if we have PSI scores
        try:
            predictions = REFERENCE_DISTRIBUTIONS['predictions']
            historical_preds = predictions[:-100]
            recent_preds = predictions[-100:]
            
            model_drift = calculate_model_drift_score(
                current_predictions=recent_preds,
                historical_predictions=historical_preds,
                feature_psi_scores=current_psi_scores
            )
            
            # Update model drift metric
            MODEL_DRIFT_SCORE.set(model_drift)
            
            # Log significant drift
            if model_drift > 0.3:  # Threshold for significant drift
                print(f"WARNING: Significant model drift detected: {model_drift}")
                
        except Exception as e:
            print(f"Error calculating model drift: {str(e)}")

def track_prediction(
    fraud_probability: float,
    is_fraud: bool,
    features: Dict[str, float],
    prediction_time:  float,
    amount: float
):
    """Track a prediction with drift monitoring"""
    # Business metrics 
    FRAUD_COUNTER.labels(
        result='fraud' if is_fraud else 'legitimate'
    ).inc()

    TRANSACTION_AMOUNT.observe(amount)
    PREDICTION_TIME.observe(prediction_time)

    # Update drift metrics
    try:
        update_drift_metrics(features, fraud_probability)
    except Exception as e:
        print(f"Error updating drift metrics: {str(e)}")

def track_request(
    status_code: int,
    response_time: float,
    endpoint: str
):
    """Track on HTTP request"""
    HTTP_REQUESTS.labels(
        endpoint=endpoint,
        status_code=status_code
    ).inc()

    RESPONSE_TIME.labels(
        endpoint=endpoint
    ).observe(response_time)


