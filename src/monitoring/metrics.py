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
    # Create histograms
    expected_hist, _ = np.histogram(expected, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bins, density=True)

    # Add small epsilon to avoid division by zero 
    epsilon = 1e-10
    expected_hist = expected_hist + epsilon
    actual_hist = actual_hist + epsilon

    # Calculate PSI
    psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
    return psi

def update_drift_metrics(
    features: Dict[str, float],    # Current feature name and values
    prediction: float,             # Current model prediction
    reference_window_size: int = 1000  # How may historical values to keep
):
    """Update drift-related metrics"""
    # Track model prediction distribution
    PREDICTION_DISTRIBUTION.observe(prediction)

    # Update feature distribution and calculate drift
    for feature_name, value in features.items():
        if feature_name not in REFERENCE_DISTRIBUTIONS:
            REFERENCE_DISTRIBUTIONS[feature_name] = []  # Initialize with empty list

        # Add new value to reference distribution
        REFERENCE_DISTRIBUTIONS[feature_name].append(float(value))

        # Keep only recent values
        if len(REFERENCE_DISTRIBUTIONS[feature_name]) > reference_window_size:
            REFERENCE_DISTRIBUTIONS[feature_name].pop(0)

        # Calculate PSI if we have enough data
        if len(REFERENCE_DISTRIBUTIONS[feature_name]) >= reference_window_size:
            reference_data = np.array(REFERENCE_DISTRIBUTIONS[feature_name][:-100]) # Older data
            current_data = np.array(REFERENCE_DISTRIBUTIONS[feature_name][-100:])  # Recent data
 
            # Calculate Population Stability Index
            try:
                psi_score = calculate_psi(reference_data, current_data)
                FEATURE_DRIFT.labels(feature_name=feature_name).set(psi_score)
            except Exception as e:
                print(f"Error calculating PSI for {feature_name}: {str(e)}")

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

