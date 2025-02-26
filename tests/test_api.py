import pytest
from datetime import datetime
import numpy as np
from src.api.schemas import TransactionRequest, BatchPredictionRequest

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_single_prediction(client, valid_single_transaction, cleanup_prediction):
    """Test single transaction prediction endpoint"""
    transaction = valid_single_transaction.copy()

    # Print reponse for debugging
    print("Request payload: ", transaction)
    response = client.post("/api/v1/transactions", json=valid_single_transaction)
    assert response.status_code == 201

    data = response.json()
    print("Response: ", data)
    assert "transaction_id" in data, "Transaction id missing in response"
    assert "fraud_probability" in data, "Fraud probability missing in response"
    assert "is_fraud" in data, "Fraud predition missing in response"
    assert "processing_time" in data, "Processing time missing in response"
    assert "timestamp" in data, "Timestamp missing in response"

    assert isinstance(data["fraud_probability"], float), f"Expected float, got type {type(data['fraud_probability'])}"
    assert isinstance(data["is_fraud"], (bool, np.bool)), f"Fraud prediction must be boolean, got {type(data['is_fraud'])}"
    assert 0 <= data["fraud_probability"] <= 1, f"Fraud probability must be between 0 and 1, got {data['fraud_probability']}"

def test_batch_prediction(client, valid_batch_transactions, cleanup_batch_predictions):
    """Test batch transaction prediction endpoint"""
    # Simplify the request creation
    request_payload = {
        "transactions": valid_batch_transactions
    }
    
    print("request:",request_payload)
    # Make the request
    response = client.post("/api/v1/transactions/batch", json=request_payload)
        
    assert response.status_code == 201

    data = response.json()
    assert isinstance(data.get("results"), list), "Expected a list of predictions"
    assert len(data) == len(valid_batch_transactions), "Number of predictions should match number of transactions"

    assert(all("transaction_id" in p for p in data["results"])), "Transaction id missing in response"
    assert(all("fraud_probability" in p for p in data["results"])), "Fraud probability missing in response"
    assert(all("is_fraud" in p for p in data["results"])), "Fraud predition missing in response"
    assert(all("processing_time" in p for p in data["results"])), "Processing time missing in response"
    assert(all("timestamp" in p for p in data["results"])), "Timestamp missing in response" 

    for p in data["results"]:
        assert isinstance(p["fraud_probability"], float), f"Expected float, got type {type(p['fraud_probability'])}"
        assert isinstance(p["is_fraud"], (bool, np.bool)), f"Fraud prediction must be boolean, got {type(p['is_fraud'])}"
        assert 0 <= p["fraud_probability"] <= 1, f"Fraud probability must be between 0 and 1, got {p['fraud_probability']}"