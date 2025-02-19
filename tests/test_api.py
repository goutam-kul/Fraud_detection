import pytest
from datetime import datetime
import numpy as np
from fastapi.testclient import TestClient
from src.api.app import app
from src.db.models import Prediction


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_single_prediction(client, valid_single_transaction):
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



