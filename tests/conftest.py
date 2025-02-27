import pytest
import numpy as np
from datetime import datetime
from src.api.app import create_app
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db.models import Prediction
from src.core.model import ModelManager
from src.core.preprocessing import TransactionPreprocessor

@pytest.fixture
def app():
    """Create test application"""
    return create_app()

@pytest.fixture
def client(app):
    """Create test client"""
    with TestClient(app=app) as test_client:
        yield test_client

@pytest.fixture(scope="session")
def model_manager():
    """Fixture that loads the actual model. Session scope mean it's loader once for all tests"""
    return ModelManager()

@pytest.fixture(scope="session")
def preprocessor(model_manager):
    """similar to model manager loads the actual preprocessor"""
    return TransactionPreprocessor(model_manager=model_manager)

@pytest.fixture
def valid_single_transaction():
    """Fixture providing a valid transaction data. These values are form actual fraud data we trained on."""
    return {
        "transaction_id": "test_tx_123",
        "amount": 150.00,
        "timestamp": "2024-02-18T10:30:00Z",
        "features": {
            "V1": -1.359807134,
            "V2": -0.072781173,
            "V3": 2.536346738,
            "V4": 1.378155708,
            "V5": -0.338321176,
            "V6": 0.462387778,
            "V7": 0.239598554,
            "V8": 0.098698315,
            "V9": 0.363787089,
            "V10": 0.090794172,
            "V11": -0.551599533,
            "V12": -0.617800856,
            "V13": -0.991389847,
            "V14": -0.311169354,
            "V15": 1.468176972,
            "V16": -0.470400525,
            "V17": 0.207971242,
            "V18": 0.025791000,
            "V19": -0.589281541,
            "V20": -0.375000000,
            "V21": -0.232083161,
            "V22": 0.003274103,
            "V23": 0.099911708,
            "V24": -0.145783041,
            "V25": -0.136767093,
            "V26": -0.088236784,
            "V27": -0.055127866,
            "V28": -0.059996371
        }
    }

@pytest.fixture
def invalid_single_transaction():
    """Contains deliberate errors to test validation."""
    transaction = {
        "transaction_id": "test_tx_invalid",
        "amount": -100.00,  # Invalid negative amount
        "timestamp": "2024-02-18T10:30:00Z",
        "features": {
            # Missing some V features to test error handling
            "V1": -1.359807134,
            "V2": -0.072781173,
            "V3": np.inf,  # Invalid infinite value
            "V4": np.nan,  # Invalid NaN value
        }
    }
    return transaction

@pytest.fixture
def valid_batch_transactions():
    """Fixture for a list of valid transactions with unique identifiers."""
    transactions = []
    timestamp = "2024-02-18T10:30:00Z"  # Already in ISO format
    for i in range(3):  # Create 3 transactions for the batch
        transactions.append({
            "transaction_id": f"test_tx_{i}" ,
            "amount": 100.0 * i + 1,
            "timestamp": timestamp,
            "features": {
                "V1": -1.359807134,
                "V2": -0.072781173,
                "V3": 2.536346738,
                "V4": 1.378155708,
                "V5": -0.338321176,
                "V6": 0.462387778,
                "V7": 0.239598554,
                "V8": 0.098698315,
                "V9": 0.363787089,
                "V10": 0.090794172,
                "V11": -0.551599533,
                "V12": -0.617800856,
                "V13": -0.991389847,
                "V14": -0.311169354,
                "V15": 1.468176972,
                "V16": -0.470400525,
                "V17": 0.207971242,
                "V18": 0.025791000,
                "V19": -0.589281541,
                "V20": -0.375000000,
                "V21": -0.232083161,
                "V22": 0.003274103,
                "V23": 0.099911708,
                "V24": -0.145783041,
                "V25": -0.136767093,
                "V26": -0.088236784,
                "V27": -0.055127866,
                "V28": -0.059996371
            }
        })
    return transactions

# Teardowns
@pytest.fixture
def cleanup_prediction(valid_single_transaction):
    """Fixture to cleanup prediction data after test"""
    transactions = valid_single_transaction.copy()
    db:  Session = next(get_db())
    try:
        prediction = db.query(Prediction).filter(
            Prediction.transaction_id == transactions["transaction_id"]
        )
        if prediction:
            prediction.delete()
            db.commit()
            print(f"Prediction data for {transactions['transaction_id']} deleted")  
    except Exception as e:
        db.rollback()
        print(f"Error deleting prediction data: {e}")
    finally:
        db.close()

@pytest.fixture
def cleanup_batch_predictions(valid_batch_transactions):
    """Fixture to add batch predictions and then remove them after the test."""
    transaction_ids = [transaction["transaction_id"] for transaction in valid_batch_transactions]
    yield transaction_ids  # Provide the transaction_ids to the test

    # Teardown code: Remove the predictions after the test
    db: Session = next(get_db())
    try:
        for transaction_id in transaction_ids:
            prediction = db.query(Prediction).filter(Prediction.transaction_id == transaction_id).first()
            if prediction:
                db.delete(prediction)
        db.commit()
        print(f"Predictions with transaction_ids {transaction_ids} removed.")
    except Exception as e:
        db.rollback()
        print(f"Error deleting predictions: {e}")
    finally:
        db.close()
