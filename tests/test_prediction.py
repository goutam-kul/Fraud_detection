import pytest
import numpy as np
from src.core.preprocessing import TransactionPreprocessor

def test_single_transaction_preprocessing(preprocessor, valid_single_transaction):
    """Test preprocessing of a single valid transaction"""
    # Preprocess the transaction
    features = preprocessor.preprocess_transaction(valid_single_transaction)

    # Test output type and shapes
    assert isinstance(features, np.ndarray), "Output should be numpy array"
    assert features.shape == (1, 30), "Shape should be (1, 30) for single transaction"
    
    # Test no invalid values
    assert np.any(np.isnan(features)) == False, "Should not contain NaN values"
    assert abs(np.any(np.isinf(features))) == False, "Should not contains infinite values"

    # Test day part calculations
    day_part = features[0, -1]
    assert isinstance(day_part, (float, np.float64)), "Day part should be integer"
    assert 0 <= day_part <= 3, "Day part should be between 0 and 3"

    # Test amount scaling
    scaled_amount = features[0, -2] 
    assert isinstance(scaled_amount, (float, np.float64)), "Scaled amount should be a float"

def test_single_prediction(model_manager, preprocessor, valid_single_transaction):
    """Test actual prediction for a single transaction"""
    # Preprocess and predict
    features = preprocessor.preprocess_transaction(valid_single_transaction)
    prediction = model_manager.predict(features)
    
    # Test prediction type and range
    assert isinstance(prediction, float), "Prediction should be float"
    assert 0 <= prediction <= 1, "Prediction should be between 0 and 1"
    
    # Test fraud classification
    is_fraud = model_manager.is_fraud(prediction)
    assert isinstance(is_fraud, bool), "Fraud classification should be boolean"

def test_prediction_consistency(model_manager, preprocessor, valid_single_transaction):
    """Test that predictions are consistent for same input"""
    # Process and predict twice
    features1 = preprocessor.preprocess_transaction(valid_single_transaction)
    features2 = preprocessor.preprocess_transaction(valid_single_transaction)
    
    pred1 = model_manager.predict(features1)
    pred2 = model_manager.predict(features2)
    
    # Test predictions are identical for same input
    assert pred1 == pred2, "Predictions should be consistent for same input"
    
    # Test feature processing is consistent
    np.testing.assert_array_equal(features1, features2, 
                                 "Feature processing should be consistent")
    

def test_invalid_transaction_handling(preprocessor, invalid_single_transaction):
    """Test handling of invalid transation data"""
    # Test should raise ValueError for invalid data 
    with pytest.raises(ValueError) as exc_info:
        preprocessor.preprocess_transaction(invalid_single_transaction)

    # Verify error message is helpful
    assert "failed" in str(exc_info.value).lower()

def test_batch_transaction_processing(preprocessor, model_manager, valid_batch_transaction):
    """Test batch predictions"""
    # Test batch preprocessing
    features = preprocessor.preprocess_batch(transactions=valid_batch_transaction)

    # Verify shape and type
    assert features.shape == (len(features), 30), f"Expected shape ({len(features)}, 30), got {features.shape}"
    assert isinstance(features, np.ndarray), "Batch features should be numpy array"

    # Test batch prediction
    predictions = model_manager.batch_predict(features=features)

    # Verify predictions
    assert len(predictions) == len(features), "Should have one prediction per transaction"
    assert all(0 <= p <= 1 for p in predictions), "All prediction values should be between 0 and 1"

    # Test fraud classifications
    fraud_flags = [model_manager.is_fraud(p) for p in predictions]
    assert all(isinstance(f, (bool, np.bool)) for f in fraud_flags), "All fraud flags should be boolean"