from typing import Optional, Dict, Any
import joblib
import numpy as np
from pathlib import Path
from src.config import  get_settings

settings = get_settings()

class ModelManager:
    """Manages the fraud detection model lifecycle."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_weights = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model and scaler from disk"""
        try:
            model_path = Path(settings.MODEL_PATH)
            scaler_path = Path(settings.SCALER_PATH)
            weights_path = Path(settings.CLASS_WEIGHTS_PATH)

            if not model_path.exists() or not scaler_path.exists() or not weights_path.exists():
                raise FileNotFoundError("Model or scaler file not found")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.class_weights = joblib.load(weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, feature: np.ndarray) -> float:
        """Make fraud prediction for a single transaction"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Get raw probability for fraud class
            probability = float(self.model.predict_proba(feature)[0, 1])
            
            # No need to reapply class weights as they're already incorporated in the model
            return probability
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
        
    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        """Make fraud predictions for a batch of transactions"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Print shape for debugging
            print(f"Features shape: {features.shape}")
            
            # Ensure features are properly formatted
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get raw probabilities directly - no need to reapply class weights
            probabilities = self.model.predict_proba(features)
            
            # Extract fraud class probabilities
            fraud_probs = probabilities[:, 1]
            
            # Debug print
            print(f"Raw fraud probabilities: {fraud_probs}")
            
            return fraud_probs
            
        except Exception as e:
            print(f"Error in batch_predict: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def is_fraud(self, probability: float) -> bool:
        """Determine if a transaction is fraudulent based on probability threshold"""
        # Use a more reasonable threshold (e.g., 0.5 or settings.FRAUD_THRESHOLD)
        return probability >= settings.FRAUD_THRESHOLD 

# Create global model manager instance
model_manager = ModelManager()