from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
from src.config.constants import FEATURE_NAMES
from .model import model_manager

class TransactionPreprocessor:
    """Preprocess transaction data for fraud detection"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def _parse_timestamp(self, timestamp: Union[str, datetime]) -> datetime:
        """Standardized timestamp parsing"""
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            try:
                # Handle both 'Z' and '+00:00' UTC indicators
                cleaned_ts = timestamp.replace('Z', '+00:00')
                return datetime.fromisoformat(cleaned_ts)
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format: {str(e)}")
        raise ValueError("Timestamp must be string or datetime")
    
    def _convert_to_day_part(self, timestamp: datetime) -> int:
        """Convert timestamp to day part (0-3)"""
        hour = timestamp.hour
        return hour // 6  # Split day into parts of 4 (0-3)

    def preprocess_transaction(self, transaction_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single transaction for prediction"""
        try:
            # 1. Convert timestamp to day_part
            raw_timestamp = transaction_data['timestamp']
            timestamp = self._parse_timestamp(raw_timestamp)
            day_part = self._convert_to_day_part(timestamp)
            
            # 2. Scale amount
            amount = float(transaction_data['amount'])
            amount_df = pd.DataFrame({'Amount': [amount]})
            scaled_amount = self.model_manager.scaler.transform(amount_df)[0][0]
            
            # 3. Create feature array in EXACT same order as training
            features = np.zeros(30)
            
            # First add V1-V28 (indices 0-27)
            for i in range(1, 29):
                feature_name = f'V{i}'
                features[i-1] = float(transaction_data['features'][feature_name])
            
            # Then add Amount (index 28)
            features[28] = scaled_amount
            
            # Finally add day_part (index 29)
            features[29] = day_part
            
            # print(features.reshape(1, -1))   # Debug
            return features.reshape(1, -1)
            
        except Exception as e:
            raise ValueError(f"Failed to preprocess transaction: {str(e)}")

    def debug_features(self, features: np.ndarray) -> Dict[str, float]:
        """Debug helper to print feature values"""
        feature_dict = {}
        # V1-V28 first
        for i in range(1, 29):
            feature_dict[f'V{i}'] = features[i-1]
        # Then Amount
        feature_dict['Amount'] = features[28]
        # day_part last
        feature_dict['day_part'] = features[29]
        return feature_dict

    def preprocess_batch(self, transactions: list[Dict[str, Any]]) -> np.ndarray:
        """Preprocess multiple transactions for prediction"""
        try:
            # Preporcess all transactions
            feature_arrays = []
            print(f"Raw Transaction: {transactions}")  # Debug
            for transaction in transactions:
                print(f"transaction: {transaction}")
                features = self.preprocess_transaction(transaction_data=transaction)
                feature_arrays.append(features.squeeze())  # (1,30) -> (30, )
            
            return np.array(feature_arrays)  # (n_transactions, 30)

        except Exception as e:
            raise ValueError(f"Failed to preprocess batch: {str(e)}")
        

# Create global preprocessor instance
preprocessor = TransactionPreprocessor(model_manager=model_manager)