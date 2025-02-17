from pydantic import BaseModel, Field, ConfigDict
from typing import List
from datetime import datetime

class TransactionFeatures(BaseModel):
    """V1-V28 features from the credit card dataset"""
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)

class TransactionRequest(BaseModel):
    """Single transaction request model"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction Amount")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    features: TransactionFeatures = Field(default_factory=TransactionFeatures)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "tx_123456789",
                "amount": 150.00,
                "timestamp": "2024-02-16T10:30:00Z",
                "features": {
                    "V1": -1.3598071336738172,
                    "V2": -0.0727811733098497,
                    "V3": 2.536346738618732,
                    "V4": 1.378155707623914,
                    # ... other features
                }
            }
        }
    )

class TransactionResponse(BaseModel):
    """Single transaction response model"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    processing_time: float  # in milliseconds
    timestamp: datetime

    class Config:
        from_attributes = True  

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    results: List[TransactionResponse]
    total_processing_time: float  # in milliseconds
    timestamp: datetime