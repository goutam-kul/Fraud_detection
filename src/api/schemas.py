from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Any
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
    timestamp: str = Field(..., description="Transaction timestamp in ISO format")
    features: TransactionFeatures = Field(default_factory=TransactionFeatures)

    @field_validator("timestamp")
    def validate_timestamp(cls, v: Any) -> str:
        """Ensure timestamp is in ISO format"""
        if isinstance(v, datetime):
            return v.isoformat().replace("+00:00", "Z")
        elif isinstance(v, str):
            try:
                # Validate string format by parsing and reformatting
                dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                # Return the original string if it's already in ISO format
                return v
            except ValueError as e:
                raise ValueError(f"Invalid timestamp format: {str(e)}")
        raise ValueError("Timestamp must be string or datetime")


    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

class TransactionResponse(BaseModel):
    """Single transaction response model"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    processing_time: float  # in milliseconds
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=1000)

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model"""
    results: List[TransactionResponse]
    total_processing_time: float  # in milliseconds
    timestamp: datetime