from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from src.db.models import Prediction
from src.db.database import get_db
from datetime import datetime

class PredictionCRUD:
    """CRUD operations for predictions."""

    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    def create_prediction(
        self,
        transaction_id: str,
        amount: float,
        fraud_probability: float,
        is_fraud: bool,
        processing_time: float
    ) -> Prediction:
        """Create a new prediction record."""
        db_prediction = Prediction(
            transaction_id=transaction_id,
            amount=amount,
            fraud_probability=fraud_probability,
            is_fraud=is_fraud,
            processing_time=processing_time
        )
        try: 
            self.db.add(db_prediction)
            self.db.commit()
            self.db.refresh(db_prediction)
            return db_prediction
        except IntegrityError:
            self.db.rollback()
            raise ValueError(f"Transaction {transaction_id} already exists")

    
    def get_prediction(self, transaction_id: str) -> Optional[Prediction]:
        """Get prediction by transaction ID"""
        return self.db.query(Prediction).filter(
            Prediction.transaction_id == transaction_id
        ).first()

    
    def list_predictions(self, skip: int = 0, limit: int = 100) -> List[Prediction]:
        """Get list of predictions with pagination"""
        return self.db.query(Prediction).offset(skip).limit(limit).all()

    
    def get_prediction_count(self) -> int:
        """Get total count of prediction"""
        return self.db.query(Prediction).count()