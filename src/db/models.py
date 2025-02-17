from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime
from sqlalchemy.sql import func
from src.db.database import Base

class Prediction(Base):
    """Prediction database model"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    amount = Column(Numeric(15, 2), nullable=False)
    fraud_probability = Column(Numeric(5, 4), nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    processing_time = Column(Numeric(10, 2), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    def __repr__(self):
        return f"<Prediction(transaction_id={self.transaction_id}, is_fraud={self.is_fraud})>"