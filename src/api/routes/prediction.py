from fastapi import APIRouter, HTTPException, Depends, status
from src.api.schemas import (
    TransactionRequest,
    TransactionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from src.core.model import model_manager
from src.core.preprocessing import preprocessor
from src.db.crud import PredictionCRUD
from datetime import datetime, timezone
import time

# Create router with prefix
router = APIRouter(prefix="/transactions", tags=["predictions"])


# Create prediction
@router.post(
    "",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    description="Submit a transaction for fraud detection"
)
async def create_prediction(
    transaction: TransactionRequest,
    crud: PredictionCRUD = Depends()
) -> TransactionResponse:
    """Create a new fraud prediction for a transaction."""
    start_time = time.time()
    try:
        # Convert transaction to model features (Preprocessing)
        features_dict = transaction.model_dump()
        features = preprocessor.preprocess_transaction(features_dict)

        # Get prediction
        probability = model_manager.predict(feature=features)
        is_fraud = model_manager.is_fraud(probability)

        # Processing time calculation
        processing_time = time.time() - start_time

        prediction = crud.create_prediction(
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            fraud_probability=probability,
            is_fraud=is_fraud,
            processing_time=processing_time
        )
        return TransactionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=probability,
            is_fraud=is_fraud,
            processing_time=processing_time,
            timestamp=prediction.created_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# Get prediction result
@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    description="Get prediction result for a specific transaction"
)
async def get_prediction(
    transaction_id: str,
    crud: PredictionCRUD = Depends()
) -> TransactionResponse:
    """Retrieve prediction result for a specific transaction."""
    prediction = crud.get_prediction(transaction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found"
        )
    
    return TransactionResponse(
        transaction_id=prediction.transaction_id,
        fraud_probability=prediction.fraud_probability,
        is_fraud=prediction.is_fraud,
        processing_time=prediction.processing_time,
        timestamp=prediction.created_at
    )


# Get list of predictions
@router.get(
    "",
    response_model=list[TransactionResponse],
    description="List prediction results with pagination"
)
async def list_predictions(
    skip: int = 0,
    limit: int = 10,
    crud: PredictionCRUD = Depends()
) -> list[TransactionResponse]:
    """List prediction results with pagination."""
    predictions = crud.list_predictions(skip=skip, limit=limit)
    return [
        TransactionResponse(
            transaction_id=p.transaction_id,
            fraud_probability=p.fraud_probability,
            is_fraud=p.is_fraud,
            processing_time=p.processing_time,
            timestamp=p.created_at
        ) for p in predictions
    ]


# Batch prediction
@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_201_CREATED,
    description="Submit multiple transactions for fraud detection"
)
async def create_batch_predictions(
    request: BatchPredictionRequest,
    crud: PredictionCRUD = Depends()
) -> BatchPredictionResponse:
    """Create fraud predictions for multiple transactions."""
    start_time = time.time()
    
    try:
        # 1. Preprocess all transactions
        transactions_dict = [tx.model_dump() for tx in request.transactions]
        features = preprocessor.preprocess_batch(transactions_dict)
        
        # 2. Get predictions - ensure they're unique per transaction
        probabilities = model_manager.batch_predict(features)
        
        # Add logging to verify probabilities
        print(f"Debug - Raw probabilities: {probabilities}")
        
        fraud_predictions = [model_manager.is_fraud(float(p)) for p in probabilities]
        
        # 3. Process each prediction and store in database
        results = []
        for idx, (tx, prob, is_fraud) in enumerate(zip(request.transactions, probabilities, fraud_predictions)):
            # Calculate individual processing time
            tx_start_time = time.time()
            
            # Store prediction in database
            prediction = crud.create_prediction(
                transaction_id=tx.transaction_id,
                amount=float(tx.amount),
                fraud_probability=float(prob),  # Ensure this is unique per transaction
                is_fraud=bool(is_fraud),
                processing_time=float((time.time() - tx_start_time) * 1000)  # Convert to ms
            )
            
            # Add to results
            results.append(
                TransactionResponse(
                    transaction_id=tx.transaction_id,
                    fraud_probability=float(prob),
                    is_fraud=bool(is_fraud),
                    processing_time=float((time.time() - tx_start_time) * 1000),
                    timestamp=prediction.created_at
                )
            )
        
        # 4. Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return BatchPredictionResponse(
            results=results,
            total_processing_time=total_processing_time,
            timestamp=datetime.now(timezone.utc)
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        # Add more detailed error logging
        print(f"Debug - Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
