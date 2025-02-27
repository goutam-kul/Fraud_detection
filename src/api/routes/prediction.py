from fastapi import APIRouter, HTTPException, Depends, status
from src.api.schemas import (
    TransactionRequest,
    TransactionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)
from src.monitoring.metrics import (
    track_prediction,
    track_request,
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
    request_start_time = time.time()
    try:
        # Convert transaction to model features (Preprocessing)
        features_dict = transaction.model_dump() 
        print("Features extracted: ", features_dict)  # Debug
        features = preprocessor.preprocess_transaction(features_dict)

        # Get prediction
        predict_start = time.time()
        probability = model_manager.predict(feature=features)
        prediction_time = time.time() - predict_start

        is_fraud = model_manager.is_fraud(probability)

        # Store prediction
        prediction = crud.create_prediction(
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            fraud_probability=probability,
            is_fraud=bool(is_fraud),
            processing_time=time.time() - predict_start
        )

        response = TransactionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=probability,
            is_fraud=bool(is_fraud),
            processing_time=time.time() - predict_start,
            timestamp=prediction.created_at
        )
        # track prediction metrics including drift
        track_prediction(
            fraud_probability=probability,
            is_fraud=bool(is_fraud),
            features=features_dict['features'],  # V1-V28 features
            prediction_time=prediction_time,
            amount=transaction.amount
        )

        # Track successful report
        track_request(
            status_code=status.HTTP_201_CREATED,
            response_time = time.time() - request_start_time,
            endpoint='create_prediction'
        )

        return response
    
    except ValueError as e:
        track_request(
            status_code=status.HTTP_400_BAD_REQUEST,
            response_time=time.time() - request_start_time,
            endpoint='create_prediction'
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        track_request(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            response_time=time.time() - request_start_time,
            endpoint='create_prediction'
        )
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
    request_start_time = time.time()
    results = []
    try:
        # Convert transactions to model features 
        transaction_data = [tx.model_dump() for tx in request.transactions]
        features = preprocessor.preprocess_batch(transaction_data)

        predict_start = time.time()
        probabilities = model_manager.batch_predict(features=features)
        prediction_time = time.time() - predict_start

        for transaction, probability in zip(request.transactions, probabilities):
            is_fraud = model_manager.is_fraud(probability)
            crud.create_prediction(
                transaction_id=transaction.transaction_id,
                amount=transaction.amount,
                fraud_probability=float(probability),
                is_fraud=bool(is_fraud),
                processing_time=prediction_time
            )
            # Track metrics for each prediction
            track_prediction(
                fraud_probability=float(probability),
                is_fraud=bool(is_fraud),
                features=transaction.features,
                prediction_time=prediction_time,
                amount=transaction.amount
            )
            # Add to results
            results.append(TransactionResponse(
                transaction_id=transaction.transaction_id,
                fraud_probability=float(probability),
                is_fraud=bool(is_fraud),
                processing_time=prediction_time,
                timestamp=datetime.now(timezone.utc)
            ))
        total_time = time.time() - request_start_time

        # Track successful report
        track_request(
            status_code=status.HTTP_201_CREATED,
            response_time = total_time,
            endpoint='create_batch_predictions'
        )

        return BatchPredictionResponse(
            results=results,
            total_processing_time=total_time,
            timestamp=datetime.now(timezone.utc)
        )
    except ValueError as e:
        track_request(
            status_code=status.HTTP_400_BAD_REQUEST,
            response_time=time.time() - request_start_time,
            endpoint='create_batch_predictions'
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        track_request(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            response_time=time.time() - request_start_time,
            endpoint='create_batch_predictions'
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
