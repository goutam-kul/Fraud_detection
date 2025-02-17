from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os 
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings(BaseSettings):
    """Application Settings.
    
    These settings can be overriden with environment variables
    """
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Fraud Detection System"

    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # Model settings
    MODEL_PATH: str = "models/model.joblib"
    SCALER_PATH: str = "models/amount_scaler.joblib"
    CLASS_WEIGHTS_PATH: str = "models/class_weights.joblib"
    FRAUD_THRESHOLD: float = 0.8   # Based on your optimal threshold

    # Performance settings
    BATCH_SIZE: int = 1000
    MAX_REQUEST_PER_MINUTE: int = 100

    # Monitoring settings
    ENABLE_METRICS: bool = True

    class Config: 
        case_sensitive = True
        env_file = ".env"

# Caching for this call 
@lru_cache()
def get_settings() -> Settings :
    """Get cached settings
    Returns:
        Settings: Application status
    """
    return Settings()

