from src.db.database import engine
from src.db.models import Base

def init_db():
    """Initialize database with required tables"""
    try:
        # Create all tables using our existing engine 
        Base.metadata.create_all(bind=engine)
        print("Successfully initialized database tables")
    except Exception as e:
        print(f"Error initializing database tables: {str(e)}")
        raise

if __name__ == "__main__":
    init_db()
