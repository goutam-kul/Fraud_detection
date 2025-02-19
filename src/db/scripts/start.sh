#!/bin/bash                                                
 
# Wait for database to be ready
while ! pg_isready -h db -U user -d frauddb; do          
    echo "Waiting for database..."
    sleep 1
done
echo "Database is ready!"

# Initialize database using existing engine configuration
python -m src.db.init_db                                  

# Start the application 
uvicorn src.api.app:app --host 0.0.0.0 --port 8000      