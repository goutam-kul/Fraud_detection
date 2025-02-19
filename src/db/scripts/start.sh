#!/bin/bash

# Wait for database to be ready
echo "Waiting for database..."
while ! curl http://db:5432/ 2>&1 | grep '52'
do 
  sleep 1
done
echo "Database is ready!"

# Initialize database using existing engine configuration
python -m src.db.init_db

# Start the application 
uvicorn src.api.app:app --host 0.0.0.0 --port 8000