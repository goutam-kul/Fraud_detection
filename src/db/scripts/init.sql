-- Only create database if it doesn't exist
SELECT 'CREATE DATABASE frauddb'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'frauddb');\gexec

\c frauddb;

-- Create predictions table if it doesn't exist
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    fraud_probability DECIMAL(5,4) NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    processing_time DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);