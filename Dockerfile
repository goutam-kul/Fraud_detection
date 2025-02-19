# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.docker.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/
COPY src/db/scripts/init.sql /app/src/db/scripts/
COPY src/db/scripts/start.sh /app/

RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# More reliable health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Modified command to ensure correct path
CMD [ "/app/start.sh" ]