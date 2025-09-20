# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app folder
COPY app/ ./app

# Set the PORT env variable (Cloud Run uses this)
ENV PORT=8080
ENV APP_ENV=production

# Expose port for Cloud Run
EXPOSE 8080

# Run Uvicorn server (0.0.0.0 ensures accessible externally)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
