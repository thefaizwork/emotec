FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables with default value
ENV PORT=8000

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application
# Command to run the application using Uvicorn directly with shell form for env var expansion
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1