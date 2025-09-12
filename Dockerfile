# Use lightweight Python base image
FROM python:3.12-slim

# Prevent Python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . /app

# Security: create a non-root user
RUN useradd -m appuser
USER appuser

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
