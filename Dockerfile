FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Create logs directory with full permissions
RUN mkdir -p /app/logs && chmod -R 777 /app/logs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose app port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]