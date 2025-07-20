# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a fake CSV file for testing (since we don't have real data file)
RUN echo "merchant_id,city,mcc_code,mcc_description,message" > fake_merchant_dataset.csv && \
    echo "1,Santos,5814,Fast Food,preciso de parceiros de frete" >> fake_merchant_dataset.csv && \
    echo "2,Sorocaba,7299,Personal Services,faço marketing digital" >> fake_merchant_dataset.csv

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "merchant_social_agents:app", "--host", "0.0.0.0", "--port", "8000"]