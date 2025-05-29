FROM python:3.10-slim

WORKDIR /app

# Create cache directories with proper permissions
RUN mkdir -p /app/cache /app/nltk_data && \
    chmod 777 /app/cache /app/nltk_data

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV NLTK_DATA=/app/nltk_data

EXPOSE 7860

CMD ["python", "app.py"]
