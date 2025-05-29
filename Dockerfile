FROM python:3.10-slim

WORKDIR /app

# Create cache directories with proper permissions
RUN mkdir -p /tmp/hf_cache /app/nltk_data &&     chmod 777 /tmp/hf_cache /app/nltk_data
# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for cache directories
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV NLTK_DATA=/app/nltk_data

EXPOSE 7860

CMD ["python", "app.py"]
