FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p data/raw_html data/raw_txt data/chroma_db data/evaluation

# Expose port
EXPOSE 7860

# Hugging Face Spaces uses port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
