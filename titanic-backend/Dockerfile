# Use the official Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# build-essential for compiling some python packages
# libgomp1 for OpenMP (required by some ML libraries like torch/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model to speed up container startup 
# and avoid runtime download failures on Render
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose the default port (Render will override this with its own PORT)
EXPOSE 8000

# Use python main.py to allow the application to handle the PORT environment variable
# correctly as implemented in main.py
CMD ["python", "main.py"]
