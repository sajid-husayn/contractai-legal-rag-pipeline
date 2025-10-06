FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing and Docling
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for uploaded documents
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]