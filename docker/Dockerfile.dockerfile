# docker/Dockerfile

# Minimal & secure base image (version patchée au 01/06/2025)
FROM python:3.11.12-alpine

# Set working directory
WORKDIR /app

# Install OS-level dependencies (with --no-cache to avoid vulnerabilities)
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    libxml2-dev \
    libxslt-dev \
    linux-headers \
    postgresql-dev

# 📁 Copy project files
COPY ../app ./app
COPY ../model_training ./model_training
COPY ../column_names.json ./column_names.json
COPY requirements.txt .

#  Install Python dependencies (clean cache for reduced image size & security)
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build tools (optional security hardening)
RUN apk del build-base gcc musl-dev linux-headers

# Expose FastAPI port
EXPOSE 8000

#  Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
