# docker/Dockerfile

# Base image: secure & minimal Python Alpine image
FROM python:3.13.4-alpine

# Set working directory inside the container
WORKDIR /app

# Install OS-level dependencies required for Python packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    build-base \
    libxml2-dev \
    libxslt-dev \
    linux-headers \
    postgresql-dev

# Copy required application files into the container
COPY ../app ./app
COPY ../model_training ./model_training
COPY ../column_names.json ./column_names.json
COPY ../requirements.txt .

# Install Python packages (no cache for smaller image & security)
RUN pip install --no-cache-dir -r requirements.txt

# Clean up unnecessary build tools for smaller, more secure image
RUN apk del build-base gcc musl-dev linux-headers

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



####### Launch the entire stack automatically (This : Build API, Automatically run Pytest tests, 
### Start API ready for testing with curl or Postman) :: docker-compose up --build 
#######

# Testing the API with curl
#curl -X POST http://localhost:8000/predict \
#  -H "Content-Type: application/json" \
#  -d '{
#    "LIMIT_BAL": 20000,
#    "SEX": 2,
#    "EDUCATION": 2,
#    "MARRIAGE": 1,
#    "AGE": 24,
#    "PAY_0": 2,
#    "PAY_2": 2,
#    "BILL_AMT1": 3913,
#    "PAY_AMT1": 0,
#    "PAY_AMT2": 0
#}'




## Launching : docker build -t bank-default-api -f docker/Dockerfile
 
