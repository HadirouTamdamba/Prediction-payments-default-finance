# Prediction payments default finance

## Project Architecture

```plaintext
Prediction-payments-default-finance/
│
├── app/                           # FastAPI app code
│   ├── main.py                   # Main API entrypoint
│   ├── predict.py                # Prediction logic
│   └── schemas.py                # Pydantic request schema
│
├── model_training/               # Model artifacts
│   ├── model_loader.py           # Loader for model + scaler
│   ├── saved_model.pkl           # Trained ML model
│   └── scaler.pkl                # Preprocessing scaler
│
├── tests/                        # Unit tests
│   └── test_api.py               # FastAPI endpoint tests
│
├── docker/                       # Docker & deployment stack
│   ├── Dockerfile                # Main API Dockerfile
│   ├── Dockerfile.test           # Pytest container
│   ├── docker-compose.yml        # Orchestration file
│   ├── requirements.txt          # Python dependencies
│   └── .dockerignore             # Ignore rules for Docker
│
├── column_names.json             # Selected features list
├── EDA/                          # EDA results & pipeline
│   ├── correlation_matrix.png
│   └── class_distribution.png
│
├── .env                          # Optional environment config
└── README.md                     # Project documentation
