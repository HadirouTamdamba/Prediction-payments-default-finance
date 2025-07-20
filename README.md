# 💳 Bank Default Payment Prediction API

A complete Machine Learning and MLOps-ready project to predict **credit card payment default risk** using a clean and containerized FastAPI REST API.
This solution includes **model training, API deployment, CI-ready test automation**, and is structured for production or scalable cloud deployment.

---

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
```plaintext


---

## 🎯 Project Objective

The goal is to **predict whether a customer will default on their credit card payment next month** based on demographic and payment history features.
Develop a fully automated **end-to-end Machine Learning pipeline** to predict the likelihood of credit card default. The project includes:
- Data cleaning, feature engineering, model training and selection
- Deployment via **FastAPI** on **AWS Lambda**, behind a **Route 53** custom domain
- CI/CD using Docker and GitHub Actions

The project uses the [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI.

---

## 🧠 Key Responsibilities & Achievements

| Responsibility | Details |
|----------------------------------------|-------------------------------------------------------------------------|
| 🧼 Data Cleaning & Feature Selection | Selected 10 most predictive features based on domain knowledge + EDA |
| ⚖️ Class Imbalance Handling | Used SMOTE to oversample minority class (defaults = 22%) |
| 🧪 Model Benchmarking | Trained and tuned 3 models: Logistic Regression, Random Forest, XGBoost |
| 📈 Model Selection | Selected **XGBoost** for best F1-score (0.487) and balanced performance |
| 🛠️ Model Export & Serving | Exported model and scaler with `pickle` and integrated in API |
| 🚀 Deployment to AWS | Deployed with Docker + FastAPI using **AWS Lambda** + **Route 53** |
| 🔬 Testing & Monitoring | Automated unit tests for API health and model response |

---

## 📊 Model Performance Summary

After applying preprocessing, scaling, and SMOTE balancing, the following results were obtained:

| Model | AUC | F1-score | Recall |
|--------------------|--------|----------|--------|
| Logistic Regression | 0.659 | 0.461 | 0.623 |
| Random Forest | 0.668 | 0.484 | 0.459 |
| **XGBoost** ✅ | **0.667** | **0.487** | **0.430** |

> 🔍 **Conclusion**: XGBoost model selected as best performer and exported for production.

---

## 🔧 Folder Structure
