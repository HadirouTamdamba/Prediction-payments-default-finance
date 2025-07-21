# 💳🏦 Bank Default Payment Prediction 

A complete Machine Learning and MLOps-ready project to predict **credit card payment default risk** using a clean and containerized FastAPI REST API.  
This solution includes **model training, API deployment, CI-ready test automation**, and is structured for production or scalable cloud deployment.
---

## 🔧 Project Architecture

```plaintext
Prediction-payments-default-finance/
│
├── app/                           # FastAPI app code
│   ├── main.py                    # Main API entrypoint
│   ├── predict.py                 # Prediction logic
│   └── schemas.py                 # Pydantic request schema
│
├── model_training/                # Model artifacts
│   ├── model_loader.py            # Loader for model + scaler
│   ├── saved_model.pkl            # Trained ML model
│   └── scaler.pkl                 # Preprocessing scaler
│
├── tests/                         # Unit tests
│   └── test_api.py                # FastAPI endpoint tests
│
├── docker/                        # Docker & deployment stack
│   ├── Dockerfile                 # Main API Dockerfile
│   ├── Dockerfile.test            # Pytest container
│   ├── docker-compose.yml         # Orchestration file
│   ├── requirements.txt           # Python dependencies
│   └── .dockerignore              # Ignore rules for Docker
|
├── data/                          # Dataset and preprocessed CSVs
│   ├── UCI_Credit_Card.csv
│   ├── processed_train.csv
│   ├── processed_test.csv
│   ├── processed_y_train.csv
│   └── processed_y_test.csv
│
├── column_names.json              # Selected features list
├── EDA/                           # EDA results & pipeline
│   ├── correlation_matrix.png
│   └── class_distribution.png
│
├── .env                           # Optional environment config
└── README.md                      # Project documentation
```

## 🎯 Project Objective

The goal is to **predict whether a customer will default on their credit card payment next month** based on demographic and payment history features.

Develop a fully automated **end-to-end Machine Learning pipeline** to predict the likelihood of credit card default. The project includes:
- Data cleaning, feature engineering, model training and selection
- Deployment via **FastAPI** on **AWS Lambda**, behind a **Route 53** custom domain
- CI/CD using Docker and GitHub Actions

The project uses the [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from UCI.

| Area                               | Description                                                                        |
| ---------------------------------- | ---------------------------------------------------------------------------------- |
| 📊 **EDA & Feature Selection**     | Used correlation matrix + domain rules to select top 10 predictors                 |
| ⚖️ **Class Imbalance**             | Applied **SMOTE** to rebalance 22% minority class (defaults)                       |
| 🧪 **Model Benchmarking**          | Trained **Logistic Regression**, **Random Forest**, and **XGBoost**                |
| 🎯 **Hyperparameter Optimization** | Applied **RandomizedSearchCV** with 5-fold CV and `roc_auc` scoring for each model |
| 🧠 **Model Selection**             | Chose **XGBoost** based on **F1-score** and **generalization performance**         |
| 🔍 **Feature Importance**          | Visualized via Random Forest to assist business interpretability                   |
| 📦 **Model Export**                | Saved best model + scaler using `pickle` for API integration                       |
| 🚀 **Cloud Deployment (Ongoing)**  | Dockerized app and deployed on **AWS Lambda**              |

---

## 📊 Model Performance + Optimization Summary
Each model was trained with hyperparameter tuning using RandomizedSearchCV (10 iterations, 5-fold CV):

| Model               | Search Space Summary                                   | AUC       | F1-score  | Recall    |
| ------------------- | ------------------------------------------------------ | --------- | --------- | --------- |
| Logistic Regression | `C` ∈ \[0.001, ..., 100]                               | 0.659     | 0.461     | 0.623     |
| Random Forest       | `n_estimators`, `max_depth`, `min_samples_split` tuned | 0.668     | 0.484     | 0.459     |
| **XGBoost** ✅       | `n_estimators`, `learning_rate`, `max_depth` tuned     | **0.667** | **0.487** | **0.430** |

> 🔍 **Conclusion**:
- XGBoost selected for its superior balance between AUC, F1-score, and robustness
- Trained on SMOTE-balanced data, then tested on untouched holdout set.

---

## 🐳 Deployment & CI/CD

- Dockerized app with **multi-stage builds**
- API and model containerized and orchestrated via **docker-compose**
- Tested using **pytest + curl** in containerized environments
- Deployed on **AWS Lambda via container image** with custom domain **(Ongoing)**
---

## 📍 Tech Stack

- Python 3.10, pandas, scikit-learn, imbalanced-learn, XGBoost
- FastAPI, Uvicorn
- Docker, Docker Compose
- AWS Lambda, AWS Route 53
- GitHub Actions (CI/CD)

---
## 🏁 Result

✅ Fully functional prediction pipeline  
✅ High-performance XGBoost model  
✅ Scalable & automated deployment  

---
## 👨‍💻 About the Author  
**Hadirou Tamdamba**  
_Machine Learning Engineer | Microsoft Certified Generative AI Engineer_  

🔗 **LinkedIn**: [Hadirou Tamdamba](https://www.linkedin.com/in/hadirou-tamdamba/)  
🔗 **GitHub**: [HadirouTamdamba](https://github.com/HadirouTamdamba)  
📧 **Email**: hadirou.tamdamba@outlook.fr  

---

📢 **Feel free to explore, contribute, or provide feedback!**  
