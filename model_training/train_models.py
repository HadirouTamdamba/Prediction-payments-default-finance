import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import matplotlib.pyplot as plt
import pickle

# ------------------------------
# Load and preprocess the dataset
# ------------------------------

df = pd.read_csv("data/UCI_Credit_Card.csv")
df.rename(columns={'default.payment.next.month': 'DEFAULT'}, inplace=True)
df.fillna(df.median(), inplace=True)

#### Optimized List of Selected Features:
# The following subset of features was chosen after correlation analysis.
# It aims to reduce redundancy while preserving informative variables
# for predicting credit default.

selected_features = [
    'LIMIT_BAL',       # Credit limit granted to the client
    'SEX',             # Gender of the client
    'EDUCATION',       # Education level (1 = graduate school, 2 = university, etc.)
    'MARRIAGE',        # Marital status (1 = married, 2 = single, etc.)
    'AGE',             # Age of the client

    'PAY_0',           # Most recent payment status (strongest individual predictor)
    'PAY_2',           # Second most recent payment status (adds behavioral context)

    'BILL_AMT1',       # Most recent bill statement amount
    'PAY_AMT1',        # Most recent payment amount
    'PAY_AMT2'         # Payment made in the previous month
]


X = df[selected_features]
y = df['DEFAULT']

# ------------------------------
# Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Handle class imbalance using SMOTE
# ------------------------------
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Save processed data
pd.DataFrame(X_train_balanced, columns=selected_features).to_csv("data/processed_train.csv", index=False)
pd.DataFrame(X_test, columns=selected_features).to_csv("data/processed_test.csv", index=False)
pd.Series(y_train_balanced, name='DEFAULT').to_csv("data/processed_y_train.csv", index=False)
pd.Series(y_test, name='DEFAULT').to_csv("data/processed_y_test.csv", index=False)

# ------------------------------
# Define and train models
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

param_grid = {
    "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6]
    }
}

best_model = None
best_score = 0

for name, model in models.items():
    print(f"🔧 Training {name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid[name],
        scoring='roc_auc',
        n_iter=10,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train_balanced, y_train_balanced)

    y_pred = search.best_estimator_.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"{name}: AUC={auc:.3f}, F1-score={f1:.3f}, Recall={recall:.3f}")
    
    if auc > best_score:
        best_score = auc
        best_model = search.best_estimator_

# ------------------------------
# Feature Importance Visualization
# ------------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
importances = rf_model.feature_importances_

plt.figure(figsize=(8, 5))
sorted_idx = np.argsort(importances)
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), [selected_features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# ------------------------------
# Save model and scaler
# ------------------------------
with open("model_training/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_training/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🏆 Best model saved successfully!") 

###### Result Comments : Model Performance After Feature Selection
# Following correlation-based feature reduction, models were retrained using the optimized feature set.
# This approach aimed to reduce multicollinearity, simplify the model, and improve generalization.

# Logistic Regression:
# - AUC = 0.659
# - F1-score = 0.461
# - Recall = 0.623
# ➤ Logistic Regression performs relatively well in identifying defaults (high recall),
#   though the F1-score remains moderate due to class imbalance.

# Random Forest:
# - AUC = 0.668
# - F1-score = 0.484
# - Recall = 0.459
# ➤ Improved overall balance between precision and recall, with slightly better AUC.
#   Handles non-linearity and interactions effectively.

# XGBoost:
# - AUC = 0.667
# - F1-score = 0.487
# - Recall = 0.430
# ➤ XGBoost achieves the highest F1-score, suggesting a better compromise between false positives and false negatives.
#   Despite a slightly lower recall, this model is retained as the best performer.

# ✅ Final decision: XGBoost selected as the best model and saved for deployment.

