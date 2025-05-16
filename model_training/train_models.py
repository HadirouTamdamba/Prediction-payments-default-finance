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

# Load dataset
df = pd.read_csv("data/UCI_Credit_Card.csv")

# Rename the target column for clarity
df.rename(columns={'default.payment.next.month': 'DEFAULT'}, inplace=True)

# Fill missing values with median values
df.fillna(df.median(), inplace=True)
print(f"🔍 Missing values after imputation: {df.isna().sum().sum()}")

# Separate features and target variable
X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

# Standardize features for model compatibility
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Handle class imbalance using SMOTE
# ------------------------------

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

# Save the processed data
pd.DataFrame(X_train_balanced).to_csv("data/processed_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed_test.csv", index=False)
pd.Series(y_train_balanced, name='DEFAULT').to_csv("data/processed_y_train.csv", index=False)
pd.Series(y_test, name='DEFAULT').to_csv("data/processed_y_test.csv", index=False)

# ------------------------------
# Define and train models
# ------------------------------

# Initialize candidate models
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Hyperparameter search space for each model
param_grid = {
    "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 6, 10]
    }
}

# Initialize variables to track the best model
best_model = None
best_score = 0

# Train and evaluate each model using RandomizedSearchCV
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
    
    # Evaluate model on test data
    y_pred = search.best_estimator_.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"{name}: AUC={auc:.3f}, F1-score={f1:.3f}, Recall={recall:.3f}")
    
    # Update best model if current one performs better
    if auc > best_score:
        best_score = auc
        best_model = search.best_estimator_

# ------------------------------
# Feature importance (Random Forest)
# ------------------------------

# Fit a Random Forest for feature importance visualization
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)

# Extract feature importances
importances = rf_model.feature_importances_
feature_names = df.drop(columns=['DEFAULT']).columns

# Plot the top 15 most important features
sorted_idx = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Top 15 Important Features (Random Forest)")
plt.tight_layout()
plt.show()

# ------------------------------
# Save the best performing model
# ------------------------------

with open("model_training/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("🏆 Best model saved successfully!")
