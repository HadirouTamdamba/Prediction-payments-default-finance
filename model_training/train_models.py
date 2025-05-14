import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
import pickle
import matplotlib.pyplot as plt
import numpy as np 

# Load dataset
df = pd.read_csv("data/UCI_Credit_Card.csv")
df.rename(columns={'default.payment.next.month': 'DEFAULT'}, inplace=True)

# Handle missing values by imputing with median
df.fillna(df.median(), inplace=True)
print(f"🔍 Missing values after median imputation: {df.isna().sum().sum()}")

# Split features and target
X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"✅ Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

# Save processed data
pd.DataFrame(X_train_balanced).to_csv("data/processed_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed_test.csv", index=False)
pd.Series(y_train_balanced, name='DEFAULT').to_csv("data/processed_y_train.csv", index=False)
pd.Series(y_test, name='DEFAULT').to_csv("data/processed_y_test.csv", index=False)


# Define models
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Expanded hyperparameter tuning grid
param_grid = {
    "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"n_estimators": [50, 100, 200, 500], "max_depth": [None, 10, 20, 30]},
    "XGBoost": {"n_estimators": [50, 100, 200, 500], "learning_rate": [0.01, 0.1, 0.2, 0.3]}
}

best_model = None
best_score = 0

# Train and evaluate models
for name, model in models.items():
    clf = RandomizedSearchCV(model, param_grid[name], scoring='roc_auc', cv=5, n_iter=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_balanced, y_train_balanced)
    
    y_pred = clf.best_estimator_.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"{name}: AUC={auc:.3f}, F1-score={f1:.3f}, Recall={recall:.3f}")
    
    if auc > best_score:
        best_score = auc
        best_model = clf.best_estimator_



# Feature importance pour RandomForest
model = RandomForestClassifier()
model.fit(X_train_balanced, y_train_balanced)

importances = model.feature_importances_
features = df.drop(columns=['DEFAULT']).columns

plt.figure(figsize=(10, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.show()


# Save best model
with open("model_training/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("🏆 Best model saved successfully!")
