import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load dataset
df= pd.read_csv("data/UCI_Credit_Card.csv")

# Rename target column
df.rename(columns={'default.payment.next.month': 'DEFAULT'}, inplace=True)

# Checking missing values
print("Missing Values:", df.isnull().sum().sum())

# Split features and target
X = df.drop(columns=['DEFAULT'])
y = df['DEFAULT']

# Standardization of numerical variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split in train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Handling of class imbalance (oversampling of the minority class)
train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='DEFAULT')], axis=1)
default_majority = train_data[train_data.DEFAULT == 0]
default_minority = train_data[train_data.DEFAULT == 1]
default_minority_upsampled = resample(default_minority, replace=True, n_samples=len(default_majority), random_state=42)
train_data_balanced = pd.concat([default_majority, default_minority_upsampled])

# Split once again
X_train_balanced = train_data_balanced.drop(columns=['DEFAULT'])
y_train_balanced = train_data_balanced['DEFAULT']

# Conversion in numpy array
X_train_balanced = X_train_balanced.to_numpy()
y_train_balanced = y_train_balanced.to_numpy()

# Size display
print(f"Train set size: {X_train_balanced.shape}, Test set size: {X_test.shape}")
