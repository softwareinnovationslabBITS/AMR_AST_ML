from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import re

# Load the dataset
file_path = 'filtered_genetic.csv'
df = pd.read_csv(file_path, low_memory=False)
# Limit the dataset for testing purposes
df = df.sample(frac=0.5, random_state=42)  # Use 50% of the dataset

# Drop irrelevant columns and rows with NA values
df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])

# Separate the features (X) and target (y)
X = df.drop(columns=['AMR'])
y = df['AMR']

# Get details of the dataset before applying SMOTENC
print(f"Dataset shape before SMOTENC: {df.shape}")
print(f"Class distribution before SMOTENC:\n{y.value_counts()}\n")

# Identify categorical feature columns (all columns are categorical)
categorical_feature_indices = list(range(X.shape[1]))

# One-hot encode categorical features
X = pd.get_dummies(X)

# Label encode the target variable (AMR)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Get class distribution before SMOTENC on training set
print(f"Class distribution in training set before SMOTENC:\n{np.bincount(y_train)}\n")

# Apply SMOTENC to balance the dataset
smote_nc = SMOTENC(categorical_features=categorical_feature_indices, random_state=42)
X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)

# Clean feature names after SMOTENC to remove special characters
X_train_resampled.columns = [re.sub(r'[\[\]<]', '', col) for col in X_train_resampled.columns]
X_test.columns = [re.sub(r'[\[\]<]', '', col) for col in X_test.columns]

# Get details after applying SMOTENC
print(f"Training set shape after SMOTENC: {X_train_resampled.shape}")
print(f"Class distribution in training set after SMOTENC:\n{np.bincount(y_train_resampled)}\n")

# Define the XGBoost model with GPU support
xgboost_model = xgb.XGBClassifier(
    eval_metric='mlogloss',
    max_depth=6,
    n_estimators=100,
    random_state=42,
    tree_method='gpu_hist',  # Use GPU for training
    predictor='gpu_predictor'  # Use GPU for prediction
)


# Train the model
xgboost_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = xgboost_model.predict(X_test)

# Print the classification report
print("\nClassification Report for XGBoost with SMOTENC:\n")
print(classification_report(y_test, y_pred))

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")