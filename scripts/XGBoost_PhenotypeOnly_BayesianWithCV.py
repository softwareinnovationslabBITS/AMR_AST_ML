import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report

# Load dataset
file_path = 'filtered.csv'
df = pd.read_csv(file_path, low_memory=False)

# Drop irrelevant columns and handle missing values
df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])
df = df.dropna()

# Prepare features and target variable
X = df.drop(columns=['AMR'])
y = df['AMR']

# One-hot encode categorical features
X = pd.get_dummies(X)

# Label encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the best hyperparameters (using GPU support)
best_params = {
    'max_depth': 9,
    'n_estimators': 384,
    'learning_rate': 0.14245663095056388,
    'subsample': 0.8144897128399538,
    'colsample_bytree': 0.9104008577987068,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'tree_method': 'gpu_hist',  # Use GPU for training
    'predictor': 'gpu_predictor'  # Use GPU for prediction
}

# Create the XGBoost model using GPU
xgb_model = xgb.XGBClassifier(**best_params)

# Perform cross-validation with GPU
scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation accuracy
print(f"Cross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Train the model using GPU
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Print the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)