import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import xgboost as xgb

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
INPUT_FILE = os.path.join(INPUT_FOLDER, "input_data.xlsx") 
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
XGB_MODEL_GENETIC_PATH = os.path.join(MODEL_FOLDER, "xgb_model_tuned_genetic.h5")
XGB_MODEL_PHENO_PATH = os.path.join(MODEL_FOLDER, "xgb_model_tuned_pheno.h5")

# Check if model files exist
if not os.path.exists(XGB_MODEL_GENETIC_PATH):
    print(f"Model file not found: {XGB_MODEL_GENETIC_PATH}")
    exit(1)

if not os.path.exists(XGB_MODEL_PHENO_PATH):
    print(f"Model file not found: {XGB_MODEL_PHENO_PATH}")
    exit(1)

# Check if input file exists
if not os.path.exists(INPUT_FILE):
    print(f"Input file not found: {INPUT_FILE}")
    exit(1)

# Load the data
print(f"Reading input Excel file from: {INPUT_FILE}")
data = pd.read_excel(INPUT_FILE)  
data = data.dropna()  # Drop missing values, if any

# Prepare features for the models
data_genetic = data.copy()
data_pheno = data.copy()

# Adjust these as per your model training setup
genetic_features = pd.get_dummies(data_genetic.drop(columns=['Isolate Id', 'Study', 'State', 'Family', 'AMR'], errors='ignore'))
pheno_features = pd.get_dummies(data_pheno.drop(columns=['Isolate Id', 'Study', 'State', 'Family', 'AMR'], errors='ignore'))

# Load XGB model for genetic data
print("Loading XGBoost (Genetic) model...")
xgb_model_genetic = load_model(XGB_MODEL_GENETIC_PATH)
xgb_preds_genetic = xgb_model_genetic.predict(genetic_features)

# Load XGB model for phenotypic data
print("Loading XGBoost (Phenotypic) model...")
xgb_model_pheno = load_model(XGB_MODEL_PHENO_PATH)
xgb_preds_pheno = xgb_model_pheno.predict(pheno_features)

# Combine results into a DataFrame
results = data[['Isolate Id']].copy() if 'Isolate Id' in data.columns else pd.DataFrame(index=data.index)
results['XGB_Prediction_Genetic'] = xgb_preds_genetic
results['XGB_Prediction_Pheno'] = xgb_preds_pheno

# Print results
print("\nPredictions:")
print(results.head())  # Display first few predictions

# Confusion matrix and classification report for both models
print("\nConfusion Matrix for Genetic XGBoost model:")
print(confusion_matrix(data['AMR'], xgb_preds_genetic))  # Replace 'AMR' with actual ground truth column name if different
print("\nClassification Report for Genetic XGBoost model:")
print(classification_report(data['AMR'], xgb_preds_genetic))  # Replace 'AMR' with actual ground truth column name if different

print("\nConfusion Matrix for Phenotypic XGBoost model:")
print(confusion_matrix(data['AMR'], xgb_preds_pheno))  # Replace 'AMR' with actual ground truth column name if different
print("\nClassification Report for Phenotypic XGBoost model:")
print(classification_report(data['AMR'], xgb_preds_pheno))  # Replace 'AMR' with actual ground truth column name if different

# Optionally save predictions to a file
output_file = os.path.join(BASE_DIR, "predictions.csv")
results.to_csv(output_file, index=False)
print(f"\nPredictions saved to: {output_file}")
