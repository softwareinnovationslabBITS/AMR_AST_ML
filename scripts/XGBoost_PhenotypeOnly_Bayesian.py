import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'filtered.csv'
df = pd.read_csv(file_path, low_memory=False)

# Drop unnecessary columns and handle missing values
df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])
df = df.dropna()

# Separate features and target
X = df.drop(columns=['AMR'])
y = df['AMR']

# One-hot encode categorical features and label encode the target variable
X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the objective function for hyperparameter optimization
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'gpu_hist',  # Use GPU for training
        'predictor': 'gpu_predictor'  # Use GPU for prediction
    }

    # Create and train the model with suggested parameters
    xgb_model = xgb.XGBClassifier(**param)
    xgb_model.fit(X_train, y_train)
    score = xgb_model.score(X_test, y_test)  # Evaluate the model on the test set
    return score

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Retrieve the best parameters from the study
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Train the final model using the best hyperparameters and GPU support
xgb_model = xgb.XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Print the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Feature importance analysis
feature_importance = xgb_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df['AbsImportance'] = feature_importance_df['Importance'].abs()

# Extract and display the top 50 most important features
top_50_features_df = feature_importance_df.sort_values(by='AbsImportance', ascending=False).head(50)
print(top_50_features_df)

# Plot the top 50 feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='AbsImportance', y='Feature', data=top_50_features_df)
plt.title("Top 50 Feature Importance (XGBoost)")
plt.show()