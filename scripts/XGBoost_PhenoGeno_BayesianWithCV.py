import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

file_path = 'filtered_genetic.csv'
df = pd.read_csv(file_path, low_memory=False)

df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])
df = df.dropna()
X = df.drop(columns=['AMR'])
y = df['AMR']

X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    xgb_model = xgb.XGBClassifier(**param)
    scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

xgb_model = xgb.XGBClassifier(**best_params, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)