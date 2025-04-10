from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'filtered.csv'  
df = pd.read_csv(file_path, low_memory=False)

df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])
df = df.dropna()

X = df.drop(columns=['AMR'])
y = df['AMR']


X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, n_estimators=100, random_state=42,tree_method='gpu_hist')
}

for model_name, model in models.items():
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\nClassification Report for {model_name}:\n")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(report)

    accuracy = accuracy_score(y_test , y_pred)
    print(f"Accuracy :{accuracy:.4f}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\nFeature Importance for {model_name}:\n")
        importances = model.feature_importances_
        feature_names = X.columns


        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df['Category'] = feature_importance_df['Feature'].apply(lambda x: x.split('_')[0])
        category_importance_df = feature_importance_df.groupby('Category')['Importance'].sum().reset_index()
        category_importance_df = category_importance_df.sort_values(by='Importance', ascending=False)
        print(category_importance_df)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Category', data=category_importance_df)
        plt.title(f"Category Importance ({model_name})")
        plt.show()
