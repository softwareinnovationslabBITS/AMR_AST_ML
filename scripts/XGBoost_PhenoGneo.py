from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = 'filtered_genetic.csv'
df = pd.read_csv(file_path, low_memory=False)

# Preprocessing
df = df.drop(columns=['Isolate Id', 'Study', 'State', , 'Family'])
df = df.dropna()
X = pd.get_dummies(df.drop(columns=['AMR']))
y = LabelEncoder().fit_transform(df['AMR'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Binarize the output
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

# Train XGBoost with OneVsRest strategy
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=6, n_estimators=100, random_state=42)
ovr_model = OneVsRestClassifier(xgb_model)
ovr_model.fit(X_train, y_train)

# Get prediction probabilities
y_proba = ovr_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(lb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and AUC
all_fpr = sorted(set([fp for class_fpr in fpr.values() for fp in class_fpr]))
mean_tpr = np.zeros_like(all_fpr, dtype=float)

for i in range(len(lb.classes_)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(lb.classes_)
macro_auc = auc(all_fpr, mean_tpr)

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(len(lb.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for XGBoost')
plt.legend(loc="lower right")
plt.show()

# Classification report
y_pred = ovr_model.predict(X_test)
print("\nClassification Report for XGBoost:\n")
print(classification_report(y_test, y_pred))

# Print macro-average AUC
print(f"\nMacro-average AUC for XGBoost: {macro_auc:.2f}")
