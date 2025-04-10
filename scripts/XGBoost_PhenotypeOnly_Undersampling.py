from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import Counter
import xgboost as xgb

# Load your data
file_path = 'filtered.csv'
df = pd.read_csv(file_path)

# Dropping unnecessary columns and handling missing values
df = df.drop(columns=['Isolate Id', 'Study', 'State', 'Family'])
df = df.dropna()

# Prepare the data
X = df.drop('AMR', axis=1)  
y = df['AMR']              

# Encoding categorical features
X = pd.get_dummies(X, drop_first=True)  
X = X.astype('float32') 

# Encoding the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify the size of the smallest class
counter = Counter(y_train)
n_minority_samples = min(counter.values())

# Separate the indices for each class
intermediate_indices = pd.Series([i for i, label in enumerate(y_train) if label == 0])  # 0 is for Intermediate
resistant_indices = pd.Series([i for i, label in enumerate(y_train) if label == 1])     # 1 is for Resistant
susceptible_indices = pd.Series([i for i, label in enumerate(y_train) if label == 2])   # 2 is for Susceptible

# Randomly undersample the Resistant and Susceptible classes to match the size of the Intermediate class
undersampled_resistant_indices = resistant_indices.sample(n=n_minority_samples, random_state=42)
undersampled_susceptible_indices = susceptible_indices.sample(n=n_minority_samples, random_state=42)

# Combine the undersampled classes with the Intermediate class
undersampled_indices = pd.concat([undersampled_resistant_indices, undersampled_susceptible_indices, intermediate_indices])

# Ensure that indices are sorted
undersampled_indices = undersampled_indices.sort_values()

# Select the undersampled training data
X_train_undersampled = X_train.iloc[undersampled_indices]
y_train_undersampled = y_train[undersampled_indices]

# Convert training and testing data into DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_undersampled, label=y_train_undersampled)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost to use GPU
params = {
    'objective': 'multi:softmax',  
    'num_class': 3,               
    'max_depth': 6,
    'eta': 0.3,                    # learning rate
    'subsample': 0.8,
    'tree_method': 'gpu_hist',     
    'predictor': 'gpu_predictor',  
    'random_state': 42
}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Make predictions on the test set
y_pred = model.predict(dtest)
y_pred = y_pred.astype(int)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")