import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Load the dataset
file_path = 'datasets/cleaned_dataset.csv'  
df = pd.read_csv(file_path)

# Display basic info before filtering
print("Initial data shape:", df.shape)

# Define the top 10 purposes to keep
valid_purposes = [
    'debt_consolidation', 'credit_card', 'home_improvement', 'other',
    'major_purchase', 'medical', 'small_business', 'car',
    'vacation', 'house'
]


# Encode the 'purpose' column into numeric labels
label_encoder = LabelEncoder()
df['purpose'] = label_encoder.fit_transform(df['purpose'])



# Select relevant features and target
X = df[['loan_amnt', 'term', 'installment']]
y = df['purpose']

# Standardize the features for SVM (important for optimal performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Reduce training set to 50% of its original size
train_sample_ratio = 0.05  # Keep 50% of training data
test_sample_ratio = 0.05   # Keep 50% of test data

# Randomly select a subset of the training data
X_train_subset, y_train_subset = X_train[:int(len(X_train) * train_sample_ratio)], y_train[:int(len(y_train) * train_sample_ratio)]
X_test_subset, y_test_subset = X_test[:int(len(X_test) * test_sample_ratio)], y_test[:int(len(y_test) * test_sample_ratio)]


print("Train set shape:", X_train_subset.shape)
print("Test set shape:", X_test_subset.shape)

# Train an SVM classifier with default parameters
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_subset, y_train_subset)

# Evaluate on test set
accuracy = svm_model.score(X_test_subset, y_test_subset)
print(f"Test Accuracy: {accuracy:.4f}")

# Optionally, save the label encoder and scaler for future predictions
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
