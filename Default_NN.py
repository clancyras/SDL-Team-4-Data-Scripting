import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Model for predicting the probability of default
class LoanDefaultPredictor(nn.Module):
    def __init__(self, input_size):
        super(LoanDefaultPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary classification output
        )

    def forward(self, x):
        return self.net(x)

    def process_output(output, y_scaler):
        """Return the processed output (probability) and the confidence. Y_scaler is not necessary"""
        probs = torch.sigmoid(output)
        return probs, 0.793




import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_model(model, feature_vectors, target_values, epochs=50, batch_size=512):
    # Convert data to PyTorch tensors
    x_train = torch.tensor(feature_vectors, dtype=torch.float32)
    y_train = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)

    # Calculate class balance
    class_counts = np.bincount(target_values.astype(int))
    ratio = class_counts[0] / class_counts[1] if class_counts[1] != 0 else 1.0
    pos_weight = torch.tensor([min(ratio, 20.0)])  # Cap to prevent instability

    # Dataloader setup
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    return model




from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import classification_report

def test_model(model, x_test, y_test):
    # Get predictions
    x_test = torch.tensor(x_test)
    predictions = model(x_test)
    predictions = (predictions >= 0.5).int()  

    with torch.no_grad():
        outputs = model(x_test)
        probs = torch.sigmoid(outputs)
        print(probs)
        predictions = (probs >= 0.5).int()  # Lower threshold → more positives predicted

    print(predictions)
    # Get true values
    true_values = y_test

    # Generate the classification report
    report = classification_report(true_values, predictions, target_names=["Fully Paid", "Charged Off"], labels=[0, 1])
    print(report)

import joblib

def get_data_from_csv(csv_loc: str, feature_names: list[str], target_name: str):
    feature_vectors = []
    target_values = []

    with open(csv_loc, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Extract column names
        feature_indexes = [header.index(f_name) for f_name in feature_names]
        target_index = header.index(target_name)

        for row in csv_reader:
            # Handle Missing Values
            if "" in [row[i] for i in feature_indexes] or row[target_index] == "":
                continue  # Skip rows with missing values
            
            feature_vector = [float(row[i]) for i in feature_indexes]
            loan_status = row[target_index]

            # Convert to binary
            target_value = 0 if loan_status == "Fully Paid" else 1  # Fully Paid = 0, Charged Off = 1
            # print(target_value)
            feature_vectors.append(feature_vector)
            target_values.append(target_value)

    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    target_values = np.array(target_values, dtype=np.float32)

    # Normalize
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)

    joblib.dump(scaler, "models/default_pred/x_scaler.pkl")

    return feature_vectors, target_values


def load_model(model_path, input_size):
    model = LoanDefaultPredictor(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_data(new_data, scaler):
    new_data = np.array(new_data, dtype=np.float32).reshape(1, -1)
    normalized_data = scaler.transform(new_data)  # Apply normalization
    return torch.tensor(normalized_data, dtype=torch.float32)  # Convert to PyTorch tensor


def predict_default_probability(model, new_data, scaler, threshold=0.5):
    processed_data = preprocess_data(new_data, scaler)

    with torch.no_grad():
        probability = model(processed_data).item()  # Now this should be between 0 and 1

    prediction = 1 if probability >= threshold else 0  # Thresholding at 0.5

    print(f"Loan Default Probability: {probability:.4f}")
    print(f"Predicted Loan Status: {'Charged Off (1)' if prediction == 1 else 'Fully Paid (0)'}")

    return prediction, probability


import random

def generate_random_inputs():
    """Random inputs for testings sake"""
    random_inputs = [
        random.uniform(1000, 50000),  # Loan Amount ($)
        random.choice([12, 24, 36, 48, 60]),  # Loan Term (Months)
        random.uniform(10000, 200000),  # Annual Income ($)
        random.uniform(300, 850),  # FICO Score
        random.randint(0, 40),  # Employment Length (Years)
        random.uniform(5.0, 30.0)  # Interest Rate (%)
    ]
    
    # convert to numpy
    new_data = np.array(random_inputs, dtype=np.float32).reshape(1, -1)
    
    return torch.tensor(new_data, dtype=torch.float32)

def test_random_loan(model, num_samples=5):
    """test a random loan"""
    
    for i in range(num_samples):
        
        # Generate random data points
        random_data = generate_random_inputs()

        # Predict
        with torch.no_grad():
            output = model(random_data)
            probability = torch.sigmoid(output).item()  # Convert logits to probability
        
        # Convert to binary prediction
        prediction = 1 if probability >= 0.5 else 0  
        
        print(f"  - Loan Default Probability: {probability:.4f}")



if __name__ == "__main__":
    # Load Data
    csv_loc = "datasets/cleaned_dataset.csv"  # CHANGE

    # Couple different tests of features
    # feature_names = ["loan_amnt", "annual_inc", "fico_range_low", "emp_length", "int_rate"] # loan_default_model_v2_500ep

    # feature_names = ["loan_amnt", "annual_inc", "fico_range_low", "int_rate"]  # loan_default_model_v2_64

    feature_names = ["annual_inc", "fico_range_low", "emp_length", "int_rate"]  # loan_default_model_v2_66

    target_name = "loan_status"

    feature_vectors, target_values = get_data_from_csv(csv_loc, feature_names, target_name)
    print(len(feature_vectors))
    # Split the data into train/test
    x_train, x_test, y_train, y_test = train_test_split(feature_vectors, target_values, test_size=0.2, stratify=target_values, random_state=42)

    # Define Model
    model = LoanDefaultPredictor(input_size=len(feature_names))

    # Train Model
    model = train_model(model, x_train, y_train, epochs=120)

    # Save Model
    torch.save(model.state_dict(), "models/loan_default_model_v2.pth")

    # Evaluate Model
    # If already saved
    # model.load_state_dict(torch.load("models/loan_default_model_v2.pth"))
    
    # If not saved
    test_model(model, x_test, y_test)

