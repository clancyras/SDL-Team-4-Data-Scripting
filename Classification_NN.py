import torch
import os
import csv
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


action_dict = {
    "other": 0,
    "debt_consolidation": 1,
    "credit_card": 2,
    # "home_improvement": 3,
    "car": 3,
    # "medical": 5,
    # "small_business": 6,
    # "car": 7,
    # "vacation": 8,
    # "house": 9,
}

class ClassifierNN(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_classes=4):
        super(ClassifierNN, self).__init__()\
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  
        return x


def get_data_from_csv(csv_loc: str, feature_names: list[str]):

    feature_vectors = []
    actions = []

    # Open and read a CSV file
    with open(csv_loc, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file) 
        header = next(csv_reader)  # Read the header row
        feature_name_indexes = [header.index(f_name) for f_name in feature_names]
        action_index = header.index('purpose')
        for row in csv_reader:  # Read remaining rows
            feature_vector = []
            for feature_name_index in feature_name_indexes:
                feature_vector.append(row[feature_name_index])
            
            if ' ' in feature_vector:
                continue
            
            actions.append(action_dict.get(row[action_index], 0))
            feature_vectors.append(feature_vector)
            

    return feature_vectors, actions


def train(model, feature_vectors, actions):
    # Dataset
    x_train = torch.tensor(feature_vectors)
    y_train = torch.tensor(actions)

    # Convert to DataLoader
    batch_size = 8
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class_counts = Counter(actions)
    total_samples = sum(class_counts.values())

    # Safely define weights for every class (handling missing classes)
    num_classes = len(action_dict.items())
    class_weights = torch.zeros(num_classes, dtype=torch.float32)  # Initialize all zeros

    for class_index in range(num_classes):
        if class_index in class_counts:
            class_weights[class_index] = total_samples / (class_counts[class_index] + 1e-6)  # Avoid 0 division
        else:
            class_weights[class_index] = 1.0

    # Check final computed weights - should NOT have huge imbalances!
    print("Class Weights:", class_weights)

    # Apply in training
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Loss and Optimizer
    # class_weights = torch.tensor([0.25, 4.0, 14.0, 14.0, 16.0, 4.0, 14.0, 14.0, 16.0], dtype=torch.float32)  # Adjust weights if need be

    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


from sklearn.metrics import accuracy_score, classification_report

def test_classification_model(model, feature_vectors, true_labels):
    # Convert inputs to PyTorch tensors
    model.eval()  # Set model to evaluation mode
    x_test = torch.tensor(feature_vectors, dtype=torch.float32)  # Features
    y_test = torch.tensor(true_labels, dtype=torch.long)  # Correct labels

    with torch.no_grad():  # Disable gradient computation for efficiency
        outputs = model(x_test)  # Get raw model outputs (logits)
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # Get predicted class labels
    
    # Convert to NumPy for easier metric calculation
    predictions_np = predictions.numpy()
    y_test_np = y_test.numpy()

    # Compute accuracy
    accuracy = accuracy_score(y_test_np, predictions_np)

    # Generate classification report
    report = classification_report(y_test_np, predictions_np, target_names=list(action_dict.keys()))

    # Print results
    print(f"Model Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)

    # Return as a dictionary
    return {"accuracy": accuracy, "classification_report": report}


import random
from collections import Counter
def balance_classes(feature_vectors, actions, target_size=None):
    # Convert to NumPy for easier handling
    feature_vectors = np.array(feature_vectors)
    actions = np.array(actions)
    
    # Count occurrences of each class
    class_counts = Counter(actions)
    print("Original class distribution:", class_counts)
    
    # Determine target_size (smallest class if not provided)
    if target_size is None:
        target_size = min(class_counts.values())  # Match smallest class
    
    balanced_features = []
    balanced_labels = []
    
    # Process each class separately
    for class_label, count in class_counts.items():
        indices = np.where(actions == class_label)[0]  # Get indices of class
        selected_indices = np.random.choice(indices, size=target_size, replace=False)  # Randomly select
        
        balanced_features.extend(feature_vectors[selected_indices])
        balanced_labels.extend(actions[selected_indices])
    
    # Shuffle dataset to prevent ordering biases
    combined = list(zip(balanced_features, balanced_labels))
    random.shuffle(combined)
    
    # Unzip back to separate feature & label lists
    balanced_features, balanced_labels = zip(*combined)
    
    print("New balanced class distribution:", Counter(balanced_labels))
    return np.array(balanced_features), np.array(balanced_labels)


if __name__ == '__main__':
    csv_loc = 'datasets/cleaned_dataset.csv'
    feature_vectors, actions = get_data_from_csv(csv_loc, ["loan_amnt", "term", "annual_inc", "fico_range_low", "emp_length", "int_rate"])
    feature_vectors, actions = balance_classes(feature_vectors, actions)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_vectors = scaler.fit_transform(feature_vectors)
    
    # Testing scaler to see if any improvements
    import pickle
    with open("models/loan_classification_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    feature_vectors_train = np.array(feature_vectors[:100000], np.float32)
    actions_train = np.array(actions[:100000], np.long)

    

    feature_vectors_test = np.array(feature_vectors[100000:110000], np.float32)
    actions_test = np.array(actions[100000:110000], np.float32)

    model = ClassifierNN(6)

    # Train
    # model = train(model, feature_vectors_train, actions_train)
    # torch.save(model.state_dict(), "models/V1.pth")

    # Test
    model.load_state_dict(torch.load("models/V1.pth"))

    result = test_classification_model(model, feature_vectors_test, actions_test)

    print(result)


