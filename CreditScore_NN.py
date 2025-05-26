import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class MultiLinearRegressor(nn.Module):
    def __init__(self, input_size):
        super(MultiLinearRegressor, self).__init__()
        """A simple multilinearregressor for credit score prediction"""
        
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  
        return x


def get_data_from_csv(csv_loc: str, feature_names: list[str], target_name: str):
    feature_vectors = []
    target_values = []

    with open(csv_loc, mode="r", newline="", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Extract column names
        feature_indexes = [header.index(f_name) for f_name in feature_names]
        target_index = header.index(target_name)

        for row in csv_reader:
            try:
                # Convert features & target to float
                feature_vector = [float(row[i]) if row[i] != "" else np.nan for i in feature_indexes]
                target_value = float(row[target_index]) if row[target_index] != "" else np.nan
                
                feature_vectors.append(feature_vector)
                target_values.append(target_value)
            except ValueError as e:
                print(f"Skipping row due to parsing error: {row}, Error: {e}")

    # Convert to numpy 
    feature_vectors = np.array(feature_vectors, dtype=np.float32)
    target_values = np.array(target_values, dtype=np.float32)

    # Remove any null
    mask = ~np.isnan(feature_vectors).any(axis=1) & ~np.isnan(target_values) 
    filtered_features = feature_vectors[mask]
    filtered_targets = target_values[mask]
    
    return filtered_features, filtered_targets


def train_regressor(model, feature_vectors, target_values, epochs=100, batch_size=16):
    # Convert to tensors
    x_train = torch.tensor(feature_vectors, dtype=torch.float32)
    y_train = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)  # Reshaped to (N,1)

    # Create data loaders
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()  # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for specified epochs
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

    print("Training Complete!")
    return model

# Normalize the predictions to the credit range
def clip_predictions(predictions):
    return torch.clamp(predictions, min=300, max=850)


from sklearn.metrics import mean_absolute_error, mean_squared_error

def test_regressor(model, feature_vectors, true_values):
    """Test the model"""
    model.eval()
    x_test = torch.tensor(feature_vectors, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(x_test)
    
    # Clip predictions to [300, 850]
    predictions = clip_predictions(predictions).numpy().flatten()
    true_values = np.array(true_values)

    # Compute metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")

    return predictions




if __name__ == "__main__":
    # Load Data
    csv_loc = "datasets/cleaned_dataset.csv"  # CHANGE
    feature_names = ["annual_inc", "emp_length", "dti"]
    target_name = "fico_range_low"  # Replace with actual column name of credit score

    feature_vectors, target_values = get_data_from_csv(csv_loc, feature_names, target_name)

    # Split Dataset (80% Training, 20% Testing)
    x_train, x_test, y_train, y_test = train_test_split(feature_vectors, target_values, test_size=0.2, random_state=42)

    # Initialize Model
    model = MultiLinearRegressor(input_size=len(feature_names))

    # Train the model
    print("Training model...")
    model = train_regressor(model, x_train, y_train, epochs=100)

    # Save Model
    torch.save(model.state_dict(), "models/regressor.pth")

    # Test the Model
    print("Testing model...")
    test_predictions = test_regressor(model, x_test, y_test)