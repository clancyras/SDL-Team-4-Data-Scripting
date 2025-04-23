import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================== MODEL ========================
class CreditScore(nn.Module):
    def __init__(self, input_size):
        super(CreditScore, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Run the model (progress the various layers)"""
        return self.net(x)
    
    def process_output(output, y_scaler):
        """Process the output accordingly (y_scaler needed)"""
        output = output.cpu().numpy().flatten()
        
        output = y_scaler.inverse_transform(output.reshape(-1, 1)).flatten()
        
        output = np.clip(output, 300, 850)
        
        return output, 26.37


# ======================== DATA ========================
def get_data_from_csv(csv_loc: str, feature_names: list[str], target_name: str):
    feature_vectors, target_values = [], []

    with open(csv_loc, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        f_idx = [header.index(f) for f in feature_names]
        t_idx = header.index(target_name)

        for row in reader:
            try:
                features = [float(row[i]) if row[i] else np.nan for i in f_idx]
                target = float(row[t_idx]) if row[t_idx] else np.nan
                feature_vectors.append(features)
                target_values.append(target)
            except ValueError:
                continue

    features = np.array(feature_vectors, dtype=np.float32)
    targets = np.array(target_values, dtype=np.float32)

    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
    return features[mask], targets[mask]


# ======================== TRAINING ========================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
    criterion = nn.HuberLoss(delta=10.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    no_improve = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model


# ======================== EVALUATION ========================
def evaluate(model, x_test, y_test, y_scaler):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        preds = model(x_test_tensor).cpu().numpy().flatten()

    print(preds)
    preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    
    preds = np.clip(preds, 300, 850)
    y_test = y_test.flatten()

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.4f}")
    return preds

import joblib
# ======================== MAIN ========================
if __name__ == "__main__":
    csv_path = "datasets/cleaned_dataset.csv"
    features = ["annual_inc", "emp_length", "dti"]
    target = "fico_range_low"

    x, y = get_data_from_csv(csv_path, features, target)

    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    
    joblib.dump(scaler, "models/x_scaler.pkl")
    joblib.dump(y_scaler, "models/y_scaler.pkl")
    
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)


    # Tensors
    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


    # Split training into train + val
    x_t, x_v, y_t, y_v = train_test_split(x_train_tensor, y_train_tensor, test_size=0.1, random_state=42)
    train_loader = DataLoader(TensorDataset(x_t, y_t), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_v, y_v), batch_size=32)

    # Init + Train Model
    model = DeepRegressor(input_size=len(features)).to(device)
    # print("Training...")
    # model = train_model(model, train_loader, val_loader, epochs=150)

    # # Save
    # torch.save(model.state_dict(), "models/deep_credit_model.pth")
    
    model.load_state_dict(torch.load("models/deep_credit_model.pth"))
    
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.tensor(np.array([50000, 5, 5], dtype=np.float32).reshape(1, -1), dtype=torch.float32).to("cuda")
        preds = model(x_test_tensor).cpu().numpy().flatten()
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        print(preds)

    
    
    # # Evaluate
    # print("Testing...")
    # evaluate(model, x_test_scaled, y_test, y_scaler)
