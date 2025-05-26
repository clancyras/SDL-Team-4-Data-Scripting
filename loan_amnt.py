import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pickle

# Regressor to predict the loan amount

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64, 32],
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# Training with ability to vary parameters for more rapid training/development

def train_model_with_metrics(
    model: nn.Module,
    csv_path: str,
    feature_names: list[str],
    target_name: str = 'loan_amnt',
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    val_split: float = 0.2,
    weight_decay: float = 1e-5
) -> StandardScaler:
    # Preprocess 
    df = pd.read_csv(csv_path).dropna(subset=feature_names + [target_name])
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target_name].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Build datasets and split 
    tX = torch.from_numpy(X)
    tY = torch.from_numpy(y).squeeze(1)
    full_ds = TensorDataset(tX, tY)
    val_size = int(len(full_ds) * val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Optimizer & loss
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Amount] Epoch {epoch:>3}/{epochs}  "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    # Final metrics
    def compute_metrics(ds, name):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                preds = model(xb).cpu().numpy().ravel()
                truth = yb.cpu().numpy().ravel()
                y_pred.append(preds)
                y_true.append(truth)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        print(f"\n{name} Metrics:")
        print(f"  MAE : {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"  RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")
        print(f"  R²  : {r2_score(y_true, y_pred):.4f}\n")

    print("Final performance on Training set:")
    compute_metrics(train_ds, "Train")
    print("Final performance on Validation set:")
    compute_metrics(val_ds,   "Validation")

    return scaler


# Save model and scaler using function

def save_model_and_scaler(
    model: nn.Module,
    scaler: StandardScaler,
    model_path: str,
    scaler_path: str
):
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved model to {model_path} and scaler to {scaler_path}")

# Quick inference function

def infer(
    model: nn.Module,
    scaler: StandardScaler,
    feature_vectors: list[list[float]]
) -> list[float]:
    model.eval()
    X = scaler.transform(np.array(feature_vectors, dtype=np.float32))
    with torch.no_grad():
        return model(torch.from_numpy(X)).tolist()


# Runs when the python file is ran (training then initial test)
# Infer the loan amount

if __name__ == "__main__":
    FEATURES = ['annual_inc', 'fico_range_low', 'dti', 'emp_length', 'mths_since_last_delinq']
    CSV_PATH = "datasets/cleaned_dataset.csv"

    amt_model = MLPRegressor(input_dim=len(FEATURES))
    amt_scaler = train_model_with_metrics(
        amt_model,
        csv_path=CSV_PATH,
        feature_names=FEATURES,
        target_name='loan_amnt'
    )

    save_model_and_scaler(
        amt_model,
        amt_scaler,
        model_path="models/loan_amnt/loan_amount_model.pth",
        scaler_path="models/loan_amnt/loan_amount_scaler.pkl"
    )

    # Example inference:
    samples = [[75000, 680, 15.2, 3, 5], [120000, 720, 8.5, 8, 17]]
    print("Predicted amounts:", infer(amt_model, amt_scaler, samples))

