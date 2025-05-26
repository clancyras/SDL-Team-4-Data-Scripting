# train_term_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

class MLPClassifier(nn.Module):
    """
    This classifier predicts the term of the loan (36 / 60 months)
    These are the terms available in the data sets, so we opted for
    a classifier to give better results.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.3,
        use_batchnorm: bool = True
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims # Control the dimensions throughout the model

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Use a net to collect and deploy all the layers
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # employ the net in the forward


# Function to trand the classifier defined above to predict loan term

def train_classifier(
    model: MLPClassifier,
    csv_path: str,
    feature_names: list[str],
    term_column: str = 'term',
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 5e-4,
    val_split: float = 0.2,
    weight_decay: float = 1e-5
) -> StandardScaler:
    # Read the csv and keep appropriate columns
    df = pd.read_csv(csv_path).dropna(subset=feature_names + [term_column])

    # Clean the column for the month
    if df[term_column].dtype == object:
        df[term_column] = df[term_column].str.extract(r'(\d+)').astype(int)
    df = df[df[term_column].isin([36, 60])].copy()

    # map to labels: 36 to 0, 60 to 1
    df['label'] = df[term_column].map({36: 0, 60: 1})

    # gather x and y for training
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy(dtype=np.int64)  # integer labels

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Build dataset and split
    tX = torch.from_numpy(X)
    tY = torch.from_numpy(y)
    full_ds = TensorDataset(tX, tY)
    val_size = int(len(full_ds) * val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(
            np.concatenate(all_labels), np.concatenate(all_preds)
        )

        # validation
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_labels.append(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(
            np.concatenate(val_labels), np.concatenate(val_preds)
        )

        print(
            f"[Term] Epoch {epoch:>3}/{epochs}  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )

    return scaler



# SAVE / LOAD UTILITIES

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


def infer(
    model: MLPClassifier,
    scaler: StandardScaler,
    feature_vectors: list[list[float]]
) -> list[int]:
    """
    Returns a list of predicted term-month values (36 or 60).
    """
    model.eval()
    X = scaler.transform(np.array(feature_vectors, dtype=np.float32))
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        classes = logits.argmax(dim=1).cpu().numpy()
    # map back 0→36, 1→60
    return [36 if c == 0 else 60 for c in classes]


# Runs initially when file is ran (Training and initial testing)

if __name__ == "__main__":
    FEATURES = ['annual_inc', 'fico_range_low', 'dti']
    CSV_PATH = "datasets/cleaned_dataset.csv"

    term_model = MLPClassifier(input_dim=len(FEATURES))
    term_scaler = train_classifier(
        term_model,
        csv_path=CSV_PATH,
        feature_names=FEATURES,
        term_column='term',
        epochs=40,
        batch_size=128,
        lr=5e-4
    )

    save_model_and_scaler(
        term_model,
        term_scaler,
        model_path="models/loan_term/loan_term_classifier.pth",
        scaler_path="models/loan_term/loan_term_scaler.pkl"
    )

    # Example inference
    samples = [[75000, 680, 15.2], [120000, 720, 8.5]]
    print("Predicted terms:", infer(term_model, term_scaler, samples))