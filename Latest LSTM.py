"""
High-standard LSTM pipeline with:
 - Robust CSV loading
 - Timestamp correction
 - Iron/Non-Iron dataset split
 - Normalization
 - Sequence dataset
 - LSTM regression model
 - Early stopping
 - Evaluation + plots
"""

# ============================================================
# 1) Imports
# ============================================================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ============================================================
# 2) Utility: CSV loader with encoding fallback
# ============================================================
def load_csv_safely(filename, encodings=None):
    """Load a CSV with multiple encoding fallbacks."""
    if encodings is None:
        encodings = ["utf-8", "latin-1", "cp1252"]

    print(f"Attempting to load: {filename}")

    for enc in encodings:
        try:
            df = pd.read_csv(filename, encoding=enc)
            print(f"Loaded successfully using encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"Encoding '{enc}' failed – trying next…")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found in directory: {os.getcwd()}"
            )

    raise UnicodeDecodeError("Unable to read file with provided encodings.")


# ============================================================
# 3) Data cleaning utilities
# ============================================================
def convert_comma_numbers(df):
    """Convert comma decimals to dots and numeric values."""
    df = df.replace(",", ".", regex=True)
    return df.apply(lambda col: pd.to_numeric(col, errors="ignore"))


def process_timestamps(df):
    """Fix timestamps, generate ordered timestamp column."""
    # Find date column
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    print(f"Detected date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # String timestamp for grouping
    df["date_str"] = df[date_col].dt.strftime("%Y-%m-%d %H:%M")
    duplicate_counts = df.groupby("date_str").cumcount()

    df["date_ordered"] = [
        ts if n == 0 else f"{ts}-{n}"
        for ts, n in zip(df["date_str"], duplicate_counts)
    ]

    # Clean output
    df = df.drop(columns=[date_col, "date_str"])
    df = df.rename(columns={"date_ordered": "date"})

    return df


def split_iron_versions(df):
    """Return df_with_iron and df_without_iron."""
    iron_col = next(
        (c for c in df.columns if "iron" in c.lower() and "concentrate" in c.lower()),
        None,
    )

    if iron_col:
        print(f"Iron concentrate column found: {iron_col}")
        return df.copy(), df.drop(columns=[iron_col])
    else:
        print("Iron concentrate column not found.")
        return df.copy(), df.copy()


# ============================================================
# 4) Dataset class for LSTM sequences
# ============================================================
class SeqDataset(Dataset):
    """Time-window dataset for LSTM."""
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
        )


# ============================================================
# 5) LSTM model
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)


# ============================================================
# 6) Train function with early stopping
# ============================================================
def train_model(model, train_dl, val_dl, device, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    patience = 5
    patience_counter = 0

    train_hist, val_hist = [], []

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_losses = []

        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # ----- Validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_losses.append(criterion(pred, yb).item())

        avg_val = np.mean(val_losses)
        train_hist.append(avg_train)
        val_hist.append(avg_val)

        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # ----- Early stopping -----
        if avg_val < best_loss:
            best_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist


# ============================================================
# 7) Main pipeline
# ============================================================
def main():
    # ---------------- Load & clean data ----------------
    df = load_csv_safely("MiningProcess_Flotation_Plant_Database.csv")
    df = convert_comma_numbers(df)
    df = process_timestamps(df)
    df_with_iron, df_without_iron = split_iron_versions(df)

    # Select dataset
    df = df_with_iron.copy()

    # ---------------- Sorting & cleaning ----------------
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # ---------------- Feature split ----------------
    target_col = "% Silica Concentrate"
    features = [c for c in df.columns if c not in ["date", target_col]]

    X = df[features].values
    y = df[target_col].values.reshape(-1, 1)

    # ---------------- Scaling ----------------
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_X.fit_transform(X)
    ys = scaler_y.fit_transform(y)

    # ---------------- Train/val split ----------------
    split_idx = int(len(df) * 0.8)
    X_train, X_val = Xs[:split_idx], Xs[split_idx:]
    y_train, y_val = ys[:split_idx], ys[split_idx:]

    # ---------------- Sequence windows ----------------
    SEQ_LEN = 32
    train_dl = DataLoader(SeqDataset(X_train, y_train, SEQ_LEN), batch_size=64)
    val_dl = DataLoader(SeqDataset(X_val, y_val, SEQ_LEN), batch_size=64)

    # ---------------- Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = LSTMRegressor(n_features=X.shape[1]).to(device)

    # ---------------- Train ----------------
    model, train_hist, val_hist = train_model(model, train_dl, val_dl, device)

    # ---------------- Validation prediction ----------------
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in val_dl:
            preds.extend(model(Xb.to(device)).cpu().numpy())

    preds = scaler_y.inverse_transform(preds)
    y_true = scaler_y.inverse_transform(y_val[SEQ_LEN:])

    # ---------------- Metrics ----------------
    mae = mean_absolute_error(y_true, preds)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)

    print("\nModel Performance")
    print("----------------------")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # ---------------- Plots ----------------
    plt.figure(figsize=(14, 5))
    plt.plot(y_true, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("LSTM Prediction vs True (% Silica Concentrate)")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_hist, label="Train")
    plt.plot(val_hist, label="Val")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


# ============================================================
# 8) Run script
# ============================================================
if __name__ == "__main__":
    main()
