import warnings
import os

# Suppress ALL warnings (including FutureWarning, UserWarning, CUDA warnings, etc.)
warnings.filterwarnings("ignore")

# Disable XGBoost debug logs
os.environ["XGBOOST_VERBOSE"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

# Clean PyTorch warning spam
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# ============================================================
# 0) Imports
# ============================================================
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Utility: CSV loader with encoding fallback
# ============================================================
def load_csv_safely(filename, encodings=None):
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
            raise FileNotFoundError(f"File not found in: {os.getcwd()}")

    raise UnicodeDecodeError("Unable to read file with provided encodings.")


# ============================================================
# 2) Data cleaning utilities
# ============================================================
def convert_comma_numbers(df):
    df = df.replace(",", ".", regex=True)
    return df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

def process_timestamps(df):
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    print(f"Detected date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_str"] = df[date_col].dt.strftime("%Y-%m-%d %H:%M")

    duplicate_counts = df.groupby("date_str").cumcount()
    df["date_ordered"] = [
        ts if n == 0 else f"{ts}-{n}"
        for ts, n in zip(df["date_str"], duplicate_counts)
    ]

    df = df.drop(columns=[date_col, "date_str"])
    df = df.rename(columns={"date_ordered": "date"})
    return df

def split_iron_versions(df):
    iron_col = next(
        (c for c in df.columns if "iron" in c.lower() and "concentrate" in c.lower()),
        None,
    )

    if iron_col:
        print(f"Iron concentrate column found: {iron_col}")
        return df.copy(), df.drop(columns=[iron_col])
    else:
        print("Iron concentrate column NOT found.")
        return df.copy(), df.copy()


# ============================================================
# 3) ---- Load + Clean + Build df_with_iron BEFORE modeling ----
# ============================================================
file_name = "MiningProcess_Flotation_Plant_Database.csv"

# 3.1 Load safely
df_raw = load_csv_safely(file_name)

# 3.2 Fix numbers
df_raw = convert_comma_numbers(df_raw)

# 3.3 Fix timestamps
df_raw = process_timestamps(df_raw)

# 3.4 Create iron vs non-iron versions
df_with_iron, df_without_iron = split_iron_versions(df_raw)

# 3.5 Final dataset used for modeling
df = df_with_iron.copy()
print("Final dataset shape:", df.shape)


# ============================================================
# 4) Clean numeric columns + sort
# ============================================================
df = df.replace(",", ".", regex=True)
df = df.apply(lambda col: pd.to_numeric(col, errors='ignore'))

df["date"] = df["date"].astype(str)
df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 5) Feature/Target split
# ============================================================
target_col = "% Silica Concentrate"
feature_cols = [c for c in df.columns if c not in ["date", target_col]]

X = df[feature_cols]
y = df[target_col]


# ============================================================
# 6) Train/Validation Split (time-based)
# ============================================================
split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]


# ============================================================
# 7) Scaling
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)


# ============================================================
# 8) MODEL 1: XGBOOST
# ============================================================
print("\n=== TRAINING XGBOOST ===")

start = time.time()

try:
    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
except:
    model_xgb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )

model_xgb.fit(
    X_train_s, y_train,
    eval_set=[(X_train_s, y_train), (X_val_s, y_val)],
    eval_metric="rmse",
    verbose=False,
)

xgb_train_time = time.time() - start

y_pred_xgb = model_xgb.predict(X_val_s)
xgb_r2 = r2_score(y_val, y_pred_xgb)
xgb_mae = mean_absolute_error(y_val, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

xgb_train_loss = model_xgb.evals_result()['validation_0']['rmse']
xgb_val_loss = model_xgb.evals_result()['validation_1']['rmse']


# ============================================================
# 9) MODEL 2: RANDOM FOREST
# ============================================================
print("\n=== TRAINING RANDOM FOREST ===")

start = time.time()

model_rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train_s, y_train)
rf_train_time = time.time() - start

y_pred_rf = model_rf.predict(X_val_s)
rf_r2 = r2_score(y_val, y_pred_rf)
rf_mae = mean_absolute_error(y_val, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))


# ============================================================
# 10) MODEL 3: LSTM
# ============================================================
print("\n=== TRAINING LSTM ===")

SEQ_LEN = 10

class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_len]
        y_t = self.y[idx+self.seq_len]
        return (
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor([y_t], dtype=torch.float32)
        )

X_train_np = X_train_s
X_val_np = X_val_s
y_train_np = y_train.values
y_val_np = y_val.values

train_ds = SeqDataset(X_train_np, y_train_np, SEQ_LEN)
val_ds = SeqDataset(X_val_np, y_val_np, SEQ_LEN)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

class LSTMReg(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = "cuda" if torch.cuda.is_available() else "cpu"
model_lstm = LSTMReg(n_features=X_train_s.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)

EPOCHS = 5
lstm_train_loss = []
lstm_val_loss = []

start = time.time()

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model_lstm.train()
    batch_losses = []
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    lstm_train_loss.append(np.mean(batch_losses))

    # ---- VALIDATION ----
    model_lstm.eval()
    v_losses = []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_lstm(xb)
            v_losses.append(criterion(pred, yb).item())
    lstm_val_loss.append(np.mean(v_losses))

    print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss={lstm_train_loss[-1]:.4f}  ValLoss={lstm_val_loss[-1]:.4f}")

lstm_train_time = time.time() - start
