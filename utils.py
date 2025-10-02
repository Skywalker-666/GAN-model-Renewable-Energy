import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def make_time_features(df, date_col="DATE"):
    # ensure datetime
    t = pd.to_datetime(df[date_col])
    df = df.copy()
    df["hour"] = t.dt.hour + t.dt.minute/60.0
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    # cyclic encodings
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    df["mon_sin"]  = np.sin(2*np.pi*(df["month"]-1)/12.0)
    df["mon_cos"]  = np.cos(2*np.pi*(df["month"]-1)/12.0)
    return df

def build_sequences(df, seq_len=48, use_covariates=True):
    # Targets: USEP (price), LOAD
    y = df[["USEP","LOAD"]].values.astype(np.float32)
    # Covariates: time features (+ PV, RP if available)
    cov_cols = ["hour_sin","hour_cos","dow_sin","dow_cos","mon_sin","mon_cos"]
    if use_covariates:
        for extra in ["PV","RP"]:
            if extra in df.columns:
                cov_cols.append(extra)
    c = df[cov_cols].values.astype(np.float32)

    # scale targets separately (to invert later)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # scale covariates (optional but helps)
    c_scaler = StandardScaler()
    c_scaled = c_scaler.fit_transform(c)

    # window into sequences
    X_c, Y = [], []
    for i in range(len(df) - seq_len + 1):
        X_c.append(c_scaled[i:i+seq_len, :])
        Y.append(y_scaled[i:i+seq_len, :])
    X_c = np.stack(X_c)  # [N, T, F_c]
    Y = np.stack(Y)      # [N, T, 2]

    return X_c, Y, y_scaler, c_scaler, cov_cols

def train_val_split(X_c, Y, val_ratio=0.1):
    n = len(X_c)
    n_val = int(n*val_ratio)
    return (X_c[:-n_val], Y[:-n_val]), (X_c[-n_val:], Y[-n_val:])