# utils.py
import numpy as np
import pandas as pd
import torch

# -----------------------------
# Repro + small helpers
# -----------------------------
def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@torch.no_grad()
def ema_update(target, source, decay=0.999):
    for p_t, p in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

# -----------------------------
# Scaling for heavy-tailed series
# -----------------------------
class RobustScaler1D:
    """MAD-based robust scaler (works well for energy price tails)."""
    def fit(self, x):
        x = np.asarray(x).astype(float)
        self.med = float(np.nanmedian(x))
        self.mad = float(np.nanmedian(np.abs(x - self.med)) + 1e-6)
        return self
    def transform(self, x):
        return (np.asarray(x).astype(float) - self.med) / (1.4826 * self.mad)
    def inverse_transform(self, x):
        return np.asarray(x).astype(float) * (1.4826 * self.mad) + self.med
    def to_dict(self):
        return {"med": self.med, "mad": self.mad}
    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.med, obj.mad = float(d["med"]), float(d["mad"])
        return obj

def log1p_signed(x):
    x = np.asarray(x).astype(float)
    return np.sign(x) * np.log1p(np.abs(x))

def expm1_signed(x):
    x = np.asarray(x).astype(float)
    return np.sign(x) * (np.expm1(np.abs(x)))

# -----------------------------
# Conditioning builders
# -----------------------------
def _sin_cos(series, period):
    x = 2.0 * np.pi * (np.asarray(series).astype(float) / period)
    return np.stack([np.sin(x), np.cos(x)], axis=0)  # [2, T]

def build_time_condition(index_like):
    """
    index_like: Pandas DatetimeIndex (length T).
    Returns cond_time: [6, T]  (hour sin/cos, weekday sin/cos, month sin/cos)
    """
    idx = pd.DatetimeIndex(index_like)
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    mon = idx.month
    return np.concatenate([_sin_cos(hour, 24), _sin_cos(dow, 7), _sin_cos(mon, 12)], axis=0)

def stack_condition_with_exogenous(cond_time, pv=None, rp=None):
    """
    cond_time: [6, T]
    pv, rp: arrays length T (optional)
    Returns [F, T]
    """
    exo = []
    if pv is not None:
        exo.append(np.asarray(pv, dtype=float)[None, :])
    if rp is not None:
        exo.append(np.asarray(rp, dtype=float)[None, :])
    if exo:
        exo = np.concatenate(exo, axis=0)
        return np.concatenate([cond_time, exo], axis=0)
    return cond_time

# -----------------------------
# Data handling (daily windows)
# -----------------------------
def read_energy_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]

    # Ensure DATE column is parsed and normalized to midnight
    if "DATE" not in df.columns:
        raise ValueError("CSV must contain a DATE column")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = df["DATE"].dt.normalize()   # ✅ key fix

    # Sort by DATE and PERIOD if present
    if "PERIOD" in df.columns:
        df = df.sort_values(["DATE", "PERIOD"])
    else:
        df = df.sort_values(["DATE"])
    return df

def make_daily_tensors(df, seq_len=48, use_rp=True):
    """
    Groups the dataframe by DATE and returns arrays per day.
    Returns:
      dates: [D] array of dates
      time_index_per_day: list of DatetimeIndex for each day (length T)
      y_usep: [D, T], y_load: [D, T]
      x_pv: [D, T] (or None), x_rp: [D, T] (or None)
    """
    out_dates, time_index = [], []
    ys_usep, ys_load, xs_pv, xs_rp = [], [], [], []
    g = df.groupby("DATE")
    for d, sub in g:
        if len(sub) < seq_len:  # skip incomplete days
            continue
        sub = sub.head(seq_len).copy()
        # Timestamp for each period (build if not present)
        if "PERIOD" in sub.columns:
            # assume each row is a half-hour period; create times within the day
            base = pd.Timestamp(d.normalize())
            times = [base + pd.Timedelta(minutes=30 * int(p - 1)) for p in sub["PERIOD"].values]
        else:
            # fallback: just create equally spaced half-hours
            base = pd.Timestamp(d.normalize())
            times = [base + pd.Timedelta(minutes=30 * i) for i in range(seq_len)]
        out_dates.append(pd.Timestamp(d))
        time_index.append(pd.DatetimeIndex(times))
        ys_usep.append(sub["USEP"].values[:seq_len].astype(float))
        ys_load.append(sub["LOAD"].values[:seq_len].astype(float))
        xs_pv.append(sub["PV"].values[:seq_len].astype(float) if "PV" in sub.columns else np.zeros(seq_len))
        if use_rp:
            xs_rp.append(sub["RP"].values[:seq_len].astype(float) if "RP" in sub.columns else np.zeros(seq_len))
    D = len(out_dates)
    y_usep = np.stack(ys_usep, axis=0) if D else np.empty((0, seq_len))
    y_load = np.stack(ys_load, axis=0) if D else np.empty((0, seq_len))
    x_pv   = np.stack(xs_pv,   axis=0) if D else np.empty((0, seq_len))
    x_rp   = np.stack(xs_rp,   axis=0) if (D and use_rp) else None
    return np.array(out_dates), time_index, y_usep, y_load, x_pv, x_rp

# -----------------------------
# Metrics (for validation)
# -----------------------------
def rmse(yhat, y):
    return float(np.sqrt(np.mean((np.asarray(yhat) - np.asarray(y)) ** 2)))

def energy_score(samples, y):
    s = np.asarray(samples)  # [S,T]
    y = np.asarray(y)        # [T]
    term1 = np.mean(np.sqrt(np.sum((s - y[None, :]) ** 2, axis=1)))
    diffs = s[:, None, :] - s[None, :, :]
    term2 = 0.5 * np.mean(np.sqrt(np.sum(diffs ** 2, axis=2)))
    return float(term1 - term2)
