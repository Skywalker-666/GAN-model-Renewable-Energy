import numpy as np, pandas as pd
import torch

# ------------------ Scaling helpers ------------------
class RobustScaler1D:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.med = np.nanmedian(X)
        self.mad = np.nanmedian(np.abs(X - self.med)) + 1e-6
        return self
    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.med) / (1.4826 * self.mad)
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * (1.4826 * self.mad) + self.med
    def to_dict(self): return {"med": self.med, "mad": self.mad}
    @classmethod
    def from_dict(cls, d):
        o = cls(); o.med = d["med"]; o.mad = d["mad"]; return o

class RobustScalerVec:
    """Channel-wise robust scaler for 2D features."""
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.med = np.nanmedian(X, axis=0)
        self.mad = np.nanmedian(np.abs(X - self.med), axis=0) + 1e-6
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.med) / (1.4826 * self.mad)
    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * (1.4826 * self.mad) + self.med
    def to_dict(self): return {"med": self.med.tolist(), "mad": self.mad.tolist()}
    @classmethod
    def from_dict(cls, d):
        o = cls(); o.med = np.array(d["med"]); o.mad = np.array(d["mad"]); return o

# ------------------ CSV Reading ------------------
def read_energy_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    if "PERIOD" in df.columns:
        df = df.sort_values(["DATE", "PERIOD"])
    else:
        df = df.sort_values(["DATE"])
    return df

# ------------------ Torch utils ------------------
def build_time_condition(times):
    """Create sin/cos time-of-day encoding"""
    times = np.asarray(times)
    sin_t = np.sin(2 * np.pi * times / 48)
    cos_t = np.cos(2 * np.pi * times / 48)
    return np.stack([sin_t, cos_t], axis=1)

def stack_condition_with_exogenous(cond_t, pv=None, rp=None):
    comps = [cond_t]
    if pv is not None: comps.append(pv[:, None])
    if rp is not None: comps.append(rp[:, None])
    return np.concatenate(comps, axis=1)

# ------------------ Torch Dataset ------------------
class DayDataset(torch.utils.data.Dataset):
    def __init__(self, dates, times, usep, load, pv, rp, seq_len):
        self.dates, self.times = dates, times
        self.usep, self.load, self.pv, self.rp = usep, load, pv, rp
        self.seq_len = seq_len

    def __len__(self): return len(self.dates)

    def __getitem__(self, i):
        y = np.stack([self.usep[i], self.load[i]], axis=0)  # [2,T]
        cond_t = build_time_condition(self.times[i])
        cond = stack_condition_with_exogenous(
            cond_t,
            pv=self.pv[i] if self.pv is not None else None,
            rp=self.rp[i] if self.rp is not None else None
        )
        y = torch.tensor(y, dtype=torch.float32)
        cond = torch.tensor(cond.T, dtype=torch.float32)
        return y, cond
