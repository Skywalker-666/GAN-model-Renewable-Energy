import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch

from utils import make_time_features, build_sequences
from model import Generator, Critic


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint hyperparams
    ckpt = torch.load(args.checkpoint, map_location=device)
    seq_len = int(ckpt["seq_len"])
    z_dim   = int(ckpt["z_dim"])
    hidden  = int(ckpt["hidden"])

    # Build time features and get scalers/columns via build_sequences (but DO NOT slice from X_c)
    df = pd.read_csv(args.data_path)
    df = make_time_features(df, "DATE")
    X_c, Y, y_scaler, c_scaler, cov_cols = build_sequences(df, seq_len=seq_len, use_covariates=True)

    # Correct way: slice from the flat covariates, then reshape to [n_days, F, T]
    total_steps = args.n_days * seq_len
    c_raw = df[cov_cols].values                          # [N_total, F]
    c_scaled = c_scaler.transform(c_raw)                 # standardize using same scaler as training
    if total_steps > len(c_scaled):
        raise ValueError(
            f"Requested n_days*seq_len={total_steps}, but only {len(c_scaled)} rows available. "
            f"Reduce --n_days or provide a longer CSV."
        )
    tail = c_scaled[-total_steps:]                       # [total_steps, F]
    cond_block = tail.reshape(args.n_days, seq_len, -1)  # [n_days, T, F]
    cond_block = np.swapaxes(cond_block, 1, 2)           # [n_days, F, T]
    cond = torch.from_numpy(cond_block).float().to(device)

    # Build generator consistent with checkpoint
    gen = Generator(z_dim=z_dim, cond_ch=cond.shape[1], out_ch=2, hidden=hidden).to(device)
    gen.load_state_dict(ckpt["gen"])
    gen.eval()

    all_samples = []
    with torch.no_grad():
        # Prepare scaler tensors for inverse transform (to real units)
        y_mean = torch.tensor(ckpt["y_scaler_mean"], device=device).view(1, 2, 1)
        y_scale = torch.tensor(ckpt["y_scaler_scale"], device=device).view(1, 2, 1)

        for _ in range(args.n_samples):
            z = torch.randn(args.n_days, z_dim, seq_len, device=device)   # [n_days, z_dim, T]
            y_hat = gen(z, cond)                                          # [n_days, 2, T]
            y_hat = y_hat * y_scale + y_mean                              # de-standardize
            all_samples.append(y_hat.cpu().numpy())

    all_samples = np.stack(all_samples)  # [S, n_days, 2, T]

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/samples_pt.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "samples": all_samples,
                "scaled": False,                 # real units now
                "seq_len": seq_len,
                "targets": ["USEP", "LOAD"],
            },
            f,
        )
    print("Saved samples to", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=48)  # kept for CLI parity; actual seq_len comes from ckpt
    p.add_argument("--n_days", type=int, default=7)
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
    main(args)
