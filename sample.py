# sample.py
import os, argparse, pickle
import numpy as np
import pandas as pd
import torch

from utils import (
    read_energy_csv, make_daily_tensors, RobustScaler1D,
    expm1_signed, build_time_condition, stack_condition_with_exogenous
)
from model import Generator

@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    usep_scaler = RobustScaler1D.from_dict(ckpt["usep_scaler"])
    load_scaler = RobustScaler1D.from_dict(ckpt["load_scaler"])

    cond_ch = 6 + (1 if cfg.get("use_pv", True) else 0) + (1 if cfg.get("use_rp", True) else 0)
    G = Generator(z_dim=cfg["z_dim"], cond_ch=cond_ch, hidden=cfg["hidden"], out_ch=2).to(device)
    G.load_state_dict(ckpt["G_ema"])  # use EMA for inference
    G.eval()

    # Prepare days to sample
    df = read_energy_csv(args.data_path)
    dates, times, _, _, x_pv, x_rp = make_daily_tensors(df, seq_len=args.seq_len, use_rp=cfg.get("use_rp", True))
    if args.start or args.end:
        mask = np.ones(len(dates), dtype=bool)
        if args.start:
            mask &= dates >= pd.to_datetime(args.start)
        if args.end:
            mask &= dates <= pd.to_datetime(args.end)
        dates = dates[mask]; times = [times[i] for i in np.where(mask)[0]]
        x_pv = x_pv[mask]; x_rp = (x_rp[mask] if x_rp is not None else None)

    D = len(dates); T = args.seq_len; S = args.n_samples
    usep_samples = np.zeros((D, S, T), dtype=float)
    load_samples = np.zeros((D, S, T), dtype=float)

    for i in range(D):
        cond_t = build_time_condition(times[i])
        cond = stack_condition_with_exogenous(cond_t,
                                              pv=x_pv[i] if cfg.get("use_pv", True) else None,
                                              rp=(x_rp[i] if (x_rp is not None and cfg.get("use_rp", True)) else None))
        cond = torch.from_numpy(cond).float().to(device)[None, ...]  # [1,F,T]
        z = torch.randn(S, cfg["z_dim"], T, device=device)
        cond_rep = cond.repeat(S, 1, 1)
        y = G(z, cond_rep).cpu().numpy()  # [S,2,T]

        # Invert scaling
        usep_hat = expm1_signed(usep_scaler.inverse_transform(y[:, 0, :]))
        load_hat = load_scaler.inverse_transform(y[:, 1, :])

        usep_samples[i] = usep_hat
        load_samples[i] = load_hat

    os.makedirs(args.outdir, exist_ok=True)
    out_npz = os.path.join(args.outdir, "samples.npz")
    np.savez_compressed(out_npz,
                        dates=np.array([pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]),
                        time=np.arange(T),
                        usep_samples=usep_samples,
                        load_samples=load_samples)
    print(f"[OK] Saved scenarios to {out_npz}")

    # Optional quick-look CSV of daily medians  ✅ NumPy 2.x safe
    med = {
        "DATE": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "USEP_MEDIAN_0": np.median(usep_samples[:, :, 0], axis=1),
        "LOAD_MEDIAN_0": np.median(load_samples[:, :, 0], axis=1),
    }
    pd.DataFrame(med).to_csv(os.path.join(args.outdir, "quick_medians.csv"), index=False)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    sample(args)

if __name__ == "__main__":
    cli()
