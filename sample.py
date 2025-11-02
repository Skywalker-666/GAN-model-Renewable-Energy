import os, argparse, torch
import numpy as np, pandas as pd
from model import Generator
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sample(args):
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]

    G = Generator(cfg["z_dim"], 8, hidden=cfg["hidden"]).to(device)
    G.load_state_dict(ckpt["G_ema"])
    G.eval()

    df = read_energy_csv(args.data_path)
    dates = sorted(df["DATE"].unique())
    days = [df[df["DATE"] == d] for d in dates]

    usep_scaler = RobustScaler1D.from_dict(ckpt["usep_scaler"])
    load_scaler = RobustScaler1D.from_dict(ckpt["load_scaler"])
    pv_scaler = RobustScalerVec.from_dict(ckpt["pv_scaler"]) if ckpt.get("pv_scaler") else None
    rp_scaler = RobustScalerVec.from_dict(ckpt["rp_scaler"]) if ckpt.get("rp_scaler") else None

    all_usep, all_load = [], []
    for d in days:
        pv = pv_scaler.transform(d["PV"].values) if pv_scaler else None
        rp = rp_scaler.transform(d["RP"].values) if rp_scaler else None
        t = np.arange(args.seq_len)
        cond_t = build_time_condition(t)
        cond = stack_condition_with_exogenous(cond_t, pv, rp)
        cond_torch = torch.tensor(cond.T[None], dtype=torch.float32, device=device)

        z = torch.randn(args.n_samples, cfg["z_dim"], args.seq_len, device=device)
        cond_rep = cond_torch.repeat(args.n_samples, 1, 1)
        y_fake = G(z, cond_rep).cpu().numpy()

        usep = usep_scaler.inverse_transform(y_fake[:,0])
        load = load_scaler.inverse_transform(y_fake[:,1])
        all_usep.append(usep)
        all_load.append(load)

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, "samples.npz"),
             dates=np.array(dates),
             usep_samples=np.array(all_usep),
             load_samples=np.array(all_load))
    print("[OK] Saved scenarios to", args.outdir)

    med = {
        "DATE": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "USEP_MEDIAN_0": np.median(np.array(all_usep)[:,:,0], axis=1),
        "LOAD_MEDIAN_0": np.median(np.array(all_load)[:,:,0], axis=1),
    }
    pd.DataFrame(med).to_csv(os.path.join(args.outdir, "quick_medians.csv"), index=False)
    print("[OK] quick_medians.csv written")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_samples", type=int, default=100)
    args = ap.parse_args()
    sample(args)
