import os, torch, argparse
import numpy as np, pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Generator, Critic
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ TRAIN FUNCTION ------------------
def train(args):
    df = read_energy_csv(args.data_path)
    dates = sorted(df["DATE"].unique())
    days = [df[df["DATE"] == d] for d in dates]

    # Prepare arrays
    usep = np.stack([d["USEP"].values for d in days])
    load = np.stack([d["LOAD"].values for d in days])
    pv   = np.stack([d["PV"].values for d in days]) if args.use_pv else None
    rp   = np.stack([d["RP"].values for d in days]) if args.use_rp else None
    times = [np.arange(args.seq_len) for _ in days]

    # Robust scalers
    usep_scaler = RobustScaler1D().fit(usep)
    load_scaler = RobustScaler1D().fit(load)
    usep_t = usep_scaler.transform(usep)
    load_t = load_scaler.transform(load)

    pv_scaler = RobustScalerVec().fit(pv) if pv is not None else None
    rp_scaler = RobustScalerVec().fit(rp) if rp is not None else None
    pv_t = pv_scaler.transform(pv) if pv is not None else pv
    rp_t = rp_scaler.transform(rp) if rp is not None else rp

    # Split
    n = len(dates)
    train_idx = np.arange(0, n - args.val_days)
    val_idx = np.arange(n - args.val_days, n)

    train_ds = DayDataset(dates[train_idx], [times[i] for i in train_idx],
                          usep_t[train_idx], load_t[train_idx],
                          pv_t[train_idx] if pv_t is not None else None,
                          rp_t[train_idx] if rp_t is not None else None,
                          seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # Models
    cond_dim = train_ds[0][1].shape[0]
    G = Generator(args.z_dim, cond_dim, hidden=args.hidden).to(device)
    D = Critic(in_ch=2, cond_dim=cond_dim, hidden=args.hidden).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.5, 0.9))

    # EMA
    G_ema = Generator(args.z_dim, cond_dim, hidden=args.hidden).to(device)
    G_ema.load_state_dict(G.state_dict())

    def ema_update(target, src, decay):
        with torch.no_grad():
            for t, s in zip(target.parameters(), src.parameters()):
                t.copy_(t * decay + s * (1 - decay))

    for epoch in range(args.epochs):
        for y, cond in train_loader:
            y, cond = y.to(device), cond.to(device)

            # --- Train Critic ---
            for _ in range(args.c_iters):
                z = torch.randn(y.size(0), args.z_dim, args.seq_len, device=device)
                y_fake = G(z, cond).detach()
                d_real = D(y, cond)
                d_fake = D(y_fake, cond)
                gp = gradient_penalty(D, y, y_fake, cond)
                loss_D = -(d_real.mean() - d_fake.mean()) + args.gp_lambda * gp
                opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # --- Train Generator ---
            z = torch.randn(y.size(0), args.z_dim, args.seq_len, device=device)
            y_fake = G(z, cond)
            d_fake = D(y_fake, cond)
            g_adv = -d_fake.mean()
            loss_anchor = torch.abs(y_fake - y).mean()
            tv = (y_fake[:,:,1:] - y_fake[:,:,:-1]).abs().mean()
            loss_G = g_adv + args.anchor * loss_anchor + 0.001 * tv
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            ema_update(G_ema, G, args.ema)

        print(f"[Epoch {epoch+1:03d}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

    # Save checkpoint
    ckpt = {
        "G": G.state_dict(),
        "G_ema": G_ema.state_dict(),
        "D": D.state_dict(),
        "cfg": vars(args),
        "usep_scaler": usep_scaler.to_dict(),
        "load_scaler": load_scaler.to_dict(),
        "pv_scaler": pv_scaler.to_dict() if pv_scaler else None,
        "rp_scaler": rp_scaler.to_dict() if rp_scaler else None,
    }
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(ckpt, os.path.join(args.outdir, "ckpt_last.pt"))
    print("[OK] Saved to", args.outdir)

# ------------------ Gradient penalty ------------------
def gradient_penalty(D, real, fake, cond):
    eps = torch.rand(real.size(0), 1, 1, device=real.device)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    out = D(inter, cond)
    grad = torch.autograd.grad(out, inter, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.norm(2, dim=(1,2)) - 1) ** 2).mean()

# ------------------ CLI ------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--val_days", type=int, default=0)
    ap.add_argument("--use_pv", action="store_true")
    ap.add_argument("--use_rp", action="store_true")
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lrG", type=float, default=1e-4)
    ap.add_argument("--lrD", type=float, default=2e-4)
    ap.add_argument("--c_iters", type=int, default=2)
    ap.add_argument("--gp_lambda", type=float, default=10.0)
    ap.add_argument("--anchor", type=float, default=0.20)
    ap.add_argument("--ema", type=float, default=0.999)
    args = ap.parse_args()
    train(args)
