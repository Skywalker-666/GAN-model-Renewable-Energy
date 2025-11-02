# train.py
import os, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import (
    set_seed, read_energy_csv, make_daily_tensors, RobustScaler1D,
    log1p_signed, build_time_condition, stack_condition_with_exogenous,
    transform_exo_for_model, ema_update
)
from model import Generator, Critic

# -----------------------------
# Dataset
# -----------------------------
class DayDataset(Dataset):
    def __init__(self, dates, times, usep, load, pv, rp, seq_len=48):
        self.dates, self.times = dates, times
        self.usep, self.load = usep, load
        self.pv, self.rp = pv, rp  # already transformed for the model (if not None)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, i):
        T = self.seq_len
        # targets
        y = np.stack([self.usep[i], self.load[i]], axis=0)  # [2, T]
        # build conditioning [F, T]
        cond_t = build_time_condition(self.times[i])        # [6, T]
        pv_i = (self.pv[i] if self.pv is not None else None)
        rp_i = (self.rp[i] if self.rp is not None else None)
        cond = stack_condition_with_exogenous(cond_t, pv=pv_i, rp=rp_i)
        return (
            torch.from_numpy(y).float(),       # [2,T]
            torch.from_numpy(cond).float(),    # [F,T]
            str(self.dates[i].date())
        )

# -----------------------------
# Training loop
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    df = read_energy_csv(args.data_path)
    dates, times, y_usep, y_load, x_pv, x_rp = make_daily_tensors(
        df, seq_len=args.seq_len, use_rp=args.use_rp
    )
    assert len(dates) > 0, "No complete days found."

    # Train/Val split by last val_days
    if args.val_days > 0:
        train_idx = np.arange(0, len(dates) - args.val_days)
        val_idx   = np.arange(len(dates) - args.val_days, len(dates))
    else:
        train_idx = np.arange(len(dates)); val_idx = np.array([], dtype=int)

    # ---- Fit scalers (on train only)
    usep_scaler = RobustScaler1D().fit(log1p_signed(y_usep[train_idx]))
    load_scaler = RobustScaler1D().fit(y_load[train_idx])

    # Exogenous scalers (match train distribution). PV & RP often heavy-tailed -> use log1p_signed.
    pv_scaler = None
    rp_scaler = None
    x_pv_t = None
    x_rp_t = None

    if args.use_pv:
        x_pv_t, pv_scaler = transform_exo_for_model(x_pv, do_log1p=True)
    if args.use_rp and x_rp is not None:
        x_rp_t, rp_scaler = transform_exo_for_model(x_rp, do_log1p=True)

    # Transform targets
    y_usep_t = usep_scaler.transform(log1p_signed(y_usep))
    y_load_t = load_scaler.transform(y_load)

    # Datasets
    train_ds = DayDataset(
        dates[train_idx], [times[i] for i in train_idx],
        y_usep_t[train_idx], y_load_t[train_idx],
        (x_pv_t[train_idx] if (args.use_pv and x_pv_t is not None) else None),
        (x_rp_t[train_idx] if (args.use_rp and x_rp_t is not None) else None),
        seq_len=args.seq_len
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    val_ds = None
    if len(val_idx):
        val_ds = DayDataset(
            dates[val_idx], [times[i] for i in val_idx],
            y_usep_t[val_idx], y_load_t[val_idx],
            (x_pv_t[val_idx] if (args.use_pv and x_pv_t is not None) else None),
            (x_rp_t[val_idx] if (args.use_rp and x_rp_t is not None) else None),
            seq_len=args.seq_len
        )

    # Model
    cond_ch = 6 + (1 if args.use_pv else 0) + (1 if args.use_rp else 0)
    G = Generator(z_dim=args.z_dim, cond_ch=cond_ch, hidden=args.hidden, out_ch=2).to(device)
    D = Critic(cond_ch=cond_ch, in_ch=2, hidden=args.hidden).to(device)
    G_ema = Generator(z_dim=args.z_dim, cond_ch=cond_ch, hidden=args.hidden, out_ch=2).to(device)
    G_ema.load_state_dict(G.state_dict())

    # Optims (TTUR)
    optG = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(0.0, 0.9))

    huber = nn.SmoothL1Loss()

    step = 0
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for (y_real, cond, _) in train_dl:
            y_real = y_real.to(device)               # [B,2,T]
            cond   = cond.to(device)                 # [B,F,T]
            B, _, T = y_real.shape

            # ----------------- Critic updates -----------------
            for _ in range(args.c_iters):
                z = torch.randn(B, args.z_dim, T, device=device)
                with torch.no_grad():
                    y_fake = G(z, cond)
                d_real = D(y_real, cond).mean()
                d_fake = D(y_fake, cond).mean()
                wd = d_real - d_fake

                # Gradient Penalty
                eps = torch.rand(B, 1, 1, device=device)
                x_hat = eps * y_real + (1 - eps) * y_fake
                x_hat.requires_grad_(True)
                d_hat = D(x_hat, cond).sum()
                g = torch.autograd.grad(d_hat, x_hat, create_graph=True)[0]
                gp = ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

                loss_D = -wd + args.gp_lambda * gp
                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

            # ----------------- Generator update -----------------
            z = torch.randn(B, args.z_dim, T, device=device)
            y_fake = G(z, cond)
            g_adv = -D(y_fake, cond).mean()

            # small anchor with tiny noise
            z_eps = 0.1 * torch.randn(B, args.z_dim, T, device=device)
            y_anchor = G(z_eps, cond)
            loss_anchor = huber(y_anchor, y_real)

            loss_G = g_adv + args.anchor * loss_anchor
            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            ema_update(G_ema, G, decay=args.ema)
            step += 1

        print(f"[Epoch {epoch:03d}] D: {loss_D.item():.4f} | G: {loss_G.item():.4f} | wd: {wd.item():.4f} | gp:{gp.item():.3f}")

        # --- light sanity signals each epoch (helps catch 'flat over T')
        with torch.no_grad():
            z_dbg = torch.randn(B, args.z_dim, T, device=device)
            y_dbg = G(z_dbg, cond).detach()
            print("   std_t(y_fake) =", y_dbg.std(dim=2).mean().item())

    # Save checkpoint (includes scalers)
    os.makedirs(args.outdir, exist_ok=True)
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
    torch.save(ckpt, os.path.join(args.outdir, "ckpt_last.pt"))
    print(f"[OK] Saved to {os.path.join(args.outdir, 'ckpt_last.pt')}")

# -----------------------------
# CLI
# -----------------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--outdir", default="checkpoints")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--val_days", type=int, default=0)
    ap.add_argument("--use_pv", action="store_true", default=True)
    ap.add_argument("--use_rp", action="store_true", default=True)

    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lrG", type=float, default=1e-4)
    ap.add_argument("--lrD", type=float, default=2e-4)
    ap.add_argument("--c_iters", type=int, default=2)
    ap.add_argument("--gp_lambda", type=float, default=10.0)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    cli()
