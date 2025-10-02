import os, argparse, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import make_time_features, build_sequences, train_val_split
from model import Generator, Critic

def gradient_penalty(critic, real, fake, cond, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    inter = alpha * real + (1-alpha) * fake
    inter.requires_grad_(True)
    score = critic(inter, cond)
    grad = torch.autograd.grad(outputs=score.sum(),
                               inputs=inter,
                               create_graph=True)[0]
    gp = ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1.0)**2).mean()
    return gp

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = make_time_features(df, "DATE")

    X_c, Y, y_scaler, c_scaler, cov_cols = build_sequences(df, seq_len=args.seq_len, use_covariates=True)
    (Xc_tr, Y_tr), (Xc_va, Y_va) = train_val_split(X_c, Y, val_ratio=0.1)

    # to torch: [N,T,C] -> [N,C,T]
    def to_ch_first(a): return torch.from_numpy(np.swapaxes(a,1,2)).float()
    tr_set = TensorDataset(to_ch_first(Xc_tr), to_ch_first(Y_tr))
    va_set = TensorDataset(to_ch_first(Xc_va), to_ch_first(Y_va))
    tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    cond_ch = Xc_tr.shape[-1]
    gen = Generator(z_dim=args.z_dim, cond_ch=cond_ch, out_ch=2, hidden=args.hidden).to(device)
    cri = Critic(in_ch=2, cond_ch=cond_ch, hidden=args.hidden).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=tuple(args.betas))
    opt_d = torch.optim.Adam(cri.parameters(), lr=args.lr_d, betas=tuple(args.betas))

    best_va = 1e9
    for epoch in range(1, args.epochs+1):
        gen.train(); cri.train()
        d_losses, g_losses = [], []

        for xc, y in tr_loader:
            xc = xc.to(device)  # [B, Cc, T]
            y  = y.to(device)   # [B, 2,  T]
            # multiple critic steps
            for _ in range(args.critic_steps):
                z = torch.randn(y.size(0), args.z_dim, y.size(-1), device=device)
                y_fake = gen(z, xc).detach()
                d_real = cri(y, xc).mean()
                d_fake = cri(y_fake, xc).mean()
                gp = gradient_penalty(cri, y, y_fake, xc, device) * args.gp_lambda
                d_loss = -(d_real - d_fake) + gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # generator step
            z = torch.randn(y.size(0), args.z_dim, y.size(-1), device=device)
            y_fake = gen(z, xc)
            g_loss = -cri(y_fake, xc).mean()

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            d_losses.append(d_loss.item()); g_losses.append(g_loss.item())

        # simple val score: L1 between real and a single sample (proxy for monitoring)
        gen.eval()
        with torch.no_grad():
            mae_list = []
            for xc, y in va_loader:
                xc = xc.to(device); y = y.to(device)
                z = torch.randn(y.size(0), args.z_dim, y.size(-1), device=device)
                y_hat = gen(z, xc)
                mae = (y_hat - y).abs().mean().item()
                mae_list.append(mae)
            va_mae = float(np.mean(mae_list))

        print(f"Epoch {epoch:03d} | D {np.mean(d_losses):.4f} | G {np.mean(g_losses):.4f} | VA-MAE {va_mae:.4f}")
        if va_mae < best_va:
            best_va = va_mae
            torch.save({
                "gen": gen.state_dict(),
                "cri": cri.state_dict(),
                "y_scaler_mean": y_scaler.mean_.tolist(),
                "y_scaler_scale": y_scaler.scale_.tolist(),
                "c_scaler_mean": y_scaler.mean_.tolist(),
                "c_scaler_scale": y_scaler.scale_.tolist(),
                "cov_cols": cov_cols,
                "seq_len": args.seq_len,
                "z_dim": args.z_dim,
                "hidden": args.hidden
            }, "checkpoints/best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=48)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=4e-4)
    p.add_argument("--betas", type=float, nargs=2, default=[0.0, 0.9])
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--critic_steps", type=int, default=5)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()
    train(args)