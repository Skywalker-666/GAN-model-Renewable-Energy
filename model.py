import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# FiLM modulation
# ------------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.scale = nn.Linear(cond_dim, num_features)
        self.shift = nn.Linear(cond_dim, num_features)

    def forward(self, x, cond):
        # x: [B, C, T], cond: [B, cond_dim]
        gamma = self.scale(cond).unsqueeze(-1)
        beta = self.shift(cond).unsqueeze(-1)
        return gamma * x + beta

# ------------------------------
# Residual Block for temporal conv
# ------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ch, ch, k, padding=pad),
        )

    def forward(self, x):
        return x + self.net(x)

# ------------------------------
# Generator
# ------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, hidden=128, out_ch=2, seq_len=48):
        super().__init__()
        self.inp = nn.Conv1d(z_dim, hidden, 1)
        self.film = FiLM(cond_dim, hidden)
        self.tcn = nn.Sequential(
            ResBlock1D(hidden),
            ResBlock1D(hidden),
            ResBlock1D(hidden),
        )
        self.out = nn.Conv1d(hidden, out_ch, 1)
        self.zskip = nn.Conv1d(z_dim, out_ch, 1)  # <== ensure noise matters

    def forward(self, z, cond):
        h = self.inp(z)
        h = self.film(h, cond)
        h = self.tcn(h)
        return self.out(h) + self.zskip(z)

# ------------------------------
# Critic (Discriminator)
# ------------------------------
class Critic(nn.Module):
    def __init__(self, in_ch=2, cond_dim=8, hidden=128):
        super().__init__()
        self.film = FiLM(cond_dim, hidden)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(hidden),
            ResBlock1D(hidden),
            nn.Conv1d(hidden, 1, 1)
        )

    def forward(self, y, cond):
        # y: [B, in_ch, T]
        h = self.film(self.net[0](y), cond)
        return self.net[1:](h).mean(dim=(1, 2))
