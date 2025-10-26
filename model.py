# model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

# ---------- Blocks ----------
class FiLM(nn.Module):
    """Feature-wise linear modulation from condition channels."""
    def __init__(self, cond_ch, hidden):
        super().__init__()
        self.affine = nn.Conv1d(cond_ch, hidden * 2, 1)

    def forward(self, h, cond):
        # h: [B,H,T], cond: [B,F,T]
        gamma_beta = self.affine(cond)
        H = h.size(1)
        gamma, beta = gamma_beta[:, :H], gamma_beta[:, H:]
        return h * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        pad = dilation
        self.main = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation),
        )
        self.skip = nn.Conv1d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))

# ---------- Generator / Critic ----------
class Generator(nn.Module):
    """
    Input:
        z:    [B, z_dim, T]
        cond: [B, F, T]
    Output:
        y: [B, 2, T]  (channel 0 = USEP, 1 = LOAD)
    """
    def __init__(self, z_dim=16, cond_ch=8, hidden=128, out_ch=2):
        super().__init__()
        self.inp = nn.Conv1d(z_dim, hidden, 1)
        self.film = FiLM(cond_ch, hidden)
        self.tcn = nn.Sequential(
            ResBlock(hidden, 1),
            ResBlock(hidden, 2),
            ResBlock(hidden, 4),
            ResBlock(hidden, 8),
        )
        self.out = nn.Conv1d(hidden, out_ch, 1)

    def forward(self, z, cond):
        h = self.inp(z)
        h = self.film(h, cond)
        h = self.tcn(h)
        return self.out(h)

class Critic(nn.Module):
    """
    Input:
        x:    [B, 2, T] (USEP, LOAD)
        cond: [B, F, T]
    Output:
        score: [B, 1]
    """
    def __init__(self, cond_ch=8, in_ch=2, hidden=128):
        super().__init__()
        ch = in_ch + cond_ch
        self.conv = nn.Sequential(
            SN(nn.Conv1d(ch, hidden, 3, padding=1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(hidden, hidden, 3, padding=1)), nn.LeakyReLU(0.2),
            SN(nn.Conv1d(hidden, hidden, 3, padding=1)), nn.LeakyReLU(0.2),
        )
        self.head = SN(nn.Conv1d(hidden, 1, 1))

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        h = self.conv(h)
        s = self.head(h).mean(dim=2, keepdim=True)  # average over time
        return s
