import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, s, p),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, k, s, p),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)

class Generator(nn.Module):
    def __init__(self, z_dim, cond_ch, out_ch=2, hidden=128):
        super().__init__()
        # IMPORTANT: no time-stacking; in-channels = z_dim + cond_ch
        self.init = nn.Conv1d(z_dim + cond_ch, hidden, 3, 1, 1)
        self.b1 = ConvBlock(hidden, hidden)
        self.b2 = ConvBlock(hidden, hidden)
        self.proj = nn.Conv1d(hidden, out_ch, 1)

    def forward(self, z, cond):
        # z, cond: [B, C, T]
        x = torch.cat([z, cond], dim=1)
        x = self.init(x)
        x = self.b1(x)
        x = self.b2(x)
        out = self.proj(x)
        return out  # [B, out_ch, T]

class Critic(nn.Module):
    def __init__(self, in_ch, cond_ch, hidden=128):
        super().__init__()
        # input: concat target(2) and condition along channels
        self.init = nn.Conv1d(in_ch + cond_ch, hidden, 3, 1, 1)
        self.b1 = ConvBlock(hidden, hidden)
        self.b2 = ConvBlock(hidden, hidden)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, y, cond):
        x = torch.cat([y, cond], dim=1)
        x = self.init(x)
        x = self.b1(x)
        x = self.b2(x)
        # global pooling over time
        x = self.head(x).mean(dim=-1)
        return x  # [B, 1]