"""
ea_model.py — reproduction of the edge-aware cross-attention surrogate of
Zhang, Zhu, Wu and Long, IEEE AWPL vol. 25, no. 6, pp. 2461-2465, Jun. 2026
(Fig. 1 and Sec. II-B).

    input 16x16 layout                       (B,   1, 16, 16)
    CNN backbone (Fig. 1):
        Conv 3x3 -> ReLU                     (B,  c0, 16, 16)
        ConvBlock x2: Conv 3x3, ReLU, MaxPool(2)   -> (B, c2, 4, 4)
        Conv 3x3 -> ReLU                     (B,  c3,  4,  4)  -> P = 16 tokens
    Linear c3 -> d_model (keys and values)   (B,  16, 256)
    K learnable frequency queries Q_emb + sinusoidal positional encoding
    multi-head cross-attention (4 heads, d_model 256, dropout 0.1)
        no residual and no normalization, as in Fig. 1 (skip=True adds q + MHA(q, K, V))
    BiGRU over frequency                     (B, 101, 2h)
    View + FC -> (a, b) per frequency
    polar head  S = sigmoid(a) [cos b + j sin b]   (|S| < 1 by construction)

The paper gives d_model = 256, 4 heads, dropout 0.1 and ~1.88 M parameters but
not the channel widths or the GRU/FC sizes; the defaults below are chosen to
match the parameter count (EA_DEFAULTS; see count_params()).
"""

import math

import torch
import torch.nn as nn

EA_DEFAULTS = dict(channels=(64, 128, 256, 256), d_model=256, heads=4, dropout=0.1,
                   gru_hidden=180, fc_hidden=256, n_queries=101, skip=False)


def sinusoidal_pe(length, dim, device):
    pe = torch.zeros(length, dim, device=device)
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:dim // 2])
    return pe


class ConvBlock(nn.Module):
    """Conv 3x3 -> ReLU -> MaxPool 2x2 (Fig. 1, bottom left)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class EdgeAwareSurrogate(nn.Module):
    """(B, 1, 16, 16) layout -> (B, K, 2) [Re S, Im S]."""

    def __init__(self, **cfg):
        super().__init__()
        cfg = {**EA_DEFAULTS, **cfg}
        self.cfg = cfg
        c0, c1, c2, c3 = cfg["channels"]
        d = cfg["d_model"]
        self.backbone = nn.Sequential(
            nn.Conv2d(1, c0, 3, 1, 1), nn.ReLU(inplace=True),
            ConvBlock(c0, c1), ConvBlock(c1, c2),
            nn.Conv2d(c2, c3, 3, 1, 1), nn.ReLU(inplace=True))
        self.token_proj = nn.Linear(c3, d)                         # "Linear" -> K, V
        self.query_embed = nn.Parameter(torch.randn(cfg["n_queries"], d) * 0.02)   # Q_emb
        self.attn = nn.MultiheadAttention(d, cfg["heads"], dropout=cfg["dropout"], batch_first=True)
        self.bigru = nn.GRU(d, cfg["gru_hidden"], batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(2 * cfg["gru_hidden"], cfg["fc_hidden"]),
                                nn.ReLU(inplace=True), nn.Linear(cfg["fc_hidden"], 2))

    def forward(self, x):
        B = x.size(0)
        f = self.backbone(x)                                      # (B, c3, 4, 4)
        kv = self.token_proj(f.flatten(2).transpose(1, 2))        # (B, 16, d)
        q = self.query_embed + sinusoidal_pe(self.query_embed.size(0), self.query_embed.size(1), x.device)
        q = q.unsqueeze(0).expand(B, -1, -1)
        a, _ = self.attn(q, kv, kv)
        h = q + a if self.cfg["skip"] else a                      # Fig. 1: no residual (skip=False)
        h, _ = self.bigru(h)                                      # (B, K, 2h)
        logit = self.fc(h)                                        # View + FC
        amp, ph = torch.sigmoid(logit[..., 0]), logit[..., 1]
        return torch.stack([amp * torch.cos(ph), amp * torch.sin(ph)], dim=-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = EdgeAwareSurrogate()
    print(f"parameters: {count_params(m):,} (paper: ~1.88 M)")
    print(m(torch.rand(2, 1, 16, 16)).shape)
