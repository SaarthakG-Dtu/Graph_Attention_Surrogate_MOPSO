"""
phase1_forward_model.py — PGA-ResNet backbone + frequency cross-attention
forward surrogate (layout -> complex reflection spectrum).

Dimension flow (resnet3conv_pga, 16x16 input):
    input                                   (B,   1, 16, 16)
    stem   Conv3x3 1->64, BN, tanh          (B,  64, 16, 16)
    stage1 3 x BasicBlock(64),  s=1 + 3 PGA (B,  64, 16, 16)  graph N=256
    stage2 4 x BasicBlock(128), s=2 + 3 PGA (B, 128,  8,  8)  N=64
    stage3 3 x BasicBlock(256), s=2 + 3 PGA (B, 256,  4,  4)  N=16
    Dropout2d -> 16 tokens of dim 256       (B,  16, 256)
    cross-attention, 101 learned queries    (B, 101, 256)
    BiGRU -> Dropout -> Linear(512 -> 2)    (B, 101, 2)
    polar head  S = sigmoid(a) * exp(j b)   |S| < 1 by construction

PGA block (Zhang et al., pixel-wise graph attention):
    R = theta(F) phi(F)^T, masked to the 8-neighbour grid adjacency (no
    self-loops), row softmax, applied to the RAW features (no value
    projection), ELU, then F' = (1 - gamma) F + gamma * PGA(F), gamma = 0 at init.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


def sinusoidal_pe(length: int, dim: int, device) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device)
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:dim // 2])
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# Grid adjacency (index arithmetic, O(kHW)), cached per shape
# ─────────────────────────────────────────────────────────────────────────────
_ADJ_CACHE: dict = {}


def build_adjacency(H: int, W: int, neighbor: int, device) -> torch.Tensor:
    key = (H, W, neighbor, str(device))
    if key not in _ADJ_CACHE:
        idx = np.arange(H * W).reshape(H, W)
        A = np.zeros((H * W, H * W), np.float32)
        if neighbor == 8:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        elif neighbor == 4:
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        else:
            raise ValueError(f"neighbor must be 4 or 8, got {neighbor}")
        for di, dj in offsets:
            src = idx[max(0, -di):H - max(0, di), max(0, -dj):W - max(0, dj)]
            dst = idx[max(0, di):H - max(0, -di) or None, max(0, dj):W - max(0, -dj) or None]
            A[src.ravel(), dst.ravel()] = 1.0
        _ADJ_CACHE[key] = torch.from_numpy(A).to(device)
    return _ADJ_CACHE[key]


class GraphAttentionLayer(nn.Module):
    """One head of spatial pixel-wise graph attention; (B,C,H,W) -> (B,C,H,W)."""

    def __init__(self, channels: int, neighbor: int = 8, down_ratio: int = 8):
        super().__init__()
        self.neighbor = neighbor
        cp = max(channels // down_ratio, 1)
        self.theta = nn.Sequential(nn.Conv2d(channels, cp, 1, bias=False),
                                   nn.BatchNorm2d(cp), nn.ReLU(inplace=True))
        self.phi = nn.Sequential(nn.Conv2d(channels, cp, 1, bias=False),
                                 nn.BatchNorm2d(cp), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Ch, H, W = x.shape
        N = H * W
        th = self.theta(x).flatten(2).transpose(1, 2)               # (B, N, C')
        ph = self.phi(x).flatten(2)                                  # (B, C', N)
        R = torch.bmm(th, ph)                                        # (B, N, N)
        A = build_adjacency(H, W, self.neighbor, x.device)
        R = R.masked_fill(A.unsqueeze(0) <= 0, -9e15)
        att = torch.softmax(R, dim=2)
        out = torch.bmm(att, x.flatten(2).transpose(1, 2))           # raw features
        return out.transpose(1, 2).reshape(B, Ch, H, W)


class PGABlock(nn.Module):
    """h = (1 - gama) x + gama * ELU(heads(x)); gama = 0 -> identity at init."""

    def __init__(self, channels: int, neighbor: int = C.P1_PGA_NEIGHBOR,
                 nheads: int = C.P1_PGA_NHEADS, down_ratio: int = C.P1_PGA_DOWN_RATIO):
        super().__init__()
        self.heads = nheads
        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(channels, neighbor, down_ratio) for _ in range(nheads)])
        self.gama = nn.Parameter(torch.zeros(1))
        self.output_down = (nn.Sequential(nn.Conv2d(channels * nheads, channels, 1, bias=False),
                                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
                            if nheads > 1 else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.heads > 1:
            h = self.output_down(torch.cat([a(x) for a in self.attentions], dim=1))
        else:
            h = self.attentions[0](x)
        return (1.0 - self.gama) * x + self.gama * F.elu(h)


class BasicBlock(nn.Module):
    """Conv-BN-LeakyReLU-Conv-BN + shortcut, then ReLU (MSDCNN residual block)."""

    def __init__(self, inplanes, planes, stride=1, downsample=None, leaky_slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.LeakyReLU(leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


def _make_stage(inplanes, planes, blocks, stride, leaky_slope):
    down = None
    if stride != 1 or inplanes != planes:
        down = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                             nn.BatchNorm2d(planes))
    layers = [BasicBlock(inplanes, planes, stride, down, leaky_slope)]
    layers += [BasicBlock(planes, planes, leaky_slope=leaky_slope) for _ in range(1, blocks)]
    return nn.Sequential(*layers)


class PGAResNetBackbone(nn.Module):
    """Stem (stride 1) -> [ResNet stage -> P1_PGA_LAYERS x PGABlock] per stage."""

    def __init__(self, planes, blocks, strides, pga, pga_layers=C.P1_PGA_LAYERS):
        super().__init__()
        assert len(planes) == len(blocks) == len(strides) == len(pga)
        self.stem = nn.Sequential(
            nn.Conv2d(1, planes[0], 3, 1, 1, bias=False), nn.BatchNorm2d(planes[0]),
            nn.Tanh() if C.P1_STEM_ACT == "tanh" else nn.ReLU(inplace=True))
        self.layers, self.pga_stacks = nn.ModuleList(), nn.ModuleList()
        inplanes = planes[0]
        for p, b, s, use_pga in zip(planes, blocks, strides, pga):
            self.layers.append(_make_stage(inplanes, p, b, s, C.P1_LEAKY_SLOPE))
            self.pga_stacks.append(nn.Sequential(*[PGABlock(p) for _ in range(pga_layers)])
                                   if use_pga else nn.Identity())
            inplanes = p
        self.out_ch = planes[-1]

    def forward(self, x):
        x = self.stem(x)
        for layer, pga in zip(self.layers, self.pga_stacks):
            x = pga(layer(x))
        return x


class PlainCNNBackbone(nn.Module):
    """Baseline: stem -> [Conv-BN-ReLU-MaxPool] x n -> Conv -> per-token Linear."""

    def __init__(self, stem_ch, channels, final_ch):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, stem_ch, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True))
        blocks, cin = [], stem_ch
        for cout in channels:
            blocks.append(nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                        nn.MaxPool2d(2)))
            cin = cout
        self.conv_blocks = nn.Sequential(*blocks)
        self.final_conv = nn.Sequential(nn.Conv2d(cin, final_ch, 3, 1, 1, bias=False),
                                        nn.BatchNorm2d(final_ch), nn.ReLU(inplace=True))
        self.proj = nn.Linear(final_ch, final_ch)
        self.out_ch = final_ch

    def forward(self, x):
        x = self.final_conv(self.conv_blocks(self.stem(x)))
        B, Ch, H, W = x.shape
        t = self.proj(x.flatten(2).transpose(1, 2))
        return t.transpose(1, 2).reshape(B, Ch, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency cross-attention head
# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionFreqLayer(nn.Module):
    """q = MHA(LN(q), kv, kv); q = Linear(LN(q)). No residual connections."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, kv):
        q, _ = self.attn(self.norm_q(q), kv, kv)
        return self.ffn(self.norm_ff(q))


class CrossAttentionFreq(nn.Module):
    """K learned frequency queries (+ sinusoidal PE) attending to geometry tokens."""

    def __init__(self, embed_dim, n_heads=C.P1_CROSS_ATTN_HEADS,
                 n_queries=C.P1_FREQ_QUERIES, n_layers=C.P1_CROSS_ATTN_LAYERS):
        super().__init__()
        self.n_queries, self.embed_dim = n_queries, embed_dim
        self.query_embed = nn.Embedding(n_queries, embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [CrossAttentionFreqLayer(embed_dim, n_heads) for _ in range(n_layers)])

    def forward(self, tokens):
        B, dev = tokens.size(0), tokens.device
        q = self.query_embed.weight + sinusoidal_pe(self.n_queries, self.embed_dim, dev)
        q = q.unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(tokens)
        for layer in self.layers:
            q = layer(q, kv)
        return q


class PolarPassivityHead(nn.Module):
    """BiGRU -> Dropout -> Linear -> [sigmoid(a) cos b, sigmoid(a) sin b].
    n_out = 2 (r_xx, r_yx): rho = sigmoid(a), psi = pi/2 sigmoid(c),
    S_xx = rho cos(psi) e^{jb}, S_yx = rho sin(psi) e^{jb'}, so that
    |S_xx|^2 + |S_yx|^2 = rho^2 < 1; output (B, 2, K, 2)."""

    def __init__(self, in_dim, gru_hidden, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.bigru = nn.GRU(in_dim, gru_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head_dropout = nn.Dropout(C.P1_DROPOUT_HEAD)
        self.mlp = nn.Linear(2 * gru_hidden, 2 if n_out == 1 else 4)

    def forward(self, ctx):
        h, _ = self.bigru(ctx)
        logit = self.mlp(self.head_dropout(h))
        a, b = torch.sigmoid(logit[..., 0]), logit[..., 1]
        if self.n_out == 1:
            return torch.stack([a * torch.cos(b), a * torch.sin(b)], dim=-1)
        psi = 0.5 * math.pi * torch.sigmoid(logit[..., 2])
        mxx, myx, byx = a * torch.cos(psi), a * torch.sin(psi), logit[..., 3]
        return torch.stack([torch.stack([mxx * torch.cos(b), mxx * torch.sin(b)], -1),
                            torch.stack([myx * torch.cos(byx), myx * torch.sin(byx)], -1)], dim=1)


class ForwardSurrogate(nn.Module):
    """(B,1,16,16) layout (binary or soft in [0,1]) -> (B,K,2) [Re S, Im S]."""

    def __init__(self, model_type: str = C.P1_MODEL_TYPE, **overrides):
        super().__init__()
        cfg = dict(C.P1_MODEL_PRESETS[model_type], **overrides)
        self.model_type = model_type
        if model_type == "plain_cnn":
            self.backbone = PlainCNNBackbone(cfg["stem_ch"], cfg["channels"], cfg["final_ch"])
        else:
            self.backbone = PGAResNetBackbone(cfg["planes"], cfg["blocks"], cfg["strides"],
                                              cfg["pga"], cfg.get("pga_layers", C.P1_PGA_LAYERS))
        dim = self.backbone.out_ch
        self.backbone_dropout = nn.Dropout2d(C.P1_DROPOUT_BACKBONE)
        self.cross_attn = CrossAttentionFreq(dim, cfg.get("heads", C.P1_CROSS_ATTN_HEADS),
                                             n_queries=cfg.get("n_queries", C.P1_FREQ_QUERIES),
                                             n_layers=cfg.get("attn_layers", C.P1_CROSS_ATTN_LAYERS))
        self.head = PolarPassivityHead(dim, cfg.get("gru_hidden", dim), cfg.get("n_out", 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone_dropout(self.backbone(images))
        tokens = feat.flatten(2).transpose(1, 2)                    # (B, N, C)
        return self.head(self.cross_attn(tokens))


def load_surrogate(ckpt_path: str, device=C.DEVICE, model_type: str = None) -> ForwardSurrogate:
    """Build the model recorded next to the checkpoint (run_config.json) and load it."""
    import json, os
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "run_config.json")
    kwargs = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            rc = json.load(fh)
        model_type = model_type or rc.get("model_type")
        kwargs = rc.get("model_overrides", {})
    model = ForwardSurrogate(model_type or C.P1_MODEL_TYPE, **kwargs).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.eval()
