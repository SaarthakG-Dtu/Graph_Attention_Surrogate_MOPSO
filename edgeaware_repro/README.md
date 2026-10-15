# Edge-aware cross-attention surrogate — reproduction (additional baseline)

Reproduction of Zhang, Zhu, Wu and Long, "Edge-Aware Cross-Attention Surrogate for
Phase-Stable Metasurface Reflection", IEEE AWPL 25(6), 2461–2465, 2026, which was
tested on one SUTD-PRCM class only. It is trained and tested here on all four classes
in the same way as our PGA-ResNet surrogate. Kept separate from the main pipeline.

## Model (`ea_model.py`, Fig. 1 and Sec. II-B of the paper)

| Part | Setting |
|---|---|
| CNN backbone | Conv 3×3 → ReLU → 2 × [Conv 3×3, ReLU, MaxPool 2] → Conv 3×3 → ReLU; channels 64/128/256/256 → 4×4 = 16 tokens |
| Tokens → keys/values | Linear 256 → d_model 256 |
| Queries | 101 learnable frequency queries + sinusoidal positional encoding |
| Cross-attention | multi-head, 4 heads, dropout 0.1; no residual and no normalization, as in Fig. 1 |
| Decoder | BiGRU (hidden 180 per direction) → FC 360 → 256 → 2 |
| Output | polar head S = σ(a)[cos b + j sin b], passive by construction |
| Parameters | 1,880,530 (paper: ≈ 1.88 M) |

The paper does not give channel widths or GRU/FC sizes; they are chosen to match its
parameter count. Like the paper, the model has no skip connection or layer normalization
around the cross-attention.

## Loss (`ea_loss.py`): the paper's eq. (5)

L = L_reim + 0.5 L_dB + 0.3 L_lin + 0.3 L_phase + 0.08 L_pass + 0.06 L_tv + 0.02 L_curv + 0.02 L_KK,
with the first and last 10 % of the 101 frequencies weighted 4×.

## Training and testing (`train_ea.py`): same protocol as our surrogate

PLG27000, PLR27000, PTN27000, RDN27000 and RDN108000: first N layouts of each class's
fixed random order, no validation set, last of 300 epochs; AdamW 2e-4, weight decay 1e-5,
cosine to 2e-6, batch 256, gradient clip 1, random mirrors, 4-mirror test-time averaging.
Every network is tested on the last 3,000 layouts of every class (`cross_class.json`).
The paper's per-epoch grid jitter is not used (our data and our surrogate use the fixed
101-point grid).

```bat
python train_ea.py            :: all five networks, then the cross-class table
```

Outputs: `runs/<CLASS>_n<N>/` (model, settings, loss history, test metrics), `cross_class.json`,
`train_ea.log`.
