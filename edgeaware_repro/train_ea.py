"""
train_ea.py — train and test the reproduced edge-aware surrogate exactly like
our PGA-ResNet surrogate (MSDCNN protocol), so the two can be compared:

  * per class, first 27,000 layouts of the fixed random order (RDN also 108,000),
    no validation set, last of 300 epochs kept;
  * tested on the last 3,000 layouts of every class (cross-class table);
  * AdamW 2e-4, weight decay 1e-5, cosine to 2e-6, batch 256, gradient clip 1,
    random mirror augmentation, 4-mirror test-time averaging;
  * loss: the paper's eq. (5) with 4x band-edge weighting (ea_loss.py).

The paper itself trained 50 epochs on an 80/20 PLG split with per-epoch grid
jitter; here the protocol of our surrogate is used for a like-for-like comparison.

    python train_ea.py                       # all five networks, then the cross-class table
    python train_ea.py --runs PTN:27000      # one network
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import config as C                                                    # noqa: E402
from dataset import get_dataloaders, get_split_arrays, freq_grid       # noqa: E402
from train_phase1 import mirror_batch, predict                         # noqa: E402
from ea_model import EdgeAwareSurrogate, count_params                  # noqa: E402
from ea_loss import EdgeAwareLoss                                      # noqa: E402

RUNS = ["PLG:27000", "PLR:27000", "PTN:27000", "RDN:27000", "RDN:108000"]
OUT = os.path.join(HERE, "runs")
EPOCHS, LR, WD, BATCH, CLIP = 300, 2e-4, 1e-5, 256, 1.0


def run_dir(spec):
    cls, n = spec.split(":")
    return os.path.join(OUT, f"{cls}_n{n}")


def metrics(p, y):
    mp, mt = np.linalg.norm(p, axis=-1), np.linalg.norm(y, axis=-1)
    dp, dt = 20 * np.log10(mp + 1e-7), 20 * np.log10(mt + 1e-7)
    K = mt.shape[1]
    e = max(1, int(round(K * 0.1)))
    edge = np.r_[0:e, K - e:K]
    Rp, Rt = p[..., 0] + 1j * p[..., 1], y[..., 0] + 1j * y[..., 1]
    return dict(lin_mse=float(np.mean((mp - mt) ** 2)),
                r2=float(1 - np.sum((mp - mt) ** 2) / np.sum((mt - mt.mean()) ** 2)),
                db_mse=float(np.mean((dp - dt) ** 2)),
                phase_mae_deg=float(np.degrees(np.abs(np.angle(Rp * np.conj(Rt)))).mean()),
                band_edge_abs_db=float(np.abs(dp - dt)[:, edge].mean()))


def train(spec, epochs=EPOCHS, skip=False):
    cls, n = spec.split(":")
    n = int(n)
    out = run_dir(spec)
    if os.path.exists(os.path.join(out, "metrics.json")):
        print(f"{spec}: done, skipped")
        return
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(C.RANDOM_SEED)
    train_loader, _, _ = get_dataloaders(cls, BATCH, "co", "physical", None, n)
    freq = freq_grid()
    model = EdgeAwareSurrogate(skip=skip).to(C.DEVICE)
    crit = EdgeAwareLoss(freq).to(C.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=LR * 0.01)
    cfg = dict(spec=spec, class_name=cls, n_train=len(train_loader.dataset), params=count_params(model),
               model=model.cfg, epochs=epochs, lr=LR, weight_decay=WD, batch=BATCH, clip=CLIP,
               augment="mirror", tta=True, test="last 3000 layouts of each class", loss="eq. (5), edge x4")
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=2, default=list)
    print(f"[{spec}] {cfg['params']:,} parameters, {cfg['n_train']} training layouts, {epochs} epochs", flush=True)
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for imgs, spec_t in train_loader:
            imgs, spec_t = imgs.to(C.DEVICE, non_blocking=True), spec_t.to(C.DEVICE, non_blocking=True)
            imgs, spec_t = mirror_batch(imgs, spec_t, "co")
            loss, _ = crit(model(imgs), spec_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += loss.item() * imgs.size(0); nb += imgs.size(0)
        sched.step()
        hist.append(tot / nb)
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d}  loss {tot / nb:.5f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save(model.state_dict(), os.path.join(out, "model_last.pth"))   # resumable record of progress
    np.save(os.path.join(out, "train_loss.npy"), np.asarray(hist))
    model.eval()
    x, y, _, _ = get_split_arrays(cls, "test", "co", "physical", 27000)    # last 3,000 layouts
    with torch.no_grad():
        m = metrics(predict(model, x, tta=True), y)
    m.update(params=cfg["params"], minutes=(time.time() - t0) / 60, n_test=int(len(x)))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"[{spec}] test (3,000 layouts): MSE {m['lin_mse']:.3e}, R2 {m['r2']:.3f}, "
          f"phase {m['phase_mae_deg']:.2f} deg, dB-MSE {m['db_mse']:.3f}, edge |dB| {m['band_edge_abs_db']:.3f}",
          flush=True)


def cross_class(runs):
    """Every trained network tested on the last 3,000 layouts of every class."""
    tests = {c: get_split_arrays(c, "test", "co", "physical", 27000) for c in C.CLASSES}
    table = {}
    for spec in runs:
        path = os.path.join(run_dir(spec), "model_last.pth")
        if not os.path.exists(os.path.join(run_dir(spec), "metrics.json")):
            continue
        cfg = json.load(open(os.path.join(run_dir(spec), "run_config.json")))
        model = EdgeAwareSurrogate(**{k: (tuple(v) if k == "channels" else v) for k, v in cfg["model"].items()}).to(C.DEVICE)
        model.load_state_dict(torch.load(path, map_location=C.DEVICE))
        model.eval()
        row = dict(params=cfg["params"])
        with torch.no_grad():
            for c, (x, y, _, _) in tests.items():
                p = predict(model, x, tta=True)
                row[c] = float(np.mean((np.linalg.norm(p, axis=-1) - np.linalg.norm(y, axis=-1)) ** 2))
        table[spec] = row
        print(f"{spec:11s} " + "  ".join(f"{c} {row[c] * 1e3:7.3f}" for c in C.CLASSES) + "  (x1e-3)")
    json.dump(table, open(os.path.join(HERE, "cross_class.json"), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--skip", action="store_true", help="add a residual q + MHA(q, K, V) (not in the paper)")
    a = ap.parse_args()
    C.be_nice()                                   # leave the CPU to the CST dataset run
    for s in a.runs:
        train(s, a.epochs, skip=a.skip)
    cross_class(RUNS)
