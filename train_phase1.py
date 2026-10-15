"""
train_phase1.py — train / evaluate the Phase 1 forward surrogate.

    python train_phase1.py --class_name RDN                 # train + evaluate
    python train_phase1.py --class_name ALL --target cross  # pooled, cross-pol
    python train_phase1.py --class_name RDN --eval_only     # evaluate a checkpoint
    python train_phase1.py --class_name RDN --tag small --model_override planes=32,64,128

Outputs in config.phase1_dir(class, target, tag):
    phase1_forward_best.pth / _last.pth, run_config.json,
    phase1_train_history.npz, phase1_loss_curves.png,
    phase1_metrics.json, phase1_test_per_sample.csv, eval figures.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim

import config as C
from dataset import get_dataloaders, get_split_arrays, class_split_indices, load_class, freq_grid, to_convention
from phase1_forward_model import ForwardSurrogate, load_surrogate
from phase1_loss import PhysicsGuidedLoss, magnitude_metrics, complex_metrics
import utils




# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def mirror_batch(imgs, spec, target):
    """Random mirror symmetry of the centred unit cell (normal incidence):
    x- or y-mirror keeps r_xx and flips the sign of r_yx; both together
    (180 deg rotation) keep both. Applied per sample."""
    B = imgs.size(0)
    fx = torch.rand(B, device=imgs.device) < 0.5
    fy = torch.rand(B, device=imgs.device) < 0.5
    imgs = torch.where(fx[:, None, None, None], imgs.flip(-1), imgs)
    imgs = torch.where(fy[:, None, None, None], imgs.flip(-2), imgs)
    if target == "cross":
        sign = 1.0 - 2.0 * (fx ^ fy).float()
        spec = spec * sign[:, None, None]
    return imgs, spec


def _run_epoch(model, loader, criterion, optimizer=None, augment=False, target="co"):
    floor = criterion.db_floor
    training = optimizer is not None
    model.train(training)
    acc, n = {}, 0
    with torch.set_grad_enabled(training):
        for imgs, spec in loader:
            imgs, spec = imgs.to(C.DEVICE, non_blocking=True), spec.to(C.DEVICE, non_blocking=True)
            if training and augment:
                imgs, spec = mirror_batch(imgs, spec, target)
            pred = model(imgs)
            loss, comps = criterion(pred, spec)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), C.P1_GRAD_CLIP)
                optimizer.step()
            with torch.no_grad():
                mp = torch.sqrt((pred ** 2).sum(-1) + 1e-7)
                mt = torch.sqrt((spec ** 2).sum(-1) + 1e-7)
                comps["lin_mse"] = ((mp - mt) ** 2).mean().item()
                dp, dt = 20 * torch.log10(mp), 20 * torch.log10(mt)
                if floor is not None:
                    dp, dt = dp.clamp(min=floor), dt.clamp(min=floor)
                comps["db_mse"] = ((dp - dt) ** 2).mean().item()
            comps["total"] = loss.item()
            B = imgs.size(0)
            for k, v in comps.items():
                acc[k] = acc.get(k, 0.0) + v * B
            n += B
    return {k: v / n for k, v in acc.items()}


def parse_overrides(items):
    """['planes=32,64,128', 'gru_hidden=64'] -> dict with int / list values."""
    def val(x):
        if x.lower() in ("true", "false"):
            return x.lower() == "true"
        return float(x) if "." in x else int(x)

    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        vals = [val(x) for x in v.split(",") if x]
        out[k] = vals if k in ("planes", "blocks", "strides", "pga", "channels") else vals[0]
    return out


def train(args):
    out_dir = C.phase1_dir(args.class_name, args.target, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "phase1_forward_best.pth")
    freq = freq_grid()
    overrides = parse_overrides(args.model_override)
    if len(freq) != C.P1_FREQ_QUERIES:
        overrides["n_queries"] = len(freq)
    epochs = args.epochs or C.P1_EPOCHS
    lr = args.lr or C.P1_LR
    wd = C.P1_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay
    patience_max = args.patience or C.P1_EARLY_STOP_PAT
    torch.manual_seed(C.RANDOM_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, _ = get_dataloaders(args.class_name, C.P1_BATCH_SIZE, args.target,
                                                  args.convention, args.n_train, args.msdcnn_train)
    use_val = len(val_loader.dataset) > 0          # MSDCNN protocol: no validation set, keep the last epoch
    model = ForwardSurrogate(args.model_type, **overrides).to(C.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    run_cfg = dict(class_name=args.class_name, target=args.target,
                   layout_convention=args.convention,
                   model_type=args.model_type,
                   model_overrides=overrides, n_params=n_params, epochs=epochs, lr=lr,
                   weight_decay=wd, batch_size=C.P1_BATCH_SIZE, patience=patience_max,
                   loss_options=C.P1_LOSS_OPTIONS[args.target], augment=args.augment,
                   n_train=len(train_loader.dataset), n_val=len(val_loader.dataset),
                   msdcnn_train=args.msdcnn_train)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(run_cfg, fh, indent=2)

    criterion = PhysicsGuidedLoss(freq, **C.P1_LOSS_OPTIONS[args.target])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    print(f"\nPhase 1 [{args.class_name} / {args.target}"
          f"{' / ' + args.tag if args.tag else ''}]  "
          f"{args.model_type} {overrides}  params={n_params:,}  "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}  device={C.DEVICE}")
    print(f"  {'ep':>4} {'tr_tot':>8} {'tr_lin':>9} {'va_tot':>8} {'va_lin':>9} {'va_dB':>8} {'lr':>9} {'s':>5}")

    best, patience, hist_tr, hist_va = float("inf"), 0, [], []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        te = time.time()
        tr = _run_epoch(model, train_loader, criterion, optimizer, args.augment, args.target)
        va = _run_epoch(model, val_loader, criterion) if use_val else {k: float("nan") for k in tr}
        scheduler.step()
        hist_tr.append(tr); hist_va.append(va)
        crit = va[C.P1_CKPT_CRITERION[args.target]] if use_val else -ep
        mark = ""
        if crit < best:
            best, patience, mark = crit, 0, " *"
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
        print(f"  {ep:4d} {tr['total']:8.5f} {tr['lin_mse']:9.6f} {va['total']:8.5f} "
              f"{va['lin_mse']:9.6f} {va['db_mse']:8.4f} {scheduler.get_last_lr()[0]:9.2e} "
              f"{time.time() - te:5.0f}{mark}", flush=True)
        if patience >= patience_max:
            print(f"  early stop at epoch {ep}")
            break

    torch.save(model.state_dict(), os.path.join(out_dir, "phase1_forward_last.pth"))
    keys = sorted(hist_tr[0])
    np.savez(os.path.join(out_dir, "phase1_train_history.npz"),
             **{f"tr_{k}": np.array([h[k] for h in hist_tr]) for k in keys},
             **{f"va_{k}": np.array([h[k] for h in hist_va]) for k in keys})
    utils.plot_training_curves(hist_tr, hist_va, os.path.join(out_dir, "phase1_loss_curves.png"))
    run_cfg.update(best_val=best, epochs_run=len(hist_tr), minutes=(time.time() - t0) / 60)
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(run_cfg, fh, indent=2)
    print(f"  done in {run_cfg['minutes']:.1f} min, best val {C.P1_CKPT_CRITERION[args.target]}={best:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, images, batch=1024, tta=False, target="co"):
    """Batched prediction; tta averages over the four mirror images (the
    cross-pol prediction of a single mirror is sign-corrected)."""
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(np.asarray(images[i:i + batch])).float().unsqueeze(1).to(C.DEVICE)
        if not tta:
            out.append(model(x).cpu().numpy())
            continue
        acc = 0
        for fx in (False, True):
            for fy in (False, True):
                xx = x.flip(-1) if fx else x
                xx = xx.flip(-2) if fy else xx
                s = -1.0 if (target == "cross" and fx != fy) else 1.0
                acc = acc + s * model(xx)
        out.append((acc / 4).cpu().numpy())
    return np.concatenate(out)


def _band_metrics(pred, tgt, band, freq):
    sel = (freq >= band[0] - 1e-9) & (freq <= band[1] + 1e-9)
    mp, mt = np.linalg.norm(pred[:, sel], axis=-1), np.linalg.norm(tgt[:, sel], axis=-1)
    return {"mag": magnitude_metrics(mp, mt),
            "power_mse": float(np.mean((mp ** 2 - mt ** 2) ** 2)), "n_points": int(sel.sum())}


def evaluate(args, model=None):
    out_dir = C.phase1_dir(args.class_name, args.target, args.tag)
    freq = freq_grid()
    ckpt = args.ckpt or os.path.join(out_dir, "phase1_forward_best.pth")
    model = model or load_surrogate(ckpt, model_type=args.model_type if args.ckpt else None)
    rc_path = os.path.join(os.path.dirname(ckpt), "run_config.json")
    rc = json.load(open(rc_path)) if os.path.exists(rc_path) else {}
    conv = rc.get("layout_convention", "file")
    msdcnn_train = rc.get("msdcnn_train")
    n_params = sum(p.numel() for p in model.parameters())

    res = {"n_params": n_params, "checkpoint": ckpt, "layout_convention": conv}
    data = {}
    for split in ("val", "test"):
        x, y, labels, idx = get_split_arrays(args.class_name, split, args.target, conv, msdcnn_train)
        if len(x) == 0:                                    # no validation set (MSDCNN protocol)
            x, y, labels, idx = get_split_arrays(args.class_name, "test", args.target, conv, msdcnn_train)
        p = predict(model, x, tta=args.tta, target=args.target)
        data[split] = (x, y, p, labels, idx)
        res[split] = dict(magnitude_metrics(np.linalg.norm(p, axis=-1), np.linalg.norm(y, axis=-1),
                                            C.P1_LOSS_OPTIONS[args.target]["db_floor"]),
                          **complex_metrics(p, y), n=len(x))
    yv = np.concatenate([data["val"][1], data["test"][1]])
    pv = np.concatenate([data["val"][2], data["test"][2]])
    res["val+test"] = magnitude_metrics(np.linalg.norm(pv, axis=-1), np.linalg.norm(yv, axis=-1))

    x, y, p, labels, idx = data["test"]
    # MSDCNN comparison: |R| MSE on a random 10% test set of the full class,
    # over 2-12 GHz (PTN, RDN) and over 9.5-12 GHz (how MSDCNN reports PLG).
    res["msdcnn_protocol"] = dict(n=len(x), full_band=_band_metrics(p, y, C.FREQ_RANGE, freq),
                                  plg_band=_band_metrics(p, y, C.MSDCNN_PLG_BAND, freq))
    if args.class_name == C.POOLED_CLASS:
        res["per_class_test"] = {
            c: magnitude_metrics(np.linalg.norm(p[labels == c], axis=-1),
                                 np.linalg.norm(y[labels == c], axis=-1)) for c in C.CLASSES}

    with open(os.path.join(out_dir, "phase1_metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    t = res["test"]
    mb = res["msdcnn_protocol"]["plg_band"]["mag"]
    print(f"[eval {args.class_name}/{args.target}] params={n_params:,}  test lin={t['lin_mse']:.3e} "
          f"dB={t['db_mse']:.3f} R2={t['r2']:.4f} phase={t['phase_mae_deg']:.1f}deg | "
          f"MSDCNN protocol 9.5-12GHz lin={mb['lin_mse']:.3e} R2={mb['r2']:.4f}")

    # per-sample table (worst first) and figures
    mp, mt = np.linalg.norm(p, axis=-1), np.linalg.norm(y, axis=-1)
    lin = ((mp - mt) ** 2).mean(1)
    dphi = np.degrees(np.abs(np.angle((p[..., 0] + 1j * p[..., 1]) * np.conj(y[..., 0] + 1j * y[..., 1])))).mean(1)
    order = np.argsort(-lin)
    with open(os.path.join(out_dir, "phase1_test_per_sample.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "folder_index", "lin_mse", "phase_mae_deg", "fill"])
        for i in order:
            w.writerow([labels[i], int(idx[i]), f"{lin[i]:.6e}", f"{dphi[i]:.2f}", f"{x[i].mean():.4f}"])
    utils.plot_eval_figures(freq, x, y, p, lin, out_dir, n_show=C.EVAL_N_PLOT,
                            title=f"{args.class_name} ({args.target})")
    return res


if __name__ == "__main__":
    C.be_nice()
    ap = argparse.ArgumentParser(description="Phase 1 forward surrogate")
    ap.add_argument("--class_name", required=True, choices=C.CLASSES + [C.POOLED_CLASS])
    ap.add_argument("--target", default="co", choices=list(C.TARGET_CHANNELS))
    ap.add_argument("--model_type", default=C.P1_MODEL_TYPE, choices=list(C.P1_MODEL_PRESETS))
    ap.add_argument("--model_override", nargs="*", help="e.g. planes=32,64,128 blocks=2,2,2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None, help="early-stop patience (epochs)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--augment", action="store_true", help="random mirror symmetry augmentation")
    ap.add_argument("--tta", action="store_true", help="average predictions over the 4 mirror images")
    ap.add_argument("--convention", default=C.LAYOUT_CONVENTION, choices=["physical", "file"])
    ap.add_argument("--ckpt", default=None, help="evaluate this checkpoint instead")
    ap.add_argument("--n_train", type=int, default=None, help="use only the first N training layouts")
    ap.add_argument("--msdcnn_train", type=int, default=None,
                    help="MSDCNN protocol: train on the first N layouts of the class, no validation set")
    args = ap.parse_args()
    if not args.eval_only:
        train(args)
    evaluate(args)
