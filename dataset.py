"""
dataset.py — SUTD-PRCM loader.

Raw layout: DATA_ROOT/<CLS>DATASET/full_data_list/Data_*/
    Integrate_image_*.npy  (M, 256)      binary layout, row-major 16x16
    Integrate_curve_*.npy  (M, 2, 1001)  complex, 2-12 GHz; [0]=R (x->x), [1]=T (x->y)

load_class() reads one class at the config.N_POINTS frequency points and
caches it in CACHE_DIR. class_split_indices() reproduces the exact
train/val/test membership used by every checkpoint:
    folder order -> torch.randperm(n, seed) -> split.
The pooled class "ALL" concatenates the per-class splits, so its test set is
the union of the four class test sets.

A batch is (image (B,1,16,16) float32, spec (B,K,2) float32 [Re, Im]).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATA_ROOT, CACHE_DIR, IMG_SIZE, FREQ_POINTS_FULL, CLASSES, POOLED_CLASS,
    FREQ_RANGE, N_POINTS, SPLIT, MSDCNN_N_TEST, RANDOM_SEED, TARGET_CHANNELS, DEVICE, LAYOUT_CONVENTION, FILE_INVERTED_CLASSES,
    RDN_STORED_AS_IS,
)


_AS_IS = None


def to_convention(images, class_name, convention=None, idx=None):
    """Layouts as stored -> requested convention ("physical": 1 = copper; "true":
    the layout as built, RDN also un-mirrored). idx: folder-order indices of the
    images (needed for "true", whose RDN reading depends on the row)."""
    global _AS_IS
    conv = convention or LAYOUT_CONVENTION
    if class_name not in FILE_INVERTED_CLASSES or conv == "file":
        return images
    if conv == "physical":
        return 1 - images
    if idx is None:
        raise ValueError("convention 'true' needs the folder-order indices of the RDN images")
    if _AS_IS is None:
        _AS_IS = np.load(RDN_STORED_AS_IS)
    out = (1 - images)[:, ::-1, :].copy()                      # flipud(1 - stored)
    keep = _AS_IS[np.asarray(idx)]
    out[keep] = images[keep]
    return out

_F_FULL = np.linspace(2.0, 12.0, FREQ_POINTS_FULL)    # raw CST grid


def freq_indices() -> np.ndarray:
    lo, hi = (int(np.argmin(np.abs(_F_FULL - f))) for f in FREQ_RANGE)
    return np.round(np.linspace(lo, hi, N_POINTS)).astype(int)


def freq_grid() -> np.ndarray:
    return _F_FULL[freq_indices()]


def _class_folders(class_name):
    root = Path(DATA_ROOT) / f"{class_name}DATASET" / "full_data_list"
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for sub in sorted(root.glob("Data_*")):
        imgs = sorted(sub.glob("Integrate_image_*.npy"))
        curves = sorted(sub.glob("Integrate_curve_*.npy"))
        if imgs and curves:
            yield imgs[0], curves[0]


def load_class(class_name: str):
    """All samples of one class in folder order.

    Returns images (N,16,16) uint8 and curves (N,2,K) complex64.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    idx = freq_indices()
    cache = os.path.join(CACHE_DIR, f"{class_name}_i{idx[0]}-{idx[-1]}_K{len(idx)}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["images"], d["curves"]

    images, curves = [], []
    for f_img, f_cur in _class_folders(class_name):
        img = np.load(f_img)
        cur = np.load(f_cur, mmap_mode="r")
        if img.shape[1] != IMG_SIZE * IMG_SIZE or cur.shape[2] != FREQ_POINTS_FULL:
            raise ValueError(f"Unexpected shapes {img.shape} / {cur.shape} in {f_img.parent}")
        images.append(img.reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.uint8))
        curves.append(np.asarray(cur[:, :, idx]).astype(np.complex64))
    images, curves = np.concatenate(images), np.concatenate(curves)
    np.savez(cache, images=images, curves=curves)
    return images, curves


def load_full_curves(class_name: str, indices) -> np.ndarray:
    """Full 1001-point complex curves (n, 2, 1001) for folder-order indices."""
    indices = np.asarray(indices)
    out = np.zeros((len(indices), 2, FREQ_POINTS_FULL), np.complex128)
    start = 0
    for _, f_cur in _class_folders(class_name):
        cur = np.load(f_cur, mmap_mode="r")
        stop = start + cur.shape[0]
        for s in np.where((indices >= start) & (indices < stop))[0]:
            out[s] = cur[indices[s] - start]
        start = stop
    return out


def class_split_indices(class_name: str) -> dict:
    """Folder-order indices of the train / val / test samples of one class."""
    _, curves = load_class(class_name)
    n = len(curves)
    n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(RANDOM_SEED)).numpy()
    return {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}


def msdcnn_split_indices(class_name: str, n_train: int) -> dict:
    """MSDCNN protocol (no validation set): the first n_train layouts of the
    class's fixed random order for training and, as in MSDCNN, the last
    MSDCNN_N_TEST (3,000) layouts of the order for testing. The test set is the
    same for every n_train, so all networks of a class share it."""
    _, curves = load_class(class_name)
    n = len(curves)
    if n_train > n - MSDCNN_N_TEST:
        raise ValueError(f"{class_name}: {n_train} training layouts overlap the last {MSDCNN_N_TEST}")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(RANDOM_SEED)).numpy()
    return {"train": perm[:n_train], "val": perm[:0], "test": perm[n - MSDCNN_N_TEST:]}


def get_split_arrays(class_name: str, split: str, target: str = "co", convention: str = None,
                     msdcnn_train: int = None):
    """images (N,16,16) uint8 in the requested layout convention, spec (N,K,2)
    float32 ((N,2,K,2) for target "both": r_xx, r_yx), class labels, folder
    indices. msdcnn_train: use the MSDCNN-style split with that many training
    layouts (see msdcnn_split_indices)."""
    names = CLASSES if class_name == POOLED_CLASS else [class_name]
    ch = TARGET_CHANNELS[target]
    xs, ys, labels, idxs = [], [], [], []
    for name in names:
        images, curves = load_class(name)
        idx = (msdcnn_split_indices(name, msdcnn_train) if msdcnn_train else class_split_indices(name))[split]
        c = curves[idx][:, list(ch)] if isinstance(ch, tuple) else curves[idx, ch]
        xs.append(to_convention(images[idx], name, convention, idx))
        ys.append(np.stack([c.real, c.imag], axis=-1).astype(np.float32))
        labels += [name] * len(idx)
        idxs.append(idx)
    return np.concatenate(xs), np.concatenate(ys), np.array(labels), np.concatenate(idxs)


def get_dataloaders(class_name: str, batch_size: int, target: str = "co", convention: str = None,
                    n_train: int = None, msdcnn_train: int = None):
    """train / val / test DataLoaders. n_train keeps only the first n_train
    training layouts (the split order is already random), e.g. to match the
    training-set size of another study; validation and test sets are unchanged.
    Data live in RAM, so num_workers=0 (worker processes also exhaust the
    Windows page file)."""
    loaders = []
    for split in ("train", "val", "test"):
        x, y, _, _ = get_split_arrays(class_name, split, target, convention, msdcnn_train)
        if split == "train" and n_train:
            x, y = x[:n_train], y[:n_train]
        ds = TensorDataset(torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y))
        loaders.append(DataLoader(ds, batch_size, shuffle=(split == "train"), num_workers=0,
                                  pin_memory=(DEVICE.type == "cuda")))
    return tuple(loaders)
