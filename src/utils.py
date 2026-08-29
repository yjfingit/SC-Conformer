import os
import random

import numpy as np


def set_mne_data():
    path = os.environ.get("ICLR_DATA", "/root/autodl-tmp/ICLR/data/processed")
    os.makedirs(path, exist_ok=True)
    mne_root = os.environ.get("MNE_DATA", "/root/autodl-tmp/ICLR/data/mne_data")
    os.environ["MNE_DATA"] = mne_root
    os.makedirs(mne_root, exist_ok=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def balanced_accuracy(y_true, y_pred, n_classes):
    """Macro recall, MOABB convention."""
    accs = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == c).mean())
    return float(np.mean(accs))


def kappa(y_true, y_pred, n_classes):
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    n = conf.sum()
    if n == 0:
        return 0.0
    po = np.trace(conf) / n
    pe = (conf.sum(0) * conf.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe + 1e-12))
