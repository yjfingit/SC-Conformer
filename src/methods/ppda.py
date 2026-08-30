"""PPDA (Zhao, Yan & Lu, AAAI 2021) — reimplementation from the published
description (no official code was released; documented in the results table).

Published description (from the paper abstract and citing works):
  - disentangle EEG representations into emotion-related *shared* features
    and subject-*private* features (a subject-classification branch makes
    the private features subject-discriminative; an orthogonality penalty
    separates the two subspaces);
  - the shared features feed the emotion classifier;
  - UDA stage: the target subject's *unlabeled* trials are pseudo-labeled by
    the current classifier; class-conditional MMD aligns source and target
    shared features, and entropy minimization sharpens target predictions.

Two versions:
  - ppda-zs : strict zero-shot (no target data whatsoever)
  - ppda-uda: original UDA (uses target unlabeled trials; flagged as such)
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.de_datasets import dataset_subjects, load_subject  # noqa
from src.metrics import all_metrics  # noqa
from src.utils import seed_all  # noqa


class PPDA(nn.Module):
    def __init__(self, n_classes=3, n_subjects=14, d=128, private=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(), nn.Linear(310 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d), nn.ReLU())
        self.emotion = nn.Sequential(nn.Linear(d, n_classes))
        self.private = nn.Sequential(nn.Linear(d, private), nn.ReLU(),
                                     nn.Linear(private, n_subjects))
        self.n_subjects = n_subjects

    def forward(self, x):
        z = self.enc(x)
        return self.emotion(z), self.private(z), z


def mmd_rbf(src, tgt, sigma=1.0):
    def k(a, b):
        d = torch.cdist(a, b) ** 2
        return torch.exp(-d / (2 * sigma ** 2))
    Kss, Ktt, Kst = k(src, src), k(tgt, tgt), k(src, tgt)
    n, m = src.shape[0], tgt.shape[0]
    return (Kss.sum() / (n * (n - 1) + 1e-8) + Ktt.sum() / (m * (m - 1) + 1e-8)
            - 2 * Kst.mean())


def load_fold_data(test_subj, seed=0, val_frac=0.1, with_target=False):
    subjects = [s for s in dataset_subjects() if s != test_subj]
    rng = np.random.RandomState(seed)
    Xs, ys, ss = [], [], []
    for si, s in enumerate(subjects):
        X, y = load_subject(s)
        Xs.append(X); ys.append(y); ss.append(np.full(len(y), si))
    X = np.concatenate(Xs).astype(np.float32)      # [N, 62, 5, W]
    y = np.concatenate(ys); sid = np.concatenate(ss)
    mu = X.mean(axis=(0, 3))[..., None]
    sd = np.maximum(X.std(axis=(0, 3)), 1e-4)[..., None]
    X = (X - mu) / sd
    X = X.reshape(len(X), -1, X.shape[-1]).copy()  # [N, 310, W]
    N = len(y)
    idx = rng.permutation(N)
    n_val = int(N * val_frac)
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    out = dict(
        Xtr=X[tr_idx], ytr=y[tr_idx], str_=sid[tr_idx],
        Xva=X[va_idx], yva=y[va_idx], sva=sid[va_idx],
        subj_map=subjects)
    if with_target:
        Xt, yt = load_subject(test_subj)
        Xt = ((Xt.astype(np.float32) - mu) / sd)
        out["Xt_unl"] = Xt.reshape(len(Xt), -1, Xt.shape[-1]).copy()
        out["yte"] = yt                            # used only for scoring
    return out


def run_ppda(test_subj, seed=0, mode="zs", epochs=30, batch=256, lr=1e-3,
             tau=0.9, w_mmd=1.0, w_ent=0.05, w_priv=0.5, w_orth=0.05,
             device="cuda"):
    assert mode in ("zs", "uda")
    seed_all(seed)
    with_target = mode == "uda"
    D = load_fold_data(test_subj, seed, with_target=with_target)
    model = PPDA(n_subjects=len(D["subj_map"])).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr = torch.tensor(D["Xtr"]).float().to(device)
    ytr = torch.tensor(D["ytr"]).to(device)
    str_ = torch.tensor(D["str_"]).to(device)
    Xva = torch.tensor(D["Xva"]).float().to(device)
    yva = torch.tensor(D["yva"]).to(device)
    tr_ds = TensorDataset(Xtr, ytr, str_)
    dl = DataLoader(tr_ds, batch, shuffle=True, drop_last=True)
    if with_target:
        Xu = torch.tensor(D["Xt_unl"]).float().to(device)
        dl_u = DataLoader(TensorDataset(Xu), batch, shuffle=True)
    best_va, best_state, bad = -1, None, 0
    import time
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        it_u = iter(dl_u) if with_target else None
        for xb, yb, sb in dl:
            eb, pb, z = model(xb)
            loss = F.cross_entropy(eb, yb)
            loss = loss + w_priv * F.cross_entropy(pb, sb)
            sim = F.cosine_similarity(
                F.normalize(z, dim=1).unsqueeze(1),
                F.normalize(z, dim=1).unsqueeze(0), dim=-1)
            loss = loss + w_orth * sim.mean()
            if with_target:
                try:
                    xu = next(it_u)[0]
                except StopIteration:
                    it_u = iter(dl_u); xu = next(it_u)[0]
                eb_u, _, z_u = model(xu)
                with torch.no_grad():
                    conf, pseudo = torch.softmax(eb_u, 1).max(1)
                m = conf > tau
                if m.sum() > 8:
                    loss_pseudo = F.cross_entropy(eb_u[m], pseudo[m])
                    loss_ent = -(torch.softmax(eb_u, 1).max(1).values
                                 .log().mean())
                    mmd = mmd_rbf(z[: xu.shape[0]], z_u)
                    loss = loss + w_mmd * (mmd + loss_pseudo) + w_ent * loss_ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            eb, _, _ = model(Xva)
            vb = (eb.argmax(1) == yva).float().mean().item()
        if vb > best_va:
            best_va, bad = vb, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    Xte = torch.tensor(D["Xt_unl"] if with_target
                       else np.concatenate(
                           [load_subject(s)[0].astype(np.float32)
                            for s in [test_subj]]))
    if not with_target:
        # strict: normalize test trials with TRAINING statistics
        subjects = [s for s in dataset_subjects() if s != test_subj]
        Xpool = np.concatenate([load_subject(s)[0].astype(np.float32)
                                for s in subjects])
        mu = Xpool.mean(axis=(0, 3))[..., None]
        sd = np.maximum(Xpool.std(axis=(0, 3)), 1e-4)[..., None]
        Xt, yte = load_subject(test_subj)
        Xte = torch.tensor(
            ((Xt.astype(np.float32) - mu) / sd).reshape(
                len(Xt), -1, Xt.shape[-1]).copy()).float().to(device)
    else:
        yte = D["yte"]
    with torch.no_grad():
        pred = []
        for i in range(0, len(Xte), 1024):
            eb, _, _ = model(Xte[i:i + 1024])
            pred.append(eb.argmax(1).cpu())
        pt = torch.cat(pred).numpy()
    m = all_metrics(np.array(yte), pt, 3)
    m.update(model=f"ppda-{'uda' if with_target else 'zs'}",
             test_subject=int(test_subj), seed=seed, train_time=time.time() - t0,
             note="reimplementation from published description; no official code")
    return m
