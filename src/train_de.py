"""Unified strict-zero-shot LOSO runner for SEED DE features.

Usage: python src/train_de.py --model eegnet-de --subject 3 --seed 0
Results saved to results_de/{model}/S{subj:02d}_seed{seed}.json with the
full metric suite; per-(subject) rows aggregate into paper tables.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.de_datasets import dataset_subjects, make_loso  # noqa: E402
from src.metrics import aggregate, all_metrics, format_summary  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from src.utils import seed_all  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results_de")

AUG = dict(noise=0.05, t_mask=0.1)


class SqueezeTime(nn.Module):
    """Averages the trailing time dim if the backbone keeps it."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = self.m(x)
        return y.mean(-1) if y.dim() == 3 else y


def build(name, n_classes=3):
    import src.compat  # noqa: F401
    from braindecode.models import EEGNetv4, Deep4Net
    if name == "eegnet-de":
        # EEGNet adapted to DE input: 310 feature-"channels" x 4 time steps;
        # temporal kernels/pools shrunk to the 4-step window.
        return SqueezeTime(EEGNetv4(
            n_outputs=n_classes, n_chans=310, n_times=15,
            kernel_length=8, depthwise_kernel_length=8,
            pool1_kernel_size=4, pool2_kernel_size=2,
            final_conv_length=1)), 1e-3
    if name == "deep4-de":
        return SqueezeTime(Deep4Net(
            n_outputs=n_classes, n_chans=310, n_times=15,
            filter_time_length=5, pool_time_length=3,
            pool_time_stride=2, filter_length_2=5,
            filter_length_3=3, filter_length_4=3,
            final_conv_length=1)), 1e-3
    if name == "tsception":
        from src.models.tsception_de import TSceptionDE
        return TSceptionDE(n_classes=n_classes, W=15), 1e-3
    if name == "scde":  # our improved model (registered separately)
        from src.models.de_model import SCDE
        return SCDE(n_classes=n_classes, W=15), 3e-4
    raise ValueError(name)


def run_torch_fold(model_name, test_subj, seed=0, epochs=30, batch=256,
                   lr=None, wd=1e-4, patience=6, device="cuda"):
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    tr, va, te = make_loso(test_subj, seed=seed, aug=AUG)
    model, lr0 = build(model_name)
    lr = lr or lr0
    model = model.to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl_tr = DataLoader(tr, batch, shuffle=True, num_workers=4, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(va, 512, False, num_workers=2)
    dl_te = DataLoader(te, 512, False, num_workers=2)
    n_steps = epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=True)
    best_va, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for x, y in dl_va:
                ps.append(model(x.to(device)).float().argmax(1).cpu()); ys.append(y)
        import numpy as np
        vb = balanced_accuracy_score(torch.cat(ys), torch.cat(ps))
        if vb > best_va:
            best_va, bad = vb, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl_te:
            ps.append(model(x.to(device)).float().argmax(1).cpu()); ys.append(y)
    yt, pt = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    m = all_metrics(yt, pt, 3)
    m.update(model=model_name, test_subject=int(test_subj), seed=seed,
             train_time=time.time() - t0, epochs_run=ep + 1)
    return m


def run_majority(test_subj, seed=0):
    from src.data.de_datasets import load_subject, dataset_subjects
    subjects = [s for s in dataset_subjects() if s != test_subj]
    ys = np.concatenate([load_subject(s)[1] for s in subjects])
    maj = int(np.bincount(ys).argmax())
    yte = load_subject(test_subj)[1]
    return all_metrics(yte, np.full(len(yte), maj), 3) | dict(
        model="majority", test_subject=int(test_subj), seed=seed)


def run_riemannian(test_subj, seed=0, mode="mdm"):
    """MDM (or tangent-space LR) per band; band log-probs averaged."""
    from src.data.de_datasets import load_subject, dataset_subjects
    import pyriemann
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM, TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    seed_all(seed)
    subjects = [s for s in dataset_subjects() if s != test_subj]
    Xtr = np.concatenate([load_subject(s)[0].astype(np.float32)
                          for s in subjects])       # [N, 62, 5, W]
    ytr = np.concatenate([load_subject(s)[1] for s in subjects])
    step = 4  # Riemannian baselines subsample training windows (cost note)
    Xtr, ytr = Xtr[::step], ytr[::step]
    Xte, yte = load_subject(test_subj)
    Xte = Xte.astype(np.float32)
    # per-band covariances: [N, 62, W] -> cov [N, 62, 62]
    probs = []
    for b in range(5):
        ctr = Covariances(estimator="oas").transform(Xtr[:, :, b])
        cte = Covariances(estimator="oas").transform(Xte[:, :, b])
        if mode == "mdm":
            # log-Euclidean MDM: closed-form mean, fast and standard
            clf = MDM(metric={"mean": "logeuclid", "distance": "logeuclid"})
        else:
            clf = Pipeline([("ts", TangentSpace(metric="riemann")),
                            ("lr", LogisticRegression(max_iter=300, C=1.0))])
        clf.fit(ctr, ytr)
        if hasattr(clf, "predict_proba"):
            probs.append(clf.predict_proba(cte))
        else:
            p = np.zeros((len(cte), 3))
            pr = clf.predict(cte)
            p[np.arange(len(cte)), pr] = 1.0
            probs.append(p)
    pt = np.sum(probs, axis=0).argmax(1)
    m = all_metrics(np.array(yte), pt, 3)
    m.update(model=f"riemannian-{mode}", test_subject=int(test_subj), seed=seed)
    return m


def run_personal_zscore(test_subj, seed=0):
    """Strict Personal-Zscore: training-set z-score + linear classifier.

    The paper's per-subject z-score uses target statistics, which the
    strict zero-shot protocol forbids; the strict version therefore
    degenerates to training-set statistics (documented in the paper).
    """
    from src.data.de_datasets import load_subject, dataset_subjects
    from sklearn.svm import LinearSVC
    seed_all(seed)
    subjects = [s for s in dataset_subjects() if s != test_subj]
    Xtr = np.concatenate([load_subject(s)[0].astype(np.float32)
                          for s in subjects])
    ytr = np.concatenate([load_subject(s)[1] for s in subjects])
    mu = Xtr.mean(axis=(0, 3), keepdims=True)
    sd = np.maximum(Xtr.std(axis=(0, 3), keepdims=True), 1e-4)
    Xn = ((Xtr - mu) / sd).reshape(len(Xtr), -1)
    clf = LinearSVC(C=0.01, max_iter=2000, dual="auto")
    clf.fit(Xn, ytr)
    Xte, yte = load_subject(test_subj)
    Xte = ((Xte.astype(np.float32) - mu) / sd).reshape(len(Xte), -1)
    pt = clf.predict(Xte)
    m = all_metrics(np.array(yte), pt, 3)
    m.update(model="personal-zscore-strict", test_subject=int(test_subj),
             seed=seed)
    return m


def run_scde_fold(test_subj, seed=0, epochs=30, batch=128, lr=3e-4,
                  wd=1e-4, patience=6, device="cuda",
                  w_supcon=0.5, w_hyp=0.5, w_adv=0.05):
    """SCDE training: two multi-scale views, SupCon + hyperbolic contrastive
    + subject-adversarial (training subjects only) + CE."""
    import torch.nn.functional as F
    from src.models.de_model import (SCDE, sup_con_loss, hyp_con_loss,
                                     grev)
    from sklearn.metrics import balanced_accuracy_score as _bas
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    tr, va, te = make_loso(test_subj, seed=seed, aug=AUG)
    model = SCDE(n_classes=3, W=tr.X.shape[-1]).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl_tr = DataLoader(tr, batch, shuffle=True, num_workers=4, drop_last=True,
                       persistent_workers=True)
    dl_va = DataLoader(va, 512, False, num_workers=2)
    dl_te = DataLoader(te, 512, False, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(dl_tr)), pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_va, best_state, bad = -1, None, 0
    t0 = time.time()
    for ep in range(epochs):
        p = ep / max(1, epochs - 1)
        lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        model.train()
        for x, y, sid in dl_tr:
            x, y = x.to(device), y.to(device)
            va_, vb_ = x.clone(), x.clone()
            va_[..., 10:] = 0          # multi-scale crop view A (frames 0-9)
            vb_[..., :10] = 0          # view B (frames 5-14, zero-padded)
            la, ea, _ = model(va_)
            lb, eb, subj = model(vb_, lamb=lamb)
            loss = (lossf(la, y) + lossf(lb, y)
                    + w_supcon * (sup_con_loss(ea, y) +
                                  sup_con_loss(eb, y))
                    + w_hyp * hyp_con_loss(ea, eb, y))
            if subj is not None:
                loss = loss + w_adv * F.cross_entropy(subj, sid.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in dl_va:
                l, _, _ = model(x.to(device))
                ps.append(l.argmax(1).cpu()); ys.append(y)
        vb = _bas(torch.cat(ys).numpy(), torch.cat(ps).numpy())
        if vb > best_va:
            best_va, bad = vb, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in dl_te:
            l, _, _ = model(x.to(device))
            ps.append(l.argmax(1).cpu()); ys.append(y)
    yt, pt = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    m = all_metrics(yt, pt, 3)
    m.update(model="scde", test_subject=int(test_subj), seed=seed,
             train_time=time.time() - t0, epochs_run=ep + 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--riemannian-mode", default="mdm",
                    choices=["mdm", "tslr"])
    ap.add_argument("--ppda-mode", default="zs", choices=["zs", "uda"])
    args = ap.parse_args()

    subjects = dataset_subjects()
    todo = [args.subject] if args.subject is not None else subjects
    per_subject = []
    for s in todo:
        if args.model == "majority":
            r = run_majority(s, args.seed)
        elif args.model.startswith("riemannian"):
            r = run_riemannian(s, args.seed, args.riemannian_mode)
        elif args.model == "personal-zscore":
            r = run_personal_zscore(s, args.seed)
        elif args.model.startswith("ppda"):
            from src.methods.ppda_paper import run_ppda_paper
            r = run_ppda_paper(s, args.seed,
                               mode="uda" if args.model == "ppda-uda" else "zs")
        elif args.model == "scde":
            r = run_scde_fold(s, args.seed, args.epochs, args.batch)
        else:
            r = run_torch_fold(args.model, s, args.seed, args.epochs,
                               args.batch)
        d = os.path.join(RESULTS, args.model)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"S{s:02d}_seed{args.seed}.json"), "w") as f:
            json.dump(r, f, indent=1)
        per_subject.append(r)
        print(f"[{args.model}] S{s:02d} acc={r['acc']*100:.2f} "
              f"bacc={r['bacc']*100:.2f} f1={r['f1']*100:.2f} "
              f"kappa={r['kappa']*100:.2f}", flush=True)
    if len(per_subject) == len(subjects):
        agg = aggregate(per_subject)
        print(format_summary(agg, len(per_subject)))


if __name__ == "__main__":
    main()
