"""LOSO training/eval for one (dataset, model, test subject, seed)."""
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

from src.data.datasets import dataset_subjects, make_loso  # noqa: E402
from src.models import build  # noqa: E402
from src.utils import balanced_accuracy, kappa, seed_all  # noqa: E402


def run_fold(ds, model_name, test_subj, seed=0, epochs=40, batch=64,
             lr=None, wd=1e-4, amp=True, patience=8, workers=4, device="cuda"):
    seed_all(seed)
    torch.backends.cudnn.benchmark = True
    tr, va, te = make_loso(ds, test_subj, seed=seed)
    model, lr0 = build(model_name, n_ch=tr.X.shape[1],
                       n_times=tr.X.shape[2], n_classes=int(tr.y.max()) + 1)
    lr = lr or lr0
    model = model.to(device).float()
    n_params = sum(p.numel() for p in model.parameters())

    dl_tr = DataLoader(tr, batch, shuffle=True, num_workers=workers,
                       drop_last=True, persistent_workers=workers > 0)
    dl_va = DataLoader(va, 256, False, num_workers=workers)
    dl_te = DataLoader(te, 256, False, num_workers=workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_steps = epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)

    best_va, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                loss = lossf(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        # val
        model.eval()
        ys, ps = [], []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
            for x, y in dl_va:
                ps.append(model(x.to(device)).float().argmax(1).cpu())
                ys.append(y)
        vb = balanced_accuracy(torch.cat(ys).numpy(), torch.cat(ps).numpy(),
                               int(va.y.max()) + 1)
        if vb > best_va:
            best_va, bad = vb, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        for x, y in dl_te:
            ps.append(model(x.to(device)).float().argmax(1).cpu())
            ys.append(y)
    yte, pte = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    n_cls = int(te.y.max()) + 1
    out = dict(dataset=ds, model=model_name, test_subject=int(test_subj),
               seed=seed, n_params=n_params, train_time=train_time,
               epochs_run=ep + 1, val_bacc=best_va,
               acc=float((yte == pte).mean()),
               bacc=balanced_accuracy(yte, pte, n_cls),
               kappa=kappa(yte, pte, n_cls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="/root/autodl-tmp/ICLR/results")
    args = ap.parse_args()

    subjects = dataset_subjects(args.dataset)
    todo = [args.subject] if args.subject is not None else subjects
    for s in todo:
        r = run_fold(args.dataset, args.model, s, seed=args.seed,
                     epochs=args.epochs, batch=args.batch)
        d = os.path.join(args.out, args.dataset, args.model)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"S{s:02d}_seed{args.seed}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=1)
        print(f"[{args.dataset}/{args.model}] S{s:02d} "
              f"bacc={r['bacc']:.4f} acc={r['acc']:.4f} "
              f"({r['train_time']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
