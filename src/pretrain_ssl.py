"""Self-supervised masked-reconstruction pretraining for SCFormer.

Pretrains on the training subjects' (unlabeled) trials: mask random
time patches, replace with zeros, encode, reconstruct stem features.
Returns a state_dict to seed LOSO fine-tuning.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import dataset_subjects, load_subject  # noqa
from src.utils import seed_all  # noqa


class RawTrials(Dataset):
    def __init__(self, ds, subjects):
        Xs, meds, mads = [], [], []
        for s in subjects:
            X, y, m, d = load_subject(ds, s)
            Xs.append(X); meds.append(m); mads.append(d)
        self.X = np.concatenate(Xs)
        self.med = np.median(np.stack(meds), 0)
        self.mad = np.median(np.stack(mads), 0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        return (x - self.med[:, None]) / self.mad[:, None]


def pretrain(model, ds, test_subj, seed=0, epochs=25, batch=128, lr=3e-4,
             mask_ratio=0.5, device="cuda", verbose=False):
    """In-place SSL pretraining of an SCFormer instance. Returns model."""
    from src.models.scformer import rope_cache
    seed_all(seed)
    subjects = [s for s in dataset_subjects(ds) if s != test_subj]
    data = RawTrials(ds, subjects)
    dl = DataLoader(data, batch, shuffle=True, num_workers=4, drop_last=True,
                    persistent_workers=True)
    m = model
    # infer stem feature dim for the reconstruction target
    with torch.no_grad():
        h0 = m.fuse(torch.cat([mod(data[0][None, None].to(device)) for mod in m.stem], 1))
        F_D = h0.shape[1]
    recon = nn.Linear(m.d_model, m.patch * F_D).to(device)
    opt = torch.optim.AdamW(list(m.parameters()) + list(recon.parameters()),
                            lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for x in dl:
            x = x.to(device, non_blocking=True)
            B = x.shape[0]
            with torch.no_grad():
                h = m.fuse(torch.cat([mod(x.unsqueeze(1)) for mod in m.stem], 1))
                h = h.squeeze(2)
                N = h.shape[-1] // m.patch
                h = h[..., : N * m.patch].reshape(B, F_D, N, m.patch)
                target = h.permute(0, 2, 3, 1).flatten(2)  # [B, N, P*F*D]
            tok = m.tok(target) + m.pos[:, :N]
            n_mask = max(1, int(N * mask_ratio))
            idx = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, idx[:, :n_mask], 1.0)
            tok = tok * (1 - mask).unsqueeze(-1)      # zero out masked
            rope = rope_cache(N, m.d_model // 4, x.device) if m.use_rope else None
            st = m.stats(x)
            e = m.stat_glob(st.flatten(1)) if (m.use_sc and m.use_film) \
                else torch.zeros(B, m.d_model, device=x.device)
            for blk in m.blocks:
                tok = blk(tok, e, rope)
            pred = recon(tok)
            denom = mask.sum() * F_D + 1e-8
            loss = ((pred - target.detach()) ** 2
                    * mask.unsqueeze(-1)).sum() / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if verbose:
            print(f"  ssl ep{ep}: {tot / max(nb, 1):.4f}", flush=True)
    return m
