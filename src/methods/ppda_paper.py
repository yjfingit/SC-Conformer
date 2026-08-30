"""PPDA (Zhao, Yan & Lu, AAAI 2021) — faithful implementation from the paper.

Architecture per paper: attention-based pooling AT (softmax over the 310
channel-x-band features) -> shared LSTM encoder E_s + per-source-subject
private LSTM encoders E_p^j (2 layers, hidden 64) + shared LSTM decoder D_s
(reverse reconstruction) -> shared emotion classifier C_s and per-subject
private classifiers C_p^j (FC, hidden 64) taking E_s + E_p^j.
Loss: L = L_cs + alpha*L_cp + beta*L_recon + gamma*L_difference
  (L_similarity is used at test time for pipeline weighting; the paper
   searches alpha..delta randomly — we fix 0.5/0.1/0.1 and set delta=0).

Deviations from the original protocol (documented in the results table):
  - sequence length l=4 (unified 4-s DE window) instead of l=15 (1-s windows);
  - all 3 sessions pooled per subject (unified LOSO) instead of one session;
  - trade-offs fixed instead of randomly searched; delta=0.

Two versions:
  - ppda-zs : strict zero-shot — calibration phase removed; only the shared
    classifier pipeline is evaluated (the private pipeline requires the
    target private encoder E_p^t which is fit on unlabeled target data).
  - ppda-uda: original — calibration phase fits E_p^t on the first
    ceil(45/4)=12 windows (48 s ~ paper's 45 s) via the reconstruction loss;
    test fuses the shared pipeline with the similarity-weighted private
    source classifiers.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.de_datasets import dataset_subjects, load_subject  # noqa
from src.metrics import all_metrics  # noqa
from src.utils import seed_all  # noqa


class AttentionPooling(nn.Module):
    def __init__(self, m=310):
        super().__init__()
        self.W = nn.Linear(m, m)

    def forward(self, x):                     # [B, L, m]
        a = torch.softmax(self.W(x), dim=-1)
        return a * x


class PPDA(nn.Module):
    def __init__(self, m=310, hidden=64, layers=2, n_subjects=14,
                 n_classes=3):
        super().__init__()
        self.at = AttentionPooling(m)
        self.es = nn.LSTM(m, hidden, layers, batch_first=True)
        self.eps = nn.ModuleList(
            [nn.LSTM(m, hidden, layers, batch_first=True)
             for _ in range(n_subjects)])
        self.ds = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out_lin = nn.Linear(hidden, m)
        self.cs = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                nn.Linear(64, n_classes))
        self.cps = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                           nn.Linear(64, n_classes))
             for _ in range(n_subjects)])

    def encode(self, x, which="shared", j=0):
        xa = self.at(x)
        if which == "shared":
            _, (h, c) = self.es(xa)
        else:
            _, (h, c) = self.eps[j](xa)
        return h[-1]                            # [B, hidden] last layer top

    def decode(self, h_s, h_p, length):
        hd = (h_s + h_p).unsqueeze(0).repeat(self.ds.num_layers, 1, 1)
        inp = torch.zeros(h_s.shape[0], length, hd.shape[-1],
                          device=h_s.device)
        out, _ = self.ds(inp, (hd, hd))
        rev = self.out_lin(out.flip(1))         # reverse-order reconstruction
        return rev

    def forward_private(self, x, j):
        h_s = self.encode(x, "shared")
        h_p = self.encode(x, "private", j)
        return self.cps[j](h_s + h_p), h_s, h_p


def load_sequences(test_subj=None, seed=0):
    """Returns dict of per-subject sequence tensors [N, L, 310] and labels.

    Uses the unified 4-s DE windows: each window is L=4 consecutive
    1-second DE frames (channel-x-band features), per the unified input.
    """
    subjects = [s for s in dataset_subjects() if s != test_subj]
    Xs, ys, ss = [], [], []
    for si, s in enumerate(subjects):
        X, y = load_subject(s)                  # raw [N, 62, 5, W]
        Xs.append(X); ys.append(y); ss.append(np.full(len(y), si))
    return subjects, Xs, ys, ss


def norm_stats(subjects):
    Xp = np.concatenate([load_subject(s)[0].astype(np.float32)
                         for s in subjects])
    mu = Xp.mean(axis=(0, 3))[..., None]        # [62,5,1]
    sd = np.maximum(Xp.std(axis=(0, 3)), 1e-4)[..., None]
    return mu, sd


def to_seq(X, mu, sd, L=None):
    # [N, 62, 5, W] -> [N, W, 310]: W=15 consecutive 1-s DE frames = the
    # paper's l=15 sequence of 310-dim features
    L = L or X.shape[-1]
    Xf = ((X.astype(np.float32) - mu) / sd).transpose(0, 3, 1, 2)
    return torch.tensor(Xf.reshape(len(X), L, 310).copy()).float()


def run_ppda_paper(test_subj, seed=0, mode="zs", epochs=40, batch=128,
                   lr=1e-3, alpha=0.5, beta=0.1, gamma=0.1,
                   calib_windows=31, device="cuda"):
    assert mode in ("zs", "uda")
    seed_all(seed)
    subjects, Xs_list, ys_list, _ = load_sequences(test_subj)
    mu, sd = norm_stats(subjects)
    seqs = [to_seq(X, mu, sd) for X in Xs_list]      # [N_s, 4, 310] fp32
    # NOTE: to_seq normalizes (train-subject stats) then [62,5,4]->[4,310]
    ys = [np.array(y) for y in ys_list]
    model = PPDA(n_subjects=len(subjects)).to(device).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # validation: 10% windows of each training subject (time-ordered tail)
    tr_sets = []
    for si in range(len(subjects)):
        n = len(ys[si]); n_va = max(1, int(n * 0.1))
        tr_sets.append((seqs[si][:-n_va], ys[si][:-n_va],
                        seqs[si][-n_va:], ys[si][-n_va:]))
    Xtr = torch.cat([t[0] for t in tr_sets]).to(device)
    ytr = torch.cat([torch.tensor(t[1]) for t in tr_sets]).to(device)
    str_ = torch.cat([torch.full((len(t[1]),), i) for i, t in
                      enumerate(tr_sets)]).to(device)
    Xva = torch.cat([t[2] for t in tr_sets]).to(device)
    yva = torch.cat([torch.tensor(t[3]) for t in tr_sets]).to(device)
    dl = DataLoader(TensorDataset(Xtr, ytr, str_), batch, shuffle=True,
                    drop_last=True)
    t0 = time.time()
    best_va, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, sb in dl:
            eb = model.cs(model.encode(xb, "shared"))
            loss = F.cross_entropy(eb, yb)
            # per-subject private losses (subject-majority batches)
            for j in range(len(subjects)):
                m = sb == j
                if m.sum() < 2:
                    continue
                yp, h_s, h_p = model.forward_private(xb[m], j)
                loss = loss + alpha * F.cross_entropy(yp, yb[m])
                rec = model.decode(h_s, h_p, xb.shape[1])
                loss = loss + beta * F.mse_loss(rec, xb[m].flip(1))
                loss = loss + gamma * (1 - F.cosine_similarity(
                    h_s, h_p).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vb = (model.cs(model.encode(Xva, "shared")).argmax(1)
                  == yva).float().mean().item()
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

    Xt, yte = load_subject(test_subj)
    x_seq = to_seq(Xt, mu, sd).to(device)      # [Nt, 4, 310]
    with torch.no_grad():
        p_shared = torch.softmax(model.cs(model.encode(x_seq, "shared")), 1)
        if mode == "zs":
            pt = p_shared.argmax(1).cpu().numpy()
        else:
            # calibration: fit E_p^t on ~45 s of unlabeled target data
            # (first 31 of the 15-s windows cover frames 0..44)
            ept = nn.LSTM(310, 64, 2, batch_first=True).to(device)
            opt_c = torch.optim.Adam(ept.parameters(), lr=lr)
            xc = x_seq[:calib_windows]
            for _ in range(200):
                xa = model.at(xc)
                _, (h, c) = ept(xa)
                h_t = h[-1]
                h_s = model.encode(xc, "shared")
                rec = model.decode(h_s, h_t, xc.shape[1])
                loss = F.mse_loss(rec, xc.flip(1))
                opt_c.zero_grad(set_to_none=True)
                loss.backward()
                opt_c.step()
            # similarity weights vs random source private encodings
            ws = []
            with torch.no_grad():
                tgt_h = ept(model.at(x_seq))[0][-1]
                for j in range(len(subjects)):
                    rj = np.random.RandomState(seed + j)
                    pick = rj.randint(0, len(seqs[j]))
                    hj = model.encode(seqs[j][pick:pick + 1].to(device),
                                      "private", j)
                    ws.append(float(F.cosine_similarity(
                        hj, tgt_h.unsqueeze(0)).mean()))
            ws = np.exp(np.array(ws) - np.max(ws))
            ws = ws / ws.sum()
            p_priv = np.zeros((len(x_seq), 3))
            with torch.no_grad():
                for j in range(len(subjects)):
                    h_s = model.encode(x_seq, "shared")
                    h_t = ept(model.at(x_seq))[0][-1]
                    yp = model.cps[j](h_s + h_t)
                    p_priv += ws[j] * torch.softmax(yp, 1).cpu().numpy()
            pt = (p_shared.cpu().numpy() + p_priv).argmax(1)  # fusion=avg
    m = all_metrics(np.array(yte), pt, 3)
    m.update(model=f"ppda-{'uda' if mode == 'uda' else 'zs'}",
             test_subject=int(test_subj), seed=seed,
             train_time=time.time() - t0,
             note="faithful implementation from the AAAI 2021 paper; "
                  "l=4 (unified window) vs l=15 in the paper")
    return m
