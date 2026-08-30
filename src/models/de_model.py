"""SCDE: our improved model — champion modules transplanted from adjacent
fields and adapted to strict zero-shot cross-subject SEED DE decoding.

Modules (each validated in top-venue prior work):
  1. (channel x band) tokenization with learnable channel/band positional
     embeddings (LaBraM / EEG-Conformer lineage);
  2. Conformer blocks — attention + convolution hybrid (Gulati et al.,
     ICLR 2021, the ASR champion architecture);
  3. Supervised contrastive learning with multi-scale temporal-crop views
     (SupCon, Khosla et al., NeurIPS 2020; multi-scale crops follow the
     MSHCL recipe, IEEE TAFFC 2025);
  4. Hyperbolic (Poincare-ball) contrastive projection (MSHCL, TAFFC 2025);
  5. Subject-adversarial invariance via a gradient-reversal layer (DANN,
     Ganin et al., JMLR 2016) applied to TRAINING subjects only — fully
     compliant with the strict zero-shot protocol (no target data).

Input: [B, 310, W=15] flattened DE windows (62 ch x 5 bands x 15 frames).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def grev(x, lamb=1.0):
    return GradReverse.apply(x, lamb)


def poincare_exp0(z, c=1.0, eps=1e-8):
    norm = torch.clamp(z.norm(dim=-1, keepdim=True), min=eps)
    gamma = torch.clamp(1.0 / math.sqrt(c) * torch.tanh(
        math.sqrt(c) * norm), max=1.0 - 1e-5)
    return gamma * z / norm


def hyp_distance(a, b, c=1.0, eps=1e-8):
    sq = (2 * c * (a - b).norm(dim=-1) ** 2 /
          ((1 - c * a.norm(dim=-1) ** 2) * (1 - c * b.norm(dim=-1) ** 2)
           + eps))
    return torch.clamp(1 + 2 / torch.clamp(sq, min=eps), min=1.0 + eps) \
        .acosh()


class ConformerBlock(nn.Module):
    def __init__(self, d, heads=8, ff=256, k=5, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ffn1 = nn.Sequential(nn.Linear(d, ff), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(ff, d),
                                  nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.ln4 = nn.LayerNorm(d)
        self.ffn2 = nn.Sequential(nn.Linear(d, ff), nn.SiLU(),
                                  nn.Dropout(dropout), nn.Linear(ff, d),
                                  nn.Dropout(dropout))

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.ln1(x))
        a, _ = self.attn(self.ln2(x), self.ln2(x), self.ln2(x),
                         need_weights=False)
        x = x + a
        h = self.bn(self.dw(self.ln3(x).transpose(1, 2))).transpose(1, 2)
        x = x + self.pw2(F.silu(h))
        x = x + 0.5 * self.ffn2(self.ln4(x))
        return x


class SCDE(nn.Module):
    def __init__(self, n_classes=3, n_ch=62, n_band=5, W=15, d=128,
                 depth=4, heads=8, ff=256, dropout=0.15, hyp_c=1.0):
        super().__init__()
        self.n_ch, self.n_band, self.W = n_ch, n_band, W
        self.tok = nn.Linear(W, d)
        self.emb_ch = nn.Parameter(torch.randn(n_ch, d) * 0.02)
        self.emb_band = nn.Parameter(torch.randn(n_band, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(
            [ConformerBlock(d, heads, ff, dropout=dropout)
             for _ in range(depth)])
        self.ln = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d, n_classes))
        self.proj = nn.Sequential(nn.Linear(d, 128), nn.GELU(),
                                  nn.Linear(128, 64))
        self.adv = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 20))   # 20 training subjects
        self.hyp_c = hyp_c

    def tokenize(self, x):
        # x: [B, 310, W] -> [B, 62, 5, W] -> tokens [B, 310, d]
        B = x.shape[0]
        g = x.reshape(B, self.n_ch, self.n_band, self.W)
        g = g.permute(1, 2, 0, 3).reshape(self.n_ch * self.n_band, B,
                                          self.W)
        tok = self.tok(g.permute(1, 0, 2))            # [B, 310, d]
        ch = self.emb_ch.unsqueeze(1).expand(self.n_ch, self.n_band, -1)
        bd = self.emb_band.unsqueeze(0).expand(self.n_ch, self.n_band, -1)
        pos = (ch + bd).reshape(self.n_ch * self.n_band, -1)
        return tok + pos.unsqueeze(0)

    def encode(self, x):
        tok = self.tokenize(x)
        tok = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], 1)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln(tok)
        q = self.query.expand(tok.shape[0], -1, -1)
        att = torch.softmax((tok @ q.transpose(1, 2)) / math.sqrt(tok.shape[-1]),
                            dim=1)
        pooled = (att * tok).sum(1)
        return pooled, tok                             # [B, d]

    def forward(self, x, lamb=0.0):
        pooled, _ = self.encode(x)
        logits = self.head(pooled)
        subj = self.adv(grev(pooled, lamb)) if lamb > 0 else None
        emb = F.normalize(self.proj(pooled), dim=-1)
        return logits, emb, subj


def sup_con_loss(z, y, temperature=0.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    mask = y.unsqueeze(0) == y.unsqueeze(1)
    mask.fill_diagonal_(False)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * mask.float()).sum(1)[mask.any(1)].mean()


def hyp_con_loss(za, zb, y, c=1.0, temperature=0.1):
    ha = poincare_exp0(za, c)
    hb = poincare_exp0(zb, c)
    d = hyp_distance(ha.unsqueeze(1), hb.unsqueeze(0), c)   # [B, B]
    sim = -d / temperature
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp.diag()).mean()                     # paired views positive
