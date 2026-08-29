"""SCFormer: Statistics-Conditioned Conformer for calibration-free EEG.

Transplanted components (all proven in speech/NLP):
  - multi-scale temporal conv stem (EEGNet/DeepConvNet lineage)
  - Conformer blocks (Gulati et al. 2020, ASR)
  - RoPE (Su et al. 2021)
  - lightweight bidirectional input-gated SSM branch (Mamba-style selective
    gating, pure PyTorch)
Novel component:
  - SCN: Statistics-Conditioned conditioning. Per-channel robust stats
    (median / MAD / band-power ratios) computed on the fly from each test
    trial feed (a) per-channel stat tokens appended to the encoder input
    and (b) adaLN-style FiLM scale/shift per block. No subject identity,
    no calibration data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = [(4, 8), (8, 13), (13, 30), (30, 40)]  # theta alpha beta gamma


# ---------------------------------------------------------------- utilities
def rope_cache(n, d, device, base=10000.0):
    half = d // 2
    freqs = 1.0 / (base ** (torch.arange(half, device=device) / half))
    ang = torch.outer(torch.arange(n, device=device), freqs)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x: [B, H, N, Dh]
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c = cos[: x.shape[-2]].view(1, 1, x.shape[-2], -1)
    s = sin[: x.shape[-2]].view(1, 1, x.shape[-2], -1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ------------------------------------------------------------------- blocks
class MHA(nn.Module):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, d * 3)
        self.out = nn.Linear(d, d)
        self.drop = dropout

    def forward(self, x, rope=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop)
        att = att.transpose(1, 2).reshape(B, N, D)
        return self.out(att)


class ConvModule(nn.Module):
    """Conformer conv module: pw -> GLU -> dw-conv -> BN -> Swish -> pw."""

    def __init__(self, d, k=15, dropout=0.1):
        super().__init__()
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        u, v = self.pw1(x).chunk(2, -1)
        x = u * torch.sigmoid(v)                       # GLU
        x = self.bn(self.dw(x.transpose(1, 2))).transpose(1, 2)
        x = self.drop(x)
        return self.pw2(F.silu(x))


class FFN(nn.Module):
    def __init__(self, d, ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, ff), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff, d),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SCNBlock(nn.Module):
    """Conformer block with adaLN conditioning driven by the stat vector."""

    def __init__(self, d, heads, ff, k=15, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ffn1 = FFN(d, ff, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.att = MHA(d, heads, dropout)
        self.ln3 = nn.LayerNorm(d)
        self.conv = ConvModule(d, k, dropout)
        self.ln4 = nn.LayerNorm(d)
        self.ffn2 = FFN(d, ff, dropout)
        # adaLN: 4 layer-norms x (gamma, beta); zero-init = identity LN
        self.ada = nn.Linear(d, 4 * d * 2)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, s, rope):
        # s: [B, d] global stat embedding
        gb = self.ada(s).view(x.shape[0], 4, 2, x.shape[-1])
        g1, b1 = gb[:, 0, 0, None], gb[:, 0, 1, None]
        g2, b2 = gb[:, 1, 0, None], gb[:, 1, 1, None]
        g3, b3 = gb[:, 2, 0, None], gb[:, 2, 1, None]
        g4, b4 = gb[:, 3, 0, None], gb[:, 3, 1, None]
        x = x + 0.5 * self.ffn1(self.ln1(x) * (1 + g1) + b1)
        x = x + self.att(self.ln2(x) * (1 + g2) + b2, rope)
        x = x + self.conv(self.ln3(x) * (1 + g3) + b3)
        x = x + 0.5 * self.ffn2(self.ln4(x) * (1 + g4) + b4)
        return x


class BiGatedSSM(nn.Module):
    """Bidirectional input-gated recurrency (selective-SSM-lite).

    h_t = a_t * h_{t-1} + (1 - a_t) * W x_t,  a_t = sigmoid(Wg x_t)
    """

    def __init__(self, d, expand=2):
        super().__init__()
        e = d * expand
        self.inp = nn.Linear(d, e)
        self.gate = nn.Linear(d, e)
        self.out = nn.Linear(e, d)

    def _scan(self, x, flip=False):
        if flip:
            x = torch.flip(x, [1])
        Bx = self.inp(x)
        a = torch.sigmoid(self.gate(x))
        h = torch.zeros_like(Bx[:, 0])
        outs = []
        for t in range(x.shape[1]):
            h = a[:, t] * h + (1 - a[:, t]) * Bx[:, t]
            outs.append(h)
        y = torch.stack(outs, 1)
        return torch.flip(y, [1]) if flip else y

    def forward(self, x):
        y = self._scan(x) + torch.flip(self._scan(x, flip=True), [1])
        return self.out(y)


# -------------------------------------------------------------------- model
class SCFormer(nn.Module):
    def __init__(self, n_ch, n_times=1000, n_classes=4, sfreq=250,
                 d_model=96, depth=4, heads=4, ff=192, patch=10, pool=4,
                 F1=16, D=2, ks=(32, 64, 128), dropout=0.2,
                 use_sc=True, use_film=True, use_ssm=True, use_rope=True,
                 use_ms_stem=True):
        super().__init__()
        assert d_model % heads == 0 and (d_model // heads) % 2 == 0
        self.use_sc, self.use_film = use_sc, use_film
        self.use_ssm, self.use_rope = use_ssm, use_rope
        self.patch, self.d_model = patch, d_model
        n_stat = 2 + len(BANDS)
        self.n_ch = n_ch

        # ---- multi-scale temporal stem -> spatial mixing -> pool
        ks = tuple(ks) if use_ms_stem else (64,)
        self.stem = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2)),
            nn.BatchNorm2d(F1), nn.GELU()) for k in ks])
        self.fuse = nn.Sequential(
            nn.Conv2d(F1 * len(ks), F1, 1),
            nn.BatchNorm2d(F1), nn.GELU(),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1),
            nn.BatchNorm2d(F1 * D), nn.GELU(),
            nn.AvgPool2d((1, pool)),
            nn.Dropout(dropout))
        t_after = (n_times // pool) // patch
        self.n_tokens = t_after
        self.tok = nn.Linear((F1 * D) * patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, t_after, d_model) * 0.02)

        # ---- SCN
        if use_sc:
            self.stat_tok = nn.Linear(n_stat, d_model)  # per-channel tokens
            if use_film:
                self.stat_glob = nn.Sequential(
                    nn.Linear(n_ch * n_stat, 256), nn.GELU(),
                    nn.Linear(256, d_model))

        # ---- encoder
        self.blocks = nn.ModuleList([
            SCNBlock(d_model, heads, ff, 15, dropout) for _ in range(depth)])

        # ---- parallel SSM branch
        if use_ssm:
            self.ssm = BiGatedSSM(d_model)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def band_powers(self, x):
        sp = torch.fft.rfft(x, dim=-1).abs() ** 2
        freqs = torch.fft.rfftfreq(x.shape[-1], device=x.device) * 250.0
        bp = [sp[..., (freqs >= lo) & (freqs < hi)].mean(-1)
              for lo, hi in BANDS]
        return torch.log1p(torch.stack(bp, -1))        # [B, C, 4]

    def stats(self, x):
        """Robust per-channel deviation profile of the (normalized) trial."""
        med = x.median(-1).values                      # [B, C]
        mad = (x - med[..., None]).abs().median(-1).values
        return torch.cat([med[..., None], mad[..., None],
                          self.band_powers(x)], -1)    # [B, C, 6]

    def forward(self, x):               # [B, C, T]
        B, C, T = x.shape
        s_tok, e = None, None
        if self.use_sc:
            st = self.stats(x)
            s_tok = self.stat_tok(st)                  # [B, C, d]
            if self.use_film:
                e = self.stat_glob(st.flatten(1))      # [B, d]
        if e is None:
            e = torch.zeros(B, self.d_model, device=x.device)

        h = self.fuse(torch.cat([m(x.unsqueeze(1)) for m in self.stem], 1))
        h = h.squeeze(2)                               # [B, F*D, T']
        N = h.shape[-1] // self.patch
        h = h[..., : N * self.patch].reshape(B, h.shape[1], N, self.patch)
        h = h.permute(0, 2, 3, 1).flatten(2)           # [B, N, P*F*D]
        tok = self.tok(h) + self.pos[:, : h.shape[1]]

        if self.use_ssm:                                # parallel branch
            tok = tok + self.ssm(self.ln_f(tok))

        if s_tok is not None:
            tok = torch.cat([s_tok, tok], 1)

        rope = rope_cache(tok.shape[1], self.d_model // 4, x.device) \
            if self.use_rope else None
        for blk in self.blocks:
            tok = blk(tok, e, rope)

        feat = tok[:, C:] if s_tok is not None else tok  # drop stat tokens
        feat = self.ln_f(feat.mean(1))
        return self.head(feat)
