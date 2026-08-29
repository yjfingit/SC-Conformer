"""PyTorch port of EEG-ATCNet (Altaheri et al., IEEE TII 2023).

EEGNet-style stem (2 blocks: temporal conv + depthwise spatial conv + pool)
-> TC block (dilated TCN + squeeze-excitation) -> Transformer encoder -> GAP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StemBlock(nn.Module):
    def __init__(self, in_ch, F, D, kt, pool, n_ch, dropout=0.3):
        super().__init__()
        self.tconv = nn.Conv2d(in_ch, F, (1, kt), padding=(0, kt // 2))
        self.bn1 = nn.BatchNorm2d(F)
        self.sconv = nn.Conv2d(F, F * D, (n_ch, 1), groups=F)
        self.bn2 = nn.BatchNorm2d(F * D)
        self.pool = nn.AvgPool2d((1, pool))
        self.drop = nn.Dropout(dropout)
        self.out_ch = F * D

    def forward(self, x):
        x = F.elu(self.bn1(self.tconv(x)))
        x = F.elu(self.bn2(self.sconv(x)))
        return self.drop(self.pool(x))


class TCBlock(nn.Module):
    """Dilated TCN (2 blocks) + squeeze-excitation."""

    def __init__(self, ch, filters, dropout=0.3):
        super().__init__()
        self.tcn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch if i == 0 else ch + filters, filters, 2,
                          dilation=2 ** (i + 1),
                          padding=2 ** (i + 1), ),
                nn.BatchNorm1d(filters), nn.GELU(), nn.Dropout(dropout))
            for i in range(2)])
        out = ch + 2 * filters
        self.se = nn.Sequential(nn.Linear(out, out // 2), nn.ReLU(),
                                nn.Linear(out // 2, out), nn.Sigmoid())
        self.out_ch = out

    def forward(self, x):          # [B, C, T]
        y = x
        for blk in self.tcn:
            y = torch.cat([y, blk(y)[..., : y.shape[-1]]], dim=1)
        w = self.se(y.mean(-1)).unsqueeze(-1)
        return y * w


class ATCNet(nn.Module):
    def __init__(self, n_ch, n_times=1000, n_classes=4, sfreq=250,
                 F1=16, D=2, layers=2, heads=2, dropout=0.3):
        super().__init__()
        pool1, pool2 = 8, 4
        self.stem1 = StemBlock(1, F1, D, 64, pool1, n_ch, dropout)
        self.stem2 = StemBlock(F1 * D, F1, D, 16, pool2, 1, dropout)
        t = (n_times // pool1) // pool2
        self.tc = TCBlock(F1 * D, 32, dropout)
        self.proj = nn.Linear(F1 * D + 64, 64)
        enc_layer = nn.TransformerEncoderLayer(
            64, heads, 128, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):                      # [B, C, T]
        x = x.unsqueeze(1)                     # [B, 1, C, T]
        h = self.stem1(x)
        h = self.stem2(h)                      # [B, F2, 1, T2]
        h = self.tc(h.squeeze(2))              # [B, C', T2]
        h = self.proj(h.transpose(1, 2))       # [B, T2, 64]
        h = self.tr(h)
        return self.head(h.mean(1))
