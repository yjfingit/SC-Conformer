"""Foundation-model comparison group.

CBraMod (Wang et al., ICLR 2025) fine-tuned under our identical LOSO
protocol. Official backbone + weights; input adapter resamples our 250 Hz
trials to their 4x200-patch layout.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbramod_official import CBraMod

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "third_party", "weights", "cbramod.pth")


class CbraModFT(nn.Module):
    """[B, C, T] @250Hz -> resample 800 -> [B, 22, 4, 200] -> CBraMod -> head."""

    def __init__(self, n_ch, n_times=1000, n_classes=4, freeze=False):
        super().__init__()
        self.n_ch, self.n_times = n_ch, n_times
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=4,
                                n_layer=12, nhead=8)
        sd = torch.load(WEIGHTS, map_location="cpu")
        self.backbone.load_state_dict(sd)
        self.backbone.proj_out = nn.Identity()
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(200, n_classes))

    def forward(self, x):                        # [B, C, T] @250Hz
        x = F.interpolate(x, size=800, mode="linear", align_corners=False)
        b, c, t = x.shape
        x = x.view(b, c, 4, 200)                 # [B, C, S, P]
        feats = self.backbone(x)                 # [B, C, S, D]
        feats = feats.permute(0, 3, 1, 2)        # [B, D, C, S]
        return self.head(feats)


class CbraModLinear(CbraModFT):
    """Linear probe: frozen backbone, train head only."""

    def __init__(self, n_ch, n_times=1000, n_classes=4):
        super().__init__(n_ch, n_times, n_classes, freeze=True)
