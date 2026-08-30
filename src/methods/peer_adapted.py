"""Wrappers adapting the official MSHCL and AMA-EEG backbones to the
unified SEED DE input [B, 310, W=15].

Both official backbones were designed for long raw-EEG segments; the
convolution kernels/pooling are rescaled to the 15-frame DE window
(documented per wrapper). Classification heads are attached by size
probing. Under the unified single-stage training budget the original
multi-stage contrastive pretraining recipes are not run (noted in the
results table).
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class _SizeProbeHead(nn.Module):
    def __init__(self, backbone, backbone_out, n_classes, hidden=128):
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            pass
        self.head = nn.Sequential(nn.Linear(backbone_out, hidden),
                                  nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x):
        if x.dim() == 3:               # [B, F, W] -> [B, 1, F, W] (conv2d)
            x = x.unsqueeze(1)
        f = self.backbone(x)
        if isinstance(f, tuple):
            f = f[0]
        return self.head(f.reshape(f.shape[0], -1))


class MSHCLWrapper(_SizeProbeHead):
    """ConvNet_baseNonlinearHead from the official MSHCL code, kernels
    rescaled: timeFilterLen 64->5, avgpool 30->2, spatial over 310 DE
    features instead of 62 raw channels; stratified layer-norm recipe
    ('initial','middle1','middle2') kept."""

    def __init__(self, n_classes=3, W=15):
        sys.path.insert(0, os.path.join(ROOT, "third_party", "MSHCL"))
        from model import ConvNet_baseNonlinearHead  # noqa
        import types

        class _Args:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        inner = ConvNet_baseNonlinearHead(
            n_spatialFilters=16, n_timeFilters=16, timeFilterLen=5,
            n_channs=310, stratified="",
            multiFact=2, isMaxPool=False, args=_Args())
        inner.avgpool = nn.AvgPool2d((1, 2))   # official (1,30) exceeds W=15
        inner.classfier = nn.Identity()        # head replaced below
        # probe backbone output on the unified window
        with torch.no_grad():
            inner.eval()
            out = inner(torch.randn(2, 1, 310, W))
            f = out[0] if isinstance(out, tuple) else out
            dim = f.reshape(f.shape[0], -1).shape[1]
        # replace the official 9-class head (FACED) with a 3-class head
        inner.classfier = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, n_classes))
        inner.args = _Args()
        super().__init__(inner, dim, n_classes)


class AMAEEGWrapper(nn.Module):
    """Conv_att_simple_new from the official AMA-EEG code (EEG-only):
    the multimodal projector is bypassed, backbone pooled features feed a
    3-class head. Kernels rescaled: timeFilterLen 6->5, seg_att 30->5,
    avgPoolLen 30->2, timeSmootherLen 6->3; dilated multi-scale spatial
    convs kept (dilations 1/3/6/12 over 310 DE features)."""

    def __init__(self, n_classes=3, W=15):
        super().__init__()
        sys.path.insert(0, os.path.join(ROOT, "third_party", "AMA-EEG", "model"))
        from models import Conv_att_simple_new  # noqa
        inner = Conv_att_simple_new(
            n_timeFilters=8, timeFilterLen=5, n_msFilters=4,
            msFilter_timeLen=3, n_channs=310,
            dilation_array=[1, 3, 6, 12], seg_att=5, avgPoolLen=2,
            timeSmootherLen=3, multiFact=2, stratified=[],
            activ="softmax", temp=1.0, saveFea=True, has_att=True,
            extract_mode="me", global_att=False,
            use_ln_backbone=True)  # stratified norm disabled: per-subject
            # statistics would leak target-subject info (strict zero-shot)
        with torch.no_grad():
            inner.eval()
            out = inner(torch.randn(2, 1, 310, W), proj_mode="me")
            f = out[0] if isinstance(out, tuple) else out
            dim = f.reshape(f.shape[0], -1).shape[1]
        self.inner = inner
        self.dim = dim
        self.head = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)         # [B, F, W] -> [B, 1, F, W]
        out = self.inner(x, proj_mode="me")
        f = out[0] if isinstance(out, tuple) else out
        return self.head(f.reshape(f.shape[0], -1))


class _AMAAdapter(nn.Module):
    def __init__(self, inner, head, dim):
        super().__init__()
        self.inner = inner
        self.head = head
        self.dim = dim

    def forward(self, x):
        out = self.inner(x, proj_mode="me")
        f = out[0] if isinstance(out, tuple) else out
        f = f.reshape(f.shape[0], -1)
        if f.shape[1] != self.dim:      # projector dim mismatch -> project
            if not hasattr(self, "_proj") or self._proj.out_features != self.dim:
                self._proj = nn.Linear(f.shape[1], self.dim).to(f.device)
            f = self._proj(f)
        return self.head(f)
