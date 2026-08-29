"""Model registry: all models take [B, C, T] -> [B, n_classes]."""
import src.compat  # noqa: F401  (must precede braindecode)
from braindecode.models import (Deep4Net, EEGConformer, EEGNetv4,
                                ShallowFBCSPNet)

from .atcnet import ATCNet
from .scformer import SCFormer

_SCFORMER_VARIANTS = {
    # ablation flags: (use_sc, use_film, use_ssm, use_rope, use_ms_stem)
    "scformer":      dict(use_sc=True,  use_film=True,  use_ssm=True,
                          use_rope=True,  use_ms_stem=True),
    "scformer+nf":   dict(use_sc=True,  use_film=True,  use_ssm=False,
                          use_rope=True,  use_ms_stem=True),
    "scformer-ff":   dict(use_sc=True,  use_film=False, use_ssm=True,
                          use_rope=True,  use_ms_stem=True),
    "scformer-nsc":  dict(use_sc=False, use_film=False, use_ssm=True,
                          use_rope=True,  use_ms_stem=True),
    "scformer-ms":   dict(use_sc=True,  use_film=True,  use_ssm=True,
                          use_rope=True,  use_ms_stem=False),
    "scformer-sin":  dict(use_sc=True,  use_film=True,  use_ssm=True,
                          use_rope=False, use_ms_stem=True),
}


def build(name, n_ch, n_times, n_classes, sfreq=250):
    kw = dict(n_ch=n_ch, n_times=n_times, n_classes=n_classes, sfreq=sfreq)
    if name == "eegnet":
        return EEGNetv4(n_outputs=n_classes, n_chans=n_ch,
                        n_times=n_times), 1e-3
    if name == "shallow":
        return ShallowFBCSPNet(n_outputs=n_classes, n_chans=n_ch,
                               n_times=n_times), 1e-3
    if name == "deep4":
        return Deep4Net(n_outputs=n_classes, n_chans=n_ch,
                        n_times=n_times), 1e-3
    if name == "conformer":
        return EEGConformer(n_outputs=n_classes, n_chans=n_ch,
                            n_times=n_times), 1e-3
    if name == "atcnet":
        return ATCNet(n_ch=n_ch, n_times=n_times,
                      n_classes=n_classes, sfreq=sfreq), 1e-3
    if name in _SCFORMER_VARIANTS:
        m = SCFormer(**kw, **_SCFORMER_VARIANTS[name])
        return m, 2e-3
    raise ValueError(name)
