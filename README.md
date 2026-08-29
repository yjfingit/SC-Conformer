# SC-Conformer: Statistics-Conditioned Conformer for Calibration-Free EEG Decoding

Code + experiment tracking for an ICLR 2027 submission on cross-subject
(catalogue-free) EEG motor-imagery decoding.

**Idea.** Transplant architecture components proven in speech/NLP — a
multi-scale temporal conv stem, Conformer blocks, RoPE, and a lightweight
bidirectional input-gated SSM branch — into an EEG backbone, and make the
backbone *condition itself* on robust per-channel statistics (median / MAD /
band-power ratios) computed on the fly from each test trial (**SCN**).
No subject identity, no calibration data: the network learns, at training
time, how to normalise itself given the deviation profile of an unseen
subject. The same mechanism is positioned as a general answer to
"test-time individual drift without identity".

**Protocol.** Strict leave-one-subject-out (LOSO) via MOABB on four
standard datasets, all baselines re-run under one identical pipeline:

| dataset | subjects | classes | channels |
|---|---|---|---|
| BCI IV-2a (`BNCI2014_001`) | 9 | 4 | 22 |
| BCI IV-2b (`BNCI2014_004`) | 9 | 2 | 3 |
| HGD (`Schirrmeister2017`) | 14 | 4 | 128 |
| OpenBMI (`Lee2019_MI`) | 54 | 2 | 62 |

## Layout

```
src/
  data/prep.py        # MOABB -> per-subject npz (4-40Hz, 250Hz, 4s epochs)
  data/datasets.py    # LOSO splits, augmentations, robust normalisation
  models/             # baselines (braindecode + ATCNet port) + SCFormer
  train.py            # one LOSO fold: train + val early-stop + test
run_all.py            # full experiment grid orchestrator
aggregate.py          # results/*.json -> paper tables (mean±std)
paper/                # ICLR 2027 LaTeX
results/              # per-fold JSON results (committed)
```

## Reproduce

```bash
pip install torch torchvision moabb braindecode einops scikit-learn
export ICLR_DATA=/root/autodl-tmp/ICLR/data/processed
python -m src.data.prep --datasets bnci_2a bnci_2b hgd openbmi
python run_all.py --datasets bnci_2a --models eegnet conformer atcnet scformer
python aggregate.py
```
