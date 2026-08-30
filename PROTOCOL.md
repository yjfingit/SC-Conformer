# Unified SEED Zero-Shot Cross-Subject Protocol (v1.0)

## Data
- SEED, official `ExtractedFeatures` DE-LDS features (62 ch x 5 bands,
  1-s non-overlapping windows; 0=neutral 1=positive... mapped to
  0/1/2 = negative/neutral/positive).
- Unified sample: sliding window of **15 consecutive 1-s DE frames**
  (stride 1 s, never crossing clip boundaries) -> [62, 5, 15] per sample.
  Window chosen so PPDA's native sequence length l=15 is preserved.
- ~9,552 samples per subject (identical for all subjects; clips are fixed).

## Split & fairness rules
- 15-fold leave-one-subject-out (LOSO); test subject fully unseen.
- **Strict zero-shot**: no target-subject labels AND no target-subject
  unlabeled data for any method, except PPDA-UDA whose calibration phase
  (per its paper) uses the first ~45 s of unlabeled target frames
  (31 of the 15-s windows cover frames 0..44). Flagged in every table.
- Normalization: per-feature z-score computed from TRAINING subjects
  only, applied to train/val/test.
- Validation split: last 10% of the (shuffled) training-subject windows;
  model selection / early stopping on this split only.

## Training budget (identical for all neural methods)
- 30 epochs max, Adam(W), OneCycle LR (peak per method), early stop
  patience 6 on validation balanced accuracy, batch 256 (TSception: 128
  per its official recipe; PPDA: 128 per its paper).
- Original multi-stage/longer recipes (e.g., MSHCL contrastive
  pretraining, PPDA 200 epochs, AMA-EEG two-phase training) are collapsed
  to the unified single-stage budget for fairness; deviations are
  documented per method below.

## Metrics
Accuracy, Balanced Accuracy, Macro-F1, Cohen's Kappa; reported as
mean +/- SD over the 15 test subjects, plus median, worst subject, and
per-subject values. Seeds: 0 (all methods), 1-2 (stability subset).

## Method adaptations & deviations (all documented, none silent)
| method | source | deviations |
|---|---|---|
| Majority | - | constant class prior of training subjects |
| EEGNet | braindecode EEGNetv4 | adapted to DE input: 310 feature-"channels" x 15 frames; kernels/pools rescaled |
| DeepConvNet | braindecode Deep4Net | same DE adaptation |
| Riemannian MDM | pyRiemann | per-band OAS covariances, log-Euclidean MDM (fast closed-form mean); training windows subsampled 1/4 (cost); test uses full windows |
| Personal-Zscore | strict reimplementation | paper unavailable -> per the protocol the per-subject z-score degenerates to training-set statistics + LinearSVC; labeled as an approximation, excluded from the peer-method comparison |
| PPDA (AAAI'21) | faithful implementation from the paper (no official code) | l=15 preserved; all 3 sessions pooled (paper: single session); trade-offs fixed (alpha=.5, beta=.1, gamma=.1) instead of random search; delta=0 |
| TSception (TAFFC'23) | official code ported to PyTorch DE | DE bands concatenated on the temporal axis (75 frames); inception kernels rescaled to 9/5/2 frames, pool 4; spatial stage identical; official T=9/S=6/hidden=128/dropout=.3/lr=1e-3/batch=128 |
| MSHCL (TAFFC'25) | official backbone (ConvNet_baseNonlinearHead) | input 310 DE features (paper: raw EEG); kernels rescaled (timeFilterLen 5, pool 2); stratified layer-norm disabled (per-subject statistics would violate strict zero-shot); contrastive pretraining stage omitted under the unified budget |
| EmT (TNNLS'25) | official model (TGC + RMPG + Transformer) | native DE input [T=15, 62, 5]; single-stage training under the unified budget (official two-phase recipe collapsed) |
| AMA-EEG (TAFFC'26) | official EEG backbone (Conv_att_simple_new) | **EEG-only version**: multimodal projector bypassed; kernels rescaled to the 15-frame window; stratified norm disabled |
| SCDE (ours) | this repo | champion-module transplant: (ch x band) tokens + Conformer blocks + SupCon + hyperbolic contrastive + subject-adversarial GRL (training subjects only) |

## Seeds
- seed 0: all methods, all folds (main table)
- seeds 1, 2: stability subset (fast + key methods)
