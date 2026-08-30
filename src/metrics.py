"""Unified metrics: accuracy, balanced accuracy, macro-F1, Cohen's kappa,
plus subject-level aggregation (mean+/-SD, median, worst, per-subject)."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)


def all_metrics(y_true, y_pred, n_classes):
    return dict(
        acc=float(accuracy_score(y_true, y_pred)),
        bacc=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, average="macro")),
        kappa=float(cohen_kappa_score(y_true, y_pred)),
    )


def aggregate(per_subject):
    """per_subject: list of metric dicts -> summary + worst subject."""
    out = {}
    for k in ("acc", "bacc", "f1", "kappa"):
        v = np.array([m[k] for m in per_subject]) * 100
        out[k] = dict(mean=float(v.mean()), sd=float(v.std()),
                      median=float(np.median(v)), worst=float(v.min()))
    return out


def format_summary(agg, n):
    lines = []
    for k, long in (("acc", "Acc"), ("bacc", "BAcc"), ("f1", "MacroF1"),
                    ("kappa", "Kappa")):
        a = agg[k]
        lines.append(f"{long}: {a['mean']:.2f}±{a['sd']:.2f} "
                     f"med={a['median']:.2f} worst={a['worst']:.2f}")
    return f"[n={n}] " + " | ".join(lines)
