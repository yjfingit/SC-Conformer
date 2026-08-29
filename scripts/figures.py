"""Paper figures: architecture schematic + results breakdowns."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.axis("off")
    boxes = [
        ("Raw trial\n$C{\\times}T$", 0.02, 0.30, "#f0f0f0"),
        ("Multi-scale\nconv stem", 0.14, 0.30, "#cfe3f7"),
        ("Patch\ntokens", 0.29, 0.30, "#cfe3f7"),
        ("SCN stat\ntokens", 0.29, 0.02, "#ffe0b3"),
        ("Conformer\n$\\times4$ + RoPE\nadaLN$\\leftarrow$SCN", 0.44, 0.30,
         "#d5ecd5"),
        ("Gated SSM\nbranch", 0.44, 0.02, "#f7cfcf"),
        ("Pool +\nhead", 0.66, 0.30, "#cfe3f7"),
        ("Logits", 0.83, 0.30, "#f0f0f0"),
    ]
    for label, x, y, c in boxes:
        w, h = 0.13, 0.52 if y > 0.1 else 0.30
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="k", lw=0.8,
                                   zorder=2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8, zorder=3)
    arrows = [(.15, .56, .29, .56), (.29, .30, .29, .32), (.35, .32, .44, .44),
              (.42, .56, .44, .56), (.57, .30, .57, .32), (.63, .32, .66, .44),
              (.57, .56, .66, .56), (.79, .56, .83, .56)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"))
    ax.annotate("FiLM $\\gamma,\\beta$ per block",
                xy=(0.50, 0.36), xytext=(0.50, 0.86), fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(os.path.join(FIG, "arch.pdf"), bbox_inches="tight")
    print("arch.pdf done")


def per_subject_box():
    """Per-subject balanced accuracy: scformer vs eegnet vs conformer."""
    rows = {}
    resdir = os.path.join(ROOT, "results")
    for path in glob_results(resdir):
        r = json.load(open(path))
        rows.setdefault(r["model"], {}).setdefault(
            (r["dataset"], r["test_subject"]), []).append(r["bacc"] * 100)
    datasets = ["bnci_2a", "bnci_2b", "seed", "seed4"]
    models = ["eegnet", "conformer", "atcnet", "scformer"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 2.2),
                             sharey=False)
    for ax, ds in zip(axes, datasets):
        data, labels = [], []
        subs = sorted({s for (d, s) in rows.get("scformer", {}) if d == ds})
        for m in models:
            vals = [np.mean(rows.get(m, {}).get((ds, s), [np.nan]))
                    for s in subs]
            data.append(vals); labels.append(m)
        bp = ax.boxplot(data, tick_labels=models, showfliers=False,
                        patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor("#d5ecd5" if "scformer" in m else "#cfe3f7")
        ax.set_title(ds)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("balanced acc. (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "per_subject.pdf"), bbox_inches="tight")
    print("per_subject.pdf done")


def glob_results(root):
    import glob
    return glob.glob(os.path.join(root, "*", "*", "S*_seed0.json"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("arch", "all"):
        fig_arch()
    if what in ("box", "all"):
        per_subject_box()
