"""Aggregate per-fold JSONs into mean±std tables + LaTeX output."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
METRICS = ["bacc", "acc", "kappa"]

LATEX_NAMES = {
    "eegnet": "EEGNet",
    "shallow": "ShallowConvNet",
    "deep4": "DeepConvNet",
    "conformer": "EEG-Conformer",
    "atcnet": "ATCNet",
    "cbramod": "CBraMod (FT)",
    "scformer-nsc": r"\method{} w/o \scn{}",
    "scformer-sin": r"\method{} w/ sin-pos",
    "scformer-ms": r"\method{} w/ 1-scale stem",
    "scformer-ff": r"\method{} w/o adaLN",
    "scformer+nf": r"\method{} w/o SSM",
    "scformer-v2": r"\method{}-S",
    "scformer": r"\method{}",
}
DATASET_LATEX = {"bnci_2a": "BCI-IV 2a", "bnci_2b": "BCI-IV 2b",
                 "bnci_2014_002": "BNCI-2014-002", "seed": "SEED"}
ORDER = ["eegnet", "shallow", "deep4", "conformer", "atcnet", "cbramod",
         "scformer-nsc", "scformer-sin", "scformer-ms", "scformer-ff",
         "scformer+nf", "scformer-v2", "scformer"]


def collect(root, seeds=(0,)):
    rows = {}
    for path in glob.glob(os.path.join(root, "*", "*", "S*_seed*.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("seed") not in seeds:
            continue
        rows.setdefault((r["dataset"], r["model"]), []).append(r)
    return rows


def fmt(rs, m):
    v = np.array([r[m] for r in rs]) * 100
    return f"{v.mean():.1f}$\\pm${v.std():.1f}"


def main_table(rows, root):
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    datasets = ["bnci_2a", "bnci_2b", "bnci_2014_002", "seed"]
    # keep models having data in >=1 dataset; custom ordering
    models = [m for m in ORDER if any((d, m) in rows for d in datasets)]
    models += sorted({m for (_, m) in rows if m not in ORDER})
    lines = ["\\begin{tabular}{l" + "c" * len(datasets) + "}", "\\toprule",
             "Method & " + " & ".join(DATASET_LATEX[d] for d in datasets)
             + " \\\\", "\\midrule"]
    means = {}
    for m in models:
        cells = []
        for d in datasets:
            rs = rows.get((d, m), [])
            if rs:
                cells.append(fmt(rs, "bacc"))
                means.setdefault(m, []).append(np.mean([r["bacc"] for r in rs]))
            else:
                cells.append("--")
        lines.append(f"{LATEX_NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(root, "tables", "main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return models, means


def ablation_table(rows, root):
    datasets = ["bnci_2a", "bnci_2b", "bnci_2014_002", "seed"]
    ablations = ["scformer", "scformer+nf", "scformer-ff", "scformer-nsc",
                 "scformer-ms", "scformer-sin", "scformer-v2"]
    lines = ["\\begin{tabular}{l" + "c" * len(datasets) + "}", "\\toprule",
             "Variant & " + " & ".join(DATASET_LATEX[d] for d in datasets)
             + " \\\\", "\\midrule"]
    for m in ablations:
        cells = []
        for d in datasets:
            rs = rows.get((d, m), [])
            cells.append(fmt(rs, "bacc") if rs else "--")
        lines.append(f"{LATEX_NAMES.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(root, "tables", "ablation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def cost_table(rows, root):
    seen = {}
    for (d, m), rs in rows.items():
        for r in rs:
            if m not in seen:
                seen[m] = (r["n_params"], r["train_time"])
    lines = ["\\begin{tabular}{lrr}", "\\toprule",
             "Method & Params & train/fold (s) \\\\", "\\midrule"]
    for m in ORDER:
        if m in seen:
            p, t = seen[m]
            lines.append(f"{LATEX_NAMES.get(m, m)} & {p/1e3:.1f}K & {t:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(root, "tables", "cost.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    seeds = tuple(int(s) for s in sys.argv[1:]) or (0,)
    rows = collect(RESULTS, seeds)
    # console summary
    for ds in sorted({k[0] for k in rows}):
        print(f"=== {ds} ===")
        for m in ORDER + sorted({m for (_, m) in rows if m not in ORDER}):
            rs = rows.get((ds, m), [])
            if rs:
                print(f"  {m:16s} n={len(rs):3d} " + " ".join(
                    f"{k}={fmt(rs, k)}" for k in METRICS[:2]))
    main_table(rows, ROOT)
    ablation_table(rows, ROOT)
    cost_table(rows, ROOT)
    print("LaTeX tables written.")


if __name__ == "__main__":
    main()
