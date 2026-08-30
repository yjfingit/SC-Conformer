"""Aggregate results_de into the final comparison tables.

Outputs:
  - console summary (all metrics, mean±SD/median/worst)
  - paper/tables/seed_main.tex  (unified-rerun main table)
  - paper/tables/seed_persub.tex (per-subject balanced accuracy)
Paper-reported numbers are maintained in PAPER_REPORTED below (no
fabrication; 'n/r' where a paper does not give the metric).
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_de")
OUT = os.path.join(ROOT, "paper", "tables")

METHODS = [
    ("majority", "Majority", "n/r"),
    ("eegnet-de", "EEGNet (DE)", "n/r"),
    ("deep4-de", "DeepConvNet (DE)", "n/r"),
    ("riemannian-mdm", "Riemannian MDM", "n/r"),
    ("tsception", "TSception (DE adapt.)", "n/r (DEAP/MAHNOB only)"),
    ("ppda-zs", "PPDA (zero-shot)", "n/r"),
    ("ppda-uda", "PPDA (UDA)", "86.7$\\pm$7.1 (Acc)"),
    ("mshcl", "MSHCL (adapt.)", "$\\approx$89.3 (Acc)"),
    ("emT", "EmT (adapt.)", "80.2 (Acc) / 82.1 (F1)"),
    ("ama-eeg", "AMA-EEG (EEG-only)", "69.5$\\pm$10.9 / 66.2$\\pm$13.8 / 54.2$\\pm$16.2"),
    ("scde", "SCDE (ours)", "-"),
]


def load(model, seed=0):
    out = {}
    for p in glob.glob(os.path.join(RES, model, f"S*_seed{seed}.json")):
        r = json.load(open(p))
        out[r["test_subject"]] = r
    return out


def agg(vals):
    v = np.array(vals) * 100
    return dict(mean=v.mean(), sd=v.std(), median=np.median(v), worst=v.min())


def main(seeds=(0,)):
    os.makedirs(OUT, exist_ok=True)
    print(f"{'method':28s} {'n':>3s} {'Acc':>15s} {'BAcc':>15s} "
          f"{'MacroF1':>15s} {'Kappa':>15s}")
    rows = {}
    for m, disp, _ in METHODS:
        data = load(m, seeds[0])
        if not data:
            continue
        subs = sorted(data)
        per = [data[s] for s in subs]
        line = {}
        for k in ("acc", "bacc", "f1", "kappa"):
            a = agg([r[k] for r in per])
            line[k] = a
        rows[m] = (len(per), line,
                   {s: data[s]["bacc"] * 100 for s in subs})
        print(f"{m:28s} {len(per):3d} "
              f"{line['acc']['mean']:6.2f}±{line['acc']['sd']:5.2f} "
              f"{line['bacc']['mean']:6.2f}±{line['bacc']['sd']:5.2f} "
              f"{line['f1']['mean']:6.2f}±{line['f1']['sd']:5.2f} "
              f"{line['kappa']['mean']:6.2f}±{line['kappa']['sd']:5.2f}")

    # LaTeX main table: paper-reported | unified rerun (Acc/BAcc/F1/Kappa)
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Method & Acc & BAcc & Macro-F1 & Kappa \\\\", "\\midrule"]
    for m, disp, paper in METHODS:
        if m not in rows:
            continue
        n, line, _ = rows[m]
        cells = " & ".join(
            f"{line[k]['mean']:.1f}$\\pm${line[k]['sd']:.1f}"
            for k in ("acc", "bacc", "f1", "kappa"))
        lines.append(f"{disp} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "seed_main.tex"), "w").write("\n".join(lines))

    # per-subject balanced accuracy table
    subs = sorted({s for m in rows for s in rows[m][2]})
    lines = ["\\begin{tabular}{l" + "c" * len(subs) + "}", "\\toprule",
             "Method & " + " & ".join(f"S{s}" for s in subs) + " \\\\",
             "\\midrule"]
    for m, disp, _ in METHODS:
        if m not in rows:
            continue
        vals = [f"{rows[m][2].get(s, float('nan')):.0f}" for s in subs]
        lines.append(f"{disp} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "seed_persub.tex"), "w").write("\n".join(lines))
    print("tables ->", OUT)


if __name__ == "__main__":
    main()
