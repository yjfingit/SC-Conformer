"""Parallel runner for DE baselines over 15 LOSO folds."""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_de")
ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--parallel", type=int, default=2)
args = ap.parse_args()
sys.path.insert(0, ROOT)
from src.data.de_datasets import dataset_subjects
subs = dataset_subjects()
jobs = []
for m in args.models:
    for s in subs:
        for sd in args.seeds:
            p = os.path.join(RES, m, f"S{s:02d}_seed{sd}.json")
            if not os.path.exists(p):
                jobs.append((m, s, sd))
print(f"{len(jobs)} folds to run", flush=True)
def run(job):
    m, s, sd = job
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "train_de.py"),
                        "--model", m, "--subject", str(s), "--seed", str(sd),
                        "--epochs", str(args.epochs)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[FAIL] {m}/S{s}: {r.stderr[-300:]}", flush=True)
    else:
        print([l for l in r.stdout.splitlines() if l.strip()][-1], flush=True)
with ThreadPoolExecutor(args.parallel) as ex:
    list(ex.map(run, jobs))
