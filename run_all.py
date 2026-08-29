"""Run the full LOSO experiment grid in parallel, skipping folds on disk."""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")


def fold_path(ds, model, subj, seed):
    return os.path.join(RESULTS, ds, model, f"S{subj:02d}_seed{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    from src.data.datasets import dataset_subjects

    jobs = []
    for ds, model, seed in itertools.product(
            args.datasets, args.models, args.seeds):
        try:
            subjects = dataset_subjects(ds)
        except FileNotFoundError:
            print(f"[skip] {ds}: no processed data yet", flush=True)
            continue
        for s in subjects:
            if os.path.exists(fold_path(ds, model, s, seed)):
                continue
            jobs.append((ds, model, s, seed))
    print(f"{len(jobs)} folds to run, parallel={args.parallel}", flush=True)

    def run(job):
        ds, model, s, seed = job
        cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"),
               "--dataset", ds, "--model", model, "--subject", str(s),
               "--seed", str(seed), "--epochs", str(args.epochs),
               "--batch", str(args.batch), "--workers", "2"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] {ds}/{model}/S{s}:\n{r.stderr[-800:]}", flush=True)
            return
        last = [l for l in r.stdout.strip().splitlines() if l.strip()][-1]
        print(last, flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
