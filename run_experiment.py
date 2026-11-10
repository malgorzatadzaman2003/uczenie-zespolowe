"""
Runs the main experiment (and optionally the Section 4.6 draw experiment).

Jobs are (dataset, fold) pairs run in parallel. Every finished fold is
saved to its own CSV, so an interrupted run resumes where it stopped.

Examples
--------
Pilot used for the 10.11 checkpoint (all datasets, 5x1 CV):
    python run_experiment.py --run pilot --n-repeats 1 --n-jobs 16

Full experiment from the plan (5x10 CV):
    python run_experiment.py --run main --n-repeats 10 --n-jobs 16

Pure bagging (max_features=None) for comparison, classification only:
    python run_experiment.py --run pilot_bagging --task classification \
        --ensemble bagging --n-repeats 1
"""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed

from src.datasets import list_datasets, load_dataset
from src.experiment import (
    _draw_fold,
    _make_splitter,
    run_fold_experiment,
)


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _main_job(path, **kwargs):
    rows = _quiet(run_fold_experiment, **kwargs)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _draw_job(path, *args):
    rows = _quiet(_draw_fold, *args)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_jobs(args, datasets, fold_dir):
    jobs = []

    for name in datasets:
        data = _quiet(load_dataset, name, max_samples=args.max_samples,
                      random_state=args.seed)
        X, y, task = data["X"], data["y"], data["task"]

        n_repeats = 1 if args.draws else args.n_repeats
        splitter = _make_splitter(task, args.n_splits, n_repeats, args.seed)

        for fold_id, (train_idx, test_idx) in enumerate(
                splitter.split(X, y), start=1):

            path = fold_dir / f"{name}__fold{fold_id:03d}.csv"

            if path.exists():
                continue

            if args.draws:
                jobs.append(delayed(_draw_job)(
                    path,
                    X.iloc[train_idx], X.iloc[test_idx],
                    y.iloc[train_idx], y.iloc[test_idx],
                    task, name, fold_id, args.draws, args.n_estimators,
                    args.seed + fold_id, args.ensemble,
                ))
            else:
                jobs.append(delayed(_main_job)(
                    path,
                    X_train=X.iloc[train_idx], X_test=X.iloc[test_idx],
                    y_train=y.iloc[train_idx], y_test=y.iloc[test_idx],
                    task=task, dataset_name=name, fold_id=fold_id,
                    n_estimators=args.n_estimators,
                    random_state=args.seed + fold_id,
                    ensemble_type=args.ensemble,
                ))

    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", default="main",
                        help="run name; results go to results/<run>/")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--ensemble", default="random_forest",
                        choices=["random_forest", "bagging"])
    parser.add_argument("--draws", type=int, default=0,
                        help="if > 0, run the Section 4.6 experiment with "
                             "this many bagging draws per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-2)
    args = parser.parse_args()

    datasets = args.datasets or list_datasets(args.task)
    datasets = [d for d in datasets if d not in args.exclude]

    out_dir = Path("results") / args.run
    fold_dir = out_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "datasets": datasets,
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    jobs = build_jobs(args, datasets, fold_dir)
    print(f"{len(jobs)} fold jobs to run -> {fold_dir}", flush=True)

    start = time.perf_counter()
    done = 0

    for path in Parallel(n_jobs=args.n_jobs,
                         return_as="generator_unordered")(jobs):
        done += 1
        print(f"[{done}/{len(jobs)}] {time.perf_counter() - start:7.0f}s "
              f"{path.name}", flush=True)

    files = sorted(fold_dir.glob("*.csv"))
    results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    results.to_csv(out_dir / "results.csv", index=False)

    print(f"Saved {len(results)} rows from {len(files)} folds to "
          f"{out_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
