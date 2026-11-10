"""
Statistical analysis and figures for a finished run.

    python analyze.py --run pilot [--bagging-run pilot_bagging]
                      [--draws-run pilot_draws]

Tables -> results/<run>/analysis/*.csv, figures -> figures/<run>/*.png
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.analysis import aggregate, run_analysis
from src.analysis_plots import make_all


def _load(run):
    if run is None:
        return None
    path = Path("results") / run / "results.csv"
    if not path.exists():
        print(f"missing {path}, skipped")
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--bagging-run", default=None)
    parser.add_argument("--draws-run", default=None)
    args = parser.parse_args()

    warnings.simplefilter("ignore")

    results = _load(args.run)
    draws = _load(args.draws_run)
    bagging = _load(args.bagging_run)

    tables = run_analysis(results, Path("results") / args.run / "analysis",
                          draws=draws)

    paths = make_all(results, tables, Path("figures") / args.run,
                     aggregated_bagging=(aggregate(bagging)
                                         if bagging is not None else None))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)

    for name in ("friedman", "control_comparisons", "control_comparisons_pooled",
                 "s3_core_test", "h1_tests", "h2_tests", "draw_tests", "power"):
        print(f"\n=== {name} ===")
        print(tables[name].round(4).to_string(index=False))

    print("\nFigures:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
