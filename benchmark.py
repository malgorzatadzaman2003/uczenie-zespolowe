import argparse
import platform
import resource
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiment import run_dataset_experiment


# ============================================================
# KONFIGURACJA BENCHMARKU
# ============================================================

BENCHMARK_DATASETS = [
    "breast_cancer",
    "adult",
    "california_housing",
    "superconductivity",
]

DEFAULT_CONFIG = {
    "n_splits": 2,
    "n_repeats": 1,
    "n_estimators": 100,
    "random_state": 42,
    "max_samples": 20000,
}

OUTPUT_DIR = Path("results/benchmark")


# ============================================================
# POMIAR PAMIĘCI
# ============================================================

def get_peak_memory_mb():
    """
    Returns the maximum resident set size (peak RSS) in MB.

    macOS reports bytes, while Linux reports kilobytes.
    """

    peak_memory = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if platform.system() == "Darwin":
        return peak_memory / (1024 ** 2)

    return peak_memory / 1024


# ============================================================
# BENCHMARK JEDNEGO ZBIORU
# ============================================================

def benchmark_dataset(
    dataset_name,
    n_splits=2,
    n_repeats=1,
    n_estimators=100,
    random_state=42,
    max_samples=20000,
):
    """
    Measures execution time, peak memory usage
    and output file size for one dataset.

    Each dataset should be benchmarked in a separate
    Python process to obtain an independent peak RSS.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK EKSPERYMENTU")
    print("=" * 70)

    print(f"Zbiór: {dataset_name}")
    print(f"Liczba podziałów: {n_splits}")
    print(f"Liczba powtórzeń: {n_repeats}")
    print(f"Liczba drzew: {n_estimators}")
    print("Liczba konfiguracji: 15")

    total_folds = n_splits * n_repeats

    print(
        "Łączna liczba drzew:",
        total_folds * 15 * n_estimators,
    )

    # --------------------------------------------------------
    # Pamięć przed eksperymentem
    # --------------------------------------------------------

    baseline_memory_mb = get_peak_memory_mb()

    # --------------------------------------------------------
    # Pomiar czasu
    # --------------------------------------------------------

    start_time = time.perf_counter()

    results = run_dataset_experiment(
        dataset_name=dataset_name,
        n_splits=n_splits,
        n_repeats=n_repeats,
        n_estimators=n_estimators,
        random_state=random_state,
        max_samples=max_samples,
    )

    elapsed_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Pomiar pamięci po eksperymencie
    # --------------------------------------------------------

    peak_memory_mb = get_peak_memory_mb()

    additional_peak_memory_mb = max(
        0.0,
        peak_memory_mb - baseline_memory_mb,
    )

    # --------------------------------------------------------
    # Zapis wyników eksperymentu
    # --------------------------------------------------------

    results_path = (
        OUTPUT_DIR / f"{dataset_name}_benchmark_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    output_size_mb = (
        results_path.stat().st_size / (1024 ** 2)
    )

    # --------------------------------------------------------
    # Przygotowanie rekordu benchmarku
    # --------------------------------------------------------

    benchmark_record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "dataset": dataset_name,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_estimators": n_estimators,
        "n_variants": 15,
        "n_trees": total_folds * 15 * n_estimators,
        "elapsed_seconds": elapsed_time,
        "elapsed_minutes": elapsed_time / 60,
        "baseline_peak_memory_mb": baseline_memory_mb,
        "peak_memory_mb": peak_memory_mb,
        "additional_peak_memory_mb": additional_peak_memory_mb,
        "output_size_mb": output_size_mb,
        "n_result_rows": len(results),
    }

    # --------------------------------------------------------
    # Zapis podsumowania benchmarku
    # --------------------------------------------------------

    summary_path = OUTPUT_DIR / "benchmark_summary.csv"

    new_record = pd.DataFrame([benchmark_record])

    if summary_path.exists():
        old_results = pd.read_csv(summary_path)

        summary = pd.concat(
            [old_results, new_record],
            ignore_index=True,
        )
    else:
        summary = new_record

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Wyświetlenie wyników
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("WYNIKI BENCHMARKU")
    print("-" * 70)

    print(f"Czas wykonania: {elapsed_time:.2f} s")
    print(f"Czas wykonania: {elapsed_time / 60:.2f} min")

    print(
        f"Pamięć szczytowa procesu: "
        f"{peak_memory_mb:.2f} MB"
    )

    print(
        f"Przyrost szczytowego zużycia pamięci: "
        f"{additional_peak_memory_mb:.2f} MB"
    )

    print(
        f"Rozmiar pliku wynikowego: "
        f"{output_size_mb:.4f} MB"
    )

    print(f"Liczba rekordów: {len(results)}")

    print("\nZapisano wyniki:")
    print(results_path)

    print("\nZapisano podsumowanie benchmarku:")
    print(summary_path)

    print("\nBenchmark zakończony.")

    return benchmark_record


# ============================================================
# URUCHOMIENIE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Benchmark czasu i pamięci eksperymentu."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=BENCHMARK_DATASETS,
        help="Nazwa zbioru danych.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_CONFIG["n_splits"],
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_CONFIG["n_repeats"],
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_CONFIG["n_estimators"],
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_CONFIG["random_state"],
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_CONFIG["max_samples"],
    )

    args = parser.parse_args()

    benchmark_dataset(
        dataset_name=args.dataset,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()