
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from src.datasets import load_dataset
from src.preprocessing import preprocess_train_test

from src.sampling import (
    generate_bootstrap_samples,
    generate_subsamples,
    generate_common_core_samples,
    generate_stratified_bootstrap_samples,
)

from src.diversity import (
    mean_pairwise_jaccard_distance,
    mean_pairwise_multiplicity_distance,
    mean_pairwise_target_distance,
    unique_fraction,
    target_representativeness,
    feature_representativeness,
)

from src.models import (
    train_ensemble,
    predict_ensemble,
)

from src.metrics import (
    evaluate_individual_models,
    evaluate_ensemble,
)


# ============================================================
# KONFIGURACJA EKSPERYMENTU
# ============================================================

DEFAULT_CONFIG = {
    "n_splits": 5,
    "n_repeats": 10,
    "n_estimators": 100,
    "random_state": 42,
}


# ============================================================
# WARIANTY PRÓBKOWANIA
# ============================================================

def generate_sampling_variants(
    n_observations,
    y_train,
    task,
    n_estimators=100,
    random_state=42,
):
    """
    Generates all 15 sampling variants.

    S1: Bootstrap sampling with replacement.
    S2: Subsampling without replacement.
    S3: Common core sampling.
    S4: Stratified bootstrap sampling.
    """

    variants = []

    # --------------------------------------------------------
    # S1 - Bootstrap
    # --------------------------------------------------------

    bootstrap_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    for ratio in bootstrap_ratios:
        samples = generate_bootstrap_samples(
            n=n_observations,
            ratio=ratio,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        variants.append({
            "name": "S1_bootstrap",
            "parameters": {"ratio": ratio},
            "samples": samples,
        })

    # --------------------------------------------------------
    # S2 - Subsampling bez powtórzeń
    # --------------------------------------------------------

    subsampling_ratios = [0.18, 0.33, 0.45, 0.55, 0.63]

    for ratio in subsampling_ratios:
        samples = generate_subsamples(
            n=n_observations,
            ratio=ratio,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        variants.append({
            "name": "S2_subsampling",
            "parameters": {"ratio": ratio},
            "samples": samples,
        })

    # --------------------------------------------------------
    # S3 - Common core
    # --------------------------------------------------------

    common_core_alphas = [0.0, 0.25, 0.5, 0.75]

    for alpha in common_core_alphas:
        samples = generate_common_core_samples(
            n=n_observations,
            ratio=0.5,
            alpha=alpha,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        variants.append({
            "name": "S3_common_core",
            "parameters": {
                "ratio": 0.5,
                "alpha": alpha,
            },
            "samples": samples,
        })

    # --------------------------------------------------------
    # S4 - Stratified bootstrap
    # --------------------------------------------------------

    samples = generate_stratified_bootstrap_samples(
        y=y_train,
        task=task,
        n_estimators=n_estimators,
        random_state=random_state,
        n_bins=10,
    )

    variants.append({
        "name": "S4_stratified_bootstrap",
        "parameters": {"ratio": 1.0},
        "samples": samples,
    })

    return variants


# ============================================================
# MIARY RÓŻNORODNOŚCI I REPREZENTATYWNOŚCI PRÓBEK
# ============================================================

def calculate_sample_diversity(
    samples,
    X_train,
    y_train,
    task,
):
    """
    Calculates sample diversity and representativeness metrics
    before training the ensemble models.
    """

    n_observations = len(y_train)

    results = {}

    # Różnorodność próbek
    results["jaccard_distance"] = (
        mean_pairwise_jaccard_distance(samples)
    )

    results["multiplicity_distance"] = (
        mean_pairwise_multiplicity_distance(
            samples,
            n_observations,
        )
    )

    results["target_distance"] = (
        mean_pairwise_target_distance(
            samples,
            y_train,
            task=task,
        )
    )

    # Średni udział unikalnych obserwacji
    results["unique_fraction"] = np.mean([
        unique_fraction(sample, n_observations)
        for sample in samples
    ])

    # Reprezentatywność względem zmiennej docelowej
    results["target_representativeness"] = np.mean([
        target_representativeness(
            sample,
            y_train,
            task=task,
        )
        for sample in samples
    ])

    # Reprezentatywność względem cech
    results["feature_representativeness"] = np.mean([
        feature_representativeness(
            sample,
            X_train,
        )
        for sample in samples
    ])

    return results


# ============================================================
# RÓŻNORODNOŚĆ PREDYKCJI DRZEW
# ============================================================

def calculate_prediction_diversity(
    predictions,
    y_true,
    task,
):
    """
    Calculates the diversity of predictions produced by
    individual models in the ensemble.

    Classification:
        Mean pairwise disagreement between trees.

    Regression:
        1 - mean pairwise correlation between prediction errors.
    """


    n_models = predictions.shape[0]

    if n_models < 2:
        return np.nan

    pairwise_values = []

    if task == "classification":

        for i, j in combinations(range(n_models), 2):
            disagreement = np.mean(
                predictions[i] != predictions[j]
            )
            pairwise_values.append(disagreement)

    elif task == "regression":

        errors = predictions - np.asarray(y_true)

        for i, j in combinations(range(n_models), 2):
            error_i = errors[i]
            error_j = errors[j]

            if (
                np.std(error_i) == 0
                or np.std(error_j) == 0
            ):
                continue

            correlation = np.corrcoef(
                error_i,
                error_j,
            )[0, 1]

            if np.isfinite(correlation):
                pairwise_values.append(1.0 - correlation)

    else:
        raise ValueError(
            f"Nieznany rodzaj zadania: {task}"
        )

    if not pairwise_values:
        return np.nan

    return float(np.mean(pairwise_values))


# ============================================================
# EKSPERYMENT DLA JEDNEGO FOLDA
# ============================================================

def run_fold_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    task,
    dataset_name,
    fold_id,
    n_estimators=100,
    random_state=42,
):
    """
    Runs all sampling variants for a single
    training/test split.
    """

    # Preprocessing dopasowujemy wyłącznie do treningu.
    X_train_processed, X_test_processed, _ = (
        preprocess_train_test(
            X_train,
            X_test,
        )
    )

    classes = (
        np.unique(y_train)
        if task == "classification"
        else None
    )

    # Generowanie wariantów próbkowania
    variants = generate_sampling_variants(
        n_observations=len(y_train),
        y_train=np.asarray(y_train),
        task=task,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    fold_results = []

    for variant_id, variant in enumerate(variants):

        samples = variant["samples"]

        # ----------------------------------------------------
        # 1. Miary próbek - przed trenowaniem
        # ----------------------------------------------------

        sample_metrics = calculate_sample_diversity(
            samples=samples,
            X_train=X_train_processed,
            y_train=np.asarray(y_train),
            task=task,
        )

        # ----------------------------------------------------
        # 2. Trenowanie zespołu
        # ----------------------------------------------------

        models = train_ensemble(
            X_train=X_train_processed,
            y_train=np.asarray(y_train),
            samples=samples,
            task=task,
            random_state=random_state + variant_id,
        )

        # ----------------------------------------------------
        # 3. Predykcje pojedynczych drzew
        # ----------------------------------------------------

        predictions = predict_ensemble(
            models,
            X_test_processed,
        )

        prediction_diversity = (
            calculate_prediction_diversity(
                predictions=predictions,
                y_true=np.asarray(y_test),
                task=task,
            )
        )

        # ----------------------------------------------------
        # 4. Jakość pojedynczych drzew
        # ----------------------------------------------------

        individual_metrics = evaluate_individual_models(
            models=models,
            X_test=X_test_processed,
            y_test=np.asarray(y_test),
            task=task,
            classes=classes,
        )

        # ----------------------------------------------------
        # 5. Jakość całego zespołu
        # ----------------------------------------------------

        ensemble_metrics = evaluate_ensemble(
            models=models,
            X_test=X_test_processed,
            y_test=np.asarray(y_test),
            task=task,
            classes=classes,
        )

        # ----------------------------------------------------
        # 6. Zapis wyników
        # ----------------------------------------------------

        result = {
            "dataset": dataset_name,
            "task": task,
            "fold": fold_id,
            "variant": variant["name"],
            **variant["parameters"],
            **sample_metrics,
            "prediction_diversity": prediction_diversity,
        }

        for metric, value in individual_metrics.items():
            result[f"individual_{metric}_mean"] = value["mean"]
            result[f"individual_{metric}_std"] = value["std"]

        for metric, value in ensemble_metrics.items():
            result[f"ensemble_{metric}"] = value

        fold_results.append(result)

    return fold_results


# ============================================================
# EKSPERYMENT DLA JEDNEGO ZBIORU DANYCH
# ============================================================

def run_dataset_experiment(
    dataset_name,
    n_splits=5,
    n_repeats=10,
    n_estimators=100,
    random_state=42,
    max_samples=20000,
):
    """
    Runs the experiment on a single dataset.

    Classification:
        Uses RepeatedStratifiedKFold.

    Regression:
        Uses RepeatedKFold.
    """


    dataset = load_dataset(
        dataset_name,
        max_samples=max_samples,
        random_state=random_state,
    )

    X = dataset["X"]
    y = dataset["y"]
    task = dataset["task"]

    print("\n" + "=" * 65)
    print(f"ZBIÓR: {dataset_name}")
    print(f"Zadanie: {task}")
    print(f"Liczba obserwacji: {len(X)}")
    print(f"Liczba cech przed preprocessingiem: {X.shape[1]}")
    print("=" * 65)

    if task == "classification":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    elif task == "regression":
        splitter = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Nieznany task: {task}")

    all_results = []

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):

        print(
            f"Fold {fold_id}/{n_splits * n_repeats}",
            end="\r",
            flush=True,
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_results = run_fold_experiment(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            task=task,
            dataset_name=dataset_name,
            fold_id=fold_id,
            n_estimators=n_estimators,
            random_state=random_state + fold_id,
        )

        all_results.extend(fold_results)

    results_df = pd.DataFrame(all_results)

    print(
        f"\nZakończono: {dataset_name}. "
        f"Liczba rekordów: {len(results_df)}"
    )

    return results_df


# ============================================================
# EKSPERYMENT DLA WIELU ZBIORÓW
# ============================================================

def run_experiments(
    dataset_names,
    n_splits=5,
    n_repeats=10,
    n_estimators=100,
    random_state=42,
    max_samples=20000,
    output_path="results/experiment_results.csv",
):
    """
    Runs experiments on a list of datasets
    and saves the combined results to a CSV file.
    """

    all_results = []

    for dataset_id, dataset_name in enumerate(dataset_names, start=1):

        print(
            f"\nDATASET {dataset_id}/{len(dataset_names)}"
        )

        dataset_results = run_dataset_experiment(
            dataset_name=dataset_name,
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=max_samples,
        )

        all_results.append(dataset_results)

        # Zapis po każdym zbiorze, aby nie stracić
        # dotychczasowych wyników w razie przerwania.
        results_df = pd.concat(
            all_results,
            ignore_index=True,
        )

        results_df.to_csv(
            output_path,
            index=False,
        )

    if not all_results:
        return pd.DataFrame()

    return pd.concat(
        all_results,
        ignore_index=True,
    )
