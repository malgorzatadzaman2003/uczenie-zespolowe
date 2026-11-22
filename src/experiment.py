"""
Main experiment loop.

For every dataset and every train/test split:
  1. preprocessing is fitted on the training part,
  2. for every sampling variant M index samples are drawn,
  3. sample measures are computed BEFORE training (no test data used),
  4. trees are trained and combined (soft voting / mean),
  5. single trees and the ensemble are evaluated on the test part.

Besides the ensemble-level (averaged) measures, every row also stores
the within-ensemble Spearman correlations needed for H1:
  * over pairs of trees: sample-pair distance vs prediction diversity,
  * over trees: sample representativeness vs tree quality.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from src.datasets import load_dataset
from src.diversity import sample_measures
from src.models import (
    get_tree_params,
    predict_tree_probabilities,
    train_ensemble,
)
from src.preprocessing import preprocess_train_test
from src.sampling import (
    generate_bootstrap_samples,
    generate_common_core_samples,
    generate_stratified_bootstrap_samples,
    generate_subsamples,
)


# ============================================================
# KONFIGURACJA EKSPERYMENTU
# ============================================================

DEFAULT_CONFIG = {
    "n_splits": 5,
    "n_repeats": 10,
    "n_estimators": 100,
    "random_state": 42,
    "ensemble_type": "random_forest",
}

BOOTSTRAP_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
SUBSAMPLING_RATIOS = [0.18, 0.33, 0.45, 0.55, 0.63]
COMMON_CORE_RATIO = 0.5
# Udział wspólnego rdzenia w próbce. W planie oznaczony jako alfa;
# w raporcie zmieniamy symbol na gamma, żeby nie mylić go z poziomem
# istotności. W kodzie kolumna nadal nazywa się "alpha".
COMMON_CORE_FRACTIONS = [0.0, 0.25, 0.5, 0.75]

BASELINE_CONFIG = "S1_bootstrap | ratio=1"


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

    S1: bootstrap with replacement, m = r n.
    S2: subsampling without replacement, m = s n.
    S3: common core of size gamma*m shared by all trees, m = 0.5 n.
    S4: stratified bootstrap (classes / 10 bins of y), m = n.

    The same seed is used for every variant (common random numbers),
    which makes comparisons between variants paired.
    """

    variants = []

    for ratio in BOOTSTRAP_RATIOS:
        variants.append({
            "name": "S1_bootstrap",
            "parameters": {"ratio": ratio},
            "samples": generate_bootstrap_samples(
                n=n_observations,
                ratio=ratio,
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        })

    for ratio in SUBSAMPLING_RATIOS:
        variants.append({
            "name": "S2_subsampling",
            "parameters": {"ratio": ratio},
            "samples": generate_subsamples(
                n=n_observations,
                ratio=ratio,
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        })

    for alpha in COMMON_CORE_FRACTIONS:
        variants.append({
            "name": "S3_common_core",
            "parameters": {"ratio": COMMON_CORE_RATIO, "alpha": alpha},
            "samples": generate_common_core_samples(
                n=n_observations,
                ratio=COMMON_CORE_RATIO,
                alpha=alpha,
                n_estimators=n_estimators,
                random_state=random_state,
            ),
        })

    variants.append({
        "name": "S4_stratified_bootstrap",
        "parameters": {"ratio": 1.0},
        "samples": generate_stratified_bootstrap_samples(
            y=y_train,
            task=task,
            n_estimators=n_estimators,
            random_state=random_state,
            n_bins=10,
        ),
    })

    return variants


def config_label(variant, parameters):
    parts = [variant]

    if parameters.get("ratio") is not None:
        parts.append(f"ratio={parameters['ratio']:g}")

    if parameters.get("alpha") is not None:
        parts.append(f"alpha={parameters['alpha']:g}")

    return " | ".join(parts)


# ============================================================
# OCENA DRZEW I ZESPOŁU
# ============================================================

def _macro_f1_from_labels(y_true_idx, y_pred_idx, n_classes):
    """
    Macro-F1 for integer-coded labels; classes with an empty
    denominator get F1 = 0 (as zero_division=0 in scikit-learn).
    """

    confusion = np.bincount(
        y_true_idx * n_classes + y_pred_idx,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)

    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    denominator = 2 * tp + fp + fn

    f1 = np.divide(
        2 * tp, denominator,
        out=np.zeros_like(tp), where=denominator > 0,
    )

    return float(f1.mean())


def _balanced_accuracy(y_true_idx, y_pred_idx, n_classes):
    recalls = []

    for c in range(n_classes):
        mask = y_true_idx == c
        if mask.any():
            recalls.append(np.mean(y_pred_idx[mask] == c))

    return float(np.mean(recalls))


def _roc_auc(y_true_idx, probabilities, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true_idx, probabilities[:, 1]))

        return float(roc_auc_score(
            y_true_idx,
            probabilities,
            labels=np.arange(n_classes),
            multi_class="ovr",
            average="macro",
        ))
    except ValueError:
        return np.nan


def _spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 3:
        return np.nan

    if np.ptp(a[mask]) == 0 or np.ptp(b[mask]) == 0:
        return np.nan

    return float(spearmanr(a[mask], b[mask])[0])


def evaluate_classification(tree_proba, y_test_idx, n_classes):
    """
    tree_proba : (M, N, K) per-tree class probabilities on the test part.
    """

    tree_pred = tree_proba.argmax(axis=2)               # (M, N)
    correct = tree_pred == y_test_idx[None, :]          # oracle outputs

    tree_f1 = np.array([
        _macro_f1_from_labels(y_test_idx, pred, n_classes)
        for pred in tree_pred
    ])
    tree_error = 1.0 - correct.mean(axis=1)

    mean_proba = tree_proba.mean(axis=0)
    soft_pred = mean_proba.argmax(axis=1)

    # Głosowanie większościowe – tylko do porównania z miękkim.
    votes = np.zeros((tree_pred.shape[1], n_classes))
    for c in range(n_classes):
        votes[:, c] = (tree_pred == c).sum(axis=0)
    hard_pred = votes.argmax(axis=1)

    ensemble = {
        "ensemble_macro_f1": _macro_f1_from_labels(
            y_test_idx, soft_pred, n_classes),
        "ensemble_balanced_accuracy": _balanced_accuracy(
            y_test_idx, soft_pred, n_classes),
        "ensemble_roc_auc": _roc_auc(y_test_idx, mean_proba, n_classes),
        "ensemble_accuracy": float(np.mean(soft_pred == y_test_idx)),
        "ensemble_macro_f1_hard_vote": _macro_f1_from_labels(
            y_test_idx, hard_pred, n_classes),
    }

    individual = {
        "individual_macro_f1_mean": float(tree_f1.mean()),
        "individual_macro_f1_std": float(tree_f1.std()),
        "individual_accuracy_mean": float(1.0 - tree_error.mean()),
        "individual_accuracy_std": float(tree_error.std()),
    }

    # Niezgodność (Kuncheva i Whitaker 2003): odsetek obserwacji, na
    # których dokładnie jedno z dwóch drzew się myli.
    c = correct.astype(np.float64)
    n_test = c.shape[1]
    both = c @ c.T
    hits = c.sum(axis=1)
    disagreement = (hits[:, None] + hits[None, :] - 2 * both) / n_test
    rows, cols = np.triu_indices(c.shape[0], k=1)
    pair_diversity = disagreement[rows, cols]

    gain = {
        # Zysk z agregacji: o ile zespół jest lepszy od przeciętnego drzewa.
        "aggregation_gain": ensemble["ensemble_macro_f1"] - tree_f1.mean(),
        "aggregation_gain_error": float(
            tree_error.mean() - np.mean(soft_pred != y_test_idx)
        ),
    }

    return ensemble, individual, gain, pair_diversity, tree_f1


def evaluate_regression(tree_pred, y_test):
    """
    tree_pred : (M, N) per-tree predictions on the test part.
    """

    errors = tree_pred - y_test[None, :]
    tree_mse = (errors ** 2).mean(axis=1)
    tree_rmse = np.sqrt(tree_mse)
    tree_mae = np.abs(errors).mean(axis=1)
    sst = np.sum((y_test - y_test.mean()) ** 2)
    tree_r2 = 1.0 - (errors ** 2).sum(axis=1) / sst

    mean_pred = tree_pred.mean(axis=0)
    ensemble_error = mean_pred - y_test
    ensemble_mse = float(np.mean(ensemble_error ** 2))

    ensemble = {
        "ensemble_rmse": float(np.sqrt(ensemble_mse)),
        "ensemble_mae": float(np.mean(np.abs(ensemble_error))),
        "ensemble_r2": float(1.0 - np.sum(ensemble_error ** 2) / sst),
    }

    individual = {
        "individual_rmse_mean": float(tree_rmse.mean()),
        "individual_rmse_std": float(tree_rmse.std()),
        "individual_mae_mean": float(tree_mae.mean()),
        "individual_mae_std": float(tree_mae.std()),
        "individual_r2_mean": float(tree_r2.mean()),
        "individual_r2_std": float(tree_r2.std()),
    }

    # Krogh i Vedelsby: średni MSE drzew - MSE zespołu = rozrzut
    # (ambiguity) predykcji drzew wokół predykcji zespołu.
    ambiguity = float(np.mean((tree_pred - mean_pred[None, :]) ** 2))

    gain = {
        "aggregation_gain": float(tree_mse.mean() - ensemble_mse),
        "aggregation_gain_relative": float(
            (tree_mse.mean() - ensemble_mse) / tree_mse.mean()
        ),
        "ambiguity": ambiguity,
    }

    # Różnorodność par: 1 - korelacja błędów.
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(errors)
    rows, cols = np.triu_indices(errors.shape[0], k=1)
    pair_diversity = 1.0 - correlation[rows, cols]

    return ensemble, individual, gain, pair_diversity, tree_r2


# ============================================================
# JEDEN ZESPÓŁ: MIARY + UCZENIE + OCENA
# ============================================================

def evaluate_samples(
    samples,
    X_train,
    y_train_model,
    X_test,
    y_test_model,
    task,
    classes_idx=None,
    max_features=None,
    random_state=42,
):
    """
    Measures, trains and evaluates one ensemble built on `samples`.

    For classification y_* are integer-coded labels 0..K-1.
    Returns a flat dict (one result row without identifiers).
    """

    # 1. Miary próbek – przed uczeniem, bez danych testowych.
    pairs, trees = sample_measures(samples, X_train, y_train_model, task)

    row = {
        "sample_size": int(np.mean([len(s) for s in samples])),
        "jaccard_distance": float(pairs["jaccard"].mean()),
        "jaccard_distance_pair_std": float(pairs["jaccard"].std()),
        "multiplicity_distance": float(pairs["multiplicity"].mean()),
        "target_distance": float(pairs["target"].mean()),
        "unique_fraction": float(trees["unique_fraction"].mean()),
        "target_representativeness": float(
            trees["target_representativeness"].mean()),
        "feature_representativeness": float(
            trees["feature_representativeness"].mean()),
    }

    # 2. Uczenie.
    models = train_ensemble(
        X_train=X_train,
        y_train=y_train_model,
        samples=samples,
        task=task,
        random_state=random_state,
        max_features=max_features,
    )

    # 3. Ocena.
    if task == "classification":
        tree_proba = predict_tree_probabilities(models, X_test, classes_idx)
        ensemble, individual, gain, pair_diversity, tree_quality = (
            evaluate_classification(tree_proba, y_test_model,
                                    len(classes_idx))
        )
    else:
        tree_pred = np.vstack([model.predict(X_test) for model in models])
        ensemble, individual, gain, pair_diversity, tree_quality = (
            evaluate_regression(tree_pred, y_test_model)
        )

    row["prediction_diversity"] = float(np.nanmean(pair_diversity))
    row.update(individual)
    row.update(ensemble)
    row.update(gain)

    # 4. H1 – korelacje wewnątrz zespołu.
    for key in ("jaccard", "multiplicity", "target"):
        row[f"h1_pair_rho_{key}"] = _spearman(pairs[key], pair_diversity)

    for key in ("unique_fraction", "target_representativeness",
                "feature_representativeness"):
        row[f"h1_tree_rho_{key}"] = _spearman(trees[key], tree_quality)

    return row


# ============================================================
# EKSPERYMENT DLA JEDNEGO FOLDA
# ============================================================

def _encode_target(y_train, y_test, task):
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if task == "regression":
        return y_train.astype(float), y_test.astype(float), None

    classes = np.unique(y_train)
    mapping = {label: i for i, label in enumerate(classes)}

    y_train_idx = np.array([mapping[v] for v in y_train])
    y_test_idx = np.array([mapping[v] for v in y_test])

    return y_train_idx, y_test_idx, np.arange(len(classes))


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
    ensemble_type="random_forest",
):
    """
    Runs all sampling variants for a single train/test split.
    """

    start = time.perf_counter()

    X_train_processed, X_test_processed, _ = preprocess_train_test(
        X_train, X_test
    )

    y_train_model, y_test_model, classes_idx = _encode_target(
        y_train, y_test, task
    )

    max_features = get_tree_params(task, ensemble_type)["max_features"]

    variants = generate_sampling_variants(
        n_observations=len(y_train_model),
        y_train=y_train_model,
        task=task,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    fold_results = []

    for variant_id, variant in enumerate(variants):

        row = evaluate_samples(
            samples=variant["samples"],
            X_train=X_train_processed,
            y_train_model=y_train_model,
            X_test=X_test_processed,
            y_test_model=y_test_model,
            task=task,
            classes_idx=classes_idx,
            max_features=max_features,
            random_state=random_state + variant_id,
        )

        fold_results.append({
            "dataset": dataset_name,
            "task": task,
            "ensemble_type": ensemble_type,
            "fold": fold_id,
            "n_train": len(y_train_model),
            "n_features": X_train_processed.shape[1],
            "variant": variant["name"],
            "ratio": variant["parameters"].get("ratio"),
            "alpha": variant["parameters"].get("alpha"),
            "sampling_config": config_label(
                variant["name"], variant["parameters"]),
            **row,
        })

    elapsed = time.perf_counter() - start

    for result in fold_results:
        result["fold_seconds"] = elapsed

    return fold_results


# ============================================================
# EKSPERYMENT DLA JEDNEGO ZBIORU DANYCH
# ============================================================

def _make_splitter(task, n_splits, n_repeats, random_state):
    if task == "classification":
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats,
            random_state=random_state)

    return RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def run_dataset_experiment(
    dataset_name,
    n_splits=5,
    n_repeats=10,
    n_estimators=100,
    random_state=42,
    max_samples=20000,
    ensemble_type="random_forest",
    n_jobs=1,
    verbose=True,
):
    """
    Runs the experiment on a single dataset (all folds, all variants).

    Folds are independent, so they run in parallel (n_jobs).
    """

    dataset = load_dataset(
        dataset_name, max_samples=max_samples, random_state=random_state)

    X, y, task = dataset["X"], dataset["y"], dataset["task"]

    if verbose:
        print(
            f"[{dataset_name}] {task}, n={len(X)}, d={X.shape[1]}, "
            f"{n_splits}x{n_repeats} CV, M={n_estimators}, "
            f"{ensemble_type}",
            flush=True,
        )

    splitter = _make_splitter(task, n_splits, n_repeats, random_state)

    jobs = [
        delayed(run_fold_experiment)(
            X_train=X.iloc[train_idx],
            X_test=X.iloc[test_idx],
            y_train=y.iloc[train_idx],
            y_test=y.iloc[test_idx],
            task=task,
            dataset_name=dataset_name,
            fold_id=fold_id,
            n_estimators=n_estimators,
            random_state=random_state + fold_id,
            ensemble_type=ensemble_type,
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(X, y), start=1)
    ]

    fold_results = Parallel(n_jobs=n_jobs)(jobs)

    results = pd.DataFrame(
        [row for fold in fold_results for row in fold])

    if verbose:
        print(
            f"[{dataset_name}] done: {len(results)} rows, "
            f"mean fold time {results['fold_seconds'].mean():.1f}s",
            flush=True,
        )

    return results


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
    ensemble_type="random_forest",
    n_jobs=1,
    output_dir="results/main",
    overwrite=False,
):
    """
    Runs experiments on many datasets.

    Every dataset is saved to its own CSV file, so an interrupted run
    can be resumed (finished datasets are skipped).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_id, dataset_name in enumerate(dataset_names, start=1):

        path = output_dir / f"{dataset_name}.csv"

        if path.exists() and not overwrite:
            print(f"[{dataset_name}] skipped (exists: {path})", flush=True)
            all_results.append(pd.read_csv(path))
            continue

        print(f"\nDATASET {dataset_id}/{len(dataset_names)}", flush=True)

        start = time.perf_counter()

        results = run_dataset_experiment(
            dataset_name=dataset_name,
            n_splits=n_splits,
            n_repeats=n_repeats,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples=max_samples,
            ensemble_type=ensemble_type,
            n_jobs=n_jobs,
        )

        results["dataset_seconds"] = time.perf_counter() - start
        results.to_csv(path, index=False)
        all_results.append(results)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


# ============================================================
# EKSPERYMENT DODATKOWY (Sekcja 4.6): KONKRETNE LOSOWANIE
# ============================================================

def _draw_fold(
    X_train, X_test, y_train, y_test, task, dataset_name, fold_id,
    n_draws, n_estimators, random_state, ensemble_type,
):
    X_train_processed, X_test_processed, _ = preprocess_train_test(
        X_train, X_test)

    y_train_model, y_test_model, classes_idx = _encode_target(
        y_train, y_test, task)

    max_features = get_tree_params(task, ensemble_type)["max_features"]

    rows = []

    for draw in range(n_draws):
        seed = random_state * 1000 + draw

        samples = generate_bootstrap_samples(
            n=len(y_train_model), ratio=1.0,
            n_estimators=n_estimators, random_state=seed)

        row = evaluate_samples(
            samples=samples,
            X_train=X_train_processed,
            y_train_model=y_train_model,
            X_test=X_test_processed,
            y_test_model=y_test_model,
            task=task,
            classes_idx=classes_idx,
            max_features=max_features,
            random_state=seed,
        )

        rows.append({
            "dataset": dataset_name,
            "task": task,
            "ensemble_type": ensemble_type,
            "fold": fold_id,
            "draw": draw,
            **row,
        })

    return rows


def run_draw_experiment(
    dataset_name,
    n_splits=5,
    n_draws=20,
    n_estimators=100,
    random_state=42,
    max_samples=20000,
    ensemble_type="random_forest",
    n_jobs=1,
):
    """
    Section 4.6: for the first repetition of the CV (n_splits splits),
    trains n_draws classical bagging ensembles that differ only in the
    random draw of bootstrap samples (and tree seeds).

    The question: within one variant, do ensembles whose samples
    "look better" (by the pre-training measures) perform better?
    """

    dataset = load_dataset(
        dataset_name, max_samples=max_samples, random_state=random_state)

    X, y, task = dataset["X"], dataset["y"], dataset["task"]

    splitter = _make_splitter(task, n_splits, 1, random_state)

    jobs = [
        delayed(_draw_fold)(
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            task, dataset_name, fold_id, n_draws, n_estimators,
            random_state + fold_id, ensemble_type,
        )
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(X, y), start=1)
    ]

    results = Parallel(n_jobs=n_jobs)(jobs)

    return pd.DataFrame([row for fold in results for row in fold])
