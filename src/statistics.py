import numpy as np
import pandas as pd

from itertools import combinations
from pathlib import Path

from scipy.stats import (
    spearmanr,
    friedmanchisquare,
    wilcoxon,
    rankdata,
)


# ============================================================
# KONFIGURACJA
# ============================================================

CLASSIFICATION_METRICS = [
    "ensemble_macro_f1",
    "ensemble_balanced_accuracy",
    "ensemble_roc_auc",
]

REGRESSION_METRICS = [
    "ensemble_rmse",
    "ensemble_mae",
    "ensemble_r2",
]

DIVERSITY_METRICS = [
    "jaccard_distance",
    "multiplicity_distance",
    "target_distance",
    "unique_fraction",
    "target_representativeness",
    "feature_representativeness",
    "prediction_diversity",
]


# ============================================================
# 1. AGREGACJA WYNIKÓW Z FOLDÓW
# ============================================================

def aggregate_fold_results(results):
    """
    Aggregates cross-validation results.

    Maintains each sampling configuration separately:
    variant family, ratio, and alpha.

    The unit for further statistical analysis is the dataset,
    not an individual fold.
    """

    required_columns = {"dataset", "task", "variant"}

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Brakuje wymaganych kolumn: {sorted(missing)}"
        )

    # Rozróżniamy konfiguracje przez ratio i alpha.
    group_columns = ["dataset", "task", "variant"]

    for column in ["ratio", "alpha"]:
        if column in results.columns:
            group_columns.append(column)

    numeric_columns = results.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        column for column in numeric_columns
        if column not in group_columns and column != "fold"
    ]

    aggregated = (
        results
        .groupby(
            group_columns,
            dropna=False
        )[numeric_columns]
        .mean()
        .reset_index()
    )

    # Czytelny identyfikator konkretnej konfiguracji.
    def make_config_label(row):
        parts = [str(row["variant"])]

        if "ratio" in aggregated.columns and pd.notna(row["ratio"]):
            parts.append(f"ratio={row['ratio']:g}")

        if "alpha" in aggregated.columns and pd.notna(row["alpha"]):
            parts.append(f"alpha={row['alpha']:g}")

        return " | ".join(parts)

    aggregated["sampling_config"] = aggregated.apply(
        make_config_label,
        axis=1,
    )

    return aggregated


# ============================================================
# 2. KORELACJE SPEARMANA
# ============================================================

def calculate_spearman_correlations(
    aggregated,
    diversity_metrics=None,
    quality_metrics=None,
):
    """
    Calculates Spearman correlations separately for each dataset.

    For a given dataset, the correlation is computed between the values
    of diversity and quality metrics obtained for individual
    sampling variants.

    Results from different datasets are not combined prior to the
    calculation of rho.
    """

    if diversity_metrics is None:
        diversity_metrics = DIVERSITY_METRICS

    if quality_metrics is None:
        quality_metrics = (
            CLASSIFICATION_METRICS + REGRESSION_METRICS
        )

    records = []

    for (dataset, task), group in aggregated.groupby(
        ["dataset", "task"]
    ):

        if task == "classification":
            task_quality_metrics = [
                metric for metric in quality_metrics
                if metric in CLASSIFICATION_METRICS
            ]

        elif task == "regression":
            task_quality_metrics = [
                metric for metric in quality_metrics
                if metric in REGRESSION_METRICS
            ]

        else:
            continue

        for diversity_metric in diversity_metrics:

            if diversity_metric not in group.columns:
                continue

            for quality_metric in task_quality_metrics:

                if quality_metric not in group.columns:
                    continue

                data = group[
                    [diversity_metric, quality_metric]
                ].dropna()

                # Potrzebujemy co najmniej 3 wariantów.
                if len(data) < 3:
                    continue

                # Korelacja nie jest zdefiniowana,
                # jeśli jedna ze zmiennych jest stała.
                if (
                    data[diversity_metric].nunique() < 2
                    or data[quality_metric].nunique() < 2
                ):
                    continue

                rho, p_value = spearmanr(
                    data[diversity_metric],
                    data[quality_metric],
                )

                records.append({
                    "dataset": dataset,
                    "task": task,
                    "diversity_metric": diversity_metric,
                    "quality_metric": quality_metric,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_variants": len(data),
                })

    return pd.DataFrame(records)


def summarize_spearman_correlations(correlations):
    """
    Summarizes the correlations calculated separately for the datasets.

    Returns, the mean and median rho, percentage of positive correlations within a given task and pair of measures.
    """

    if correlations.empty:
        return pd.DataFrame()

    records = []

    group_columns = [
        "task",
        "diversity_metric",
        "quality_metric",
    ]

    for keys, group in correlations.groupby(group_columns):

        rhos = group["spearman_rho"].dropna().to_numpy()

        if len(rhos) == 0:
            continue

        # Testujemy, czy korelacje z poszczególnych zbiorów
        # różnią się systematycznie od zera.
        if len(rhos) >= 3 and not np.allclose(rhos, 0):
            try:
                statistic, p_value = wilcoxon(
                    rhos,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                statistic = np.nan
                p_value = np.nan
        else:
            statistic = np.nan
            p_value = np.nan

        records.append({
            "task": keys[0],
            "diversity_metric": keys[1],
            "quality_metric": keys[2],
            "n_datasets": len(rhos),
            "mean_rho": np.mean(rhos),
            "median_rho": np.median(rhos),
            "positive_rho_fraction": np.mean(rhos > 0),
            "negative_rho_fraction": np.mean(rhos < 0),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_value": p_value,
        })

    return pd.DataFrame(records)


# ============================================================
# 3. TEST FRIEDMANA
# ============================================================

def friedman_tests(
    aggregated,
    quality_metrics=None,
):
    """
    Compares all sampling variants using the Friedman test.

    Datasets serve as the blocks, while the sampling variants
    are the methods being compared.

    Classification and regression are analyzed separately.
    """

    if quality_metrics is None:
        quality_metrics = (
            CLASSIFICATION_METRICS + REGRESSION_METRICS
        )

    records = []

    for task in aggregated["task"].unique():

        task_data = aggregated[
            aggregated["task"] == task
        ]

        task_metrics = (
            CLASSIFICATION_METRICS
            if task == "classification"
            else REGRESSION_METRICS
        )

        for metric in task_metrics:

            if metric not in quality_metrics:
                continue

            if metric not in task_data.columns:
                continue

            pivot = task_data.pivot_table(
                index="dataset",
                columns="sampling_config",
                values=metric,
                aggfunc="mean",
            )

            # Friedman wymaga kompletnych bloków.
            pivot = pivot.dropna(axis=0, how="any")

            n_datasets = len(pivot)
            n_variants = len(pivot.columns)

            if n_datasets < 3 or n_variants < 3:
                continue

            statistic, p_value = friedmanchisquare(
                *[
                    pivot[variant].to_numpy()
                    for variant in pivot.columns
                ]
            )

            records.append({
                "task": task,
                "metric": metric,
                "n_datasets": n_datasets,
                "n_variants": n_variants,
                "friedman_statistic": statistic,
                "p_value": p_value,
            })

    return pd.DataFrame(records)


# ============================================================
# 4. KOREKTA HOLMA
# ============================================================

def holm_correction(p_values):
    """
    Holm correction for multiple comparisons.

    Returns adjusted p-values in the input order.
    """

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.where(np.isfinite(p_values))[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    m = len(sorted_p)

    adjusted_sorted = np.maximum.accumulate(
        [
            min(1.0, (m - i) * p)
            for i, p in enumerate(sorted_p)
        ]
    )

    adjusted_valid = np.empty(m)
    adjusted_valid[order] = adjusted_sorted

    adjusted[valid_indices] = adjusted_valid

    return adjusted


# ============================================================
# 5. WIELKOŚĆ EFEKTU
# ============================================================

def paired_rank_biserial_effect(a, b):
    """
    Calculates the rank-biserial effect size for dependent samples.

    A positive value indicates that variant A outperforms variant B
    in terms of the numerical value of the metric.
    """

    differences = np.asarray(a) - np.asarray(b)

    differences = differences[
        np.isfinite(differences)
    ]

    differences = differences[differences != 0]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(differences))

    positive_rank_sum = np.sum(
        ranks[differences > 0]
    )

    negative_rank_sum = np.sum(
        ranks[differences < 0]
    )

    denominator = positive_rank_sum + negative_rank_sum

    if denominator == 0:
        return 0.0

    return (
        positive_rank_sum - negative_rank_sum
    ) / denominator


# ============================================================
# 6. TESTY WILCOXONA DLA PAR WARIANTÓW
# ============================================================

def wilcoxon_pairwise_tests(
    aggregated,
    quality_metrics=None,
):
    """
    Compares sampling variants in pairs using the Wilcoxon test.

    Each pair is compared using the same datasets.
    The results are further adjusted using the Holm method.
    """

    if quality_metrics is None:
        quality_metrics = (
            CLASSIFICATION_METRICS + REGRESSION_METRICS
        )

    all_records = []

    for task in aggregated["task"].unique():

        task_data = aggregated[
            aggregated["task"] == task
        ]

        task_metrics = (
            CLASSIFICATION_METRICS
            if task == "classification"
            else REGRESSION_METRICS
        )

        for metric in task_metrics:

            if metric not in quality_metrics:
                continue

            if metric not in task_data.columns:
                continue

            pivot = task_data.pivot_table(
                index="dataset",
                columns="sampling_config",
                values=metric,
                aggfunc="mean",
            )

            variants = list(pivot.columns)

            for variant_a, variant_b in combinations(
                variants, 2
            ):

                pair = pivot[
                    [variant_a, variant_b]
                ].dropna()

                if len(pair) < 3:
                    continue

                a = pair[variant_a].to_numpy()
                b = pair[variant_b].to_numpy()

                differences = a - b

                if np.allclose(differences, 0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    try:
                        result = wilcoxon(
                            a,
                            b,
                            alternative="two-sided",
                            zero_method="wilcox",
                            method="auto",
                        )

                        statistic = result.statistic
                        p_value = result.pvalue

                    except ValueError:
                        statistic = np.nan
                        p_value = np.nan

                effect_size = paired_rank_biserial_effect(
                    a, b
                )

                all_records.append({
                    "task": task,
                    "metric": metric,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "n_datasets": len(pair),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "rank_biserial_effect": effect_size,
                    "mean_difference_a_minus_b": np.mean(
                        differences
                    ),
                })

    results = pd.DataFrame(all_records)

    if results.empty:
        return results

    # Korekta Holma osobno dla każdej kombinacji zadania
    # i metryki jakości.
    results["p_value_holm"] = np.nan

    for _, indices in results.groupby(
        ["task", "metric"]
    ).groups.items():

        results.loc[indices, "p_value_holm"] = (
            holm_correction(
                results.loc[indices, "p_value"].to_numpy()
            )
        )

    return results


# ============================================================
# 7. GŁÓWNA FUNKCJA ANALIZY STATYSTYCZNEJ
# ============================================================

def run_statistical_analysis(
    results,
    output_dir="results",
):
    """
    Runs a full statistical analysis and saves the results as CSV files.

    Parameters
    ----------
    results : pandas.DataFrame
        Results returned by run_experiments or
        run_dataset_experiment.

    output_dir : str
        Directory where the result tables will be saved.

    Returns
    -------
    dict
        A dictionary containing all the result tables.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Agregacja foldów.
    aggregated = aggregate_fold_results(results)

    # 2. Korelacje Spearmana.
    correlations = calculate_spearman_correlations(
        aggregated
    )

    correlation_summary = summarize_spearman_correlations(
        correlations
    )

    # 3. Test Friedmana.
    friedman = friedman_tests(aggregated)

    # 4. Testy Wilcoxona i wielkość efektu.
    wilcoxon_results = wilcoxon_pairwise_tests(
        aggregated
    )

    tables = {
        "aggregated_results": aggregated,
        "spearman_by_dataset": correlations,
        "spearman_summary": correlation_summary,
        "friedman_tests": friedman,
        "wilcoxon_pairwise": wilcoxon_results,
    }

    # Zapis każdej tabeli do osobnego pliku.
    for name, table in tables.items():
        file_path = output_path / f"{name}.csv"
        table.to_csv(file_path, index=False)

    return tables