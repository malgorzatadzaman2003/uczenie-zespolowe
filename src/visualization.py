import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# KONFIGURACJA WYKRESÓW
# ============================================================

DIVERSITY_METRICS = [
    "jaccard_distance",
    "multiplicity_distance",
    "unique_fraction",
]

REPRESENTATIVENESS_METRICS = [
    "target_distance",
    "target_representativeness",
    "feature_representativeness",
]

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


# Nazwy używane na osiach i w tytułach.
METRIC_LABELS = {
    "jaccard_distance": "Jaccard distance",
    "multiplicity_distance": "Multiplicity distance",
    "unique_fraction": "Unique fraction",
    "target_distance": "Target distribution distance",
    "target_representativeness": "Target representativeness (RY)",
    "feature_representativeness": "Feature representativeness (RX)",
    "prediction_diversity": "Prediction diversity",
    "ensemble_macro_f1": "Macro-F1",
    "ensemble_balanced_accuracy": "Balanced accuracy",
    "ensemble_roc_auc": "ROC-AUC",
    "ensemble_rmse": "RMSE",
    "ensemble_mae": "MAE",
    "ensemble_r2": "R²",
}


# ============================================================
# 1. FUNKCJE POMOCNICZE
# ============================================================

def _get_config_column(results):
    """
    Identifies the column that specifies the sampling variant.

    Supports both sampling_config and variant.
    """

    if "sampling_config" in results.columns:
        return "sampling_config"

    if "variant" in results.columns:
        return "variant"

    raise ValueError(
        "Nie znaleziono kolumny 'sampling_config' ani 'variant'."
    )


def _get_variant_name(config):
    """
    Extracts the strategy name from the configuration.

    Example:
    'S1_bootstrap | ratio=0.2' -> 'S1_bootstrap'
    """

    return str(config).split("|")[0].strip()


def _prepare_results(results):
    """
    Prepares results for visualization.

    If the data contains multiple folds, it calculates the averages
    for each combination of dataset, task, and configuration.
    """

    if results.empty:
        raise ValueError("Tabela wynikowa jest pusta.")

    config_column = _get_config_column(results)

    required_columns = {
        "dataset",
        "task",
        config_column,
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Brakuje wymaganych kolumn: {sorted(missing)}"
        )

    # Jeśli wyniki zawierają foldy, agregujemy je.
    if "fold" in results.columns:

        numeric_columns = results.select_dtypes(
            include=np.number
        ).columns.tolist()

        numeric_columns = [
            column
            for column in numeric_columns
            if column != "fold"
        ]

        grouping_columns = [
            "dataset",
            "task",
            config_column,
        ]

        prepared = (
            results
            .groupby(grouping_columns, dropna=False)[
                numeric_columns
            ]
            .mean()
            .reset_index()
        )

    else:
        prepared = results.copy()

    prepared["strategy"] = prepared[
        config_column
    ].apply(_get_variant_name)

    return prepared, config_column


def _save_figure(fig, output_path):
    """
    Saves the figure at a high resolution and closes it.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def _metric_label(metric):
    """
    Returns a readable name for the metric.
    """

    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def _available_metrics(results, metrics):
    """
    Returns only those metrics that appear in the table.
    """

    return [
        metric for metric in metrics
        if metric in results.columns
    ]


# ============================================================
# 2. WYKRESY RÓŻNORODNOŚCI PRÓBEK
# ============================================================

def plot_sample_diversity(
    results,
    output_dir="figures",
):
    """
    Creates plots of sample diversity metrics
    depending on the sampling variant.

    Results are displayed as boxplots,
    since each variant may have results
    from multiple datasets.
    """

    output_dir = Path(output_dir) / "diversity"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _available_metrics(
        results,
        DIVERSITY_METRICS,
    )

    for metric in metrics:

        data = results[
            ["strategy", metric]
        ].dropna()

        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.boxplot(
            data=data,
            x="strategy",
            y=metric,
            ax=ax,
        )

        ax.set_title(
            f"Sample diversity: {_metric_label(metric)}"
        )

        ax.set_xlabel("Sampling strategy")
        ax.set_ylabel(_metric_label(metric))

        ax.tick_params(
            axis="x",
            rotation=25,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{metric}.png",
        )


# ============================================================
# 3. WYKRESY REPREZENTATYWNOŚCI PRÓBEK
# ============================================================

def plot_representativeness(
    results,
    output_dir="figures",
):
    """
    Creates plots of sample representativeness metrics
    depending on the sampling variant.
    """

    output_dir = Path(output_dir) / "representativeness"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _available_metrics(
        results,
        REPRESENTATIVENESS_METRICS,
    )

    for metric in metrics:

        data = results[
            ["strategy", metric]
        ].dropna()

        if data.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        sns.boxplot(
            data=data,
            x="strategy",
            y=metric,
            ax=ax,
        )

        ax.set_title(
            f"Sample representativeness: {_metric_label(metric)}"
        )

        ax.set_xlabel("Sampling strategy")
        ax.set_ylabel(_metric_label(metric))

        ax.tick_params(
            axis="x",
            rotation=25,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{metric}.png",
        )


# ============================================================
# 4. WYKRESY JAKOŚCI KLASYFIKACJI I REGRESJI
# ============================================================

def plot_model_quality(
    results,
    output_dir="figures",
):
    """
    Creates separate plots of ensemble quality
    for classification and regression.
    """

    output_dir = Path(output_dir) / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_metrics = {
        "classification": CLASSIFICATION_METRICS,
        "regression": REGRESSION_METRICS,
    }

    for task, metrics in task_metrics.items():

        task_data = results[
            results["task"] == task
        ]

        if task_data.empty:
            continue

        available = _available_metrics(
            task_data,
            metrics,
        )

        for metric in available:

            data = task_data[
                ["strategy", metric]
            ].dropna()

            if data.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 6))

            sns.boxplot(
                data=data,
                x="strategy",
                y=metric,
                ax=ax,
            )

            ax.set_title(
                f"{task.capitalize()} - {_metric_label(metric)}"
            )

            ax.set_xlabel("Sampling strategy")
            ax.set_ylabel(_metric_label(metric))

            ax.tick_params(
                axis="x",
                rotation=25,
            )

            ax.grid(
                axis="y",
                alpha=0.3,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir / f"{task}_{metric}.png",
            )


# ============================================================
# 5. ZALEŻNOŚĆ RÓŻNORODNOŚCI OD JAKOŚCI
# ============================================================

def plot_diversity_quality_relationship(
    results,
    output_dir="figures",
):
    """
    Creates scatter plots showing the relationship
    between sample diversity and ensemble quality.

    For classification, uses F1-score.
    For regression, uses RMSE.

    Each point represents one combination
    of dataset and sampling variant.
    """

    output_dir = (
        Path(output_dir) / "diversity_quality"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    diversity_metrics = (
        DIVERSITY_METRICS
        + REPRESENTATIVENESS_METRICS
        + ["prediction_diversity"]
    )

    task_quality_metrics = {
        "classification": "ensemble_macro_f1",
        "regression": "ensemble_rmse",
    }

    for task, quality_metric in task_quality_metrics.items():

        task_data = results[
            results["task"] == task
        ]

        if quality_metric not in task_data.columns:
            continue

        for diversity_metric in diversity_metrics:

            if diversity_metric not in task_data.columns:
                continue

            data = task_data[
                [
                    diversity_metric,
                    quality_metric,
                    "strategy",
                ]
            ].dropna()

            if len(data) < 2:
                continue

            fig, ax = plt.subplots(figsize=(9, 6))

            sns.scatterplot(
                data=data,
                x=diversity_metric,
                y=quality_metric,
                hue="strategy",
                ax=ax,
            )

            ax.set_title(
                f"{task.capitalize()}: "
                f"{_metric_label(diversity_metric)} "
                f"vs {_metric_label(quality_metric)}"
            )

            ax.set_xlabel(
                _metric_label(diversity_metric)
            )

            ax.set_ylabel(
                _metric_label(quality_metric)
            )

            ax.grid(alpha=0.3)
            ax.legend(
                title="Sampling strategy",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir
                / f"{task}_{diversity_metric}_vs_{quality_metric}.png",
            )


# ============================================================
# 6. PORÓWNANIE STRATEGII PRÓBKOWANIA
# ============================================================

def plot_strategy_comparison(
    results,
    output_dir="figures",
):
    """
    It compares four main sampling strategies:
    S1, S2, S3, and S4.

    The charts show the distribution of results for the individual
    variants belonging to a given strategy.
    """

    output_dir = Path(output_dir) / "strategies"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        CLASSIFICATION_METRICS
        + REGRESSION_METRICS
        + DIVERSITY_METRICS
    )

    for metric in metrics:

        if metric not in results.columns:
            continue

        data = results[
            ["strategy", "task", metric]
        ].dropna()

        if data.empty:
            continue

        for task in data["task"].unique():

            task_data = data[
                data["task"] == task
            ]

            if task_data.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.boxplot(
                data=task_data,
                x="strategy",
                y=metric,
                ax=ax,
            )

            ax.set_title(
                f"{task.capitalize()} - "
                f"Comparison of sampling strategies: "
                f"{_metric_label(metric)}"
            )

            ax.set_xlabel("Sampling strategy")
            ax.set_ylabel(_metric_label(metric))

            ax.tick_params(
                axis="x",
                rotation=25,
            )

            ax.grid(
                axis="y",
                alpha=0.3,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir / f"{task}_{metric}.png",
            )


# ============================================================
# 7. HEATMAPA KORELACJI SPEARMANA
# ============================================================

def plot_spearman_heatmap(
    correlations,
    output_dir="figures",
):
    """
    Creates a heatmap of mean Spearman correlations.

    First, it averages the rho coefficients for datasets
    belonging to the same task.
    """

    if correlations is None or correlations.empty:
        print(
            "Pominięto heatmapę: brak wyników korelacji."
        )
        return

    output_dir = Path(output_dir) / "correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "task",
        "diversity_metric",
        "quality_metric",
        "spearman_rho",
    }

    if not required.issubset(correlations.columns):
        print(
            "Pominięto heatmapę: "
            "brakuje wymaganych kolumn."
        )
        return

    for task in correlations["task"].unique():

        task_data = correlations[
            correlations["task"] == task
        ]

        pivot = task_data.pivot_table(
            index="diversity_metric",
            columns="quality_metric",
            values="spearman_rho",
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            center=0,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title(
            f"Mean Spearman correlations - {task}"
        )

        ax.set_xlabel("Ensemble quality metric")
        ax.set_ylabel("Diversity metric")

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{task}_spearman_heatmap.png",
        )


# ============================================================
# 8. GŁÓWNA FUNKCJA WIZUALIZACJI
# ============================================================

def run_visualizations(
    results,
    statistical_results=None,
    output_dir="figures",
):
    """
    Generates all project plots.

    Parameters
    ----------
    results : pandas.DataFrame
        Experimental results at the fold level,
        or already aggregated results.

    statistical_results : dict, optional
        A dictionary returned by `run_statistical_analysis`.
        If provided, a Spearman correlation heatmap
        will also be generated.

    output_dir : str
        Directory where the plots will be saved.

    Returns
    -------
    pandas.DataFrame
        Prepared, aggregated results.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prepared, config_column = _prepare_results(
        results
    )

    print("\n" + "=" * 60)
    print("GENEROWANIE WYKRESÓW")
    print("=" * 60)

    print(
        f"Liczba rekordów po agregacji: {len(prepared)}"
    )

    print(
        f"Kolumna konfiguracji: {config_column}"
    )

    plot_sample_diversity(
        prepared,
        output_dir=output_path,
    )

    plot_representativeness(
        prepared,
        output_dir=output_path,
    )

    plot_model_quality(
        prepared,
        output_dir=output_path,
    )

    plot_diversity_quality_relationship(
        prepared,
        output_dir=output_path,
    )

    plot_strategy_comparison(
        prepared,
        output_dir=output_path,
    )

    if statistical_results is not None:
        correlations = statistical_results.get(
            "spearman_by_dataset"
        )

        plot_spearman_heatmap(
            correlations,
            output_dir=output_path,
        )

    print("\nZakończono generowanie wykresów.")
    print(f"Wykresy zapisano w katalogu: {output_path}")

    return prepared