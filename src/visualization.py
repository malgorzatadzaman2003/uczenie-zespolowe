
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. GLOBALNY STYL WIZUALIZACJI
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.1,
)

# Stałe kolory dla strategii próbkowania.
# Te same kolory są wykorzystywane na wszystkich wykresach.

STRATEGY_COLORS = {
    "S1_bootstrap": "#4477AA",
    "S2_subsampling": "#EE7733",
    "S3_common_core": "#228833",
    "S4_stratified_bootstrap": "#AA3377",
}

STRATEGY_ORDER = [
    "S1_bootstrap",
    "S2_subsampling",
    "S3_common_core",
    "S4_stratified_bootstrap",
]

# Kierunek interpretacji metryk jakości.

QUALITY_DIRECTIONS = {
    "ensemble_macro_f1": "higher is better",
    "ensemble_balanced_accuracy": "higher is better",
    "ensemble_roc_auc": "higher is better",
    "ensemble_rmse": "lower is better",
    "ensemble_mae": "lower is better",
    "ensemble_r2": "higher is better",
}


def _style_axis(ax):
    """
    Applies a consistent visual style to plot axes.
    """

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)

    ax.tick_params(
        axis="both",
        labelsize=10,
    )


def _strategy_palette(strategies):
    """
    Returns colors only for strategies present in the data.
    """

    return {
        strategy: STRATEGY_COLORS.get(
            strategy,
            "#999999",
        )
        for strategy in strategies
    }


def _add_boxplot(
    data,
    metric,
    ax,
    order=None,
):
    """
    Draws a colored boxplot with individual observations.

    The boxplot displays:
    - median,
    - interquartile range,
    - mean as a white diamond,
    - individual observations as dark points.
    """

    if order is None:
        order = [
            strategy
            for strategy in STRATEGY_ORDER
            if strategy in data["strategy"].unique()
        ]

    palette = _strategy_palette(order)

    sns.boxplot(
        data=data,
        x="strategy",
        y=metric,
        order=order,
        palette=palette,
        width=0.6,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 7,
        },
        medianprops={
            "color": "#222222",
            "linewidth": 2,
        },
        whiskerprops={
            "linewidth": 1.3,
        },
        capprops={
            "linewidth": 1.3,
        },
        ax=ax,
    )

    sns.stripplot(
        data=data,
        x="strategy",
        y=metric,
        order=order,
        color="#333333",
        alpha=0.5,
        size=4.5,
        jitter=0.16,
        ax=ax,
        zorder=2,
    )

    _style_axis(ax)

    ax.set_xlabel("Sampling strategy")
    ax.set_ylabel(_metric_label(metric))

    ax.tick_params(
        axis="x",
        rotation=20,
    )


# ============================================================
# 2. KONFIGURACJA METRYK
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

METRIC_LABELS = {
    "jaccard_distance": "Jaccard distance",
    "multiplicity_distance": "Multiplicity distance",
    "unique_fraction": "Unique fraction",
    "target_distance": "Target distribution distance",
    "target_representativeness": "Target representativeness (RY)",
    "feature_representativeness": "Feature representativeness (RX)",
    "prediction_diversity": "Prediction diversity",
    "ensemble_macro_f1": "F1-score",
    "ensemble_balanced_accuracy": "Balanced accuracy",
    "ensemble_roc_auc": "ROC-AUC",
    "ensemble_rmse": "RMSE",
    "ensemble_mae": "MAE",
    "ensemble_r2": "R²",
}


# ============================================================
# 3. FUNKCJE POMOCNICZE
# ============================================================

def _get_config_column(results):
    """
    Identifies the column specifying the sampling configuration.

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
    Prepares experimental results for visualization.

    If fold-level results are provided, calculates averages
    for each dataset, task, and sampling configuration.
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
            .groupby(
                grouping_columns,
                dropna=False,
            )[numeric_columns]
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
    Saves a figure at high resolution and closes it.
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
        facecolor="white",
    )

    plt.close(fig)


def _metric_label(metric):
    """
    Returns a readable metric name.
    """

    return METRIC_LABELS.get(
        metric,
        metric.replace("_", " ").title(),
    )


def _available_metrics(results, metrics):
    """
    Returns metrics that are available in the results table.
    """

    return [
        metric
        for metric in metrics
        if metric in results.columns
    ]


def _quality_direction(metric):
    """
    Returns the preferred direction for a quality metric.
    """

    return QUALITY_DIRECTIONS.get(
        metric,
        "",
    )


# ============================================================
# 4. WYKRESY RÓŻNORODNOŚCI PRÓBEK
# ============================================================

def plot_sample_diversity(
    results,
    output_dir="figures",
):
    """
    Creates boxplots of sample diversity metrics
    for different sampling strategies.
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

        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        _add_boxplot(
            data=data,
            metric=metric,
            ax=ax,
        )

        ax.set_title(
            f"Sample diversity: {_metric_label(metric)}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{metric}.png",
        )


# ============================================================
# 5. WYKRESY REPREZENTATYWNOŚCI PRÓBEK
# ============================================================

def plot_representativeness(
    results,
    output_dir="figures",
):
    """
    Creates boxplots of sample representativeness metrics.
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

        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        _add_boxplot(
            data=data,
            metric=metric,
            ax=ax,
        )

        ax.set_title(
            f"Sample representativeness: {_metric_label(metric)}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{metric}.png",
        )


# ============================================================
# 6. WYKRESY JAKOŚCI KLASYFIKACJI I REGRESJI
# ============================================================

def plot_model_quality(
    results,
    output_dir="figures",
):
    """
    Creates separate quality plots for classification
    and regression.
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

            fig, ax = plt.subplots(
                figsize=(11, 6),
            )

            _add_boxplot(
                data=data,
                metric=metric,
                ax=ax,
            )

            direction = _quality_direction(metric)

            title = (
                f"{task.capitalize()} performance: "
                f"{_metric_label(metric)}"
            )

            if direction:
                title += f" ({direction})"

            ax.set_title(
                title,
                fontsize=14,
                fontweight="bold",
                pad=15,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir / f"{task}_{metric}.png",
            )


# ============================================================
# 7. ZALEŻNOŚĆ RÓŻNORODNOŚCI OD JAKOŚCI
# ============================================================

def plot_diversity_quality_relationship(
    results,
    output_dir="figures",
):
    """
    Creates scatter plots showing the relationship
    between sample diversity and ensemble quality.

    Classification:
        Macro-F1 is used as the quality metric.

    Regression:
        RMSE is used as the quality metric.

    Each point represents one dataset and sampling configuration.
    """

    output_dir = Path(output_dir) / "diversity_quality"
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

            fig, ax = plt.subplots(
                figsize=(9, 6),
            )

            strategies = [
                strategy
                for strategy in STRATEGY_ORDER
                if strategy in data["strategy"].unique()
            ]

            sns.scatterplot(
                data=data,
                x=diversity_metric,
                y=quality_metric,
                hue="strategy",
                style="strategy",
                hue_order=strategies,
                palette=_strategy_palette(strategies),
                s=100,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                ax=ax,
            )

            _style_axis(ax)

            ax.set_title(
                f"{task.capitalize()}: "
                f"{_metric_label(diversity_metric)} "
                f"vs {_metric_label(quality_metric)}",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax.set_xlabel(
                _metric_label(diversity_metric)
            )

            ax.set_ylabel(
                _metric_label(quality_metric)
            )

            ax.legend(
                title="Sampling strategy",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir
                / f"{task}_{diversity_metric}_vs_{quality_metric}.png",
            )


# ============================================================
# 8. PORÓWNANIE STRATEGII PRÓBKOWANIA
# ============================================================

def plot_strategy_comparison(
    results,
    output_dir="figures",
):
    """
    Compares sampling strategies using boxplots.

    The plots show the distribution of results
    for individual configurations within each strategy.
    """

    output_dir = Path(output_dir) / "strategies"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        CLASSIFICATION_METRICS
        + REGRESSION_METRICS
        + DIVERSITY_METRICS
        + REPRESENTATIVENESS_METRICS
        + ["prediction_diversity"]
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

            fig, ax = plt.subplots(
                figsize=(10, 6),
            )

            _add_boxplot(
                data=task_data,
                metric=metric,
                ax=ax,
            )

            ax.set_title(
                f"{task.capitalize()} - "
                f"Sampling strategy comparison: "
                f"{_metric_label(metric)}",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir / f"{task}_{metric}.png",
            )


# ============================================================
# 9. ŚREDNIA JAKOŚĆ STRATEGII - WYKRESY SŁUPKOWE
# ============================================================

def plot_strategy_mean_quality(
    results,
    output_dir="figures",
):
    """
    Creates barplots comparing the mean ensemble quality
    of sampling strategies across datasets.

    First, configurations belonging to the same strategy
    are averaged within each dataset.

    Then, the mean and standard deviation are calculated
    across datasets.

    Each dataset therefore has equal weight,
    regardless of the number of configurations.
    """

    output_dir = Path(output_dir) / "mean_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_metrics = {
        "classification": CLASSIFICATION_METRICS,
        "regression": REGRESSION_METRICS,
    }

    for task, metrics in task_metrics.items():

        task_data = results[
            results["task"] == task
        ].copy()

        if task_data.empty:
            continue

        available = _available_metrics(
            task_data,
            metrics,
        )

        for metric in available:

            data = task_data[
                ["dataset", "strategy", metric]
            ].dropna()

            if data.empty:
                continue

            # Najpierw uśredniamy warianty strategii
            # w obrębie każdego zbioru danych.

            dataset_means = (
                data
                .groupby(
                    ["dataset", "strategy"],
                    as_index=False,
                )[metric]
                .mean()
            )

            # Następnie obliczamy średnią i odchylenie
            # standardowe pomiędzy zbiorami danych.

            summary = (
                dataset_means
                .groupby("strategy")[metric]
                .agg(["mean", "std", "count"])
                .reindex(STRATEGY_ORDER)
                .dropna(subset=["mean"])
            )

            if summary.empty:
                continue

            strategies = summary.index.tolist()

            colors = [
                STRATEGY_COLORS.get(
                    strategy,
                    "#999999",
                )
                for strategy in strategies
            ]

            means = summary["mean"].to_numpy()

            # Dla jednego zbioru danych odchylenie standardowe
            # nie jest określone, dlatego pokazujemy słupki bez błędu.

            errors = (
                summary["std"]
                .fillna(0)
                .to_numpy()
            )

            fig, ax = plt.subplots(
                figsize=(10, 6),
            )

            bars = ax.bar(
                strategies,
                means,
                color=colors,
                width=0.65,
                edgecolor="white",
                linewidth=1.2,
                yerr=errors,
                capsize=5,
                error_kw={
                    "elinewidth": 1.3,
                    "capthick": 1.3,
                },
            )

            _style_axis(ax)

            # Wartości średnie nad słupkami.

            for bar, value in zip(bars, means):

                ax.annotate(
                    f"{value:.3f}",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            direction = _quality_direction(metric)

            title = (
                f"Mean {task} performance: "
                f"{_metric_label(metric)}"
            )

            if direction:
                title += f" ({direction})"

            ax.set_title(
                title,
                fontsize=13,
                fontweight="bold",
                pad=15,
            )

            ax.set_xlabel("Sampling strategy")
            ax.set_ylabel(_metric_label(metric))

            ax.tick_params(
                axis="x",
                rotation=20,
            )

            fig.tight_layout()

            _save_figure(
                fig,
                output_dir / f"{task}_{metric}_mean.png",
            )


# ============================================================
# 10. HEATMAPA KORELACJI SPEARMANA
# ============================================================

def plot_spearman_heatmap(
    correlations,
    output_dir="figures",
):
    """
    Creates heatmaps of mean Spearman correlations.

    Correlations are averaged across datasets
    belonging to the same task.

    The color scale ranges from -1 to 1.
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

    for task in correlations["task"].dropna().unique():

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

        fig, ax = plt.subplots(
            figsize=(10, 7),
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            center=0,
            vmin=-1,
            vmax=1,
            cmap="RdYlBu_r",
            linewidths=0.7,
            linecolor="white",
            cbar_kws={
                "label": "Spearman correlation (rho)",
            },
            ax=ax,
        )

        ax.set_title(
            f"Mean Spearman correlations - {task}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        ax.set_xlabel("Ensemble quality metric")
        ax.set_ylabel("Diversity metric")

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        ax.tick_params(
            axis="y",
            rotation=0,
        )

        fig.tight_layout()

        _save_figure(
            fig,
            output_dir / f"{task}_spearman_heatmap.png",
        )


# ============================================================
# 11. GŁÓWNA FUNKCJA WIZUALIZACJI
# ============================================================

def run_visualizations(
    results,
    statistical_results=None,
    output_dir="figures",
):
    """
    Generates all project visualizations.

    Parameters
    ----------
    results : pandas.DataFrame
        Experimental results at the fold level,
        or already aggregated results.

    statistical_results : dict, optional
        Dictionary returned by run_statistical_analysis.
        If provided, a Spearman correlation heatmap
        will also be generated.

    output_dir : str
        Directory where figures will be saved.

    Returns
    -------
    pandas.DataFrame
        Prepared and aggregated results.
    """

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print(
        "Liczba zbiorów danych: "
        f"{prepared['dataset'].nunique()}"
    )

    print(
        "Liczba strategii: "
        f"{prepared['strategy'].nunique()}"
    )

    # 1. Różnorodność próbek

    plot_sample_diversity(
        prepared,
        output_dir=output_path,
    )

    # 2. Reprezentatywność próbek

    plot_representativeness(
        prepared,
        output_dir=output_path,
    )

    # 3. Jakość modeli

    plot_model_quality(
        prepared,
        output_dir=output_path,
    )

    # 4. Zależność różnorodności od jakości

    plot_diversity_quality_relationship(
        prepared,
        output_dir=output_path,
    )

    # 5. Porównanie strategii

    plot_strategy_comparison(
        prepared,
        output_dir=output_path,
    )

    # 6. Średnia jakość strategii pomiędzy zbiorami danych

    plot_strategy_mean_quality(
        prepared,
        output_dir=output_path,
    )

    # 7. Heatmapa korelacji Spearmana

    if statistical_results is not None:

        correlations = statistical_results.get(
            "spearman_by_dataset"
        )

        plot_spearman_heatmap(
            correlations,
            output_dir=output_path,
        )

    print("\nZakończono generowanie wykresów.")

    print(
        f"Wykresy zapisano w katalogu: {output_path}"
    )

    return prepared
