"""
Figures for the checkpoint / report. Complements src/visualization.py
(generic per-strategy boxplots) with figures tied to the hypotheses.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    H1_PAIR,
    H1_TREE,
    SAMPLE_MEASURES,
    TREE_QUALITY,
)
from src.diversity import pairwise_jaccard_vector, sample_count_matrix
from src.experiment import BASELINE_CONFIG
from src.sampling import generate_bootstrap_samples


COLORS = {
    "S1_bootstrap": "#4477AA",
    "S2_subsampling": "#EE7733",
    "S3_common_core": "#228833",
    "S4_stratified_bootstrap": "#AA3377",
}
TASK_LABEL = {"classification": "Klasyfikacja", "regression": "Regresja"}
SCORE_LABEL = {
    "classification": "Δ macro-F1 vs klasyczny bagging",
    "regression": "względna poprawa RMSE vs klasyczny bagging",
}
MEASURE_LABEL = {
    "jaccard_distance": "$D_J$",
    "multiplicity_distance": "$D_C$",
    "target_distance": "$D_Y$",
    "unique_fraction": "$U$",
    "target_representativeness": "$R_Y$",
    "feature_representativeness": "$R_X$",
    "prediction_diversity": "różn.\npred.",
    "h1_pair_rho_jaccard": "pary: $D_J$",
    "h1_pair_rho_multiplicity": "pary: $D_C$",
    "h1_pair_rho_target": "pary: $D_Y$",
    "h1_tree_rho_unique_fraction": "drzewa: $U$",
    "h1_tree_rho_target_representativeness": "drzewa: $R_Y$",
    "h1_tree_rho_feature_representativeness": "drzewa: $R_X$",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _tasks(data):
    return [t for t in ("classification", "regression")
            if t in set(data["task"])]


# ============================================================
# 1. DLACZEGO MIARY KLASYCZNEGO BAGGINGU SĄ PRAWIE STAŁE
# ============================================================

def plot_concentration(results, output_dir, seed=0):
    """
    (a) Distribution of pairwise Jaccard distance between bootstrap
        samples for growing n, with the limit 1 - p/(2 - p),
        p = 1 - e^{-1}.
    (b) Spread of D_J between pairs within one ensemble vs n (pilot data),
        compared with the c / sqrt(n) rate.
    """

    p = 1 - np.exp(-1)
    limit = 1 - p / (2 - p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    for n, color in zip((100, 1000, 10000), ("#BBBBBB", "#66CCEE", "#4477AA")):
        samples = generate_bootstrap_samples(n, 1.0, 100, seed)
        d = pairwise_jaccard_vector(sample_count_matrix(samples, n))
        ax.hist(d, bins=60, density=True, alpha=0.75, color=color,
                label=f"n = {n}: sd = {d.std():.4f}")
    ax.axvline(limit, color="black", ls="--", lw=1,
               label=f"granica {limit:.3f}")
    ax.set_xlabel("odległość Jaccarda pary próbek bootstrapowych")
    ax.set_ylabel("gęstość")
    ax.set_title("(a) 4950 par próbek, M = 100")
    ax.legend(fontsize=8)

    ax = axes[1]
    base = results[results["sampling_config"] == BASELINE_CONFIG]
    per_dataset = base.groupby("dataset").agg(
        n_train=("n_train", "mean"),
        pair_sd=("jaccard_distance_pair_std", "mean"),
        ensemble_sd=("jaccard_distance", "std"),
    )
    ax.scatter(per_dataset["n_train"], per_dataset["pair_sd"], color="#4477AA",
               label="sd $D_J$ między parami w zespole")
    ax.scatter(per_dataset["n_train"], per_dataset["ensemble_sd"],
               color="#EE7733", marker="s",
               label="sd średniego $D_J$ między foldami")
    grid = np.logspace(np.log10(per_dataset["n_train"].min()),
                       np.log10(per_dataset["n_train"].max()), 50)
    c = np.median(per_dataset["pair_sd"] * np.sqrt(per_dataset["n_train"]))
    ax.plot(grid, c / np.sqrt(grid), color="black", lw=1, ls="--",
            label=r"$c/\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (część treningowa)")
    ax.set_ylabel("odchylenie standardowe")
    ax.set_title("(b) zbiory z pilotażu, klasyczny bagging")
    ax.legend(fontsize=8)

    return _save(fig, Path(output_dir) / "01_concentration.png")


# ============================================================
# 2. JAKOŚĆ VS ROZMIAR PRÓBKI (S1, S2) I VS RDZEŃ (S3)
# ============================================================

def plot_size_and_core(aggregated, output_dir):
    tasks = _tasks(aggregated)
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11, 3.6 * len(tasks)),
                             squeeze=False)

    for row, task in enumerate(tasks):
        data = aggregated[aggregated["task"] == task]

        ax = axes[row, 0]
        for family in ("S1_bootstrap", "S2_subsampling"):
            fam = data[data["variant"] == family]
            for _, g in fam.groupby("dataset"):
                g = g.sort_values("unique_fraction")
                ax.plot(g["unique_fraction"], g["score"],
                        color=COLORS[family], alpha=0.15, lw=0.8)
            med = (fam.groupby("ratio")[["unique_fraction", "score"]]
                   .median().sort_values("unique_fraction"))
            ax.plot(med["unique_fraction"], med["score"], "o-",
                    color=COLORS[family], lw=2.2,
                    label=f"{family} (mediana)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("udział unikalnych obserwacji U")
        ax.set_ylabel(SCORE_LABEL[task])
        ax.set_title(f"{TASK_LABEL[task]}: rozmiar próbki")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        core = data[data["variant"] == "S3_common_core"]
        for _, g in core.groupby("dataset"):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g["score"], color=COLORS["S3_common_core"],
                    alpha=0.15, lw=0.8)
        med = core.groupby("alpha")["score"].median()
        ax.plot(med.index, med.values, "o-", color=COLORS["S3_common_core"],
                lw=2.2, label="mediana po zbiorach")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(r"udział wspólnego rdzenia $\gamma$ (m = 0.5n)")
        ax.set_ylabel(SCORE_LABEL[task])
        ax.set_title(f"{TASK_LABEL[task]}: pokrywanie się próbek (S3)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "02_size_and_core.png")


# ============================================================
# 3. SIŁA VS NIEZALEŻNOŚĆ
# ============================================================

def plot_strength_diversity(aggregated, output_dir):
    """
    Mean tree quality vs prediction diversity, both z-scored within the
    dataset (so that datasets are comparable), colored by family.
    """

    tasks = _tasks(aggregated)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.5 * len(tasks), 4.2),
                             squeeze=False)

    for col, task in enumerate(tasks):
        data = aggregated[aggregated["task"] == task].copy()
        quality = TREE_QUALITY[task]

        def z(s):
            return (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)

        data["z_quality"] = data.groupby("dataset")[quality].transform(z)
        data["z_div"] = data.groupby("dataset")[
            "prediction_diversity"].transform(z)

        ax = axes[0, col]
        sc = None
        for family, g in data.groupby("variant"):
            sc = ax.scatter(g["z_div"], g["z_quality"], c=g["score"],
                            cmap="RdBu", vmin=-np.nanmax(np.abs(data["score"])),
                            vmax=np.nanmax(np.abs(data["score"])), s=28,
                            edgecolor=COLORS[family], linewidth=1.2,
                            label=family)
        ax.set_xlabel("różnorodność predykcji drzew (z w obrębie zbioru)")
        ax.set_ylabel("średnia jakość drzew (z w obrębie zbioru)")
        ax.set_title(f"{TASK_LABEL[task]}: siła vs niezależność")
        ax.legend(fontsize=7, loc="lower left")
        fig.colorbar(sc, ax=ax, label="wynik zespołu vs bagging")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "03_strength_vs_diversity.png")


# ============================================================
# 4. H1: KORELACJE WEWNĄTRZ ZESPOŁU
# ============================================================

def plot_h1(h1_per_dataset, output_dir):
    tasks = _tasks(h1_per_dataset)
    columns = [c for c in H1_PAIR + H1_TREE if c in h1_per_dataset.columns]

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 3.8),
                             squeeze=False, sharey=True)

    rng = np.random.default_rng(0)

    for col, task in enumerate(tasks):
        data = h1_per_dataset[h1_per_dataset["task"] == task]
        ax = axes[0, col]
        for i, column in enumerate(columns):
            values = data[column].dropna().to_numpy()
            if len(values) == 0:
                continue
            ax.boxplot(values, positions=[i], widths=0.5, showfliers=False)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values,
                       s=14, alpha=0.7,
                       color="#4477AA" if "pair" in column else "#EE7733")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([MEASURE_LABEL[c] for c in columns], rotation=30,
                           ha="right")
        ax.set_ylabel(r"mediana $\rho$ Spearmana (po foldach)")
        ax.set_title(f"{TASK_LABEL[task]}: H1, klasyczny bagging")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "04_h1_within_ensemble.png")


# ============================================================
# 5. ŚREDNIE RANGI
# ============================================================

def plot_ranks(ranks, output_dir):
    tasks = _tasks(ranks)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.6),
                             squeeze=False)

    for col, task in enumerate(tasks):
        data = ranks[ranks["task"] == task].sort_values("average_rank",
                                                        ascending=False)
        ax = axes[0, col]
        colors = [COLORS[c.split(" |")[0]] for c in data["sampling_config"]]
        bars = ax.barh(data["sampling_config"].str.replace("_", " "),
                       data["average_rank"], color=colors)
        for bar, config in zip(bars, data["sampling_config"]):
            if config == BASELINE_CONFIG:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
        ax.set_xlabel("średnia ranga (1 = najlepszy)")
        ax.set_title(f"{TASK_LABEL[task]}: H3 (czarna ramka = bagging)")
        ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "05_average_ranks.png")


# ============================================================
# 6. H2: KORELACJE MIAR Z JAKOŚCIĄ MIĘDZY WARIANTAMI
# ============================================================

def plot_h2(h2_per_dataset, output_dir):
    tasks = _tasks(h2_per_dataset)
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.5 * len(tasks), 9),
                             squeeze=False)

    for col, task in enumerate(tasks):
        for row, kind in enumerate(("rho", "partial_rho")):
            data = h2_per_dataset[h2_per_dataset["task"] == task]
            pivot = data.pivot_table(index="dataset", columns="measure",
                                     values=kind)
            order = [m for m in SAMPLE_MEASURES + ["prediction_diversity"]
                     if m in pivot.columns]
            pivot = pivot[order]
            ax = axes[row, col]
            im = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-1, vmax=1,
                           aspect="auto")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([MEASURE_LABEL[m] for m in order])
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            for (i, j), v in np.ndenumerate(pivot.to_numpy()):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7)
            title = ("Spearman miara ~ jakość zespołu" if kind == "rho"
                     else "cząstkowa, kontrola średniej jakości drzew")
            ax.set_title(f"{TASK_LABEL[task]}: {title}", fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "06_h2_correlations.png")


# ============================================================
# 7. LAS LOSOWY VS CZYSTY BAGGING
# ============================================================

def plot_rf_vs_bagging(aggregated_rf, aggregated_bagging, output_dir):
    rf = aggregated_rf[aggregated_rf["task"] == "classification"]
    bg = aggregated_bagging[aggregated_bagging["task"] == "classification"]

    if rf.empty or bg.empty:
        return None

    merged = rf.merge(bg, on=["dataset", "sampling_config"],
                      suffixes=("_rf", "_bag"))
    configs = list(dict.fromkeys(rf["sampling_config"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for i, (name, suffix, color) in enumerate(
            (("las losowy (sqrt)", "_rf", "#4477AA"),
             ("bagging (wszystkie cechy)", "_bag", "#EE7733"))):
        med = merged.groupby("sampling_config")[f"score{suffix}"].median()
        ax.plot(range(len(configs)), [med[c] for c in configs], "o-",
                color=color, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace("_", " ") for c in configs], rotation=70,
                       ha="right", fontsize=7)
    ax.set_ylabel("mediana Δ macro-F1 vs S1 r=1 (ta sama rodzina)")
    ax.set_title("Wpływ wariantu próbkowania")
    ax.legend(fontsize=8)

    ax = axes[1]
    base = merged[merged["sampling_config"] == BASELINE_CONFIG]
    ax.scatter(base["prediction_diversity_bag"],
               base["prediction_diversity_rf"], color="#4477AA")
    lim = [0, max(base["prediction_diversity_rf"].max(),
                  base["prediction_diversity_bag"].max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlabel("różnorodność predykcji: bagging")
    ax.set_ylabel("różnorodność predykcji: las losowy")
    ax.set_title("Klasyczny bootstrap, 12 zbiorów")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "07_rf_vs_bagging.png")


def make_all(results, tables, output_dir, aggregated_bagging=None):
    paths = [
        plot_concentration(results, output_dir),
        plot_size_and_core(tables["aggregated"], output_dir),
        plot_strength_diversity(tables["aggregated"], output_dir),
        plot_h1(tables["h1_per_dataset"], output_dir),
        plot_ranks(tables["average_ranks"], output_dir),
        plot_h2(tables["h2_per_dataset"], output_dir),
    ]

    if aggregated_bagging is not None:
        paths.append(plot_rf_vs_bagging(tables["aggregated"],
                                        aggregated_bagging, output_dir))

    return [p for p in paths if p is not None]
