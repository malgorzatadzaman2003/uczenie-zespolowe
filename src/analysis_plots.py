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
    "classification": "Δ macro-F1 względem S1 r = 1",
    "regression": "względna poprawa RMSE względem S1 r = 1",
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
# 1. KONCENTRACJA MIAR PRÓBEK BOOTSTRAPOWYCH (m = n)
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
                label=f"n = {n}: odch. std. = {d.std():.4f}")
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
               label="odch. std. $D_J$ między parami w zespole")
    ax.scatter(per_dataset["n_train"], per_dataset["ensemble_sd"],
               color="#EE7733", marker="s",
               label="odch. std. średniego $D_J$ między podziałami")
    grid = np.logspace(np.log10(per_dataset["n_train"].min()),
                       np.log10(per_dataset["n_train"].max()), 50)
    c = np.median(per_dataset["pair_sd"] * np.sqrt(per_dataset["n_train"]))
    ax.plot(grid, c / np.sqrt(grid), color="black", lw=1, ls="--",
            label=r"$c/\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (część treningowa)")
    ax.set_ylabel("odchylenie standardowe")
    ax.set_title("(b) 24 zbiory, S1 r = 1 (średnia po podziałach)")
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
                    label=f"{FAMILY_LABEL[family]}, mediana")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("udział różnych obserwacji U")
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
                            label=FAMILY_LABEL[family])
        ax.set_xlabel("różnorodność predykcji drzew (standaryzowana w zbiorze)")
        ax.set_ylabel("średnia jakość drzew (standaryzowana w zbiorze)")
        ax.set_title(f"{TASK_LABEL[task]}: siła a niezależność drzew")
        ax.legend(fontsize=7, loc="lower left")
        fig.colorbar(sc, ax=ax, label="wynik zespołu względem S1 r = 1")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "03_strength_vs_diversity.png")


# ============================================================
# 4. H1: KORELACJE WEWNĄTRZ ZESPOŁU
# ============================================================

ENSEMBLE_LABEL = {"random_forest": "las losowy", "bagging": "bagging"}
FAMILY_LABEL = {"S1_bootstrap": "S1 ze zwracaniem",
                "S2_subsampling": "S2 bez zwracania",
                "S3_common_core": "S3 wspólny rdzeń",
                "S4_stratified_bootstrap": "S4 warstwowy"}


def plot_h1(h1_per_dataset, output_dir, ensemble_label="las losowy",
            filename="04_h1_within_ensemble.png"):
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
        ax.set_ylabel(r"mediana $\rho$ Spearmana (po podziałach)")
        ax.set_title(f"{TASK_LABEL[task]}: H1, {ensemble_label}, bootstrap r = 1")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / filename)


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
        bars = ax.barh([_short_config(c) for c in data["sampling_config"]],
                       data["average_rank"], color=colors)
        for bar, config in zip(bars, data["sampling_config"]):
            if config == BASELINE_CONFIG:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
        ax.set_xlabel("średnia ranga (1 = najlepszy)")
        ax.set_title(f"{TASK_LABEL[task]}: średnie rangi (czarna ramka: S1 r = 1)")
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
            title = ("korelacja miary z jakością zespołu" if kind == "rho"
                     else "korelacja cząstkowa (kontrola jakości drzew)")
            ax.set_title(f"{TASK_LABEL[task]}: {title}", fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "06_h2_correlations.png")


# ============================================================
# 7. LAS LOSOWY A BAGGING (BEZ LOSOWANIA CECH)
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
            (("las losowy (√d cech w węźle)", "_rf", "#4477AA"),
             ("bagging (wszystkie cechy)", "_bag", "#EE7733"))):
        med = merged.groupby("sampling_config")[f"score{suffix}"].median()
        ax.plot(range(len(configs)), [med[c] for c in configs], "o-",
                color=color, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([_short_config(c) for c in configs], rotation=60,
                       ha="right", fontsize=8)
    ax.set_ylabel("mediana Δ macro-F1 względem S1 r = 1")
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
    ax.set_title("S1 r = 1, 12 zbiorów")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "07_rf_vs_bagging.png")


# ============================================================
# 8. JAKOŚĆ KAŻDEGO WARIANTU W KAŻDYM ZBIORZE
# ============================================================

def _short_config(config):
    family, *params = config.split(" | ")
    short = {"S1_bootstrap": "S1", "S2_subsampling": "S2",
             "S3_common_core": "S3", "S4_stratified_bootstrap": "S4"}[family]
    params = [q.replace("ratio=", {"S1": "r=", "S2": "s="}.get(short, ""))
              .replace("alpha=", "γ=") for q in params]
    if short == "S3":
        params = params[1:]
    if short == "S4":
        params = []
    return " ".join([short] + params)


def plot_per_dataset(aggregated, output_dir):
    """Score vs classical bootstrap for every dataset and variant."""

    tasks = _tasks(aggregated)
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5 * len(tasks), 5.2),
                             squeeze=False)
    configs = list(dict.fromkeys(aggregated["sampling_config"]))

    for col, task in enumerate(tasks):
        data = aggregated[aggregated["task"] == task]
        pivot = data.pivot_table(index="dataset", columns="sampling_config",
                                 values="score")[configs]
        values = pivot.to_numpy() * 100
        limit = np.nanpercentile(np.abs(values), 95)
        ax = axes[0, col]
        im = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit,
                       aspect="auto")
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([_short_config(c) for c in configs], rotation=60,
                           ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6)
        ax.grid(False)
        unit = ("Δ macro-F1 [p.p.]" if task == "classification"
                else "poprawa RMSE [%]")
        ax.set_title(f"{TASK_LABEL[task]}: {unit} względem S1 r = 1")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "08_per_dataset.png")


# ============================================================
# 9. H1 NA POZIOMIE PAR DRZEW: PRZYKŁAD JEDNEGO ZESPOŁU
# ============================================================

def plot_pair_example(output_dir, datasets=("breast_cancer", "diabetes"),
                      ensemble_type="random_forest", seed=42):
    """
    One split, one ensemble (bootstrap r = 1): every dot is a pair of
    trees, x = Jaccard distance of their samples, y = diversity of their
    predictions. Shows what a single H1 correlation is computed from.
    """

    from sklearn.model_selection import KFold, StratifiedKFold

    from src.datasets import load_dataset
    from src.diversity import sample_measures
    from src.experiment import (
        _encode_target, evaluate_classification, evaluate_regression)
    from src.models import (
        get_tree_params, predict_tree_probabilities, train_ensemble)
    from src.preprocessing import preprocess_train_test

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5.2 * len(datasets), 3.9),
                             squeeze=False)

    for col, name in enumerate(datasets):
        data = load_dataset(name, random_state=seed)
        X, y, task = data["X"], data["y"], data["task"]
        splitter = (StratifiedKFold if task == "classification"
                    else KFold)(5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y))
        X_tr, X_te, _ = preprocess_train_test(X.iloc[train_idx],
                                              X.iloc[test_idx])
        y_tr, y_te, classes = _encode_target(y.iloc[train_idx],
                                             y.iloc[test_idx], task)
        samples = generate_bootstrap_samples(len(y_tr), 1.0, 100, seed)
        pairs, _ = sample_measures(samples, X_tr, y_tr, task)
        models = train_ensemble(
            X_train=X_tr, y_train=y_tr, samples=samples, task=task,
            random_state=seed,
            max_features=get_tree_params(task, ensemble_type)["max_features"])
        if task == "classification":
            proba = predict_tree_probabilities(models, X_te, classes)
            *_, diversity, _ = evaluate_classification(proba, y_te,
                                                       len(classes))
            ylabel = "niezgodność predykcji pary"
        else:
            pred = np.vstack([m.predict(X_te) for m in models])
            *_, diversity, _ = evaluate_regression(pred, y_te)
            ylabel = "1 − korelacja błędów pary"

        ax = axes[0, col]
        ax.scatter(pairs["jaccard"], diversity, s=4, alpha=0.35,
                   color="#4477AA")
        rho = pd.Series(pairs["jaccard"]).corr(pd.Series(diversity),
                                                method="spearman")
        ax.set_xlabel("odległość Jaccarda próbek pary $D_J$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name}: 4950 par, ρ = {rho:.2f}")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "09_pair_example.png")


# ============================================================
# 10. H3: MIARY A ŚREDNIE RANGI WARIANTÓW
# ============================================================

def plot_measures_vs_ranks(measures_ranks, output_dir,
                           measures=("unique_fraction", "jaccard_distance",
                                     "prediction_diversity")):
    tasks = _tasks(measures_ranks)
    fig, axes = plt.subplots(len(tasks), len(measures),
                             figsize=(4.2 * len(measures), 3.5 * len(tasks)),
                             squeeze=False)

    for row, task in enumerate(tasks):
        data = measures_ranks[measures_ranks["task"] == task]
        for col, measure in enumerate(measures):
            ax = axes[row, col]
            for _, r in data.iterrows():
                family = r["sampling_config"].split(" |")[0]
                ax.scatter(r[measure], r["average_rank"],
                           color=COLORS[family], s=30)
                ax.annotate(_short_config(r["sampling_config"]),
                            (r[measure], r["average_rank"]), fontsize=6,
                            xytext=(3, 2), textcoords="offset points")
            ax.invert_yaxis()
            ax.set_xlabel(f"średnia {MEASURE_LABEL[measure]}".replace("\n", " "))
            ax.set_ylabel("średnia ranga (1 = najlepszy)")
            ax.set_title(TASK_LABEL[task], fontsize=9)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "10_measures_vs_ranks.png")


# ============================================================
# 11. EKSPERYMENT DODATKOWY: 20 LOSOWAŃ
# ============================================================

def plot_draws(draw_per_dataset, output_dir):
    if draw_per_dataset is None or draw_per_dataset.empty:
        return None

    tasks = _tasks(draw_per_dataset)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 3.8),
                             squeeze=False, sharey=True)
    rng = np.random.default_rng(0)

    for col, task in enumerate(tasks):
        data = draw_per_dataset[draw_per_dataset["task"] == task]
        ax = axes[0, col]
        for i, measure in enumerate(SAMPLE_MEASURES):
            values = data.loc[data["measure"] == measure, "rho"].to_numpy()
            if len(values) == 0:
                continue
            ax.boxplot(values, positions=[i], widths=0.5, showfliers=False)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values,
                       s=14, alpha=0.7, color="#228833")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(SAMPLE_MEASURES)))
        ax.set_xticklabels([MEASURE_LABEL[m] for m in SAMPLE_MEASURES])
        ax.set_ylabel("mediana ρ miary i jakości zespołu (20 losowań)")
        ax.set_title(f"{TASK_LABEL[task]}: konkretne losowanie, 20 zespołów")

    fig.tight_layout()
    return _save(fig, Path(output_dir) / "11_draws.png")


def make_all(results, tables, output_dir, aggregated_bagging=None,
             bagging_tables=None):
    ensemble = results["ensemble_type"].iloc[0]
    paths = [
        plot_concentration(results, output_dir),
        plot_size_and_core(tables["aggregated"], output_dir),
        plot_strength_diversity(tables["aggregated"], output_dir),
        plot_h1(tables["h1_per_dataset"], output_dir,
                ensemble_label=ENSEMBLE_LABEL[ensemble]),
        plot_ranks(tables["average_ranks"], output_dir),
        plot_h2(tables["h2_per_dataset"], output_dir),
        plot_per_dataset(tables["aggregated"], output_dir),
        plot_pair_example(output_dir, ensemble_type=ensemble),
        plot_measures_vs_ranks(tables["measures_vs_ranks"], output_dir),
        plot_draws(tables.get("draw_per_dataset"), output_dir),
    ]

    if aggregated_bagging is not None:
        paths.append(plot_rf_vs_bagging(tables["aggregated"],
                                        aggregated_bagging, output_dir))

    if bagging_tables is not None:
        paths.append(plot_h1(bagging_tables["h1_per_dataset"], output_dir,
                             ensemble_label="bagging",
                             filename="04b_h1_bagging.png"))

    return [p for p in paths if p is not None]
