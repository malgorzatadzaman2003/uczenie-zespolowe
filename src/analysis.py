"""
Statistical analysis following Section 11 of the plan, revised after
the feedback on the final plan.

Changes with respect to src/statistics.py (kept for compatibility):
  * variants are compared only with the control (classical bagging,
    S1 r=1.0) – 14 tests instead of 105 pairs, as recommended by
    Demšar (2006) and as written in the plan;
  * H1 and H2 are tested explicitly (within-ensemble correlations,
    correlations across variants, partial correlations controlling for
    mean tree quality);
  * because the unit of the test is the dataset (only 12 per task), we
    (a) report the smallest attainable p-values (power table),
    (b) add a pooled analysis of all 24 datasets on a task-independent
        relative score,
    (c) add the Bayesian signed-rank test with a region of practical
        equivalence (Benavoli et al., 2017), which gives graded evidence
        instead of a binary significant / not significant answer.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

from src.experiment import BASELINE_CONFIG


PRIMARY = {"classification": "ensemble_macro_f1", "regression": "ensemble_rmse"}
TREE_QUALITY = {
    "classification": "individual_macro_f1_mean",
    "regression": "individual_r2_mean",
}
HIGHER_IS_BETTER = {"classification": True, "regression": False}

SAMPLE_MEASURES = [
    "jaccard_distance",
    "multiplicity_distance",
    "target_distance",
    "unique_fraction",
    "target_representativeness",
    "feature_representativeness",
]
H1_PAIR = ["h1_pair_rho_jaccard", "h1_pair_rho_multiplicity",
           "h1_pair_rho_target"]
H1_TREE = ["h1_tree_rho_unique_fraction",
           "h1_tree_rho_target_representativeness",
           "h1_tree_rho_feature_representativeness"]

# Region praktycznej równoważności: 1 punkt procentowy macro-F1
# albo 1% RMSE.
ROPE = 0.01


# ============================================================
# NARZĘDZIA
# ============================================================

def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]

    if len(valid) == 0:
        return out

    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.maximum.accumulate(
        [min(1.0, (m - i) * p[k]) for i, k in enumerate(order)])
    out[order] = adjusted

    return out


def wilcoxon_p(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 3 or np.allclose(x, 0):
        return np.nan

    return float(wilcoxon(x, zero_method="wilcox",
                          alternative="two-sided").pvalue)


def rank_biserial(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]

    if len(x) == 0:
        return 0.0

    ranks = rankdata(np.abs(x))

    return float((ranks[x > 0].sum() - ranks[x < 0].sum()) / ranks.sum())


def bayesian_signed_rank(x, rope=ROPE, prior_strength=0.5, n_samples=20000,
                         seed=0):
    """
    Bayesian signed-rank test (Benavoli, Corani, Demšar, Zaffalon, 2017).

    x : paired differences (positive = variant better than control).
    Returns posterior probabilities (P(worse), P(rope), P(better)).
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    z = np.concatenate([[0.0], x])               # pseudo-obserwacja priora
    n = len(z)

    # Średnie Walsha dla par i <= j (jak w bibliotece baycomp).
    walsh = (z[:, None] + z[None, :]) / 2.0
    upper = np.triu(np.ones((n, n)))
    left = (walsh < -rope) * upper
    right = (walsh > rope) * upper
    middle = upper - left - right

    rng = np.random.default_rng(seed)
    alpha = np.concatenate([[prior_strength], np.ones(n - 1)])
    weights = rng.dirichlet(alpha, size=n_samples)   # (S, n)

    p_left = np.einsum("si,ij,sj->s", weights, left, weights)
    p_mid = np.einsum("si,ij,sj->s", weights, middle, weights)
    p_right = np.einsum("si,ij,sj->s", weights, right, weights)

    winner = np.argmax(np.stack([p_left, p_mid, p_right]), axis=0)

    return tuple(float(np.mean(winner == k)) for k in range(3))


def partial_spearman(x, y, z):
    """
    Spearman correlation between x and y controlling for z
    (partial Pearson correlation of ranks).
    """

    x, y, z = (rankdata(v) for v in (x, y, z))
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denominator = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    if denominator == 0 or not np.isfinite(denominator):
        return np.nan

    return float((rxy - rxz * ryz) / denominator)


# ============================================================
# 1. AGREGACJA
# ============================================================

def aggregate(results):
    """
    Mean and standard deviation over splits for every
    (dataset, sampling configuration).
    """

    keys = ["dataset", "task", "ensemble_type", "variant", "ratio", "alpha",
            "sampling_config"]
    keys = [k for k in keys if k in results.columns]

    numeric = [c for c in results.select_dtypes(np.number).columns
               if c not in keys + ["fold", "draw"]]

    grouped = results.groupby(keys, dropna=False)[numeric]
    mean = grouped.mean()
    std = grouped.std().add_suffix("_std")

    out = pd.concat([mean, std], axis=1).reset_index()
    out["n_folds"] = results.groupby(keys, dropna=False).size().to_numpy()

    return add_relative_score(out)


def add_relative_score(aggregated):
    """
    score = improvement over classical bagging on the primary metric,
    positive = better:
      classification: macro-F1(variant) - macro-F1(baseline),
      regression:     (RMSE(baseline) - RMSE(variant)) / RMSE(baseline).
    Comparable across datasets and tasks (used for pooled analysis).
    """

    aggregated = aggregated.copy()
    aggregated["score"] = np.nan

    for (dataset, task), group in aggregated.groupby(["dataset", "task"]):
        metric = PRIMARY[task]
        baseline = group.loc[group["sampling_config"] == BASELINE_CONFIG,
                             metric]

        if baseline.empty:
            continue

        base = float(baseline.iloc[0])

        if task == "classification":
            score = group[metric] - base
        else:
            score = (base - group[metric]) / base

        aggregated.loc[group.index, "score"] = score

    return aggregated


def quality_table(aggregated):
    """Primary metric, mean ± std over splits, dataset x configuration."""

    rows = []

    for (dataset, task), group in aggregated.groupby(["dataset", "task"]):
        metric = PRIMARY[task]
        for _, r in group.iterrows():
            rows.append({
                "task": task, "dataset": dataset,
                "sampling_config": r["sampling_config"],
                "metric": metric,
                "mean": r[metric], "std": r[f"{metric}_std"],
                "score_vs_baseline": r["score"],
            })

    return pd.DataFrame(rows)


# ============================================================
# 2. H3: FRIEDMAN, RANGI, PORÓWNANIA Z KONTROLĄ
# ============================================================

def friedman_and_ranks(aggregated):
    friedman_rows, rank_rows = [], []

    for task, data in aggregated.groupby("task"):
        pivot = data.pivot_table(index="dataset", columns="sampling_config",
                                 values="score").dropna()

        if len(pivot) < 3:
            continue

        stat, p = friedmanchisquare(*[pivot[c] for c in pivot.columns])
        friedman_rows.append({"task": task, "n_datasets": len(pivot),
                              "n_variants": pivot.shape[1],
                              "friedman_statistic": stat, "p_value": p})

        # Ranga 1 = najlepszy wariant w danym zbiorze.
        ranks = pivot.rank(axis=1, ascending=False).mean()
        for config, rank in ranks.items():
            rank_rows.append({"task": task, "sampling_config": config,
                              "average_rank": rank})

    return pd.DataFrame(friedman_rows), pd.DataFrame(rank_rows)


def control_comparisons(aggregated, pooled=False):
    """
    Every variant vs classical bagging (S1 r=1.0) on the relative score.
    Wilcoxon signed-rank + Holm (within task), effect sizes and the
    Bayesian signed-rank test.
    """

    data = aggregated.copy()

    if pooled:
        data["task"] = "pooled (24)"

    rows = []

    for task, group in data.groupby("task"):
        pivot = group.pivot_table(index="dataset", columns="sampling_config",
                                  values="score")

        for config in pivot.columns:
            if config == BASELINE_CONFIG:
                continue

            diff = pivot[config].dropna().to_numpy()
            p_worse, p_rope, p_better = bayesian_signed_rank(diff)

            rows.append({
                "task": task,
                "sampling_config": config,
                "n_datasets": len(diff),
                "median_score": float(np.median(diff)),
                "mean_score": float(np.mean(diff)),
                "wins": int(np.sum(diff > 0)),
                "losses": int(np.sum(diff < 0)),
                "rank_biserial": rank_biserial(diff),
                "p_value": wilcoxon_p(diff),
                "bayes_p_worse": p_worse,
                "bayes_p_rope": p_rope,
                "bayes_p_better": p_better,
            })

    table = pd.DataFrame(rows)

    for task, idx in table.groupby("task").groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_value"])

    return table


def s3_core_test(aggregated):
    """H2 (second part): S3 gamma=0 vs gamma=0.75 – same sample size."""

    rows = []
    low = "S3_common_core | ratio=0.5 | alpha=0"
    high = "S3_common_core | ratio=0.5 | alpha=0.75"

    for task, group in list(aggregated.groupby("task")) + [
            ("pooled (24)", aggregated)]:
        pivot = group.pivot_table(index="dataset", columns="sampling_config",
                                  values="score")
        if low not in pivot or high not in pivot:
            continue

        diff = (pivot[low] - pivot[high]).dropna().to_numpy()
        p_worse, p_rope, p_better = bayesian_signed_rank(diff)

        rows.append({
            "task": task,
            "comparison": "gamma=0 minus gamma=0.75 (positive: less overlap better)",
            "n_datasets": len(diff),
            "median_difference": float(np.median(diff)),
            "wins_gamma0": int(np.sum(diff > 0)),
            "p_value": wilcoxon_p(diff),
            "bayes_p_gamma075_better": p_worse,
            "bayes_p_rope": p_rope,
            "bayes_p_gamma0_better": p_better,
        })

    return pd.DataFrame(rows)


# ============================================================
# 3. H1: KORELACJE WEWNĄTRZ ZESPOŁU
# ============================================================

def h1_tests(results, config=BASELINE_CONFIG):
    """
    For classical bagging: median over splits of the within-ensemble
    Spearman correlations is the dataset result; Wilcoxon over datasets
    tests whether its median differs from 0 (Holm within task).
    """

    data = results[results["sampling_config"] == config]
    columns = [c for c in H1_PAIR + H1_TREE if c in data.columns]

    per_dataset = (data.groupby(["task", "dataset"])[columns]
                   .median().reset_index())

    rows = []

    for task, group in per_dataset.groupby("task"):
        for column in columns:
            values = group[column].dropna().to_numpy()
            if len(values) == 0:
                continue
            rows.append({
                "task": task,
                "correlation": column,
                "n_datasets": len(values),
                "median_rho": float(np.median(values)),
                "min_rho": float(np.min(values)),
                "max_rho": float(np.max(values)),
                "n_positive": int(np.sum(values > 0)),
                "p_value": wilcoxon_p(values),
            })

    table = pd.DataFrame(rows)

    if not table.empty:
        for task, idx in table.groupby("task").groups.items():
            table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_value"])

    return per_dataset, table


# ============================================================
# 4. H2: MIARY PRÓBEK A JAKOŚĆ ZESPOŁU (MIĘDZY WARIANTAMI)
# ============================================================

def h2_tests(aggregated):
    """
    Per dataset: Spearman correlation between a sample measure and the
    ensemble score across the 15 variants, plus the partial correlation
    controlling for mean tree quality. Wilcoxon across datasets + Holm.
    """

    per_dataset = []

    for (task, dataset), group in aggregated.groupby(["task", "dataset"]):
        quality = group["score"].to_numpy()
        trees = group[TREE_QUALITY[task]].to_numpy()

        for measure in SAMPLE_MEASURES + ["prediction_diversity"]:
            values = group[measure].to_numpy()
            if np.ptp(values) == 0:
                continue
            per_dataset.append({
                "task": task, "dataset": dataset, "measure": measure,
                "rho": float(spearmanr(values, quality)[0]),
                "partial_rho": partial_spearman(values, quality, trees),
            })

    per_dataset = pd.DataFrame(per_dataset)
    rows = []

    for (task, measure), group in per_dataset.groupby(["task", "measure"]):
        for kind in ("rho", "partial_rho"):
            values = group[kind].dropna().to_numpy()
            rows.append({
                "task": task, "measure": measure, "type": kind,
                "n_datasets": len(values),
                "median": float(np.median(values)),
                "n_positive": int(np.sum(values > 0)),
                "p_value": wilcoxon_p(values),
            })

    table = pd.DataFrame(rows)

    for _, idx in table.groupby(["task", "type"]).groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_value"])

    return per_dataset, table


# ============================================================
# 5. EKSPERYMENT DODATKOWY (4.6)
# ============================================================

def draw_tests(draws):
    """
    Within each split, Spearman correlation across the 20 draws between
    a sample measure and ensemble quality; median over splits per
    dataset, Wilcoxon across datasets.
    """

    if draws is None or draws.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []

    for (task, dataset, fold), group in draws.groupby(
            ["task", "dataset", "fold"]):
        metric = PRIMARY[task]
        sign = 1.0 if HIGHER_IS_BETTER[task] else -1.0
        quality = sign * group[metric].to_numpy()

        for measure in SAMPLE_MEASURES:
            values = group[measure].to_numpy()
            if np.ptp(values) == 0:
                continue
            rows.append({"task": task, "dataset": dataset, "fold": fold,
                         "measure": measure,
                         "rho": float(spearmanr(values, quality)[0])})

    per_split = pd.DataFrame(rows)
    per_dataset = (per_split.groupby(["task", "dataset", "measure"])["rho"]
                   .median().reset_index())

    summary = []
    for (task, measure), group in per_dataset.groupby(["task", "measure"]):
        values = group["rho"].to_numpy()
        summary.append({"task": task, "measure": measure,
                        "n_datasets": len(values),
                        "median_rho": float(np.median(values)),
                        "p_value": wilcoxon_p(values)})

    summary = pd.DataFrame(summary)
    for task, idx in summary.groupby("task").groups.items():
        summary.loc[idx, "p_holm"] = holm(summary.loc[idx, "p_value"])

    return per_dataset, summary


# ============================================================
# 6. MOC TESTÓW PRZY 12 / 24 ZBIORACH
# ============================================================

def power_table(n_tests=14, alpha=0.05):
    """
    Smallest attainable two-sided Wilcoxon p-value for n datasets and
    the largest minority rank sum T that still survives Holm at the
    strictest step (alpha / n_tests).
    """

    rows = []

    for n in (12, 24):
        # Rozkład dokładny statystyki T przez programowanie dynamiczne.
        max_sum = n * (n + 1) // 2
        counts = np.zeros(max_sum + 1)
        counts[0] = 1
        for k in range(1, n + 1):
            counts[k:] = counts[k:] + counts[:-k].copy()
        cdf = np.cumsum(counts) / 2 ** n
        p_two_sided = np.minimum(1.0, 2 * cdf)

        threshold = alpha / n_tests
        t_max = int(np.max(np.where(p_two_sided <= threshold)[0]))

        rows.append({
            "n_datasets": n,
            "min_p_value": float(p_two_sided[0]),
            "holm_strictest_threshold": threshold,
            "max_minority_rank_sum": t_max,
            "uncorrected_max_minority_rank_sum": int(
                np.max(np.where(p_two_sided <= alpha)[0])),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. CAŁOŚĆ
# ============================================================

def run_analysis(results, output_dir, draws=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate(results)
    friedman, ranks = friedman_and_ranks(aggregated)
    h1_per_dataset, h1 = h1_tests(results)
    h2_per_dataset, h2 = h2_tests(aggregated)
    draw_per_dataset, draw_summary = draw_tests(draws)

    tables = {
        "aggregated": aggregated,
        "quality_table": quality_table(aggregated),
        "friedman": friedman,
        "average_ranks": ranks,
        "control_comparisons": control_comparisons(aggregated),
        "control_comparisons_pooled": control_comparisons(aggregated,
                                                          pooled=True),
        "s3_core_test": s3_core_test(aggregated),
        "h1_per_dataset": h1_per_dataset,
        "h1_tests": h1,
        "h2_per_dataset": h2_per_dataset,
        "h2_tests": h2,
        "draw_per_dataset": draw_per_dataset,
        "draw_tests": draw_summary,
        "power": power_table(),
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    return tables
