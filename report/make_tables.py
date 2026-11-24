"""
Builds LaTeX tables and copies figures for the report from the analysis
outputs (results/<run>/analysis/*.csv, figures/<run>/*.png).

    python report/make_tables.py [--run main] [--bagging-run main_bagging]
                                 [--draws-run main_draws]
"""

import argparse

import typografia
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
TABLES = REPORT / "tables"

NAMES = {
    "breast_cancer": "Breast Cancer Wisconsin",
    "ionosphere": "Ionosphere",
    "sonar": "Sonar",
    "spambase": "Spambase",
    "adult": "Adult",
    "bank_marketing": "Bank Marketing",
    "german_credit": "German Credit",
    "glass": "Glass",
    "online_shoppers": "Online Shoppers",
    "waveform": "Waveform-5000",
    "vehicle": "Vehicle",
    "phoneme": "Phoneme",
    "california_housing": "California Housing",
    "diabetes": "Diabetes",
    "abalone": "Abalone",
    "auto_mpg": "Auto MPG",
    "concrete": "Concrete",
    "energy_efficiency": "Energy Efficiency (Y1)",
    "yacht": "Yacht Hydrodynamics",
    "wine_quality_red": "Wine Quality -- Red",
    "airfoil": "Airfoil Self-Noise",
    "superconductivity": "Superconductivity",
    "kin8nm": "Kin8nm",
    "power_plant": "Combined Cycle Power Plant",
}
SOURCE = {
    "breast_cancer": "sklearn", "diabetes": "sklearn",
    "california_housing": "sklearn", "online_shoppers": "UCI 468",
    "power_plant": "UCI 294",
}
MEASURE = {
    "jaccard_distance": "$D_J$",
    "multiplicity_distance": "$D_C$",
    "target_distance": "$D_Y$",
    "unique_fraction": "$U$",
    "target_representativeness": "$R_Y$",
    "feature_representativeness": "$R_X$",
    "prediction_diversity": "różnorodność predykcji",
}
H1_NAME = {
    "h1_pair_rho_jaccard": "pary: $D_J$ ~ różn.\\ predykcji",
    "h1_pair_rho_multiplicity": "pary: $D_C$ ~ różn.\\ predykcji",
    "h1_pair_rho_target": "pary: $D_Y$ ~ różn.\\ predykcji",
    "h1_tree_rho_unique_fraction": "drzewa: $U$ ~ jakość drzewa",
    "h1_tree_rho_target_representativeness": "drzewa: $R_Y$ ~ jakość drzewa",
    "h1_tree_rho_feature_representativeness": "drzewa: $R_X$ ~ jakość drzewa",
}
TASK = {"classification": "Klasyfikacja", "regression": "Regresja",
        "pooled (24)": "Łącznie (24 zbiory)"}


def config_name(config):
    family, *params = config.split(" | ")
    values = {k: v for k, v in (p.split("=") for p in params)}
    if family == "S1_bootstrap":
        return f"S1 $r={values['ratio']}$"
    if family == "S2_subsampling":
        return f"S2 $s={values['ratio']}$"
    if family == "S3_common_core":
        return f"S3 $\\gamma={values['alpha']}$"
    return "S4"


def config_order(configs):
    rank = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}
    return sorted(configs, key=lambda c: (rank[c[:2]], config_name(c)))


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def bold_if(text, condition):
    return f"\\textbf{{{text}}}" if condition else text


def write(name, text):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(typografia.popraw(text))


def score_fmt(value, task):
    # klasyfikacja: punkty procentowe macro-F1; regresja: % RMSE
    return f"{100 * value:+.2f}"


# ------------------------------------------------------------------

def table_datasets(analysis):
    data = pd.read_csv(analysis / "datasets.csv")
    rows = []
    for task, label in (("classification", "Klasyfikacja"),
                        ("regression", "Regresja")):
        rows.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\emph{{{label}}}}}\\\\")
        for _, r in data[data["task"] == task].iterrows():
            note = (f"{int(r['n_classes'])} kl.: {r['class_counts']}"
                    if task == "classification" else "")
            rows.append(
                f"{NAMES[r['dataset']]} & {SOURCE.get(r['dataset'], 'OpenML')} & "
                f"{r['n_raw']:,} & {r['removed_missing_rows'] or '--'} & "
                f"{r['removed_duplicate_rows'] or '--'} & {r['n']:,} & "
                f"{r['d']} / {r['d_after_one_hot']} & {note}\\\\"
                .replace(",", "\\,"))
    write("datasets", f"""\\begin{{table}}[!htbp]
\\centering\\scriptsize\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Zbiory danych. $n_0$ oznacza liczbę obserwacji w źródle, kolumny ,,braki'' i ,,duplikaty'' podają liczbę usuniętych wierszy, $n$ to liczba obserwacji użytych w eksperymencie (zbiory większe niż 20\\,000 obserwacji ograniczono losowo do 20\\,000), a $d$ to liczba cech przed kodowaniem one-hot i po nim. Ostatnia kolumna zawiera liczebności klas po przygotowaniu danych.}}
\\label{{tab:zbiory}}
\\begin{{tabular}}{{llrrrrrl}}
\\toprule
\\textbf{{Zbiór}} & \\textbf{{Źródło}} & $n_0$ & \\textbf{{braki}} & \\textbf{{duplikaty}} & $n$ & $d$ & \\textbf{{klasy}} \\\\
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_power(analysis):
    data = pd.read_csv(analysis / "power.csv")
    rows = [
        f"{int(r['n_datasets'])} & {r['min_p_value']:.2g} & "
        f"{int(r['max_minority_rank_sum'])} & "
        f"{int(r['uncorrected_max_minority_rank_sum'])} \\\\"
        for _, r in data.iterrows()
    ]
    write("power", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{Moc testu Wilcoxona przy 12 i 24 zbiorach. Podano najmniejszą osiągalną dwustronną wartość $p$ oraz największą sumę rang $T$ zbiorów o przeciwnym znaku różnicy, przy której wynik jest jeszcze istotny w najostrzejszym kroku procedury Holma (14 testów, próg $0.05/14$) i bez korekty (próg $0.05$).}}
\\label{{tab:moc}}
\\begin{{tabular}}{{rrrr}}
\\toprule
\\textbf{{zbiory}} & \\textbf{{min.\\ $p$}} & \\textbf{{maks.\\ $T$ (Holm, krok 1)}} & \\textbf{{maks.\\ $T$ (bez korekty)}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_control(analysis):
    data = pd.concat([pd.read_csv(analysis / "control_comparisons.csv"),
                      pd.read_csv(analysis / "control_comparisons_pooled.csv")])
    practical = pd.read_csv(analysis / "practical_significance.csv")
    ranks = pd.read_csv(analysis / "average_ranks.csv")
    rows = []
    for task in ("classification", "regression", "pooled (24)"):
        part = data[data["task"] == task].set_index("sampling_config")
        unit = {"classification": "$\\Delta$ macro-F1 [p.p.]",
                "regression": "poprawa RMSE [\\%]",
                "pooled (24)": "wynik względny [p.p. / \\%]"}[task]
        rows.append(f"\\midrule\n\\multicolumn{{10}}{{l}}{{\\emph{{{TASK[task]}}} -- {unit}}}\\\\")
        for config in config_order(part.index):
            r = part.loc[config]
            rank = ranks[(ranks["task"] == task)
                         & (ranks["sampling_config"] == config)]
            rank = f"{rank['average_rank'].iloc[0]:.1f}" if len(rank) else "--"
            pr = practical[(practical["task"] == task)
                           & (practical["sampling_config"] == config)]
            sd = (f"{pr['median_sd_ratio'].iloc[0]:.2f}" if len(pr) else "--")
            pr = (f"{int(pr['relevant_better'].iloc[0])}/"
                  f"{int(pr['relevant_worse'].iloc[0])}" if len(pr) else "--")
            significant = r["p_holm"] < 0.05
            rows.append(
                f"{config_name(config)} & {rank} & "
                f"{bold_if(score_fmt(r['median_score'], task), significant)} & "
                f"{int(r['wins'])}/{int(r['losses'])} & "
                f"{r['rank_biserial']:+.2f} & {fmt_p(r['p_value'])} & "
                f"{bold_if(fmt_p(r['p_holm']), significant)} & "
                f"{r['bayes_p_worse']:.2f} / {r['bayes_p_rope']:.2f} / "
                f"{r['bayes_p_better']:.2f} & {pr} & {sd} \\\\")
    write("control", f"""{{\\scriptsize\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{longtable}}{{lrrrrrrcrr}}
\\caption{{Porównanie wariantów z wariantem odniesienia S1 $r=1$ (las losowy). Ranga to średnia ranga w teście Friedmana (1 oznacza najlepszy z 15 wariantów). Mediana to mediana wyniku względnego po zbiorach; wartość dodatnia oznacza wariant lepszy od wariantu odniesienia. W/P to liczba zbiorów, w których wariant wygrał i przegrał. $r_{{rb}}$ to dwuseryjny współczynnik korelacji rang, a $p_H$ to wartość $p$ po korekcie Holma (pogrubiono $p_H<0.05$). Kolumna Bayes zawiera prawdopodobieństwa a posteriori, że wariant jest gorszy, praktycznie równoważny (ROPE $\\pm1$) i lepszy. Kolumna ,,Istotne prakt.'' podaje liczbę zbiorów, w których różnica na korzyść i na niekorzyść wariantu przekracza odchylenie standardowe wyniku wariantu odniesienia między 50 podziałami. $\\sigma/\\sigma_0$ to mediana po zbiorach stosunku odchyleń standardowych metryki głównej wariantu i wariantu odniesienia między podziałami; wartość poniżej 1 oznacza wariant stabilniejszy.}}
\\label{{tab:kontrola}}\\\\
\\toprule
\\textbf{{Wariant}} & \\textbf{{Ranga}} & \\textbf{{Mediana}} & \\textbf{{W/P}} & $r_{{rb}}$ & $p$ & $p_H$ & \\textbf{{Bayes}} & \\textbf{{Istotne prakt.}} & $\\sigma/\\sigma_0$ \\\\
\\endfirsthead
\\toprule
\\textbf{{Wariant}} & \\textbf{{Ranga}} & \\textbf{{Mediana}} & \\textbf{{W/P}} & $r_{{rb}}$ & $p$ & $p_H$ & \\textbf{{Bayes}} & \\textbf{{Istotne prakt.}} & $\\sigma/\\sigma_0$ \\\\
\\endhead
{chr(10).join(rows)}
\\bottomrule
\\end{{longtable}}
}}
""")


def table_friedman(analysis):
    data = pd.read_csv(analysis / "friedman.csv")
    rows = [
        f"{TASK[r['task']]} & {int(r['n_datasets'])} & {r['friedman_statistic']:.1f} & "
        f"{fmt_p(r['p_value'])} & {r['iman_davenport_F']:.2f} & "
        f"{fmt_p(r['iman_davenport_p'])} \\\\"
        for _, r in data.iterrows()
    ]
    write("friedman", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{Test Friedmana dla 15 wariantów (las losowy) i jego wersja Imana--Davenporta.}}
\\label{{tab:friedman}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
\\textbf{{Zadanie}} & \\textbf{{zbiory}} & $\\chi^2_F$ & $p$ & $F_{{ID}}$ & $p$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_s3(analysis):
    data = pd.read_csv(analysis / "s3_core_test.csv")
    rows = [
        f"{TASK[r['task']]} & {int(r['n_datasets'])} & "
        f"{100 * r['median_difference']:+.2f} & {int(r['wins_gamma0'])} & "
        f"{fmt_p(r['p_value'])} & {r['bayes_p_gamma075_better']:.2f} / "
        f"{r['bayes_p_rope']:.2f} / {r['bayes_p_gamma0_better']:.2f} \\\\"
        for _, r in data.iterrows()
    ]
    write("s3", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{Bezpośredni test drugiej części H2. Wariant S3 z $\\gamma=0$ (próbki niezależne) porównano z $\\gamma=0.75$ (75\\% każdej próbki wspólne dla wszystkich drzew) przy tym samym rozmiarze $m=0.5n$. Dodatnia różnica oznacza, że mniejsze pokrywanie się próbek daje lepszy zespół. Kolumna Bayes zawiera kolejno prawdopodobieństwa a posteriori, że lepsze jest $\\gamma=0.75$, że różnica mieści się w ROPE i że lepsze jest $\\gamma=0$.}}
\\label{{tab:s3}}
\\begin{{tabular}}{{lrrrrc}}
\\toprule
\\textbf{{Zadanie}} & \\textbf{{zbiory}} & \\textbf{{mediana różnicy}} & \\textbf{{wygrane $\\gamma=0$}} & $p$ & \\textbf{{Bayes}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_h1(analysis, bagging_analysis):
    rf = pd.read_csv(analysis / "h1_tests.csv")
    bag = (pd.read_csv(bagging_analysis / "h1_tests.csv")
           if bagging_analysis and (bagging_analysis / "h1_tests.csv").exists()
           else None)
    rows = []
    for task in ("classification", "regression"):
        rows.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\emph{{{TASK[task]}}}}}\\\\")
        for _, r in rf[rf["task"] == task].iterrows():
            cells = (f"{H1_NAME[r['correlation']]} & {r['median_rho']:+.3f} & "
                     f"{int(r['n_positive'])}/{int(r['n_datasets'])} & "
                     f"{bold_if(fmt_p(r['p_holm']), r['p_holm'] < 0.05)}")
            if bag is not None and task == "classification":
                b = bag[bag["correlation"] == r["correlation"]].iloc[0]
                cells += (f" & {b['median_rho']:+.3f} & "
                          f"{int(b['n_positive'])}/{int(b['n_datasets'])} & "
                          f"{bold_if(fmt_p(b['p_holm']), b['p_holm'] < 0.05)}")
            else:
                cells += " & \\multicolumn{3}{c}{= las losowy}" if task == "regression" else " & & &"
            rows.append(cells + " \\\\")
    write("h1", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{H1: korelacje Spearmana wewnątrz zespołu w wariancie odniesienia S1 $r=1$. Podano medianę po zbiorach (z median po 50 podziałach), liczbę zbiorów z dodatnią korelacją oraz $p_H$, czyli wartość $p$ testu Wilcoxona po zbiorach z korektą Holma (6 testów w zadaniu). W regresji las losowy jest baggingiem, więc kolumny dla baggingu pominięto.}}
\\label{{tab:h1}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{\\textbf{{Las losowy}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Bagging}}}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
\\textbf{{Korelacja}} & $\\tilde\\rho$ & $\\rho>0$ & $p_H$ & $\\tilde\\rho$ & $\\rho>0$ & $p_H$ \\\\
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_h2(analysis, label, caption_extra=""):
    data = pd.read_csv(analysis / "h2_tests.csv")
    rows = []
    for task in ("classification", "regression"):
        part = data[data["task"] == task]
        if part.empty:
            continue
        rows.append(f"\\midrule\n\\multicolumn{{7}}{{l}}{{\\emph{{{TASK[task]}}}}}\\\\")
        for measure in MEASURE:
            m = part[part["measure"] == measure].set_index("type")
            if m.empty:
                continue
            a, b = m.loc["rho"], m.loc["partial_rho"]
            rows.append(
                f"{MEASURE[measure]} & {a['median']:+.2f} & "
                f"{int(a['n_positive'])}/{int(a['n_datasets'])} & "
                f"{bold_if(fmt_p(a['p_holm']), a['p_holm'] < 0.05)} & "
                f"{b['median']:+.2f} & {int(b['n_positive'])}/{int(b['n_datasets'])} & "
                f"{bold_if(fmt_p(b['p_holm']), b['p_holm'] < 0.05)} \\\\")
    write(label, f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{H2: korelacja Spearmana między średnią wartością miary a wynikiem zespołu po 15 wariantach, liczona w każdym zbiorze osobno{caption_extra}. Podano medianę po zbiorach, liczbę zbiorów z dodatnią korelacją i wartość $p_H$ testu Wilcoxona z korektą Holma (7 miar). Korelacja cząstkowa uwzględnia średnią jakość drzew.}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{\\textbf{{korelacja}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{cząstkowa}}}} \\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-7}}
\\textbf{{Miara}} & $\\tilde\\rho$ & $\\rho>0$ & $p_H$ & $\\tilde\\rho$ & $\\rho>0$ & $p_H$ \\\\
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_measures_ranks(analysis):
    data = pd.read_csv(analysis / "measures_vs_ranks_correlation.csv")
    rows = []
    for measure in MEASURE:
        cells = [MEASURE[measure]]
        for task in ("classification", "regression"):
            r = data[(data["task"] == task) & (data["measure"] == measure)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [f"{r['rho_with_rank']:+.2f}",
                      bold_if(fmt_p(r["p_holm"]), r["p_holm"] < 0.05)]
        rows.append(" & ".join(cells) + " \\\\")
    write("measures_ranks", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{Związek miar ze średnimi rangami wariantów: korelacja Spearmana między średnią wartością miary wariantu (po zbiorach) a jego średnią rangą dla 15 wariantów. Korelacja ujemna oznacza, że wariant o wyższej wartości miary ma lepszą rangę. Wartości $p_H$ podano po korekcie Holma.}}
\\label{{tab:miary-rangi}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{Klasyfikacja}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Regresja}}}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}
\\textbf{{Miara}} & $\\rho$ & $p_H$ & $\\rho$ & $p_H$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_draws(analysis):
    path = analysis / "draw_tests.csv"
    if not path.exists() or path.stat().st_size < 5:
        write("draws", "")
        return
    data = pd.read_csv(path)
    rows = []
    for measure in list(MEASURE)[:6]:
        cells = [MEASURE[measure]]
        for task in ("classification", "regression"):
            r = data[(data["task"] == task) & (data["measure"] == measure)]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [f"{r['median_rho']:+.3f}",
                      bold_if(fmt_p(r["p_holm"]), r["p_holm"] < 0.05)]
        rows.append(" & ".join(cells) + " \\\\")
    write("draws", f"""\\begin{{table}}[!htbp]
\\centering\\small
\\caption{{Eksperyment dodatkowy: korelacja Spearmana między miarą próbek a jakością zespołu dla 20 losowań próbek w tym samym podziale (bagging, wariant S1 $r=1$). Podano medianę po zbiorach z median po 5 podziałach oraz wartość $p_H$ testu Wilcoxona z korektą Holma (6 miar).}}
\\label{{tab:losowania}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{\\textbf{{Klasyfikacja}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Regresja}}}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}
\\textbf{{Miara}} & $\\tilde\\rho$ & $p_H$ & $\\tilde\\rho$ & $p_H$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def table_per_dataset(analysis):
    """Appendix: mean (sd) of the primary metric for every dataset/variant."""

    data = pd.read_csv(analysis / "quality_table.csv")
    out = []
    for task, metric_label in (("classification", "macro-F1"),
                               ("regression", "RMSE")):
        part = data[data["task"] == task]
        configs = config_order(part["sampling_config"].unique())
        for chunk in (configs[:8], configs[8:]):
            header = " & ".join(config_name(c) for c in chunk)
            rows = []
            for dataset, g in part.groupby("dataset"):
                g = g.set_index("sampling_config")
                scale = 1.0
                if task == "regression":
                    base = g["mean"].abs().max()
                    digits = 3 if base < 10 else (2 if base < 100 else 1)
                else:
                    digits = 3
                cells = [NAMES[dataset]]
                best = (g.loc[chunk, "mean"].max() if task == "classification"
                        else g.loc[chunk, "mean"].min())
                all_best = (g["mean"].max() if task == "classification"
                            else g["mean"].min())
                for c in chunk:
                    m, s = g.loc[c, "mean"] * scale, g.loc[c, "std"] * scale
                    text = f"{m:.{digits}f}"
                    if np.isclose(g.loc[c, "mean"], all_best):
                        text = f"\\textbf{{{text}}}"
                    cells.append(f"\\makecell{{{text}\\\\{{\\tiny ({s:.{digits}f})}}}}")
                rows.append(" & ".join(cells) + " \\\\")
            cols = "l" + "c" * len(chunk)
            out.append(f"""{{\\scriptsize\\setlength{{\\tabcolsep}}{{1.5pt}}
\\begin{{longtable}}{{{cols}}}
\\caption{{{TASK[task]}: {metric_label} zespołu (las losowy), średnia i w nawiasie odchylenie standardowe z 50 podziałów. Pogrubiono najlepszy z 15 wariantów w zbiorze.}}\\\\
\\toprule
\\textbf{{Zbiór}} & {header} \\\\
\\midrule
\\endhead
{chr(10).join(rows)}
\\bottomrule
\\end{{longtable}}}}
""")
    write("per_dataset", "\n".join(out))


def cpu_hours(runs):
    total = 0.0
    for run in runs:
        path = ROOT / "results" / run / "results.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path, usecols=lambda c: c in (
            "dataset", "fold", "fold_seconds", "draw"))
        if "fold_seconds" in r:
            total += r.groupby(["dataset", "fold"])["fold_seconds"].first().sum()
    write("cpu_hours", f"{total / 3600:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--bagging-run", default="main_bagging")
    parser.add_argument("--draws-run", default="main_draws")
    args = parser.parse_args()

    analysis = ROOT / "results" / args.run / "analysis"
    bagging = ROOT / "results" / args.bagging_run / "analysis"

    table_datasets(analysis)
    table_power(analysis)
    table_control(analysis)
    table_friedman(analysis)
    table_s3(analysis)
    table_h1(analysis, bagging)
    table_h2(analysis, "h2")
    if (bagging / "h2_tests.csv").exists():
        table_h2(bagging, "h2bag", " (bagging, klasyfikacja)")
    table_measures_ranks(analysis)
    table_draws(analysis)
    table_per_dataset(analysis)
    cpu_hours([args.run, args.bagging_run, args.draws_run])

    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    for png in (ROOT / "figures" / args.run).glob("*.png"):
        shutil.copy(png, figures / png.name)

    print("tables ->", TABLES, "| figures ->", figures)


if __name__ == "__main__":
    main()
