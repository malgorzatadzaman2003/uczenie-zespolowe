import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.sampling import (
    generate_bootstrap_samples,
    generate_subsamples
)

from src.sampling import (
    generate_bootstrap_samples,
    generate_subsamples,
    generate_common_core_samples,
    generate_stratified_bootstrap_samples
)

from src.diversity import (
    jaccard_distance,
    multiplicity_distance,
    mean_pairwise_jaccard_distance,
    mean_pairwise_multiplicity_distance,
    target_distribution_distance,
    mean_pairwise_target_distance,
    unique_fraction,
    target_representativeness,
    feature_representativeness
)

from src.models import (
    create_decision_tree,
    train_ensemble,
    predict_ensemble,
    aggregate_predictions
)

from src.metrics import (
    classification_metrics,
    regression_metrics,
    evaluate_individual_models,
    evaluate_ensemble
)

from src.datasets import list_datasets, load_dataset

from src.preprocessing import (
    preprocess_train_test,
    get_feature_names
)

from src.experiment import run_dataset_experiment

from src.statistics import run_statistical_analysis

# ============================================================
# TEST WIZUALIZACJI
# ============================================================

from src.visualization import run_visualizations

def main():
    print("=" * 50)
    print("UCZENIE ZESPOLOWE - ANALIZA BAGGINGU")
    print("=" * 50)

    print("\nTEST PRÓBKOWANIA")

    n = 1000

    bootstrap_samples = generate_bootstrap_samples(
        n=n,
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    subsamples = generate_subsamples(
        n=n,
        ratio=0.63,
        n_estimators=100,
        random_state=42
    )

    print("Liczba próbek bootstrap:", len(bootstrap_samples))
    print("Rozmiar pierwszej próbki:", len(bootstrap_samples[0]))
    print("Unikalne obserwacje w bootstrap:",
          len(np.unique(bootstrap_samples[0])))

    print("\nLiczba podpróbek:", len(subsamples))
    print("Rozmiar pierwszej podpróbki:", len(subsamples[0]))
    print("Unikalne obserwacje w podpróbce:",
          len(np.unique(subsamples[0])))


def test_sampling():

    print("\n" + "=" * 60)
    print("TESTOWANIE WARIANTOW PROBKOWANIA")
    print("=" * 60)

    n = 1000
    n_estimators = 100
    seed = 42

    # S1: Bootstrap
    bootstrap = generate_bootstrap_samples(
        n=n,
        ratio=1.0,
        n_estimators=n_estimators,
        random_state=seed
    )

    print("\nS1 - Bootstrap")
    print("Liczba próbek:", len(bootstrap))
    print("Rozmiar próbki:", len(bootstrap[0]))
    print(
        "Liczba unikalnych obserwacji:",
        len(np.unique(bootstrap[0]))
    )

    # S2: Subsampling
    subsamples = generate_subsamples(
        n=n,
        ratio=0.63,
        n_estimators=n_estimators,
        random_state=seed
    )

    print("\nS2 - Subsampling")
    print("Liczba próbek:", len(subsamples))
    print("Rozmiar próbki:", len(subsamples[0]))
    print(
        "Liczba unikalnych obserwacji:",
        len(np.unique(subsamples[0]))
    )

    # S3: Common core
    common_core = generate_common_core_samples(
        n=n,
        ratio=0.5,
        alpha=0.5,
        n_estimators=n_estimators,
        random_state=seed
    )

    print("\nS3 - Common core")
    print("Liczba próbek:", len(common_core))
    print("Rozmiar próbki:", len(common_core[0]))
    print(
        "Liczba wspólnych obserwacji:",
        len(
            set(common_core[0]).intersection(
                set(common_core[1])
            )
        )
    )

    # S4: Stratified bootstrap - klasyfikacja
    y_class = np.array(
        [0] * 700 + [1] * 300
    )

    stratified_class = generate_stratified_bootstrap_samples(
        y=y_class,
        task="classification",
        n_estimators=n_estimators,
        random_state=seed
    )

    print("\nS4 - Stratified bootstrap (klasyfikacja)")
    print("Liczba próbek:", len(stratified_class))
    print("Rozmiar próbki:", len(stratified_class[0]))

    print(
        "Liczba obserwacji klasy 0:",
        np.sum(y_class[stratified_class[0]] == 0)
    )

    print(
        "Liczba obserwacji klasy 1:",
        np.sum(y_class[stratified_class[0]] == 1)
    )

    # S4: Stratified bootstrap - regresja
    y_regression = np.linspace(0, 100, 1000)

    stratified_reg = generate_stratified_bootstrap_samples(
        y=y_regression,
        task="regression",
        n_estimators=n_estimators,
        random_state=seed
    )

    print("\nS4 - Stratified bootstrap (regresja)")
    print("Liczba próbek:", len(stratified_reg))
    print("Rozmiar próbki:", len(stratified_reg[0]))

    print("\nTesty próbkowania zakończone.")


def test_diversity():

    print("\n" + "=" * 60)
    print("TESTOWANIE MIAR ROZNORODNOSCI")
    print("=" * 60)

    # -----------------------------------------
    # Test 1: Odległość Jaccarda
    # -----------------------------------------

    sample_a = np.array([0, 1, 2, 2, 3])
    sample_b = np.array([0, 1, 4, 4, 5])

    jaccard = jaccard_distance(sample_a, sample_b)

    print("\nTest 1 - Jaccard distance")
    print("Próbka A:", sample_a)
    print("Próbka B:", sample_b)
    print("Odległość Jaccarda:", jaccard)

    # Oczekiwany wynik: 1 - 2/6 = 0.6667
    assert np.isclose(jaccard, 2 / 3)

    # -----------------------------------------
    # Test 2: Identyczne zbiory
    # -----------------------------------------

    sample_c = np.array([0, 0, 1, 2])
    sample_d = np.array([2, 1, 0, 0])

    jaccard_identical = jaccard_distance(
        sample_c,
        sample_d
    )

    print("\nTest 2 - Identyczne zbiory")
    print("Odległość Jaccarda:", jaccard_identical)

    assert np.isclose(jaccard_identical, 0.0)

    # -----------------------------------------
    # Test 3: Odległość krotności
    # -----------------------------------------

    sample_e = np.array([0, 0, 1, 2])
    sample_f = np.array([0, 1, 1, 3])

    multiplicity = multiplicity_distance(
        sample_e,
        sample_f,
        n_observations=4
    )

    print("\nTest 3 - Multiplicity distance")
    print("Próbka E:", sample_e)
    print("Próbka F:", sample_f)
    print("Odległość krotności:", multiplicity)

    # Oczekiwany wynik: 0.5
    assert np.isclose(multiplicity, 0.5)

    # -----------------------------------------
    # Test 4: Średnia dla wszystkich par
    # -----------------------------------------

    samples = [
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 3]),
        np.array([2, 2, 3, 3])
    ]

    mean_jaccard = mean_pairwise_jaccard_distance(
        samples
    )

    mean_multiplicity = (
        mean_pairwise_multiplicity_distance(
            samples,
            n_observations=4
        )
    )

    print("\nTest 4 - Średnie miary zespołu")
    print("Średnia odległość Jaccarda:", mean_jaccard)
    print("Średnia odległość krotności:", mean_multiplicity)

    assert 0 <= mean_jaccard <= 1
    assert 0 <= mean_multiplicity <= 1

    print("\nWszystkie testy miar zakończone poprawnie!")


    # -----------------------------------------
    # Test 5: DY dla klasyfikacji
    # -----------------------------------------

    y_class = np.array([0, 0, 1, 1])

    sample_a = np.array([0, 0, 1, 2])
    sample_b = np.array([0, 2, 2, 3])

    dy_class = target_distribution_distance(
        sample_a,
        sample_b,
        y_class,
        task="classification"
    )

    print("\nTest 5 - Target distance (classification)")
    print("Odległość DY:", dy_class)

    # Próbka A: 75% klasy 0, 25% klasy 1.
    # Próbka B: 25% klasy 0, 75% klasy 1.
    # Oczekiwany wynik: 0.5.
    assert np.isclose(dy_class, 0.5)

    # -----------------------------------------
    # Test 6: DY dla regresji
    # -----------------------------------------

    y_reg = np.array([0, 1, 2, 3, 4], dtype=float)

    sample_a = np.array([0, 1])
    sample_b = np.array([2, 3])

    dy_reg = target_distribution_distance(
        sample_a,
        sample_b,
        y_reg,
        task="regression"
    )

    print("\nTest 6 - Target distance (regression)")
    print("Odległość DY:", dy_reg)

    # Wartości próbek: [0, 1] oraz [2, 3].
    # Odległość Wassersteina wynosi 2.
    # Odchylenie standardowe y_reg wynosi sqrt(2).
    # Oczekiwany wynik: sqrt(2).
    assert np.isclose(dy_reg, np.sqrt(2))

    # -----------------------------------------
    # Test 7: Średnia DY dla wszystkich par
    # -----------------------------------------

    samples_class = [
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 3]),
        np.array([2, 2, 3, 3])
    ]

    mean_dy_class = mean_pairwise_target_distance(
        samples_class,
        y_class,
        task="classification"
    )

    print("\nTest 7 - Mean target distance")
    print("Średnia DY:", mean_dy_class)

    assert 0 <= mean_dy_class <= 1

    print("\nTesty DY zakończone poprawnie!")


    # -----------------------------------------
    # Test 8: Udział unikalnych obserwacji Ui
    # -----------------------------------------

    sample = np.array([0, 0, 1, 2, 2, 3])

    ui = unique_fraction(
        sample,
        n_observations=10
    )

    print("\nTest 8 - Unique fraction")
    print("Udział unikalnych obserwacji:", ui)

    # 4 unikalne obserwacje / 10 wszystkich.
    assert np.isclose(ui, 0.4)

    # -----------------------------------------
    # Test 9: Reprezentatywność targetu - klasyfikacja
    # -----------------------------------------

    y_class = np.array([0, 0, 1, 1])

    # Próbka zawiera te same proporcje klas co cały zbiór.
    sample = np.array([0, 1, 2, 3])

    ry_class = target_representativeness(
        sample,
        y_class,
        task="classification"
    )

    print("\nTest 9 - Target representativeness (classification)")
    print("RY:", ry_class)

    # Rozkłady klas są identyczne.
    assert np.isclose(ry_class, 0.0)

    # -----------------------------------------
    # Test 10: Reprezentatywność targetu - regresja
    # -----------------------------------------

    y_reg = np.array([0, 1, 2, 3, 4], dtype=float)

    # Próbka zawiera cały zbiór treningowy.
    sample = np.arange(len(y_reg))

    ry_reg = target_representativeness(
        sample,
        y_reg,
        task="regression"
    )

    print("\nTest 10 - Target representativeness (regression)")
    print("RY:", ry_reg)

    # Identyczne rozkłady dają odległość zero.
    assert np.isclose(ry_reg, 0.0)

    # -----------------------------------------
    # Test 11: Reprezentatywność cech RX
    # -----------------------------------------

    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0]
    ])

    # Próbka zawiera wszystkie obserwacje.
    sample = np.arange(len(X_train))

    rx = feature_representativeness(
        sample,
        X_train
    )

    print("\nTest 11 - Feature representativeness")
    print("RX:", rx)

    # Rozkłady cech są identyczne.
    assert np.isclose(rx, 0.0)

    # -----------------------------------------
    # Test 12: RX dla próbki niereprezentatywnej
    # -----------------------------------------

    sample = np.array([2, 3])

    rx_partial = feature_representativeness(
        sample,
        X_train
    )

    print("\nTest 12 - Feature representativeness (partial sample)")
    print("RX:", rx_partial)

    # Próbka zawiera wyłącznie obserwacje o większych
    # wartościach cech, więc odległość powinna być dodatnia.
    assert rx_partial > 0

    print(
        "\nTesty miar różnorodności i reprezentatywności "
        "zakończone poprawnie!"
    )

def test_models():
    print("\n" + "=" * 60)
    print("TESTOWANIE MODELI")
    print("=" * 60)

    # ==========================================
    # TEST 1: KLASYFIKACJA
    # ==========================================

    print("\nTEST 1 - Klasyfikacja")

    X_class, y_class = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    samples_class = generate_bootstrap_samples(
        n=len(X_class),
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    models_class = train_ensemble(
        X_train=X_class,
        y_train=y_class,
        samples=samples_class,
        task="classification",
        random_state=42
    )

    predictions_class = predict_ensemble(
        models=models_class,
        X=X_class
    )

    final_class = aggregate_predictions(
        predictions=predictions_class,
        task="classification"
    )

    print("Liczba drzew:", len(models_class))
    print("Rozmiar macierzy predykcji:", predictions_class.shape)
    print("Liczba predykcji zespołu:", len(final_class))
    print("Przykładowe predykcje:", final_class[:10])

    assert len(models_class) == 100
    assert predictions_class.shape == (100, 200)
    assert len(final_class) == 200

    # ==========================================
    # TEST 2: REGRESJA
    # ==========================================

    print("\nTEST 2 - Regresja")

    X_reg, y_reg = make_regression(
        n_samples=200,
        n_features=8,
        n_informative=6,
        noise=10,
        random_state=42
    )

    samples_reg = generate_bootstrap_samples(
        n=len(X_reg),
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    models_reg = train_ensemble(
        X_train=X_reg,
        y_train=y_reg,
        samples=samples_reg,
        task="regression",
        random_state=42
    )

    predictions_reg = predict_ensemble(
        models=models_reg,
        X=X_reg
    )

    final_reg = aggregate_predictions(
        predictions=predictions_reg,
        task="regression"
    )

    print("Liczba drzew:", len(models_reg))
    print("Rozmiar macierzy predykcji:", predictions_reg.shape)
    print("Liczba predykcji zespołu:", len(final_reg))
    print("Przykładowe predykcje:", final_reg[:5])

    assert len(models_reg) == 100
    assert predictions_reg.shape == (100, 200)
    assert len(final_reg) == 200

    print("\nWszystkie testy modeli zakończone poprawnie!")


def test_metrics():
    print("\n" + "=" * 60)
    print("TESTOWANIE METRYK")
    print("=" * 60)

    # ==========================================
    # TEST 1: KLASYFIKACJA
    # ==========================================

    print("\nTEST 1 - Klasyfikacja")

    X_class, y_class = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        test_size=0.25,
        random_state=42,
        stratify=y_class
    )

    classes = np.unique(y_train)

    samples = generate_bootstrap_samples(
        n=len(X_train),
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    models = train_ensemble(
        X_train=X_train,
        y_train=y_train,
        samples=samples,
        task="classification",
        random_state=42
    )

    individual_results = evaluate_individual_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        task="classification",
        classes=classes
    )

    ensemble_results = evaluate_ensemble(
        models=models,
        X_test=X_test,
        y_test=y_test,
        task="classification",
        classes=classes
    )

    print("\nŚrednie wyniki pojedynczych drzew:")
    for metric, values in individual_results.items():
        print(
            f"{metric}: "
            f"średnia = {values['mean']:.4f}, "
            f"odchylenie = {values['std']:.4f}"
        )

    print("\nWyniki całego zespołu:")
    for metric, value in ensemble_results.items():
        print(f"{metric}: {value:.4f}")

    assert len(individual_results) == 3
    assert len(ensemble_results) == 3

    # ==========================================
    # TEST 2: REGRESJA
    # ==========================================

    print("\nTEST 2 - Regresja")

    X_reg, y_reg = make_regression(
        n_samples=300,
        n_features=8,
        n_informative=6,
        noise=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.25,
        random_state=42
    )

    samples = generate_bootstrap_samples(
        n=len(X_train),
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    models = train_ensemble(
        X_train=X_train,
        y_train=y_train,
        samples=samples,
        task="regression",
        random_state=42
    )

    individual_results = evaluate_individual_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        task="regression"
    )

    ensemble_results = evaluate_ensemble(
        models=models,
        X_test=X_test,
        y_test=y_test,
        task="regression"
    )

    print("\nŚrednie wyniki pojedynczych drzew:")
    for metric, values in individual_results.items():
        print(
            f"{metric}: "
            f"średnia = {values['mean']:.4f}, "
            f"odchylenie = {values['std']:.4f}"
        )

    print("\nWyniki całego zespołu:")
    for metric, value in ensemble_results.items():
        print(f"{metric}: {value:.4f}")

    assert len(individual_results) == 3
    assert len(ensemble_results) == 3

    print("\nWszystkie testy metryk zakończone poprawnie!")


def test_datasets():
    print("\n" + "=" * 70)
    print("AUDYT WSZYSTKICH ZBIORÓW DANYCH")
    print("=" * 70)

    from pathlib import Path

    # Pobieramy listy zbiorów dla obu rodzajów zadań.
    classification_datasets = list_datasets("classification")
    regression_datasets = list_datasets("regression")

    print("\nLiczba zbiorów klasyfikacyjnych:",
          len(classification_datasets))
    print("Liczba zbiorów regresyjnych:",
          len(regression_datasets))

    assert len(classification_datasets) == 12
    assert len(regression_datasets) == 12

    # Zapisujemy wyniki audytu.
    audit_results = []

    # Błędy uniemożliwiające wykorzystanie zbioru.
    failed_datasets = []

    # Ostrzeżenia wymagające późniejszej weryfikacji.
    warnings = []

    datasets_to_check = [
        ("classification", name)
        for name in classification_datasets
    ] + [
        ("regression", name)
        for name in regression_datasets
    ]

    print("\n" + "=" * 70)
    print("WCZYTYWANIE I SPRAWDZANIE ZBIORÓW")
    print("=" * 70)

    for expected_task, dataset_name in datasets_to_check:

        print("\n" + "-" * 70)
        print(f"Zbiór: {dataset_name}")
        print(f"Oczekiwany typ zadania: {expected_task}")

        try:
            # -----------------------------------------
            # 1. Wczytanie zbioru
            # -----------------------------------------

            dataset = load_dataset(dataset_name)

            X = dataset["X"]
            y = dataset["y"]
            actual_task = dataset["task"]

            dataset_warnings = []
            dataset_errors = []

            # -----------------------------------------
            # 2. Sprawdzenie podstawowej struktury
            # -----------------------------------------

            if not isinstance(X, pd.DataFrame):
                dataset_errors.append(
                    "X nie jest obiektem pandas DataFrame"
                )

            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            if len(X) != len(y):
                dataset_errors.append(
                    "Liczba obserwacji X i y jest różna"
                )

            if len(X) == 0:
                dataset_errors.append(
                    "Zbiór nie zawiera obserwacji"
                )

            if X.shape[1] == 0:
                dataset_errors.append(
                    "Zbiór nie zawiera żadnych cech"
                )

            # -----------------------------------------
            # 3. Sprawdzenie typu zadania
            # -----------------------------------------

            if actual_task != expected_task:
                dataset_errors.append(
                    f"Nieprawidłowy typ zadania: {actual_task}"
                )

            # -----------------------------------------
            # 4. Sprawdzenie zmiennej docelowej
            # -----------------------------------------

            missing_targets = int(y.isna().sum())

            if missing_targets > 0:
                dataset_errors.append(
                    f"Zmienna docelowa zawiera "
                    f"{missing_targets} brakujących wartości"
                )

            if expected_task == "classification":

                n_classes = int(y.nunique())

                if n_classes < 2:
                    dataset_errors.append(
                        "Klasyfikacja ma mniej niż 2 klasy"
                    )

                class_counts = y.value_counts()

                min_class_size = (
                    int(class_counts.min())
                    if len(class_counts) > 0
                    else 0
                )

                if min_class_size < 5:
                    dataset_warnings.append(
                        "Co najmniej jedna klasa ma mniej niż "
                        "5 obserwacji"
                    )

            else:
                n_classes = np.nan
                min_class_size = np.nan

                if not pd.api.types.is_numeric_dtype(y):
                    dataset_errors.append(
                        "Zmienna docelowa regresji nie jest numeryczna"
                    )

                else:
                    y_values = pd.to_numeric(
                        y,
                        errors="coerce"
                    ).to_numpy(dtype=float)

                    if not np.isfinite(y_values).all():
                        dataset_errors.append(
                            "Zmienna docelowa zawiera wartości "
                            "nieskończone lub niepoprawne"
                        )

            # -----------------------------------------
            # 5. Sprawdzenie brakujących danych w X
            # -----------------------------------------

            missing_features = int(X.isna().sum().sum())

            missing_columns = int(
                (X.isna().sum() > 0).sum()
            )

            if missing_features > 0:
                dataset_warnings.append(
                    f"Braki danych w X: {missing_features} "
                    f"wartości w {missing_columns} kolumnach"
                )

            # -----------------------------------------
            # 6. Sprawdzenie wartości nieskończonych
            # -----------------------------------------

            numeric_X = X.select_dtypes(include=[np.number])

            if numeric_X.shape[1] > 0:

                numeric_values = numeric_X.to_numpy(
                    dtype=float
                )

                infinite_values = int(
                    np.isinf(numeric_values).sum()
                )

            else:
                infinite_values = 0

            if infinite_values > 0:
                dataset_errors.append(
                    f"Znaleziono {infinite_values} "
                    "nieskończonych wartości w cechach"
                )

            # -----------------------------------------
            # 7. Powtarzające się kolumny
            # -----------------------------------------

            duplicated_columns = int(
                X.columns.duplicated().sum()
            )

            if duplicated_columns > 0:
                dataset_errors.append(
                    f"Znaleziono {duplicated_columns} "
                    "powtarzających się nazw kolumn"
                )

            # -----------------------------------------
            # 8. Kolumny stałe
            # -----------------------------------------

            constant_columns = [
                column
                for column in X.columns
                if X[column].nunique(dropna=False) <= 1
            ]

            if constant_columns:
                dataset_warnings.append(
                    "Kolumny stałe: "
                    + ", ".join(map(str, constant_columns))
                )

            # -----------------------------------------
            # 9. Potencjalne kolumny identyfikacyjne
            # -----------------------------------------

            possible_id_columns = [
                column
                for column in X.columns
                if any(
                    token in str(column).lower()
                    for token in [
                        "id",
                        "identifier",
                        "index",
                        "customer_number"
                    ]
                )
            ]

            if possible_id_columns:
                dataset_warnings.append(
                    "Potencjalne kolumny identyfikacyjne: "
                    + ", ".join(map(str, possible_id_columns))
                )

            # -----------------------------------------
            # 10. Wyniki
            # -----------------------------------------

            status = (
                "OK"
                if len(dataset_errors) == 0
                else "ERROR"
            )

            print("\nWyniki:")
            print("Status:", status)
            print("Liczba obserwacji:", len(X))
            print("Liczba cech:", X.shape[1])
            print("Typ zadania:", actual_task)

            if expected_task == "classification":
                print("Liczba klas:", n_classes)
                print("Najmniejsza klasa:", min_class_size)

            print("Brakujące wartości w X:", missing_features)
            print("Wartości nieskończone:", infinite_values)

            if dataset_warnings:
                print("\nOstrzeżenia:")

                for warning in dataset_warnings:
                    print(" -", warning)

                    warnings.append({
                        "dataset": dataset_name,
                        "warning": warning
                    })

            if dataset_errors:
                print("\nBłędy:")

                for error in dataset_errors:
                    print(" -", error)

            else:
                print("\nZbiór przeszedł podstawowe testy.")

            if dataset_errors:
                failed_datasets.append({
                    "dataset": dataset_name,
                    "errors": dataset_errors
                })

            audit_results.append({
                "dataset": dataset_name,
                "expected_task": expected_task,
                "actual_task": actual_task,
                "n_observations": len(X),
                "n_features": X.shape[1],
                "n_classes": n_classes,
                "missing_features": missing_features,
                "missing_columns": missing_columns,
                "infinite_values": infinite_values,
                "constant_columns": len(constant_columns),
                "duplicated_columns": duplicated_columns,
                "possible_id_columns": len(possible_id_columns),
                "status": status,
                "warnings": " | ".join(dataset_warnings),
                "errors": " | ".join(dataset_errors)
            })

        except Exception as error:

            print("\nBŁĄD WCZYTYWANIA ZBIORU!")
            print(type(error).__name__, ":", error)

            failed_datasets.append({
                "dataset": dataset_name,
                "errors": [str(error)]
            })

            audit_results.append({
                "dataset": dataset_name,
                "expected_task": expected_task,
                "actual_task": None,
                "n_observations": None,
                "n_features": None,
                "n_classes": None,
                "missing_features": None,
                "missing_columns": None,
                "infinite_values": None,
                "constant_columns": None,
                "duplicated_columns": None,
                "possible_id_columns": None,
                "status": "LOAD_ERROR",
                "warnings": "",
                "errors": str(error)
            })

    # ========================================================
    # PODSUMOWANIE
    # ========================================================

    audit_df = pd.DataFrame(audit_results)

    Path("results").mkdir(
        parents=True,
        exist_ok=True
    )

    audit_df.to_csv(
        "results/dataset_audit.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("PODSUMOWANIE AUDYTU")
    print("=" * 70)

    print("\nLiczba sprawdzonych zbiorów:", len(audit_df))

    print(
        "Zbiory z poprawnym statusem:",
        (audit_df["status"] == "OK").sum()
    )

    print(
        "Zbiory z błędami:",
        (audit_df["status"] != "OK").sum()
    )

    print("Liczba ostrzeżeń:", len(warnings))

    if failed_datasets:

        print("\nZBIORY WYMAGAJĄCE POPRAWY:")

        for item in failed_datasets:
            print("\n", item["dataset"])

            for error in item["errors"]:
                print(" -", error)

    else:
        print("\nWszystkie zbiory zostały poprawnie wczytane.")

    print("\nPodsumowanie zapisano w:")
    print("results/dataset_audit.csv")

    print("\n" + "=" * 70)

def test_preprocessing():
    print("\n" + "=" * 60)
    print("TESTOWANIE PREPROCESSINGU")
    print("=" * 60)

    # ==========================================
    # Dane testowe z brakami
    # ==========================================

    X_train = pd.DataFrame({
        "age": [20, 30, np.nan, 40, 50, 60],
        "income": [2000, 3000, 2500, np.nan, 5000, 6000],
        "city": ["Warsaw", "Krakow", None, "Warsaw", "Gdansk", "Krakow"],
        "education": ["high", "medium", "high", None, "low", "medium"]
    })

    X_test = pd.DataFrame({
        "age": [25, np.nan, 55],
        "income": [2200, 4500, np.nan],
        "city": ["Warsaw", "Poznan", None],
        "education": ["high", "very_high", "low"]
    })

    print("\nLiczba obserwacji treningowych:", len(X_train))
    print("Liczba obserwacji testowych:", len(X_test))

    print("\nBraki przed preprocessingiem:")
    print("Trening:", X_train.isna().sum().sum())
    print("Test:", X_test.isna().sum().sum())

    # ==========================================
    # Preprocessing
    # ==========================================

    X_train_processed, X_test_processed, preprocessor = (
        preprocess_train_test(X_train, X_test)
    )

    feature_names = get_feature_names(preprocessor)

    print("\nWymiary po preprocessingu:")
    print("Trening:", X_train_processed.shape)
    print("Test:", X_test_processed.shape)

    print("\nNazwy cech po preprocessingu:")
    print(feature_names)

    print("\nLiczba wartości NaN po preprocessingu:")
    print("Trening:", np.isnan(X_train_processed).sum())
    print("Test:", np.isnan(X_test_processed).sum())

    # ==========================================
    # Asercje
    # ==========================================

    assert X_train_processed.shape[0] == len(X_train)
    assert X_test_processed.shape[0] == len(X_test)

    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    assert np.isfinite(X_train_processed).all()
    assert np.isfinite(X_test_processed).all()

    # Sprawdzamy, czy nowa kategoria "Poznan"
    # nie spowodowała błędu.
    assert "city_Poznan" not in feature_names

    # Sprawdzamy, czy powstała kolumna dla kategorii brakującej.
    assert any(
        "missing" in str(name)
        for name in feature_names
    )

    print("\nWszystkie testy preprocessingu zakończone poprawnie!")

def test_experiment():
    print("\n" + "=" * 60)
    print("TESTOWANIE EKSPERYMENTU")
    print("=" * 60)

    results = run_dataset_experiment(
        dataset_name="breast_cancer",
        n_splits=2,
        n_repeats=1,
        n_estimators=5,
        random_state=42,
    )

    print("\nPierwsze wyniki eksperymentu:")
    print(results.head())

    print("\nWymiary tabeli wynikowej:")
    print(results.shape)

    results.to_csv(
        "results/test_experiment.csv",
        index=False,
    )

    print("\nWyniki zapisano w results/test_experiment.csv")


def test_statistical_analysis_and_visualization():
    test_results = pd.read_csv("results/test_experiment.csv")

    statistical_results = run_statistical_analysis(
        results=test_results,
        output_dir="results/statistics",
    )

    print("\n" + "=" * 60)
    print("AGREGACJA WYNIKÓW")
    print("=" * 60)

    print(
        statistical_results["aggregated_results"].head()
    )

    print("\n" + "=" * 60)
    print("KORELACJE SPEARMANA")
    print("=" * 60)

    print(
        statistical_results["spearman_by_dataset"].head()
    )

    print("\n" + "=" * 60)
    print("PODSUMOWANIE KORELACJI")
    print("=" * 60)

    print(
        statistical_results["spearman_summary"].head()
    )

    print("\n" + "=" * 60)
    print("TEST FRIEDMANA")
    print("=" * 60)

    print(
        statistical_results["friedman_tests"]
    )

    print("\n" + "=" * 60)
    print("TESTY WILCOXONA")
    print("=" * 60)

    print(
        statistical_results["wilcoxon_pairwise"].head()
    )

    test_results_vis = pd.read_csv(
    "results/test_experiment.csv"
    )

    run_visualizations(
        results=statistical_results["aggregated_results"],
        statistical_results=statistical_results,
        output_dir="figures/test",
    )


if __name__ == "__main__":
    test_sampling()
    test_diversity()
    test_models()
    test_metrics()
    test_datasets()
    test_preprocessing()
    test_experiment()
    test_statistical_analysis_and_visualization()
