import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib

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
    mean_pairwise_multiplicity_distance
)

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


if __name__ == "__main__":
    test_sampling()
    test_diversity()