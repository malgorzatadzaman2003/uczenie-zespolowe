import numpy as np
from itertools import combinations
from scipy.stats import wasserstein_distance

def jaccard_distance(sample_a, sample_b):
    """
    Calculates the Jaccard distance between two samples.

    We compare sets of unique observation indices;
    therefore, multiple occurrences are not taken into account.

    Returns a value in the range [0, 1].
    """

    # Zamieniamy próbki na tablice NumPy.
    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)

    # Sprawdzamy, czy próbki nie są puste.
    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("Próbki nie mogą być puste.")

    # Usuwamy powtórzenia, ponieważ Jaccard
    # porównuje zbiory unikalnych obserwacji.
    set_a = set(sample_a)
    set_b = set(sample_b)

    # Część wspólna próbek.
    intersection = set_a.intersection(set_b)

    # Suma zbiorów.
    union = set_a.union(set_b)

    # Odległość Jaccarda.
    distance = 1 - len(intersection) / len(union)

    return distance



def multiplicity_distance(sample_a, sample_b, n_observations=None):
    """
    Calculates the multiplicity-based distance between two samples.

    Takes into account the number of occurrences of each observation.

    The samples must be of the same size.

    n_observations:
        The number of observations in the entire training set.
    """

    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)

    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("Próbki nie mogą być puste.")

    if len(sample_a) != len(sample_b):
        raise ValueError(
            "Próbki muszą mieć jednakowy rozmiar."
        )

    # Indeksy obserwacji muszą być nieujemnymi liczbami całkowitymi.
    if not np.issubdtype(sample_a.dtype, np.integer):
        raise ValueError("Indeksy próbek muszą być całkowite.")

    if not np.issubdtype(sample_b.dtype, np.integer):
        raise ValueError("Indeksy próbek muszą być całkowite.")

    if np.any(sample_a < 0) or np.any(sample_b < 0):
        raise ValueError("Indeksy nie mogą być ujemne.")

    # Ustalamy rozmiar całego zbioru treningowego.
    if n_observations is None:
        n_observations = int(
            max(np.max(sample_a), np.max(sample_b)) + 1
        )

    if n_observations <= 0:
        raise ValueError("n_observations musi być dodatnie.")

    if np.any(sample_a >= n_observations):
        raise ValueError("Indeks poza zakresem zbioru.")

    if np.any(sample_b >= n_observations):
        raise ValueError("Indeks poza zakresem zbioru.")

    # Liczymy, ile razy wystąpiła każda obserwacja.
    counts_a = np.bincount(
        sample_a,
        minlength=n_observations
    )

    counts_b = np.bincount(
        sample_b,
        minlength=n_observations
    )

    # Zamieniamy liczby wystąpień na udziały.
    m = len(sample_a)

    proportions_a = counts_a / m
    proportions_b = counts_b / m

    # Obliczamy połowę sumy różnic bezwzględnych.
    distance = 0.5 * np.sum(
        np.abs(proportions_a - proportions_b)
    )

    return distance



def mean_pairwise_jaccard_distance(samples):
    """
    Calculates the mean Jaccard distance
    for all pairs of samples in the set.
    """

    if len(samples) < 2:
        raise ValueError(
            "Potrzebne są co najmniej dwie próbki."
        )

    distances = []

    for sample_a, sample_b in combinations(samples, 2):
        distance = jaccard_distance(sample_a, sample_b)
        distances.append(distance)

    return float(np.mean(distances))


def mean_pairwise_multiplicity_distance(
    samples,
    n_observations
):
    """
    Calculates the mean multiplicity distance
    for all pairs of samples in the set.
    """

    if len(samples) < 2:
        raise ValueError(
            "Potrzebne są co najmniej dwie próbki."
        )

    distances = []

    for sample_a, sample_b in combinations(samples, 2):

        distance = multiplicity_distance(
            sample_a,
            sample_b,
            n_observations=n_observations
        )

        distances.append(distance)

    return float(np.mean(distances))



def target_distribution_distance(
    sample_a,
    sample_b,
    y_train,
    task="classification"
):
    """
    Computes the distance between the target-variable distributions
    of two samples.

    For classification:
        Total Variation Distance.

    For regression:
        Normalized Wasserstein distance.

    sample_a, sample_b:
        Indices of observations belonging to the samples.

    y_train:
        Target-variable values in the entire training set.

    task:
        "classification" or "regression".
    """

    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)
    y_train = np.asarray(y_train)

    if y_train.ndim != 1:
        raise ValueError("y_train musi być tablicą jednowymiarową.")

    if len(sample_a) == 0 or len(sample_b) == 0:
        raise ValueError("Próbki nie mogą być puste.")

    if np.any(sample_a < 0) or np.any(sample_b < 0):
        raise ValueError("Indeksy nie mogą być ujemne.")

    if np.any(sample_a >= len(y_train)):
        raise ValueError("Indeks próbki A poza zakresem y_train.")

    if np.any(sample_b >= len(y_train)):
        raise ValueError("Indeks próbki B poza zakresem y_train.")

    if task not in ("classification", "regression"):
        raise ValueError(
            "task musi mieć wartość 'classification' lub 'regression'."
        )

    # Pobieramy wartości zmiennej docelowej
    # odpowiadające obserwacjom w obu próbkach.
    y_a = y_train[sample_a]
    y_b = y_train[sample_b]

    # -----------------------------------------
    # KLASYFIKACJA
    # -----------------------------------------

    if task == "classification":

        # Wszystkie klasy występujące w zbiorze treningowym.
        classes = np.unique(y_train)

        # Obliczamy proporcje klas w obu próbkach.
        proportions_a = np.array([
            np.mean(y_a == label)
            for label in classes
        ])

        proportions_b = np.array([
            np.mean(y_b == label)
            for label in classes
        ])

        # Total Variation Distance.
        distance = 0.5 * np.sum(
            np.abs(proportions_a - proportions_b)
        )

        return float(distance)

    # -----------------------------------------
    # REGRESJA
    # -----------------------------------------

    if not np.issubdtype(y_train.dtype, np.number):
        raise ValueError(
            "Dla regresji y_train musi zawierać wartości liczbowe."
        )

    # Odchylenie standardowe całego zbioru treningowego.
    y_std = np.std(y_train)

    if y_std == 0:
        raise ValueError(
            "Nie można normalizować odległości: "
            "odchylenie standardowe y_train wynosi zero."
        )

    # Odległość Wassersteina między rozkładami
    # wartości docelowych obu próbek.
    raw_distance = wasserstein_distance(y_a, y_b)

    # Normalizacja względem zmienności y w zbiorze treningowym.
    distance = raw_distance / y_std

    return float(distance)


def mean_pairwise_target_distance(
    samples,
    y_train,
    task="classification"
):
    """
    Calculates the mean target-distribution distance
    for all pairs of samples in the ensemble.
    """

    if len(samples) < 2:
        raise ValueError(
            "Potrzebne są co najmniej dwie próbki."
        )

    distances = []

    for sample_a, sample_b in combinations(samples, 2):

        distance = target_distribution_distance(
            sample_a,
            sample_b,
            y_train,
            task=task
        )

        distances.append(distance)

    return float(np.mean(distances))



def unique_fraction(sample, n_observations):
    """
    Calculates the proportion of unique observations in the sample.

    Ui = number of unique indices / number of observations
         in the entire training set.
    """

    sample = np.asarray(sample)

    if len(sample) == 0:
        raise ValueError("Próbka nie może być pusta.")

    if n_observations <= 0:
        raise ValueError("n_observations musi być dodatnie.")

    if not np.issubdtype(sample.dtype, np.integer):
        raise ValueError("Indeksy próbki muszą być całkowite.")

    if np.any(sample < 0) or np.any(sample >= n_observations):
        raise ValueError("Indeks próbki poza zakresem zbioru.")

    return float(len(np.unique(sample)) / n_observations)


def target_representativeness(
    sample,
    y_train,
    task="classification"
):
    """
    Calculates the representativeness of a sample with respect to the 
    entire training distribution of the target variable.

    Classification:
        Total Variation Distance.

    Regression:
        Normalized Wasserstein distance.

    A lower value indicates greater representativeness.
    """

    sample = np.asarray(sample)
    y_train = np.asarray(y_train)

    # Indeksy wszystkich obserwacji treningowych.
    all_indices = np.arange(len(y_train))

    # Porównujemy próbkę z całym zbiorem treningowym.
    return target_distribution_distance(
        sample,
        all_indices,
        y_train,
        task=task
    )


def feature_representativeness(sample, X_train):
    """
    Calculates the representativeness of a sample with respect to the
    distributions of all features in the entire training set.

    For each feature:
        1. Standardize the values ​​based on X_train.
        2. Calculate the Wasserstein distance between the sample
           and the entire training set.

    The result is the average distance across all features.

    X_train must be a 2D numerical matrix that has undergone preprocessing.
    """

    sample = np.asarray(sample)
    X_train = np.asarray(X_train, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train musi być macierzą dwuwymiarową.")

    if len(sample) == 0:
        raise ValueError("Próbka nie może być pusta.")

    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        raise ValueError("X_train nie może być pusty.")

    if not np.issubdtype(sample.dtype, np.integer):
        raise ValueError("Indeksy próbki muszą być całkowite.")

    if np.any(sample < 0) or np.any(sample >= len(X_train)):
        raise ValueError("Indeks próbki poza zakresem X_train.")

    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train zawiera NaN lub wartości nieskończone.")

    # Pobieramy obserwacje należące do próbki.
    # Powtórzenia z bootstrapu pozostają zachowane.
    X_sample = X_train[sample]

    # Parametry standaryzacji wyznaczamy wyłącznie
    # na podstawie całego zbioru treningowego.
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)

    # Dla cech stałych przyjmujemy skalę 1,
    # aby uniknąć dzielenia przez zero.
    stds[stds == 0] = 1.0

    # Standaryzacja obu zbiorów tymi samymi parametrami.
    X_train_scaled = (X_train - means) / stds
    X_sample_scaled = (X_sample - means) / stds

    # Odległość Wassersteina osobno dla każdej cechy.
    feature_distances = []

    for feature_idx in range(X_train.shape[1]):

        distance = wasserstein_distance(
            X_sample_scaled[:, feature_idx],
            X_train_scaled[:, feature_idx]
        )

        feature_distances.append(distance)

    # Średnia odległość dla wszystkich cech.
    return float(np.mean(feature_distances))
