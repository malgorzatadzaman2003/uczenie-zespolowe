import numpy as np
from itertools import combinations

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
