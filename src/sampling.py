import numpy as np


def bootstrap_sample(n, sample_size, random_state=None):
    """
    Draws a bootstrap sample with replacement.

    Parameters
    ----------
    n : int
    Number of observations in the original training set.

    sample_size : int
    Number of observations in the generated sample.

    random_state : int or None
    Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
    Indices of observations selected for the sample.
    """

    if n <= 0:
        raise ValueError("n must be positive")

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rng = np.random.default_rng(random_state)

    indices = rng.choice(
        n,
        size=sample_size,
        replace=True
    )

    return indices


def generate_bootstrap_samples(n, ratio, n_estimators=100, random_state=None):
    """
    Generates multiple bootstrap samples.

    Each sample contains approximately ratio * n observations.
    Sampling is performed with replacement.
    """

    sample_size = round(ratio * n)

    rng = np.random.default_rng(random_state)

    samples = []

    for _ in range(n_estimators):
        seed = rng.integers(0, 2**32)

        sample = bootstrap_sample(
            n=n,
            sample_size=sample_size,
            random_state=int(seed)
        )

        samples.append(sample)

    return samples

def generate_subsamples(n, ratio, n_estimators=100,
                        random_state=None):
    """
    Generates multiple subsamples without replacement.

    Each sample contains approximately ratio * n observations.
    An observation cannot occur more than once in one sample.
    """

    sample_size = round(ratio * n)

    if n <= 0:
        raise ValueError("n must be positive")

    if sample_size <= 0 or sample_size > n:
        raise ValueError("Invalid sample size")

    rng = np.random.default_rng(random_state)

    samples = []

    for _ in range(n_estimators):
        sample = rng.choice(
        n,
        size=sample_size,
        replace=False
        )

        samples.append(sample)

    return samples


def generate_common_core_samples(
    n,
    ratio=0.5,
    alpha=0.5,
    n_estimators=100,
    random_state=None
):
    """
    Generates samples with a common core (S3).

    Each sample has the same size.
    'alpha' determines the proportion of shared observations.
    Sampling is performed without replacement.
    """

    if n <= 0:
        raise ValueError("n musi być dodatnie.")

    if not 0 < ratio <= 1:
        raise ValueError("ratio musi należeć do (0, 1].")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha musi należeć do [0, 1].")

    if n_estimators <= 0:
        raise ValueError("n_estimators musi być dodatnie.")

    sample_size = round(ratio * n)
    common_size = round(alpha * sample_size)
    private_size = sample_size - common_size

    if sample_size > n:
        raise ValueError("Rozmiar próbki nie może przekraczać n.")

    if common_size > n:
        raise ValueError("Wspólny rdzeń jest zbyt duży.")

    rng = np.random.default_rng(random_state)

    # Losujemy wspólny rdzeń dla wszystkich drzew.
    common_core = rng.choice(
        n,
        size=common_size,
        replace=False
    )

    # Obserwacje spoza wspólnego rdzenia.
    remaining = np.setdiff1d(
        np.arange(n),
        common_core
    )

    samples = []

    for _ in range(n_estimators):

        # Każde drzewo otrzymuje własną część próbki.
        private_part = rng.choice(
            remaining,
            size=private_size,
            replace=False
        )

        sample = np.concatenate([
            common_core,
            private_part
        ])

        # Mieszamy kolejność obserwacji.
        rng.shuffle(sample)

        samples.append(sample)

    return samples



def generate_stratified_bootstrap_samples(
    y,
    task="classification",
    n_estimators=100,
    random_state=None,
    n_bins=10
):
    """
    Generates stratified bootstrap samples (S4).

    Classification:
        Sampling is performed separately within each class.

    Regression:
        Observations are divided into groups based on their y-values,
        and sampling is performed separately within each group.
    """

    y = np.asarray(y)

    if len(y) == 0:
        raise ValueError("y nie może być puste.")

    if n_estimators <= 0:
        raise ValueError("n_estimators musi być dodatnie.")

    if task not in ("classification", "regression"):
        raise ValueError(
            "task musi mieć wartość 'classification' lub 'regression'."
        )

    rng = np.random.default_rng(random_state)

    # Tworzymy grupy obserwacji.
    if task == "classification":

        groups = [
            np.where(y == label)[0]
            for label in np.unique(y)
        ]

    else:

        if n_bins <= 0:
            raise ValueError("n_bins musi być dodatnie.")

        # Sortujemy indeksy według wartości zmiennej y.
        sorted_indices = np.argsort(y)

        # Dzielimy obserwacje na maksymalnie n_bins grup.
        groups = [
            group
            for group in np.array_split(
                sorted_indices,
                min(n_bins, len(y))
            )
            if len(group) > 0
        ]

    samples = []

    for _ in range(n_estimators):

        sample_parts = []

        # Bootstrap osobno w każdej grupie.
        for group in groups:

            sampled_group = rng.choice(
                group,
                size=len(group),
                replace=True
            )

            sample_parts.append(sampled_group)

        sample = np.concatenate(sample_parts)

        rng.shuffle(sample)

        samples.append(sample)

    return samples
