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