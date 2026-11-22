"""
Checks that the vectorized sample measures agree with the
straightforward pairwise implementations in src/diversity.py.

Run: .venv/bin/python -m pytest tests  (or python tests/test_diversity_vectorized.py)
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import diversity as dv
from src.sampling import (
    generate_bootstrap_samples,
    generate_common_core_samples,
    generate_stratified_bootstrap_samples,
)


def _reference(samples, X, y, task):
    pairs = list(combinations(samples, 2))
    return {
        "jaccard": np.array([dv.jaccard_distance(a, b) for a, b in pairs]),
        "multiplicity": np.array(
            [dv.multiplicity_distance(a, b, len(y)) for a, b in pairs]
        ),
        "target": np.array(
            [dv.target_distribution_distance(a, b, y, task) for a, b in pairs]
        ),
        "unique_fraction": np.array(
            [dv.unique_fraction(s, len(y)) for s in samples]
        ),
        "target_representativeness": np.array(
            [dv.target_representativeness(s, y, task) for s in samples]
        ),
        "feature_representativeness": np.array(
            [dv.feature_representativeness(s, X) for s in samples]
        ),
    }


def _check(samples, X, y, task):
    pairs, trees = dv.sample_measures(samples, X, y, task)
    ref = _reference(samples, X, y, task)
    for key, value in {**pairs, **trees}.items():
        np.testing.assert_allclose(value, ref[key], rtol=1e-9, atol=1e-12,
                                   err_msg=key)


def test_classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:, 2] = np.round(X[:, 2])  # ties
    y = rng.integers(0, 3, size=300)
    _check(generate_bootstrap_samples(300, 1.0, 8, 1), X, y, "classification")
    _check(generate_stratified_bootstrap_samples(y, "classification", 8, 2),
           X, y, "classification")


def test_regression():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(250, 3))
    y = np.round(rng.normal(size=250), 1)  # ties in y
    _check(generate_bootstrap_samples(250, 0.6, 8, 3), X, y, "regression")
    _check(generate_common_core_samples(250, 0.5, 0.5, 8, 4), X, y,
           "regression")


if __name__ == "__main__":
    test_classification()
    test_regression()
    print("OK")
