"""Training decision-tree ensembles on the project's sampling variants."""

from typing import Dict, List, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.sampling import (
    generate_bootstrap_samples,
    generate_common_core_samples,
    generate_stratified_bootstrap_samples,
    generate_subsamples,
)


DecisionTree = Union[DecisionTreeClassifier, DecisionTreeRegressor]


def _validate_training_data(X_train, y_train, task):
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    if task not in ("classification", "regression"):
        raise ValueError(
            "task must be 'classification' or 'regression'."
        )
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X_train must be a non-empty 2D matrix.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            "y_train must be 1D and have the same number of rows as X_train."
        )
    if task == "regression" and not np.issubdtype(y.dtype, np.number):
        raise ValueError("Regression targets must be numeric.")

    return X, y


def train_decision_trees(
    X_train,
    y_train,
    samples,
    task="classification",
    random_state=None,
) -> List[DecisionTree]:
    """Fit one decision tree per set of training observation indices."""
    X, y = _validate_training_data(X_train, y_train, task)

    if not samples:
        raise ValueError("At least one sample is required.")

    rng = np.random.default_rng(random_state)
    models: List[DecisionTree] = []

    for sample in samples:
        indices = np.asarray(sample)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("Each sample must be a non-empty 1D index array.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Sample indices must be integers.")
        if np.any(indices < 0) or np.any(indices >= len(y)):
            raise ValueError("Sample index is outside the training data.")

        seed = int(rng.integers(0, 2**32))
        if task == "classification":
            model = DecisionTreeClassifier(random_state=seed)
        else:
            model = DecisionTreeRegressor(random_state=seed)

        model.fit(X[indices], y[indices])
        models.append(model)

    return models


def train_models_for_sampling_variants(
    X_train,
    y_train,
    task="classification",
    n_estimators=100,
    random_state=None,
    n_bins=10,
) -> Dict[str, List[DecisionTree]]:
    """
    Train a decision-tree ensemble for each sampling variant (S1-S4).

    S1 uses bootstrap samples, S2 uses 63% subsamples, S3 uses 50% samples
    with a 50% common core, and S4 uses stratified bootstrap samples.
    """
    X, y = _validate_training_data(X_train, y_train, task)

    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32, size=8, dtype=np.uint32)
    n_observations = len(y)

    samples_by_variant = {
        "S1_bootstrap": generate_bootstrap_samples(
            n=n_observations,
            ratio=1.0,
            n_estimators=n_estimators,
            random_state=int(seeds[0]),
        ),
        "S2_subsampling": generate_subsamples(
            n=n_observations,
            ratio=0.63,
            n_estimators=n_estimators,
            random_state=int(seeds[1]),
        ),
        "S3_common_core": generate_common_core_samples(
            n=n_observations,
            ratio=0.5,
            alpha=0.5,
            n_estimators=n_estimators,
            random_state=int(seeds[2]),
        ),
        "S4_stratified_bootstrap": generate_stratified_bootstrap_samples(
            y=y,
            task=task,
            n_estimators=n_estimators,
            random_state=int(seeds[3]),
            n_bins=n_bins,
        ),
    }

    return {
        variant: train_decision_trees(
            X_train=X,
            y_train=y,
            samples=samples,
            task=task,
            random_state=int(seeds[4 + index]),
        )
        for index, (variant, samples) in enumerate(samples_by_variant.items())
    }