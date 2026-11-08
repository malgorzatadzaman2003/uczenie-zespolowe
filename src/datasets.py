import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing
)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# Nazwy zbiorów i rodzaje zadań.
# Zbiory oznaczone jako "openml" będą pobierane z OpenML.

DATASETS = {
    # ==========================================
    # KLASYFIKACJA
    # ==========================================

    "breast_cancer": {
        "task": "classification",
        "source": "sklearn"
    },
    "ionosphere": {
        "task": "classification",
        "source": "openml",
        "openml_name": "ionosphere"
    },
    "sonar": {
        "task": "classification",
        "source": "openml",
        "openml_name": "sonar"
    },
    "spambase": {
        "task": "classification",
        "source": "openml",
        "openml_name": "spambase"
    },
    "adult": {
        "task": "classification",
        "source": "openml",
        "openml_name": "adult"
    },
    "bank_marketing": {
        "task": "classification",
        "source": "openml",
        "openml_name": "bank-marketing",
        "drop_columns": ["duration"]
    },
    "german_credit": {
        "task": "classification",
        "source": "openml",
        "openml_name": "credit-g"
    },
    "glass": {
        "task": "classification",
        "source": "openml",
        "openml_name": "glass"
    },
    "online_shoppers": {
        "task": "classification",
        "source": "openml",
        "openml_name": "Online Shoppers Purchasing Intention"
    },
    "waveform": {
        "task": "classification",
        "source": "openml",
        "openml_name": "waveform-5000"
    },
    "vehicle": {
        "task": "classification",
        "source": "openml",
        "openml_name": "vehicle"
    },
    "phoneme": {
        "task": "classification",
        "source": "openml",
        "openml_name": "phoneme"
    },

    # ==========================================
    # REGRESJA
    # ==========================================

    "california_housing": {
        "task": "regression",
        "source": "sklearn"
    },
    "diabetes": {
        "task": "regression",
        "source": "sklearn"
    },
    "abalone": {
        "task": "regression",
        "source": "openml",
        "openml_name": "abalone"
    },
    "auto_mpg": {
        "task": "regression",
        "source": "openml",
        "openml_name": "autoMpg",
        "drop_columns": ["name", "car_name"]
    },
    "concrete": {
        "task": "regression",
        "source": "openml",
        "openml_name": "concrete_compressive_strength"
    },
    "energy_efficiency": {
        "task": "regression",
        "source": "openml",
        "openml_name": "energy_efficiency"
    },
    "yacht": {
        "task": "regression",
        "source": "openml",
        "openml_name": "yacht_hydrodynamics"
    },
    "wine_quality_red": {
        "task": "regression",
        "source": "openml",
        "openml_name": "wine-quality-red"
    },
    "airfoil": {
        "task": "regression",
        "source": "openml",
        "openml_name": "airfoil_self_noise"
    },
    "superconductivity": {
        "task": "regression",
        "source": "openml",
        "openml_name": "superconductivity"
    },
    "kin8nm": {
        "task": "regression",
        "source": "openml",
        "openml_name": "kin8nm"
    },
    "power_plant": {
        "task": "regression",
        "source": "openml",
        "openml_name": "combined-cycle-power-plant"
    }
}


def list_datasets(task=None):
    """
    Returns the names of available datasets.

    task:
        None - all datasets,
        "classification" - classification only,
        "regression" - regression only.
    """

    if task is not None and task not in (
        "classification",
        "regression"
    ):
        raise ValueError(
            "task musi mieć wartość 'classification', "
            "'regression' albo None."
        )

    if task is None:
        return list(DATASETS.keys())

    return [
        name
        for name, info in DATASETS.items()
        if info["task"] == task
    ]


def _load_sklearn_dataset(name):
    """
    Loads a dataset available directly within scikit-learn.
    """

    if name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)

    elif name == "diabetes":
        dataset = load_diabetes(as_frame=True)

    elif name == "california_housing":
        dataset = fetch_california_housing(as_frame=True)

    else:
        raise ValueError(
            f"Nieobsługiwany zbiór sklearn: {name}"
        )

    X = dataset.data.copy()
    y = dataset.target.copy()

    return X, y


def _load_openml_dataset(openml_name):
    """
    Retrieves a dataset from OpenML.

    The data is returned as a pandas DataFrame and Series.
    """

    dataset = fetch_openml(
        name=openml_name,
        version="active",
        as_frame=True
    )

    X = dataset.data.copy()
    y = dataset.target.copy()

    return X, y


def _limit_dataset_size(
    X,
    y,
    task,
    max_samples=20000,
    random_state=42
):
    """
    Limits the dataset to a maximum number of observations.

    Classification:
        sampling while preserving class proportions.

    Regression:
        random sampling of observations without replacement.
    """

    if max_samples is None or len(X) <= max_samples:
        return X, y

    if task == "classification":
        X_reduced, _, y_reduced, _ = train_test_split(
            X,
            y,
            train_size=max_samples,
            random_state=random_state,
            stratify=y
        )

    else:
        rng = np.random.default_rng(random_state)

        selected_indices = rng.choice(
            len(X),
            size=max_samples,
            replace=False
        )

        X_reduced = X.iloc[selected_indices]
        y_reduced = y.iloc[selected_indices]

    return X_reduced, y_reduced


def load_dataset(
    name,
    max_samples=20000,
    random_state=42
):
    """
    Loads and performs basic preprocessing on a selected dataset.

    Parameters:
    name : str
        The dataset key as defined in DATASETS.
    max_samples : int or None
        The maximum number of observations.
        None indicates no limit.
    random_state : int
        The random seed used when limiting the dataset.

    Returns:
    dict:
        name - the name of the dataset,
        task - the type of task,
        X - the feature matrix as a DataFrame,
        y - the target variable as a Series.
    """

    if name not in DATASETS:
        raise ValueError(
            f"Nieznany zbiór danych: {name}. "
            f"Dostępne zbiory: {list_datasets()}"
        )

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples musi być dodatnie albo None.")

    info = DATASETS[name]
    task = info["task"]

    # Wczytanie danych z odpowiedniego źródła.
    if info["source"] == "sklearn":
        X, y = _load_sklearn_dataset(name)

    elif info["source"] == "openml":
        try:
            X, y = _load_openml_dataset(info["openml_name"])
        except Exception as error:
            raise RuntimeError(
                f"Nie udało się pobrać zbioru '{name}' "
                f"z OpenML. Szczegóły: {error}"
            ) from error

    else:
        raise ValueError(
            f"Nieznane źródło danych: {info['source']}"
        )

    # Upewniamy się, że dane mają spójny format.
    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()

    # Usuwamy kolumny, które zgodnie z planem nie powinny
    # być wykorzystywane jako cechy.
    columns_to_drop = info.get("drop_columns", [])

    columns_to_drop = [
        column
        for column in X.columns
        if str(column).lower() in [
            name.lower() for name in columns_to_drop
        ]
    ]

    if columns_to_drop:
        X = X.drop(columns=columns_to_drop)

    # Usuwamy obserwacje bez wartości docelowej.
    valid_target = y.notna()

    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    # W regresji target musi być numeryczny.
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")

        valid_target = y.notna()

        X = X.loc[valid_target].reset_index(drop=True)
        y = y.loc[valid_target].reset_index(drop=True)

    # Ograniczamy duże zbiory do 20 000 obserwacji.
    X, y = _limit_dataset_size(
        X=X,
        y=y,
        task=task,
        max_samples=max_samples,
        random_state=random_state
    )

    # Reset indeksów ułatwia późniejsze operowanie na danych.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if len(X) == 0:
        raise ValueError(
            f"Zbiór '{name}' nie zawiera żadnych poprawnych obserwacji."
        )

    return {
        "name": name,
        "task": task,
        "X": X,
        "y": y
    }