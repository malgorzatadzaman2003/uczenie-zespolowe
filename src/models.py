import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ============================================================
# TYPY ZESPOŁÓW
# ============================================================
#
# "random_forest" – domyślne hiperparametry drzew z RandomForestClassifier
#     i RandomForestRegressor (scikit-learn): max_features="sqrt" w
#     klasyfikacji i 1.0 (wszystkie cechy) w regresji. To główna
#     konfiguracja po uwagach do planu (Tabela 4).
# "bagging" – czysty bagging drzew (max_features=None), czyli konfiguracja
#     z pierwotnego planu. Służy jako porównanie: czy losowanie cech
#     "przykrywa" wpływ różnorodności próbek.
#
# W regresji obie konfiguracje są identyczne, bo domyślny las losowy
# w scikit-learn używa wszystkich cech.

ENSEMBLE_TYPES = {
    "random_forest": {
        "classification": {"max_features": "sqrt"},
        "regression": {"max_features": 1.0},
    },
    "bagging": {
        "classification": {"max_features": None},
        "regression": {"max_features": None},
    },
}


def get_tree_params(task, ensemble_type="random_forest"):
    """
    Returns decision-tree hyperparameters for a given ensemble type.
    """

    if ensemble_type not in ENSEMBLE_TYPES:
        raise ValueError(
            f"Nieznany typ zespołu: {ensemble_type}. "
            f"Dostępne: {list(ENSEMBLE_TYPES)}"
        )

    return dict(ENSEMBLE_TYPES[ensemble_type][task])


def create_decision_tree(task, random_state=None, max_features=None):
    """
    Creates a single decision tree for classification or regression.

    Parameters:
    task : str
        Type of task: "classification" or "regression".
    random_state : int or None
        Seed for the random number generator.

    Returns:
    model : DecisionTreeClassifier or DecisionTreeRegressor
        An untrained decision tree.
    """

    if task == "classification":
        model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_leaf=1,
            max_features=max_features,
            random_state=random_state
        )

    elif task == "regression":
        model = DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=1,
            max_features=max_features,
            random_state=random_state
        )

    else:
        raise ValueError(
            "task musi mieć wartość 'classification' albo 'regression'."
        )

    return model


def train_ensemble(
    X_train,
    y_train,
    samples,
    task="classification",
    random_state=None,
    max_features=None,
):
    """
    Trains an ensemble of decision trees based on prepared samples.

    Each tree is trained on a separate sample of indices.
    Repetitions of indices within the bootstrap samples are preserved.

    Parameters:
    X_train : array-like
        Matrix of training features.
    y_train : array-like
        Vector of target values.
    samples : list of arrays
        List of index samples returned by sampling.py.
    task : str
        "classification" or "regression".
    random_state : int or None
        Seed for the random number generator.

    Returns:
    models : list
        List of trained decision trees.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if X_train.ndim != 2:
        raise ValueError("X_train musi być macierzą dwuwymiarową.")

    if y_train.ndim != 1:
        raise ValueError("y_train musi być wektorem jednowymiarowym.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train i y_train muszą mieć tyle samo obserwacji.")

    if len(samples) == 0:
        raise ValueError("Lista próbek nie może być pusta.")

    if task not in ("classification", "regression"):
        raise ValueError(
            "task musi mieć wartość 'classification' albo 'regression'."
        )

    rng = np.random.default_rng(random_state)

    models = []

    for sample_indices in samples:
        sample_indices = np.asarray(sample_indices)

        if sample_indices.ndim != 1 or len(sample_indices) == 0:
            raise ValueError("Każda próbka musi być niepustym wektorem indeksów.")

        if not np.issubdtype(sample_indices.dtype, np.integer):
            raise ValueError("Indeksy próbek muszą być liczbami całkowitymi.")

        if np.any(sample_indices < 0) or np.any(sample_indices >= len(X_train)):
            raise ValueError("Próbka zawiera indeks spoza zbioru treningowego.")

        # Pobieramy obserwacje zgodnie z indeksami próbki.
        # Jeżeli indeks występuje kilka razy, obserwacja również
        # zostanie powtórzona w danych treningowych drzewa.
        X_sample = X_train[sample_indices]
        y_sample = y_train[sample_indices]

        # Każde drzewo otrzymuje własne ziarno losowe.
        tree_seed = int(rng.integers(0, 2**32))

        model = create_decision_tree(
            task=task,
            random_state=tree_seed,
            max_features=max_features,
        )

        model.fit(X_sample, y_sample)

        models.append(model)

    return models


def predict_ensemble(models, X):
    """
    Calculates the predictions of each tree in the ensemble.

    Returns a matrix with dimensions:
    (number of trees, number of observations).
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X musi być macierzą dwuwymiarową.")

    if len(models) == 0:
        raise ValueError("Zespół modeli nie może być pusty.")

    predictions = []

    for model in models:
        predictions.append(model.predict(X))

    return np.asarray(predictions)


def aggregate_predictions(predictions, task="classification"):
    """
    Combines tree predictions into a single ensemble prediction.

    Classification:
        majority (hard) voting. The main experiment uses soft voting
        (averaged probabilities, see aggregate_probabilities), as in
        scikit-learn's RandomForestClassifier; hard voting is kept for
        comparison.

    Regression:
        arithmetic mean of the tree predictions.
    """

    predictions = np.asarray(predictions)

    if predictions.ndim != 2:
        raise ValueError(
            "Predykcje muszą być macierzą: liczba drzew x liczba obserwacji."
        )

    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("Macierz predykcji nie może być pusta.")

    if task == "regression":
        return np.mean(predictions, axis=0)

    elif task == "classification":
        aggregated = []

        # Dla każdej obserwacji zbieramy głosy wszystkich drzew.
        for column in predictions.T:
            labels, counts = np.unique(column, return_counts=True)

            # Przy remisie wybierana jest pierwsza etykieta
            # w uporządkowanym wyniku np.unique.
            majority_label = labels[np.argmax(counts)]

            aggregated.append(majority_label)

        return np.asarray(aggregated)

    else:
        raise ValueError(
            "task musi mieć wartość 'classification' albo 'regression'."
        )

def predict_ensemble_proba(models, X, classes=None):
    """
    Calculates the averaged class probabilities for an ensemble of trees.

    Each tree may have a slightly different set of classes—for instance,
    if a rare class is missing from its bootstrap sample.
    Therefore, we align the probability columns to a common
    set of classes.

    Parameters:
    models : list
        A list of trained classification trees.
    X : array-like
        The data for which probabilities are to be calculated.
    classes : array-like or None
        The complete set of classes from the training dataset.

    Returns:
    probabilities : ndarray
        A matrix of shape: number of observations × number of classes.
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X musi być macierzą dwuwymiarową.")

    if len(models) == 0:
        raise ValueError("Lista modeli nie może być pusta.")

    if classes is None:
        classes = np.unique(
            np.concatenate([model.classes_ for model in models])
        )
    else:
        classes = np.asarray(classes)

    if len(classes) < 2:
        raise ValueError("Klasyfikacja wymaga co najmniej dwóch klas.")

    class_to_column = {
        label: index for index, label in enumerate(classes)
    }

    ensemble_probabilities = np.zeros(
        (len(X), len(classes)),
        dtype=float
    )

    for model in models:
        model_probabilities = model.predict_proba(X)

        for model_column, label in enumerate(model.classes_):
            if label not in class_to_column:
                raise ValueError(
                    "Model zawiera klasę spoza przekazanego zestawu classes."
                )

            ensemble_column = class_to_column[label]

            ensemble_probabilities[:, ensemble_column] += (
                model_probabilities[:, model_column]
            )

    ensemble_probabilities /= len(models)

    return ensemble_probabilities


def aggregate_probabilities(probabilities, classes):
    """
    Soft voting: averages class probabilities of individual trees and
    returns the class with the highest mean probability.

    probabilities : ndarray (n_trees, n_observations, n_classes)
        or (n_observations, n_classes) if already averaged.
    """

    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)

    if probabilities.ndim == 3:
        probabilities = probabilities.mean(axis=0)

    return classes[np.argmax(probabilities, axis=1)]


def predict_tree_probabilities(models, X, classes):
    """
    Returns per-tree class probabilities aligned to a common set of
    classes: array of shape (n_trees, n_observations, n_classes).

    A tree trained on a sample without some class gives it 0.
    """

    X = np.asarray(X)
    classes = np.asarray(classes)

    class_to_column = {label: i for i, label in enumerate(classes)}

    output = np.zeros((len(models), len(X), len(classes)), dtype=float)

    for t, model in enumerate(models):
        proba = model.predict_proba(X)
        columns = [class_to_column[label] for label in model.classes_]
        output[t][:, columns] = proba

    return output
