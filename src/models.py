import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def create_decision_tree(task, random_state=None):
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
            max_features=None,
            random_state=random_state
        )

    elif task == "regression":
        model = DecisionTreeRegressor(
            max_depth=None,
            min_samples_leaf=1,
            max_features=None,
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
    random_state=None
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
            random_state=tree_seed
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
        majority voting.

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