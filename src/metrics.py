import numpy as np

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.models import (
    predict_ensemble,
    predict_ensemble_proba,
    aggregate_predictions
)


def classification_metrics(y_true, y_pred, y_proba=None, classes=None):
    """
    Calculates classification quality metrics.

    Parameters:
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : ndarray or None
        Class probabilities.
    classes : array-like or None
        The full set of classes, corresponding to the column order in y_proba.

    Returns:
    dict
        F1-score, balanced accuracy, and ROC-AUC.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true i y_pred muszą być wektorami.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true i y_pred muszą mieć tę samą długość.")

    if len(y_true) == 0:
        raise ValueError("Wektory nie mogą być puste.")

    if classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.asarray(classes)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    roc_auc = np.nan

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        if y_proba.ndim != 2:
            raise ValueError("y_proba musi być macierzą dwuwymiarową.")

        if y_proba.shape != (len(y_true), len(classes)):
            raise ValueError(
                "Wymiary y_proba nie pasują do liczby obserwacji i klas."
            )

        try:
            if len(classes) == 2:
                roc_auc = roc_auc_score(
                    y_true,
                    y_proba[:, 1]
                )
            else:
                roc_auc = roc_auc_score(
                    y_true,
                    y_proba,
                    labels=classes,
                    multi_class="ovr",
                    average="macro"
                )

        except ValueError:
            # ROC-AUC nie jest określone np. wtedy, gdy w y_true
            # występuje tylko jedna klasa.
            roc_auc = np.nan

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc
    }


def regression_metrics(y_true, y_pred):
    """
    Calculates regression quality metrics.

    Returns:
    dict
        RMSE, MAE, and R2.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true i y_pred muszą być wektorami.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true i y_pred muszą mieć tę samą długość.")

    if len(y_true) == 0:
        raise ValueError("Wektory nie mogą być puste.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def evaluate_individual_models(
    models,
    X_test,
    y_test,
    task="classification",
    classes=None
):
    """
    Calculates quality metrics for each tree individually.

    Returns the mean and standard deviation of the metrics
    for all trees in the ensemble.
    """

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if len(models) == 0:
        raise ValueError("Lista modeli nie może być pusta.")

    if task == "classification" and classes is None:
        classes = np.unique(
            np.concatenate([model.classes_ for model in models])
        )

    all_metrics = []

    for model in models:
        predictions = model.predict(X_test)

        if task == "classification":
            model_classes = np.asarray(classes)

            probabilities = np.zeros(
                (len(X_test), len(model_classes)),
                dtype=float
            )

            class_to_column = {
                label: index
                for index, label in enumerate(model_classes)
            }

            model_probabilities = model.predict_proba(X_test)

            for column, label in enumerate(model.classes_):
                probabilities[:, class_to_column[label]] = (
                    model_probabilities[:, column]
                )

            metrics = classification_metrics(
                y_true=y_test,
                y_pred=predictions,
                y_proba=probabilities,
                classes=model_classes
            )

        elif task == "regression":
            metrics = regression_metrics(
                y_true=y_test,
                y_pred=predictions
            )

        else:
            raise ValueError(
                "task musi mieć wartość 'classification' albo 'regression'."
            )

        all_metrics.append(metrics)

    metric_names = all_metrics[0].keys()

    summary = {}

    for name in metric_names:
        values = np.array([
            metrics[name] for metrics in all_metrics
        ])

        summary[name] = {
            "mean": np.nanmean(values),
            "std": np.nanstd(values)
        }

    return summary


def evaluate_ensemble(
    models,
    X_test,
    y_test,
    task="classification",
    classes=None
):
    """
    Calculates quality metrics for the entire ensemble of trees.
    """

    predictions = predict_ensemble(
        models=models,
        X=X_test
    )

    final_predictions = aggregate_predictions(
        predictions=predictions,
        task=task
    )

    if task == "classification":
        if classes is None:
            classes = np.unique(
                np.concatenate([model.classes_ for model in models])
            )

        probabilities = predict_ensemble_proba(
            models=models,
            X=X_test,
            classes=classes
        )

        return classification_metrics(
            y_true=y_test,
            y_pred=final_predictions,
            y_proba=probabilities,
            classes=classes
        )

    elif task == "regression":
        return regression_metrics(
            y_true=y_test,
            y_pred=final_predictions
        )

    else:
        raise ValueError(
            "task musi mieć wartość 'classification' albo 'regression'."
        )