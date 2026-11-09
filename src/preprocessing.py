import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _clean_missing_values(X):
    """
    Standardizes the representation of missing values in the data.

    `None` values, `pd.NA`, and specific text-based
    placeholders for missing data are converted to `np.nan`.
    """
    X = pd.DataFrame(X).copy()

    missing_tokens = [
        "?",
        "",
        "NA",
        "N/A",
        "nan",
        "NaN",
        "null",
        "None"
    ]

    for column in X.columns:
        if (
            pd.api.types.is_object_dtype(X[column])
            or isinstance(X[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[column])
        ):
            X[column] = X[column].replace(missing_tokens, np.nan)

        X[column] = X[column].replace({None: np.nan, pd.NA: np.nan})

    return X


def create_preprocessor(X_train):
    """
    Creates a transformer tailored to the structure of the training data.

    Numerical columns:
        - imputation of missing values using the median.

    Categorical columns:
        - imputation of missing values with the category 'missing',
        - One-Hot Encoding.

    Returns:
        preprocessor - a transformer that has not yet been fitted.
    """

    X_train = _clean_missing_values(X_train)

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        try:
            numeric_imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        except TypeError:
            numeric_imputer = SimpleImputer(
                strategy="median"
            )

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", numeric_imputer)
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:
        # sparse_output=False jest dostępne w nowszych wersjach
        # scikit-learn. Dla starszych używamy sparse=False.
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="missing"
                    )
                ),
                (
                    "onehot",
                    encoder
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        raise ValueError(
            "Nie znaleziono żadnych kolumn do przetworzenia."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def preprocess_train_test(X_train, X_test):
    """
    Fits the preprocessing steps to the training data,
    and then transforms both the training and test data.

    Parameters:
        X_train - training DataFrame or feature matrix.
        X_test - test DataFrame or feature matrix.

    Returns:
        X_train_processed - processed training data.
        X_test_processed - processed test data.
        preprocessor - the fitted transformer.
    """

    X_train = _clean_missing_values(X_train)
    X_test = _clean_missing_values(X_test)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "X_train i X_test muszą być macierzami dwuwymiarowymi."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train i X_test muszą mieć te same kolumny "
            "w tej samej kolejności."
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Zbiory treningowy i testowy nie mogą być puste."
        )

    preprocessor = create_preprocessor(X_train)

    # Dopasowanie wyłącznie na danych treningowych.
    X_train_processed = preprocessor.fit_transform(X_train)

    # Test jest tylko przekształcany.
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=float
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=float
    )

    if X_train_processed.shape[1] != X_test_processed.shape[1]:
        raise ValueError(
            "Dane treningowe i testowe mają różną liczbę cech "
            "po preprocessingu."
        )

    if X_train_processed.shape[1] == 0:
        raise ValueError(
            "Po preprocessingu nie pozostała żadna cecha."
        )
    
    if not np.isfinite(X_train_processed).all():
        raise ValueError(
            "Po preprocessingu w danych treningowych "
            "pozostały wartości NaN lub nieskończone."
        )

    if not np.isfinite(X_test_processed).all():
        raise ValueError(
            "Po preprocessingu w danych testowych "
            "pozostały wartości NaN lub nieskończone."
        )

    return (
        X_train_processed,
        X_test_processed,
        preprocessor
    )


def get_feature_names(preprocessor):
    """
    Returns the names of the features after preprocessing.
    """

    return preprocessor.get_feature_names_out()