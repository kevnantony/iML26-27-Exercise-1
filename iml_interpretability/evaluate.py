"""Train every model on every dataset and record test accuracy plus our metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from iml_interpretability.data import Dataset


def evaluate(datasets: list[Dataset], models: dict, metrics: list, seed: int = 0) -> pd.DataFrame:
    """Fit every model on every dataset; measure accuracy on held-out data and every metric.

    :param datasets: Datasets to use.
    :param models: Model name -> untrained model (from :func:`get_models`).
    :param metrics: Metric objects with a ``name`` and a ``score(model, dataset)`` method.
    :param seed: Random seed for the 75/25 train/test split.
    :return: One row per (dataset, model): dataset, model, accuracy, one column per metric.
    :rtype: pd.DataFrame
    """
    # List to collect result dictionaries for each (dataset, model) combination.
    rows = []

    # Iterate over each dataset in the provided list.
    for dataset in datasets:
        # Split the data into 75% training and 25% test sets.
        # `stratify=dataset.y` ensures class distributions remain identical in both splits.
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=0.25, random_state=seed, stratify=dataset.y
        )

        # Wrap training data in a Dataset object; metrics like simulation load evaluate on train data.
        train = Dataset(name=dataset.name, X=X_train, y=y_train)

        # Iterate over all models in the dictionary.
        for model_name, model in models.items():
            # Create a fresh, untrained clone of the model architecture so runs are completely independent.
            fresh_model = clone(model)

            # Fit (train) the model on the training data.
            fitted = fresh_model.fit(X_train, y_train)

            # Generate predictions on the unseen test set to evaluate generalisation performance.
            predictions = fitted.predict(X_test)

            # Compute the classification test accuracy (fraction of correct predictions).
            test_accuracy = accuracy_score(y_test, predictions)

            # Start building the record row for this model on this dataset.
            row = {
                "dataset": dataset.name,
                "model": model_name,
                "accuracy": test_accuracy,
            }

            # Evaluate each requested interpretability metric on the fitted model.
            for metric in metrics:
                # Call metric.score(fitted_model, training_data) and store under metric.name.
                row[metric.name] = metric.score(fitted, train)

            # Append the completed row to our collection.
            rows.append(row)

    # Convert the list of rows into a clean Pandas DataFrame and return it.
    return pd.DataFrame(rows)


def split_model_name(results: pd.DataFrame) -> pd.DataFrame:
    """Add columns family / depth / n_trees, read from model names like ``forest_d3_n10``."""
    # Use regular expression extraction on the 'model' column:
    # (?P<family>[a-z]+) captures the model family (tree, forest, or boosting)
    # (?P<depth>\w+) captures the depth string ('1', '2', ..., 'max')
    # (?P<n_trees>\d+) captures the integer tree count
    parts = results["model"].str.extract(r"(?P<family>[a-z]+)_d(?P<depth>\w+)_n(?P<n_trees>\d+)")

    # Cast the extracted 'n_trees' string into integer format.
    parts["n_trees"] = parts["n_trees"].astype(int)

    # Concatenate the new columns horizontally to the original results table.
    return pd.concat([results, parts], axis=1)
