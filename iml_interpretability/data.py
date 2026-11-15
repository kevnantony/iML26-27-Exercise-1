"""Loading datasets.

Every dataset is turned into a :class:`Dataset`: a purely numeric feature table with no missing
values, plus integer class labels. That way every tree model can be trained on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer as sklearn_breast_cancer
from sklearn.model_selection import train_test_split

# Directory where OpenML downloads are locally cached on disk.
# This prevents downloading datasets repeatedly across experiment runs.
DATA_HOME = Path("data")

# Task 1: Six representative datasets spanning different difficulty levels.
# Maps dataset name -> OpenML dataset ID (or None for built-in datasets).
TASK1_DATASETS = {
    "banknote": 1462,       # 4 continuous features, easily separable
    "breast_cancer": None,  # Built directly into scikit-learn (no network download needed)
    "titanic": 40945,       # Classic passenger survival dataset (mixed numeric and categorical)
    "diabetes": 37,         # 8 numeric medical diagnostic features
    "credit_g": 31,         # German credit dataset with many categorical attributes
    "phoneme": 1489,        # 5 audio acoustic features with complex non-linear boundaries
}

# Task 2: 32 OpenML classification datasets of varied sizes, dimensions, and difficulty.
# Maps dataset name -> OpenML dataset ID.
TASK2_DATASETS = {
    "credit_g": 31, "diabetes": 37, "banknote": 1462, "phoneme": 1489, "blood_transfusion": 1464,
    "ilpd": 1480, "qsar_biodeg": 1494, "wdbc": 1510, "kc2": 1063, "kc1": 1067, "pc1": 1068,
    "pc4": 1049, "climate_crashes": 40994, "steel_plates": 40982, "vehicle": 54, "balance_scale": 11,
    "breast_w": 15, "cmc": 23, "credit_approval": 29, "eucalyptus": 188, "vowel": 307, "spambase": 44,
    "tic_tac_toe": 50, "kr_vs_kp": 3, "wilt": 40983, "churn": 40701, "phishing": 4534, "adult": 1590,
    "bank_marketing": 1461, "car": 40975, "ozone": 1487, "titanic": 40945,
}  # fmt: skip

# Columns to discard from Titanic:
# 'boat' and 'body' directly reveal survival status (target leakage).
# 'name', 'ticket', 'cabin', and 'home.dest' are unhelpful high-cardinality text strings.
TITANIC_DROP = ["boat", "body", "home.dest", "name", "ticket", "cabin"]


@dataclass(frozen=True)
class Dataset:
    """A ready-to-use classification dataset.

    :param name: Short name used in result tables.
    :param X: Numeric features, no missing values.
    :param y: Integer class labels 0, 1, 2, ...
    :param n_categorical: How many columns of ``X`` are one-hot columns made from categorical features.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    n_categorical: int = 0


def load_dataset(name: str, data_id: int | None, max_samples: int | None = None, seed: int = 0) -> Dataset:
    """Download (or read from cache) one dataset and make it numeric.

    :param name: Short name for the dataset.
    :param data_id: OpenML id; ``None`` only for breast cancer, which ships with scikit-learn.
    :param max_samples: If set, keep at most this many rows (random, class proportions kept).
    :param seed: Random seed for that subsampling.
    :return: The preprocessed dataset.
    :rtype: Dataset
    """
    # Special case: Breast cancer is bundled inside scikit-learn.
    # It has no missing values and is already fully numeric.
    if data_id is None:
        bunch = sklearn_breast_cancer(as_frame=True)
        return Dataset(name=name, X=bunch.data, y=bunch.target)

    # Fetch dataset from OpenML (uses local cache in `DATA_HOME`).
    # `parser="liac-arff"` is required because the default parser can fail on certain OpenML files.
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff", data_home=DATA_HOME)

    # Create a local copy of the feature dataframe.
    X = bunch.data.copy()

    # If loading Titanic, drop the target leakage and arbitrary text columns.
    if name == "titanic":
        X = X.drop(columns=TITANIC_DROP)

    # 1. Handle numeric columns:
    # Identify all columns that already contain numerical values.
    numeric_cols = X.select_dtypes(include="number").columns
    # Fill any missing values in numeric columns with the column's median.
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # 2. Handle categorical columns:
    # Convert categorical variables into binary dummy (one-hot) columns.
    # `drop_first=True` avoids collinearity by dropping one redundant category level.
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # 3. Handle remaining NaNs:
    # In case any column was entirely null or missed, fill remaining NaNs with 0.0.
    X = X.fillna(0.0)

    # Calculate how many columns in the final table originate from categorical features.
    n_categorical = X.shape[1] - len(numeric_cols)

    # 4. Standardize target labels:
    # Cast targets to string to handle mixed or categorical types cleanly.
    labels = bunch.target.astype(str)
    # Map distinct class labels to contiguous integers (0, 1, ..., C-1) in alphabetical order.
    label_to_int = {label: i for i, label in enumerate(sorted(labels.unique()))}
    y = labels.map(label_to_int).astype(int)

    # 5. Optional subsampling for large datasets:
    # If a maximum sample size is specified and the dataset exceeds it, subsample with stratification.
    if max_samples is not None and len(y) > max_samples:
        keep_indices, _ = train_test_split(
            range(len(y)), train_size=max_samples, stratify=y, random_state=seed
        )
        # Re-index X and y to the retained subset.
        X = X.iloc[keep_indices].reset_index(drop=True)
        y = y.iloc[keep_indices].reset_index(drop=True)

    # Return the clean, ready-to-train Dataset object.
    return Dataset(name=name, X=X, y=y, n_categorical=n_categorical)


def load_task1_datasets() -> list[Dataset]:
    """The six datasets of Task 1, without subsampling."""
    # Load all six Task 1 benchmark datasets without sample restrictions.
    return [load_dataset(name, data_id) for name, data_id in TASK1_DATASETS.items()]
