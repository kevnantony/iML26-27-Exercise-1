"""Task 2: describe each dataset with meta-features, and summarise how interpretable it can be modelled.

Meta-features (computed from the data alone, before choosing a model):
    size        : number of samples, number of features, number of classes
    statistics  : class balance, share of categorical columns, PCA dimensions needed,
                  correlation of the single best feature with the target
    landmarkers : accuracy of very cheap models (always guess the majority class, one split,
                  logistic regression, 1-nearest-neighbour) - they show how hard the problem is

Outcomes (computed from our 65-model results on that dataset):
    price               : best accuracy - best accuracy among models with load <= 7
    load_needed         : smallest load of any model within 1 accuracy point of the best
    interpretable_enough: 1 if load_needed <= 7, else 0
    full_tree_load      : load of a fully grown single tree
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.data import Dataset
from iml_interpretability.metric import THRESHOLD


def cv_accuracy(model, X, y) -> float:
    """Average accuracy of ``model`` over 5-fold cross-validation."""
    # Split data into 5 folds, train on 4, evaluate on 1, and return the mean score across all 5 folds.
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


def compute_meta_features(dataset: Dataset) -> dict[str, float]:
    """All meta-features of one dataset.

    :param dataset: A preprocessed dataset.
    :return: Meta-feature name -> value.
    :rtype: dict[str, float]
    """
    # Convert features to a continuous 2D float NumPy array.
    X = dataset.X.to_numpy(dtype=float)
    # Convert integer targets to a 1D NumPy array.
    y = dataset.y.to_numpy()

    # Get sample count (number of rows) and feature count (number of columns).
    n_samples, n_features = X.shape

    # 1. Class Distribution & Entropy:
    # Count occurrences of each integer class label, divided by sample count to get proportion.
    class_share = np.bincount(y) / n_samples
    # Drop any zero counts for classes that do not appear in the dataset.
    class_share = class_share[class_share > 0]
    # Count total distinct classes present.
    n_classes = len(class_share)
    # Shannon entropy of the class proportions: measures how unpredictable the classes are.
    entropy = -(class_share * np.log2(class_share)).sum()
    # Normalize entropy by theoretical maximum log2(n_classes) so balance is bounded in [0, 1].
    # Value 1.0 means perfectly balanced classes; close to 0.0 means extreme class imbalance.
    class_balance = entropy / np.log2(n_classes) if n_classes > 1 else 0.0

    # 2. PCA Intrinsic Dimensionality:
    # Standardize each feature to zero mean and unit variance.
    # Replace any potential NaNs (e.g. from constant columns) with 0.
    X_scaled = np.nan_to_num(StandardScaler().fit_transform(X))
    # Fit PCA and compute the cumulative explained variance ratio across principal components.
    explained_variance = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
    # Find how many components are needed to explain at least 95% of total variance.
    pca_dims = int(np.searchsorted(explained_variance, 0.95) + 1)

    # 3. Best Feature Correlation:
    # Create a binary indicator: 1 for samples in the most common class, 0 for all others.
    most_common_class = np.bincount(y).argmax()
    is_majority = (y == most_common_class).astype(float)
    # Compute absolute Pearson correlation between each feature column and the majority indicator.
    with np.errstate(invalid="ignore", divide="ignore"):
        corrs = [abs(np.corrcoef(X_scaled[:, j], is_majority)[0, 1]) for j in range(n_features)]
    # Extract the highest correlation found across any single feature.
    best_corr = float(np.nanmax(corrs)) if np.isfinite(corrs).any() else 0.0

    # 4. Landmarker Models (fast probes indicating problem difficulty):
    # Baseline 1: Accuracy achieved by simply always predicting the majority class.
    acc_majority = float(class_share.max())
    # Baseline 2: Single decision split (decision stump with max_depth=1).
    acc_stump = cv_accuracy(DecisionTreeClassifier(max_depth=1, random_state=0), X, y)
    # Baseline 3: Linear model (standardized logistic regression).
    acc_logreg = cv_accuracy(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y)
    # Baseline 4: Nearest neighbor model (1-NN, measuring local cluster purity).
    acc_1nn = cv_accuracy(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)), X, y)

    # Compile all meta-features into a dictionary:
    return {
        # Log10 of sample count (captures order of magnitude).
        "log_n_samples": float(np.log10(n_samples)),
        # Log10 of feature count.
        "log_n_features": float(np.log10(n_features)),
        # Total number of classes.
        "n_classes": n_classes,
        # Normalized class entropy balance [0, 1].
        "class_balance": float(class_balance),
        # Proportion of features that originated from one-hot categorical encoding.
        "share_categorical": dataset.n_categorical / n_features,
        # Proportion of PCA components needed to retain 95% variance (intrinsic dimensionality).
        "pca_dims_95": pca_dims / n_features,
        # Strongest correlation between any single feature and the target class.
        "best_feature_corr": best_corr,
        # Accuracy of trivial majority-class guessing.
        "acc_majority": acc_majority,
        # Accuracy gain of making 1 decision split over naive majority guessing.
        "stump_gain": acc_stump - acc_majority,
        # Accuracy gain of combining features linearly over a single split.
        "linear_gain": acc_logreg - acc_stump,
        # Accuracy gain of local non-linear 1-NN over a single split.
        "knn_gain": acc_1nn - acc_stump,
    }


def interpretability_outcomes(results: pd.DataFrame) -> dict[str, float]:
    """Summarise the 65 models trained on ONE dataset.

    :param results: One row per model with columns family, depth, accuracy, simulation_load.
    :return: Outcome name -> value (see the module docstring).
    :rtype: dict[str, float]
    """
    # 1. Best accuracy achieved by any of the 65 models on this dataset.
    best_acc = results.accuracy.max()

    # 2. Filter down to only interpretable models (simulation load <= 7.0).
    interpretable_models = results[results.simulation_load <= THRESHOLD]

    # 3. Find models whose accuracy is within 1 percentage point (0.01) of the overall best.
    good_enough_models = results[results.accuracy >= best_acc - 0.01]

    # 4. Out of those near-optimal models, find the minimum simulation load needed.
    load_needed = good_enough_models.simulation_load.min()

    # 5. Extract the row for the fully grown single decision tree (depth='max').
    full_tree = results[(results.family == "tree") & (results.depth == "max")]

    # Calculate key outcome measures:
    return {
        # Maximum accuracy reached by any model.
        "best_accuracy": best_acc,
        # Price of interpretability: accuracy sacrificed when restricted to load <= 7.
        "price": best_acc - interpretable_models.accuracy.max(),
        # Minimum cognitive load needed to achieve near-optimal (within 1%) accuracy.
        "load_needed": load_needed,
        # Log-transformed load needed (bounded at 1.0 to avoid non-positive log).
        "log_load_needed": float(np.log10(max(load_needed, 1.0))),
        # Binary outcome: 1.0 if an interpretable model is within 1% of the best, else 0.0.
        "interpretable_enough": float(load_needed <= THRESHOLD),
        # Simulation load of the unconstrained fully grown decision tree.
        "full_tree_load": float(full_tree.simulation_load.iloc[0]),
    }
