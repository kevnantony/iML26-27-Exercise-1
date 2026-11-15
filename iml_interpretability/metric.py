"""Our interpretability metric: simulation load.

Question: how many mental steps does a person need to work out ONE prediction of the model by hand?

    For one input x:  load(x) = sum over trees of path_cost(tree, x)  +  (number of trees - 1)

    path_cost: walk x from the root to its leaf and count the questions ("is age < 40?"):
        - question about a feature not seen yet on this path: 1 step   (look the value up, compare)
        - question about a feature already seen on this path: 0.5 step (value known, only compare)
    (number of trees - 1): combining the trees' answers (counting votes / adding scores)

    Simulation load of a model = average of load(x) over the data. Lower = more interpretable.
    Threshold: interpretable if load <= 7 (working memory holds about 7 +- 2 items, Miller 1956).

We also define two simple baselines (number of leaves, number of features used) for comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import BaseDecisionTree

from iml_interpretability.data import Dataset

# Threshold: a human can comfortably keep around 7 items in working memory at once.
# Any model requiring more than 7 cognitive steps per prediction is considered non-interpretable.
THRESHOLD = 7.0


def get_trees(model) -> list[BaseDecisionTree]:
    """List the individual trees inside any of our models (1 tree, forest or boosting).

    :param model: A trained decision tree, random forest or gradient boosting model.
    :return: The trees a person would have to walk through.
    :rtype: list[BaseDecisionTree]
    """
    # Case 1: If the model is already a single decision tree, put it in a 1-element list.
    if isinstance(model, BaseDecisionTree):
        return [model]

    # Case 2: Random forest stores its individual decision trees in the `estimators_` list.
    if isinstance(model, RandomForestClassifier):
        return list(model.estimators_)

    # Case 3: Gradient boosting stores trees in a 2-D array of shape (n_estimators, n_classes).
    # We flatten this array using `.ravel()` so we get a simple flat list of all trees.
    if isinstance(model, GradientBoostingClassifier):
        return list(np.asarray(model.estimators_).ravel())

    # If the model is not any recognized tree type, raise an error explaining the issue.
    raise TypeError(f"not a tree model: {type(model).__name__}")


def path_cost(tree: BaseDecisionTree, X: np.ndarray) -> np.ndarray:
    """For every row of X: the cost of walking that row from the root of ``tree`` to its leaf.

    cost = 1 * (number of distinct features asked) + 0.5 * (number of repeated questions)
         = 0.5 * (number of questions + number of distinct features)

    :param tree: A trained tree.
    :param X: Inputs, one row per sample.
    :return: One cost per row.
    :rtype: np.ndarray
    """
    # Access scikit-learn's underlying C-level tree structure (`tree_`).
    t = tree.tree_

    # In scikit-learn, leaf nodes have children_left == -1 (they have no child nodes).
    # Any node with children_left != -1 is a decision node that asks a question about a feature.
    is_question = t.children_left != -1

    # Get the integer node IDs of all decision nodes that ask questions.
    question_nodes = np.flatnonzero(is_question)

    # Edge case: If there are no question nodes at all (the tree is just 1 leaf node),
    # then no questions are ever asked, so every sample has a cost of exactly 0.0.
    if question_nodes.size == 0:
        return np.zeros(len(X))

    # Trace which nodes each sample visits on its path from root to leaf.
    # scikit-learn may emit a harmless UserWarning about missing feature names, which we ignore.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # `visited` is a sparse boolean matrix of shape (n_samples, n_nodes).
        # Entry (i, j) is True (1) if sample i passes through node j.
        visited = tree.decision_path(X)

    # Build a lookup matrix that maps each node to the feature index it asks about.
    # Shape: (total_nodes, total_features).
    # Entry (node, feature) is 1.0 if that node splits on that feature.
    node_asks_feature = sparse.csr_matrix(
        (np.ones(len(question_nodes)), (question_nodes, t.feature[question_nodes])),
        shape=(t.node_count, tree.n_features_in_),
    )

    # Multiply the paths by the feature map:
    # (n_samples, n_nodes) @ (n_nodes, n_features) -> (n_samples, n_features).
    # For each sample and each feature, this gives how many times that feature was asked along the path.
    questions_per_feature = visited @ node_asks_feature

    # Total questions asked along the path = sum of questions across all features for each row.
    n_questions = np.asarray(questions_per_feature.sum(axis=1)).ravel()

    # Total distinct features asked along the path = count of features asked at least once (> 0).
    n_distinct = np.asarray((questions_per_feature > 0).sum(axis=1)).ravel()

    # Cost formula:
    # First time a feature is checked: 1.0 step (look up the value in memory/table + compare).
    # Each repeated check of the same feature: 0.5 step (value is already in mind, only compare).
    # Algebraically: 1.0 * distinct + 0.5 * (questions - distinct) = 0.5 * (questions + distinct).
    return 0.5 * (n_questions + n_distinct)


def load_per_sample(model, X) -> np.ndarray:
    """load(x) for every row x of X: walk all trees, then add one step per extra tree to combine them.

    :param model: A trained tree model.
    :param X: Inputs, one row per sample.
    :return: One load value per row.
    :rtype: np.ndarray
    """
    # Ensure input data is a float32 NumPy array as required by scikit-learn's tree methods.
    X = np.asarray(X, dtype=np.float32)

    # Extract all individual decision trees that make up this model.
    trees = get_trees(model)

    # Sum the path costs for walking down each individual tree in the model.
    load = sum(path_cost(tree, X) for tree in trees)

    # Adding up tree outputs also takes mental effort:
    # 1 tree needs 0 additions (0 extra steps).
    # 2 trees need 1 addition to combine.
    # N trees need (N - 1) additions/mental aggregations.
    return load + (len(trees) - 1)


class SimulationLoad:
    """The metric: average (or a quantile) of load(x) over the data. Lower = more interpretable.

    :param quantile: ``None`` = average load; e.g. ``0.9`` = 90% of inputs need at most this many steps.
    """

    def __init__(self, quantile: float | None = None):
        # Store whether we want the mean (None) or a specific percentile (e.g., 0.9 for 90th percentile).
        self.quantile = quantile
        # Give the metric a descriptive name for columns in output tables.
        self.name = "simulation_load" if quantile is None else f"simulation_load_q{round(quantile * 100)}"

    def score(self, model, dataset: Dataset) -> float:
        # Calculate the simulation load for every single sample in the dataset.
        load = load_per_sample(model, dataset.X)

        # Return the mean load if no quantile is requested, otherwise return the quantile.
        if self.quantile is None:
            return float(load.mean())
        return float(np.quantile(load, self.quantile))


class InterpretabilityScore:
    """Simulation load squeezed into (0, 1]: 1 / (1 + load / 7). Higher = more interpretable, 0.5 = threshold."""

    name = "interpretability"

    def score(self, model, dataset: Dataset) -> float:
        # Compute the raw average simulation load for this model.
        raw_load = SimulationLoad().score(model, dataset)
        # Squeeze the score into the range (0, 1]:
        # - Load 0 steps -> Score = 1.0 (maximally interpretable).
        # - Load 7 steps -> Score = 0.5 (exactly at the human memory limit).
        # - Load infinity -> Score approaches 0.0 (completely uninterpretable).
        return 1.0 / (1.0 + raw_load / THRESHOLD)


class NumberOfLeaves:
    """Baseline: total number of leaves over all trees. Lower = more interpretable."""

    name = "n_leaves"

    def score(self, model, dataset: Dataset) -> float:
        # Ask each tree how many leaves it has, and sum them together.
        return float(sum(tree.get_n_leaves() for tree in get_trees(model)))


class NumberOfFeaturesUsed:
    """Baseline (the "trivial idea" from the exercise): how many different features the model uses."""

    name = "n_features_used"

    def score(self, model, dataset: Dataset) -> float:
        # Use a Python set to collect the unique indices of all features queried.
        used = set()
        for tree in get_trees(model):
            # `tree_.feature` contains the feature index used at each node (or -2 for leaf nodes).
            features = tree.tree_.feature
            # Keep only valid non-negative feature indices (filtering out leaves where feature is -2).
            valid_features = features[features >= 0]
            used.update(valid_features.tolist())
        # Return the total count of distinct features that the model ever looks at.
        return float(len(used))
