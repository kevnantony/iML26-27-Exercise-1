"""The 65 tree models we compare: from a single split up to 100-tree ensembles.

All of them belong to one model class (trees), in three forms:
single decision tree, random forest, gradient boosting.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Depths tested for single decision trees:
# Depth 1 is a single split (stump); None means unconstrained (grows until leaves are pure).
TREE_DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, None]

# Number of individual trees in each ensemble (random forest or gradient boosting).
# Ranges from 1 tree up to 100 trees.
ENSEMBLE_SIZES = [1, 2, 3, 5, 10, 30, 100]

# Maximum depth allowed for individual trees inside random forests.
FOREST_DEPTHS = [1, 2, 3, 5, None]

# Maximum depth allowed for individual trees in gradient boosting (typically very shallow).
BOOSTING_DEPTHS = [1, 2, 3]


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Create all (untrained) models. Names look like ``forest_d3_n10`` = forest, depth 3, 10 trees.

    9 single trees + 7 sizes x 5 depths forests + 7 sizes x 3 depths boosting = 65 models.

    :param seed: Random seed given to every model.
    :return: Model name -> untrained scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    # Dictionary that will hold model_name -> model_instance.
    models = {}

    # 1. Single Decision Trees:
    # Loop over each depth level to create 9 single decision trees.
    for depth in TREE_DEPTHS:
        # If depth is None, label it 'max' in the name; otherwise use the integer depth.
        depth_label = depth or "max"
        model_name = f"tree_d{depth_label}_n1"
        # Instantiate the decision tree with the given depth and random seed.
        models[model_name] = DecisionTreeClassifier(max_depth=depth, random_state=seed)

    # 2. Ensembles (Random Forests & Gradient Boosting):
    # Loop over each ensemble size (number of trees).
    for n in ENSEMBLE_SIZES:
        # For each size, loop over all tree depths for Random Forest (5 depths x 7 sizes = 35 forests).
        for depth in FOREST_DEPTHS:
            depth_label = depth or "max"
            model_name = f"forest_d{depth_label}_n{n}"
            # Instantiate the random forest with n estimators, max depth, and random seed.
            models[model_name] = RandomForestClassifier(
                n_estimators=n, max_depth=depth, random_state=seed
            )

        # For each size, loop over all tree depths for Gradient Boosting (3 depths x 7 sizes = 21 models).
        for depth in BOOSTING_DEPTHS:
            model_name = f"boosting_d{depth}_n{n}"
            # Gradient boosting learning rate adjustment:
            # Standard learning rate 0.1 is too slow for very small ensembles (1-5 trees),
            # causing them to just predict the majority class.
            # So for small n, we scale the learning rate as 1/n (with a minimum of 0.1).
            learning_rate = max(0.1, 1.0 / n)
            # Instantiate the gradient boosting classifier.
            models[model_name] = GradientBoostingClassifier(
                n_estimators=n, max_depth=depth, learning_rate=learning_rate, random_state=seed
            )

    # Return the complete dictionary of 65 model configurations.
    return models
