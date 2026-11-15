"""Tests that run offline on small synthetic data. Run with: uv run pytest"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import (
    Dataset,
    InterpretabilityScore,
    SimulationLoad,
    compute_meta_features,
    evaluate,
    get_models,
    get_trees,
    interpretability_outcomes,
    load_per_sample,
)


@pytest.fixture
def dataset() -> Dataset:
    """Fixture providing a fast, reproducible synthetic classification dataset."""
    # Generate 100 random samples with 5 features and 2 classes using scikit-learn.
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)

    # Return as a wrapped Dataset with feature column names x0, x1, x2, x3, x4.
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_grid_has_65_models():
    """Verify that the model factory generates exactly 65 tree architectures."""
    # 9 single decision trees + 35 random forests + 21 gradient boosting models = 65 models.
    all_models = get_models()
    assert len(all_models) == 65


def test_evaluate_gives_one_row_per_model(dataset):
    """Verify that evaluate() produces the expected table structure."""
    # Define a single test model (decision stump).
    models = {"stump": DecisionTreeClassifier(max_depth=1)}

    # Run evaluation using SimulationLoad metric.
    results = evaluate([dataset], models, [SimulationLoad()])

    # Check that required columns exist in the resulting DataFrame.
    assert list(results.columns) == ["dataset", "model", "accuracy", "simulation_load"]

    # Exactly 1 model on 1 dataset should produce 1 row.
    assert len(results) == 1


def test_stump_needs_one_step(dataset):
    """Verify that a single split (decision stump) always requires exactly 1 cognitive step."""
    # Fit a decision tree restricted to max_depth=1 (a single binary question).
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)

    # A single question on a new feature takes 1 lookup step and 0 ensemble combination steps = 1.0.
    assert SimulationLoad().score(stump, dataset) == 1.0

    # Interpretability score formula: 1 / (1 + load / 7). With load = 1.0, score = 1 / (1 + 1/7).
    expected_score = 1.0 / (1.0 + 1.0 / 7.0)
    assert InterpretabilityScore().score(stump, dataset) == pytest.approx(expected_score)


def test_load_never_exceeds_depth_and_grows_with_depth(dataset):
    """Verify simulation load bounds: load cannot exceed depth and grows monotonically with tree depth."""
    loads = []

    # Test trees of increasing maximum depth.
    for depth in [1, 2, 3, 5]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(dataset.X, dataset.y)
        load = SimulationLoad().score(tree, dataset)

        # Simulation load cannot exceed tree depth because each decision level asks at most 1 question.
        assert load <= tree.get_depth()

        loads.append(load)

    # Deeper trees should never require fewer simulation steps than shallower trees.
    assert loads == sorted(loads)


def test_repeated_feature_costs_half_a_step():
    """Verify cognitive discount: repeated questions on the same feature cost 0.5 instead of 1.0."""
    # Create dataset with only ONE single feature 'x', forcing depth-2 tree to split on 'x' twice.
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0] * 10})
    y = pd.Series([0, 1, 0, 1] * 10)

    # Fit depth-2 tree.
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

    # Compute per-sample simulation load.
    load = load_per_sample(tree, X)

    # Samples reaching depth 2 make 1 initial lookup (1.0 step) + 1 repeated check (0.5 step) = 1.5 steps.
    deep_path_loads = set(load[load > 1].tolist())
    assert deep_path_loads == {1.5}


def test_ensembles_pay_for_combining(dataset):
    """Verify that ensemble models correctly pay a penalty of (N - 1) steps to combine tree predictions."""
    # 1. Random Forest with 4 depth-1 stumps:
    # 4 trees x 1.0 step each + 3 combination additions = 7.0 total steps.
    forest = RandomForestClassifier(n_estimators=4, max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert SimulationLoad().score(forest, dataset) == 7.0

    # 2. Gradient Boosting with 3 depth-1 stumps:
    # 3 trees x 1.0 step each + 2 combination additions = 5.0 total steps.
    boosting = GradientBoostingClassifier(n_estimators=3, max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert len(get_trees(boosting)) == 3
    assert SimulationLoad().score(boosting, dataset) == 5.0


def test_meta_features_are_in_range(dataset):
    """Verify that dataset meta-features lie within expected statistical intervals."""
    meta = compute_meta_features(dataset)

    # Synthetic binary classification data must have 2 classes.
    assert meta["n_classes"] == 2

    # Class balance and accuracy must be within valid bounded ranges (0, 1].
    assert 0.0 < meta["class_balance"] <= 1.0
    assert 0.0 < meta["acc_majority"] <= 1.0
    assert 0.0 < meta["pca_dims_95"] <= 1.0


def test_outcomes_on_a_hand_made_example():
    """Verify interpretability outcome calculations on a synthetic test table."""
    # Create mock results representing 3 models.
    models = pd.DataFrame(
        {
            "family": ["tree", "tree", "forest"],
            "depth": ["1", "max", "3"],
            "accuracy": [0.70, 0.75, 0.80],
            "simulation_load": [1.0, 6.0, 50.0],
        }
    )

    out = interpretability_outcomes(models)

    # Price = Best overall accuracy (0.80) - Best interpretable accuracy (load <= 7 -> 0.75) = 0.05.
    assert out["price"] == pytest.approx(0.05)

    # Within 1 point of best (>= 0.79): only the forest (accuracy 0.80) qualifies, with load 50.0.
    assert out["load_needed"] == 50.0

    # Since load_needed (50.0) exceeds 7.0, interpretable_enough must be 0.0.
    assert out["interpretable_enough"] == 0.0

    # Full tree load (depth='max') is 6.0.
    assert out["full_tree_load"] == 6.0
