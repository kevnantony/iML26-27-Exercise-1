"""Task 2: which dataset meta-features go together with expensive (= hard to keep interpretable) datasets?

Reads results/task2_datasets.csv and writes:
    results/task2_correlations.csv   Spearman correlation of every meta-feature with every outcome
    results/task2_heatmap.png        the same as a picture
    results/task2_meta_tree.txt      a depth-2 tree that predicts "interpretable enough" from meta-features
    results/task2_slide.json         numbers for the slides

Usage::

    uv run python scripts/analyse_task2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Use 'Agg' non-interactive backend to generate figures without a GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

# Directory where result files are stored.
RESULTS = Path("results")

# List of all 11 dataset meta-features analyzed:
META_FEATURES = [
    "log_n_samples", "log_n_features", "n_classes", "class_balance", "share_categorical", "pca_dims_95",
    "best_feature_corr", "acc_majority", "stump_gain", "linear_gain", "knn_gain",
]  # fmt: skip

# List of interpretability outcome variables computed across the 65 models:
OUTCOMES = ["price", "log_load_needed", "full_tree_load", "best_accuracy"]


def correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation (and p-value) between every meta-feature and every outcome."""
    rows = []

    # Iterate over each meta-feature column.
    for meta in META_FEATURES:
        # Iterate over each outcome column.
        for outcome in OUTCOMES:
            # Spearman rank correlation rho and two-tailed p-value:
            # Non-parametric and robust against outliers and non-linear monotonic trends.
            rho, p = spearmanr(table[meta], table[outcome])

            # Record result.
            rows.append({"meta_feature": meta, "outcome": outcome, "rho": rho, "p": p})

    # Return as DataFrame.
    return pd.DataFrame(rows)


def plot_heatmap(corr: pd.DataFrame, n_datasets: int) -> None:
    """Colour = correlation; a star marks p < 0.05."""
    # Pivot correlation table into a 2D matrix (meta_features x outcomes).
    rho = corr.pivot(index="meta_feature", columns="outcome", values="rho").loc[META_FEATURES, OUTCOMES]
    # Pivot p-values into matching 2D matrix.
    p = corr.pivot(index="meta_feature", columns="outcome", values="p").loc[META_FEATURES, OUTCOMES]

    # Create figure.
    fig, ax = plt.subplots(figsize=(8, 7))

    # Display heatmap: Diverging colormap RdBu_r (-1 = dark blue, 0 = white, +1 = dark red).
    image = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)

    # Annotate each cell with its numerical correlation value, adding an asterisk '*' if p < 0.05.
    for i in range(len(META_FEATURES)):
        for j in range(len(OUTCOMES)):
            is_significant = p.values[i, j] < 0.05
            star = "*" if is_significant else ""
            cell_text = f"{rho.values[i, j]:.2f}{star}"
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=9)

    # Label x and y axes with outcome names and meta-feature names.
    ax.set_xticks(range(len(OUTCOMES)), OUTCOMES, rotation=20, ha="right")
    ax.set_yticks(range(len(META_FEATURES)), META_FEATURES)
    ax.set_title(f"Spearman ρ over {n_datasets} datasets (* = p < 0.05)")

    # Add colorbar indicator.
    fig.colorbar(image, ax=ax, shrink=0.7)

    # Format layout and save PNG.
    fig.tight_layout()
    fig.savefig(RESULTS / "task2_heatmap.png", dpi=150)
    plt.close(fig)


def meta_tree(table: pd.DataFrame) -> tuple[DecisionTreeClassifier, float, float]:
    """Depth-2 tree predicting "interpretable enough" from meta-features (small, so readable itself).

    :return: The tree fitted on all datasets, its leave-one-out accuracy, and the accuracy of always
        guessing the more common answer.
    """
    # Features for the meta-tree: the 11 dataset meta-features.
    X = table[META_FEATURES]

    # Binary classification target: 1 if the dataset can be modelled accurately with load <= 7, else 0.
    y = table.interpretable_enough.astype(int)

    # Constrain the meta-tree to depth 2 with at least 3 datasets per leaf for robust, readable rules.
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=0)

    # Evaluate generalisation with Leave-One-Out Cross-Validation (LOO-CV):
    # Train on 31 datasets, predict on the 1 held-out dataset, repeated 32 times.
    predicted = cross_val_predict(tree, X, y, cv=LeaveOneOut())

    # Calculate leave-one-out cross-validation accuracy.
    loo_accuracy = float((predicted == y).mean())

    # Baseline accuracy: what you get by always guessing the majority class.
    class_1_ratio = y.mean()
    baseline = float(max(class_1_ratio, 1.0 - class_1_ratio))

    # Fit the final tree on all 32 datasets and return the fitted model along with accuracies.
    return tree.fit(X, y), loo_accuracy, baseline


def main() -> None:
    # 1. Load the Task 2 dataset summary table.
    table = pd.read_csv(RESULTS / "task2_datasets.csv")

    # 2. Compute Spearman correlations between meta-features and outcomes.
    corr = correlations(table)
    corr.to_csv(RESULTS / "task2_correlations.csv", index=False)

    # 3. Plot and save correlation heatmap.
    plot_heatmap(corr, len(table))

    # 4. Train and evaluate the interpretable meta-decision tree.
    tree, loo_accuracy, baseline = meta_tree(table)

    # 5. Export meta-tree logic as human-readable if/else text rules.
    rules = export_text(tree, feature_names=META_FEATURES)
    (RESULTS / "task2_meta_tree.txt").write_text(
        f"leave-one-out accuracy {loo_accuracy:.3f} (always guessing: {baseline:.3f})\n\n{rules}"
    )

    # 6. Rank meta-features by correlation magnitude with the price of interpretability.
    price = corr[corr.outcome == "price"].sort_values("rho", key=abs, ascending=False)
    print(price.round(3).to_string(index=False))
    print(f"\ninterpretable enough: {int(table.interpretable_enough.sum())}/{len(table)} datasets")
    print(f"meta-tree leave-one-out accuracy {loo_accuracy:.3f} vs. {baseline:.3f}\n{rules}")

    # 7. Package structured numerical data required by the presentation slides.
    t = tree.tree_
    slide = {
        # Spearman rho values for price of interpretability.
        "rho": {m: corr[(corr.meta_feature == m) & (corr.outcome == "price")].rho.item() for m in META_FEATURES},
        # Significance p-values for price of interpretability.
        "p": {m: corr[(corr.meta_feature == m) & (corr.outcome == "price")].p.item() for m in META_FEATURES},
        # Scatter plot coordinates for individual datasets.
        "points": table[["dataset", "price", "n_classes", "acc_majority"]].round(4).to_dict(orient="records"),
        # LOO cross-validation accuracy of the meta-tree.
        "loo": loo_accuracy,
        # Majority-class baseline accuracy.
        "base": baseline,
        # Low-level tree architecture arrays:
        "tree": {
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "threshold": t.threshold.round(3).tolist(),
            "value": [[round(c * n) for c in v[0]] for v, n in zip(t.value, t.n_node_samples, strict=True)],
        },
    }

    # Write slide data to JSON.
    (RESULTS / "task2_slide.json").write_text(json.dumps(slide))


if __name__ == "__main__":
    main()
