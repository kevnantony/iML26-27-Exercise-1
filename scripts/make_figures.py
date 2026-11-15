"""Task 1: turn results/scores.csv into figures, a summary table and the data for the slides.

Usage::

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Use 'Agg' non-interactive backend to render plots directly to disk without requiring a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from iml_interpretability import THRESHOLD  # noqa: E402

# Folder where input scores are read from and output figures/tables are saved.
RESULTS = Path("results")

# Palette colours for model families, matching the presentation theme:
# Blue for single trees, Green for random forests, Red for gradient boosting.
COLORS = {"tree": "#3333b2", "forest": "#2e8b57", "boosting": "#c62828"}

# Ordering datasets roughly from easiest to most complex for visual layout.
ORDER = ["banknote", "breast_cancer", "titanic", "diabetes", "credit_g", "phoneme"]


def average_over_seeds() -> pd.DataFrame:
    """One row per (dataset, model): the mean of all scores over the 3 random splits."""
    # Read the full raw scores file produced by run_experiment.py.
    scores = pd.read_csv(RESULTS / "scores.csv")

    # Group by model architecture identifiers and dataset name.
    keys = ["dataset", "model", "family", "depth", "n_trees"]

    # Compute the arithmetic mean of all numeric metrics across the random seeds.
    averaged = scores.groupby(keys, as_index=False).mean(numeric_only=True)

    # Drop the seed column as it is no longer meaningful after averaging.
    return averaged.drop(columns="seed")


def summarise(g: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: best model, best interpretable model (load <= 7) and the accuracy difference."""
    rows = []

    # Iterate through the datasets in fixed order.
    for name in ORDER:
        # Filter rows for this specific dataset.
        d = g[g.dataset == name]

        # Find the overall most accurate model (unconstrained, highest test accuracy).
        best = d.loc[d.accuracy.idxmax()]

        # Filter models that satisfy the interpretability constraint (simulation load <= 7).
        interpretable = d[d.simulation_load <= THRESHOLD]

        # Find the best performing model within the interpretable subset.
        best_ok = interpretable.loc[interpretable.accuracy.idxmax()]

        # Compute price of interpretability: how much accuracy is sacrificed to stay interpretable.
        price = best.accuracy - best_ok.accuracy

        # Record summary information for this dataset.
        rows.append(
            {
                "dataset": name,
                "best_model": best.model,
                "best_accuracy": best.accuracy,
                "best_load": best.simulation_load,
                "best_interpretable_model": best_ok.model,
                "best_interpretable_accuracy": best_ok.accuracy,
                "best_interpretable_load": best_ok.simulation_load,
                "price_of_interpretability": price,
            }
        )

    # Return summary as a DataFrame.
    return pd.DataFrame(rows)


def plot_accuracy_vs_load(g: pd.DataFrame) -> None:
    """One scatter plot per dataset: every model as a dot, the interpretable zone shaded."""
    # Create a 2-row, 3-column grid of subplots (one subplot per dataset).
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)

    # Iterate over each subplot axis and corresponding dataset.
    for ax, name in zip(axes.ravel(), ORDER, strict=True):
        d = g[g.dataset == name]

        # Shade the region where simulation load <= 7 in soft blue to highlight the interpretable zone.
        ax.axvspan(0.8, THRESHOLD, color="#e6e6f6", zorder=0)

        # Plot each model family in its designated color (tree=blue, forest=green, boosting=red).
        for family, color in COLORS.items():
            f = d[d.family == family]
            ax.scatter(f.simulation_load, f.accuracy, s=22, color=color, label=family)

        # Use a logarithmic x-axis since simulation load ranges from 1 to >1000 steps.
        ax.set_xscale("log")
        ax.set_title(name)
        ax.grid(alpha=0.3)

    # Set x-axis labels on the bottom row.
    for ax in axes[1]:
        ax.set_xlabel("simulation load (steps per prediction, log scale)")

    # Set y-axis labels on the leftmost column.
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")

    # Add legend to the first subplot.
    axes[0, 0].legend(loc="lower right")

    # Adjust layout and save high-resolution figure to disk.
    fig.tight_layout()
    fig.savefig(RESULTS / "frontier.png", dpi=150)
    plt.close(fig)


def plot_price(summary: pd.DataFrame) -> None:
    """Bar chart: accuracy lost per dataset when only interpretable models are allowed."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Convert accuracy difference to percentage points (* 100) and plot as horizontal bars.
    percentage_lost = 100 * summary.price_of_interpretability
    ax.barh(summary.dataset, percentage_lost, color="#3333b2")

    # Invert the vertical axis so the first dataset appears at the top.
    ax.invert_yaxis()
    ax.set_xlabel("accuracy lost by staying at load ≤ 7 (percentage points)")

    # Save figure to disk.
    fig.tight_layout()
    fig.savefig(RESULTS / "price_of_interpretability.png", dpi=150)
    plt.close(fig)


def median_load_per_config(g: pd.DataFrame) -> pd.DataFrame:
    """Median load over the 6 datasets for every (family, depth, number of trees) configuration."""
    # Group by architecture configuration and take the median load across the 6 benchmark datasets.
    return g.groupby(["family", "depth", "n_trees"], as_index=False).simulation_load.median()


def plot_ensemble_grid(grid: pd.DataFrame) -> None:
    """Heatmap per ensemble type: median load for each (depth, number of trees); blue = load <= 7."""
    # Create two side-by-side heatmaps: one for Random Forest, one for Gradient Boosting.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, family in zip(axes, ["forest", "boosting"], strict=True):
        # Pivot the table: rows = depth, columns = n_trees, values = median load.
        table = grid[grid.family == family].pivot(index="depth", columns="n_trees", values="simulation_load")

        # Color cell blue if load <= 7 (interpretable), otherwise white.
        ax.imshow(table.values <= THRESHOLD, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")

        # Overlay the numeric median load value in each grid cell.
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{table.values[i, j]:.0f}", ha="center", va="center", fontsize=9)

        # Configure axis ticks and labels.
        ax.set_xticks(range(table.shape[1]), table.columns)
        ax.set_yticks(range(table.shape[0]), table.index)
        ax.set_xlabel("number of trees")
        ax.set_ylabel("max depth")
        ax.set_title(f"{family}: median load (blue = ≤ 7)")

    fig.tight_layout()
    fig.savefig(RESULTS / "ensemble_grid.png", dpi=150)
    plt.close(fig)


def main() -> None:
    # 1. Calculate average scores across the random seeds.
    g = average_over_seeds()

    # 2. Compute summary table (price of interpretability per dataset) and write to CSV.
    summary = summarise(g)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    # 3. Calculate median simulation load for every model configuration.
    grid = median_load_per_config(g)

    # 4. Generate all figures for Task 1.
    plot_accuracy_vs_load(g)
    plot_price(summary)
    plot_ensemble_grid(grid)

    # 5. Measure rank correlation between simulation load and baseline metrics:
    # Check how strongly simulation load agrees with number of leaves and number of features.
    rho_leaves = spearmanr(g.simulation_load, g.n_leaves)[0]
    rho_features = spearmanr(g.simulation_load, g.n_features_used)[0]

    # Print summary to stdout.
    print(summary.round(3).to_string(index=False))
    print(f"Spearman(load, n_leaves) = {rho_leaves:.2f}, Spearman(load, n_features_used) = {rho_features:.2f}")

    # 6. Export structured JSON data consumed by the presentation slides builder.
    slide_data = {
        "points": g[["dataset", "family", "accuracy", "simulation_load"]].round(4).to_dict(orient="records"),
        "summary": summary[["dataset", "price_of_interpretability"]].round(4).to_dict(orient="records"),
        "grid": grid.round(2).to_dict(orient="records"),
    }
    (RESULTS / "slide_data.json").write_text(json.dumps(slide_data))


if __name__ == "__main__":
    main()
