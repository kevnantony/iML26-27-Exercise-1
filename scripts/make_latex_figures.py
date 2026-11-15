"""Make the plots used by the LaTeX (Beamer) version of the slides: presentation/latex/figures/*.pdf

Needs results from make_figures.py and analyse_task2.py. Usage::

    uv run python scripts/make_latex_figures.py
    cd presentation/latex && tectonic slides.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

# Render directly to vector files without opening an interactive GUI window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paths for input data and output PDF vector figure directory.
RESULTS = Path("results")
OUT = Path("presentation/latex/figures")

# Colors selected to coordinate harmoniously with the LaTeX Beamer 'Madrid' color scheme:
BLUE, GREEN, RED, GREY, LIGHT = "#3333b2", "#2e8b57", "#c62828", "#8f8fa8", "#e6e6f6"
COLORS = {"tree": BLUE, "forest": GREEN, "boosting": RED}

# Set global matplotlib typography and clean border styling (no top or right axis spines).
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    """Save a figure as a vector PDF (sharp at any size) and close it."""
    # Ensure all elements fit snugly without clipping.
    fig.tight_layout()
    # Save as high-quality vector PDF.
    fig.savefig(OUT / f"{name}.pdf")
    # Close figure object to release memory.
    plt.close(fig)


def scatter_two_datasets(task1: dict) -> None:
    """Accuracy vs. load for Titanic and Phoneme, with best overall vs. best interpretable marked."""
    # Create two side-by-side subplots for Titanic and Phoneme.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, name in zip(axes, ["titanic", "phoneme"], strict=True):
        # Extract data points for this dataset.
        pts = [p for p in task1["points"] if p["dataset"] == name]

        # Highlight the interpretable region (load between 0.8 and 7) with light blue shading.
        ax.axvspan(0.8, 7, color=LIGHT, zorder=0)

        # Plot scatter points for each model family.
        for family, color in COLORS.items():
            f = [p for p in pts if p["family"] == family]
            ax.scatter(
                [p["simulation_load"] for p in f],
                [100 * p["accuracy"] for p in f],
                s=28,
                color=color,
                label={"tree": "1 tree"}.get(family, family),
            )

        # Find the global maximum accuracy and the maximum accuracy within load <= 7.
        best = max(p["accuracy"] for p in pts)
        best_ok = max(p["accuracy"] for p in pts if p["simulation_load"] <= 7)

        # Draw horizontal dashed lines for both benchmark accuracies.
        for v in (best, best_ok):
            ax.axhline(100 * v, color="black", ls="--", lw=1)

        # Annotate the price difference in percentage points on the right margin.
        price_diff = 100 * (best - best_ok)
        text_color = BLUE if (best - best_ok) < 0.02 else RED
        ax.text(1100, 100 * (best + best_ok) / 2, f"−{price_diff:.1f}", va="center", fontsize=15,
                fontweight="bold", color=text_color)

        # Format logarithmic x-axis and titles.
        ax.set_xscale("log")
        ax.set_xlim(0.8, 1000)
        ax.set_title(name.capitalize(), fontweight="bold")
        ax.set_xlabel("load (steps, log scale)")

    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=11)
    save(fig, "scatter")


def price_bars(task1: dict) -> None:
    """Accuracy lost per dataset when only load <= 7 is allowed."""
    # Sort datasets in ascending order of interpretability price.
    rows = sorted(task1["summary"], key=lambda r: r["price_of_interpretability"])
    # Convert accuracy differences to percentage points.
    values = [100 * r["price_of_interpretability"] for r in rows]

    # Create horizontal bar plot.
    fig, ax = plt.subplots(figsize=(9, 4))
    dataset_labels = [r["dataset"].replace("_", " ") for r in rows]
    bar_colors = [BLUE if v < 2 else RED for v in values]
    bars = ax.barh(dataset_labels, values, color=bar_colors)

    # Add numeric labels at the ends of the bars.
    ax.bar_label(bars, fmt="%.1f", padding=4, fontweight="bold")

    # Invert y-axis so smallest sacrifice appears on top.
    ax.invert_yaxis()
    ax.set_xlabel("accuracy lost with load ≤ 7 (percentage points)")
    save(fig, "price")


def threshold_dots(task1: dict) -> None:
    """Filled dot = median load <= 7, hollow = above, for every (family, depth, number of trees)."""
    # Create lookup map for (family, depth, n_trees) -> simulation load.
    load = {(g["family"], g["depth"], g["n_trees"]): g["simulation_load"] for g in task1["grid"]}
    sizes = [1, 2, 3, 5, 10, 30, 100]

    # 3 panel plot for single tree, random forest, and boosting.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 3, 3]})
    panels = [
        ("tree", ["1", "2", "3", "4", "5", "6", "8", "10", "max"], [1]),
        ("forest", ["1", "2", "3", "5", "max"], sizes),
        ("boosting", ["1", "2", "3"], sizes),
    ]

    for ax, (family, depths, ns) in zip(axes, panels, strict=True):
        for i, d in enumerate(depths):
            for j, n in enumerate(ns):
                # Solid circle if model is interpretable (load <= 7), hollow circle if not.
                ok = load[(family, d, n)] <= 7
                fill_color = COLORS[family] if ok else "white"
                ax.scatter(j, i, s=170, color=fill_color, edgecolor=COLORS[family], lw=2)

        # Set tick marks and axes.
        ax.set_yticks(range(len(depths)), depths)
        ax.set_xticks(range(len(ns)), ns)
        ax.invert_yaxis()
        ax.set_ylabel("depth")
        ax.set_xlabel("trees" if len(ns) > 1 else "")
        ax.set_title({"tree": "1 tree"}.get(family, family), color=COLORS[family], fontweight="bold")
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

    save(fig, "threshold")


def task2_correlations(task2: dict) -> None:
    """Correlation of each meta-feature with the price; solid = p < 0.05."""
    # Friendly labels for dataset meta-features.
    names = {
        "log_n_samples": "# samples",
        "log_n_features": "# features",
        "n_classes": "# classes",
        "class_balance": "class balance",
        "share_categorical": "% categorical",
        "pca_dims_95": "intrinsic dim.",
        "best_feature_corr": "best single feature",
        "acc_majority": "majority-class acc.",
        "stump_gain": "gain of 1 split",
        "linear_gain": "gain of linear model",
        "knn_gain": "gain of 1-NN",
    }

    # Sort meta-features by correlation rho with price.
    rows = sorted(names, key=lambda m: task2["rho"][m])
    rho = np.array([task2["rho"][m] for m in rows])
    sig = np.array([task2["p"][m] < 0.05 for m in rows])

    fig, ax = plt.subplots(figsize=(9, 4.6))

    # Color bars: Red for positive correlation (increases price), Blue for negative.
    # Solid opacity (1.0) if statistically significant (p < 0.05), faded (0.3) if not.
    colors = [
        matplotlib.colors.to_rgba(RED if r > 0 else BLUE, 1.0 if s else 0.3)
        for r, s in zip(rho, sig, strict=True)
    ]
    ax.barh([names[m] for m in rows], rho, color=colors)

    # Annotate numeric correlation values next to the bars.
    for i, (r, s) in enumerate(zip(rho, sig, strict=True)):
        offset = 0.02 if r > 0 else -0.02
        halign = "left" if r > 0 else "right"
        weight = "bold" if s else "normal"
        ax.text(r + offset, i, f"{r:+.2f}", va="center", ha=halign, fontsize=11, fontweight=weight)

    # Add vertical zero baseline.
    ax.axvline(0, color="black", lw=1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("← cheaper     Spearman ρ with accuracy lost     more expensive →   (solid: p < 0.05)")
    save(fig, "task2_corr")


def task2_scatter(task2: dict) -> None:
    """Price vs. number of classes and vs. majority-class accuracy."""
    pts = task2["points"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        (axes[0], "n_classes", "# classes", {"vowel", "steel_plates", "balance_scale"}),
        (axes[1], "acc_majority", "majority-class accuracy", {"vowel", "wilt", "ozone"}),
    ]

    for ax, key, label, names in panels:
        # Scatter plot price vs meta-feature.
        ax.scatter([p[key] for p in pts], [100 * p["price"] for p in pts], s=40, color=BLUE)

        # Label notable representative datasets.
        for p in pts:
            if p["dataset"] in names:
                ax.annotate(p["dataset"].replace("_", " "), (p[key], 100 * p["price"]),
                            xytext=(5, 3), textcoords="offset points", fontsize=11)

        ax.set_xlabel(label)
        ax.set_title(f"ρ = {task2['rho'][key]:+.2f}", color=RED, fontweight="bold")

    axes[0].set_ylabel("accuracy lost at load ≤ 7 (pts)")
    save(fig, "task2_scatter")


def main() -> None:
    # Ensure figure output directory exists.
    OUT.mkdir(parents=True, exist_ok=True)

    # Load calculated results exported by make_figures.py and analyse_task2.py.
    task1 = json.loads((RESULTS / "slide_data.json").read_text())
    task2 = json.loads((RESULTS / "task2_slide.json").read_text())

    # Build and save each vector PDF plot for LaTeX slides.
    scatter_two_datasets(task1)
    price_bars(task1)
    threshold_dots(task1)
    task2_correlations(task2)
    task2_scatter(task2)

    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
