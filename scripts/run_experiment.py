"""Task 1: train all 65 models on the 6 datasets (3 random splits) and save every score.

Usage::

    uv run python scripts/run_experiment.py     # writes results/scores.csv (about 1 minute)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iml_interpretability import (
    InterpretabilityScore,
    NumberOfFeaturesUsed,
    NumberOfLeaves,
    SimulationLoad,
    evaluate,
    get_models,
    load_task1_datasets,
    split_model_name,
)

# Destination file path where the detailed experiment scores are saved.
OUTPUT = Path("results/scores.csv")

# Three distinct random seeds used for 75/25 train/test splits.
# Repeating across 3 splits prevents results from being biased by one lucky split.
SEEDS = [0, 1, 2]

# The set of evaluation metrics tracked for every model:
METRICS = [
    # Primary metric: mean cognitive simulation load (average mental steps per prediction).
    SimulationLoad(),
    # 90th percentile simulation load: upper-bound steps required for 90% of inputs.
    SimulationLoad(quantile=0.9),
    # Normalized score bounded in (0, 1] with 0.5 at the load <= 7 threshold.
    InterpretabilityScore(),
    # Baseline 1: Total number of leaf nodes across all trees in the model.
    NumberOfLeaves(),
    # Baseline 2: Number of distinct input features queried anywhere in the model.
    NumberOfFeaturesUsed(),
]


def main() -> None:
    # Step 1: Load all six Task 1 benchmark datasets (cached in data/ after first run).
    datasets = load_task1_datasets()

    # List to store the results DataFrame from each seed repetition.
    runs = []

    # Step 2: Loop over each random seed.
    for seed in SEEDS:
        print(f"seed {seed} ...", flush=True)

        # Generate fresh instances of all 65 models initialized with this seed.
        models = get_models(seed=seed)

        # Train and evaluate every model on every dataset using the specified metrics.
        scores = evaluate(datasets, models, METRICS, seed=seed)

        # Record which seed produced this set of scores.
        scores["seed"] = seed

        # Add this run to our collection.
        runs.append(scores)

    # Step 3: Combine all runs into a single table.
    combined_scores = pd.concat(runs, ignore_index=True)

    # Step 4: Parse the 'model' column to add explicit 'family', 'depth', and 'n_trees' columns.
    results = split_model_name(combined_scores)

    # Step 5: Ensure the destination directory exists and save the table to CSV.
    OUTPUT.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print(f"wrote {len(results)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
