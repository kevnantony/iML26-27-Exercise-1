"""Task 2: train all 65 models on 32 OpenML datasets, and compute meta-features + outcomes per dataset.

Usage::

    uv run python scripts/run_task2.py      # about 5 minutes, datasets run in parallel
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from iml_interpretability import (
    TASK2_DATASETS,
    SimulationLoad,
    compute_meta_features,
    evaluate,
    get_models,
    interpretability_outcomes,
    load_dataset,
    split_model_name,
)

# Output directory for saving Task 2 result tables.
RESULTS = Path("results")

# Two random seeds per dataset for train/test splits (averaging across 32 datasets provides robust estimates).
SEEDS = [0, 1]

# Subsample cap for very large datasets to keep total runtime around 5 minutes.
MAX_SAMPLES = 5000


def run_dataset(name: str, data_id: int) -> tuple[pd.DataFrame, dict]:
    """Everything for one dataset: train the 65 models, then compute its meta-features and outcomes."""
    # Suppress non-critical convergence/version warnings to keep terminal output readable.
    warnings.filterwarnings("ignore")

    # Step 1: Download or load the preprocessed dataset from local cache.
    dataset = load_dataset(name, data_id, max_samples=MAX_SAMPLES)

    # Step 2: Train all 65 models across both seeds and compute their test accuracy & simulation load.
    runs = [
        evaluate([dataset], get_models(seed), [SimulationLoad()], seed=seed)
        for seed in SEEDS
    ]

    # Step 3: Average accuracy and simulation load across the random seeds for each model.
    models = pd.concat(runs).groupby(["dataset", "model"], as_index=False).mean(numeric_only=True)

    # Step 4: Parse model names to append explicit columns for family, depth, and n_trees.
    models = split_model_name(models)

    # Step 5: Compute dataset meta-features and interpretability outcomes (e.g. price, load needed).
    meta_features = compute_meta_features(dataset)
    outcomes = interpretability_outcomes(models)

    # Merge everything into one summary row for this dataset.
    row = {"dataset": name, **meta_features, **outcomes}

    print(f"done {name} {dataset.X.shape}", flush=True)
    return models, row


def main() -> None:
    # Run all 32 datasets in parallel using all available CPU cores (`n_jobs=-1`).
    outputs = Parallel(n_jobs=-1)(
        delayed(run_dataset)(name, data_id) for name, data_id in TASK2_DATASETS.items()
    )

    # Ensure output results directory exists.
    RESULTS.mkdir(exist_ok=True)

    # Concatenate model evaluation tables across all datasets and save to CSV.
    all_models = pd.concat([models for models, _ in outputs])
    all_models.to_csv(RESULTS / "task2_models.csv", index=False)

    # Combine summary rows (meta-features + outcomes) for all datasets into one table and save to CSV.
    dataset_table = pd.DataFrame([row for _, row in outputs])
    dataset_table.to_csv(RESULTS / "task2_datasets.csv", index=False)

    # Print rounded summary table to stdout for immediate inspection.
    print(dataset_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
