# [OWN WORK] Experiment runner for the full ablation study
# Runs 4 architectures x 3 losses x 3 seeds = 36 training experiments

import os
import json
import pandas as pd
from train import train

# Experiment configuration
ARCHITECTURES = ["unet", "attention_unet", "resunet", "smp_resnet18"]
LOSSES = ["bce", "dice", "bce_dice"]
SEEDS = [42, 123, 456]

DEFAULT_CONFIG = {
    "epochs": 50,
    "lr": 1e-4,
    "batch_size": 16,
    "patience": 10,
    "data_root": "./data",
    "output_dir": "./results",
}


def run_all_experiments(architectures=None, losses=None, seeds=None, config=None):
    """Run the full ablation study.

    Args:
        architectures: List of architecture names (default: all 4).
        losses: List of loss function names (default: all 3).
        seeds: List of random seeds (default: [42, 123, 456]).
        config: Dict of training config overrides.
    """
    if architectures is None:
        architectures = ARCHITECTURES
    if losses is None:
        losses = LOSSES
    if seeds is None:
        seeds = SEEDS
    if config is None:
        config = DEFAULT_CONFIG

    total = len(architectures) * len(losses) * len(seeds)
    completed = 0
    all_results = []

    for arch in architectures:
        for loss in losses:
            for seed in seeds:
                completed += 1
                print(f"\n{'='*60}")
                print(f"Experiment {completed}/{total}: {arch} | {loss} | seed={seed}")
                print(f"{'='*60}")

                run_config = {**config, "architecture": arch, "loss": loss, "seed": seed}

                try:
                    result = train(run_config)
                    all_results.append(result)
                except Exception as e:
                    print(f"FAILED: {arch}_{loss}_seed{seed}: {e}")
                    all_results.append({
                        "architecture": arch,
                        "loss": loss,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save combined results
    results_path = os.path.join(config["output_dir"], "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save as CSV for easy analysis
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(config["output_dir"], "results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"All {total} experiments complete.")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    df = run_all_experiments()
    print("\nSummary (mean Dice per architecture x loss):")
    pivot = df.pivot_table(values="test_dice", index="architecture", columns="loss", aggfunc=["mean", "std"])
    print(pivot)
