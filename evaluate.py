# [OWN WORK] Evaluation and statistical analysis script
# Loads experiment results and performs statistical tests + generates report figures

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

from utils.visualization import (
    plot_training_curves,
    plot_ablation_bar_chart,
    plot_loss_comparison,
)


def load_results(results_dir="./results"):
    """Load all experiment results from the results directory."""
    csv_path = os.path.join(results_dir, "results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    # Fall back to JSON
    json_path = os.path.join(results_dir, "all_results.json")
    with open(json_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_statistics(df, metric="test_dice"):
    """Compute mean, std, and pairwise statistical tests.

    Args:
        df: DataFrame with experiment results.
        metric: Column name of the metric to analyze.

    Returns:
        summary: DataFrame with mean +/- std per (architecture, loss).
        pairwise: DataFrame with Wilcoxon test p-values between architectures.
    """
    # Summary statistics
    summary = df.groupby(["architecture", "loss"])[metric].agg(["mean", "std", "count"]).reset_index()
    summary["mean_pm_std"] = summary.apply(
        lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1
    )
    print(f"\n{'='*60}")
    print(f"Summary Statistics for {metric}")
    print(f"{'='*60}")
    pivot = summary.pivot(index="architecture", columns="loss", values="mean_pm_std")
    print(pivot.to_string())

    # Pairwise Wilcoxon signed-rank tests between architectures
    architectures = df["architecture"].unique()
    n_arch = len(architectures)
    pairwise_results = []

    # Compare using the best loss for each architecture
    best_loss = "bce_dice"  # Use combined loss for fair comparison
    for i in range(n_arch):
        for j in range(i + 1, n_arch):
            a1 = architectures[i]
            a2 = architectures[j]

            scores1 = df[(df["architecture"] == a1) & (df["loss"] == best_loss)][metric].values
            scores2 = df[(df["architecture"] == a2) & (df["loss"] == best_loss)][metric].values

            if len(scores1) >= 3 and len(scores2) >= 3 and len(scores1) == len(scores2):
                # Wilcoxon signed-rank test (paired)
                try:
                    stat_val, p_value = stats.wilcoxon(scores1, scores2)
                except Exception:
                    p_value = 1.0
                    stat_val = 0.0

                # Cohen's d effect size
                diff = scores1 - scores2
                d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

                # Bonferroni correction
                n_comparisons = n_arch * (n_arch - 1) / 2
                p_corrected = min(p_value * n_comparisons, 1.0)

                pairwise_results.append({
                    "architecture_1": a1,
                    "architecture_2": a2,
                    "loss_used": best_loss,
                    "mean_diff": round(np.mean(scores1) - np.mean(scores2), 6),
                    "cohens_d": round(d, 4),
                    "wilcoxon_p": round(p_value, 6),
                    "bonferroni_p": round(p_corrected, 6),
                    "significant_005": p_corrected < 0.05,
                })

    pairwise_df = pd.DataFrame(pairwise_results)

    print(f"\n{'='*60}")
    print(f"Pairwise Comparisons (Wilcoxon signed-rank, loss={best_loss})")
    print(f"{'='*60}")
    if not pairwise_df.empty:
        print(pairwise_df.to_string(index=False))
    else:
        print("Not enough data for pairwise tests.")

    return summary, pairwise_df


def generate_report_figures(results_dir="./results", output_dir="./figures"):
    """Generate all figures for the research report."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_results(results_dir)

    architectures = df["architecture"].unique().tolist()
    losses = df["loss"].unique().tolist()

    print("Generating report figures...")

    # 1. Training curves for each loss function
    for loss in losses:
        plot_training_curves(
            results_dir, architectures, loss,
            output_path=os.path.join(output_dir, f"training_curves_{loss}.png")
        )

    # 2. Ablation bar chart
    plot_ablation_bar_chart(
        df, output_path=os.path.join(output_dir, "ablation_bar_chart.png")
    )

    # 3. Loss function comparison for each architecture
    for arch in architectures:
        plot_loss_comparison(
            results_dir, arch, losses,
            output_path=os.path.join(output_dir, f"loss_comparison_{arch}.png")
        )

    # 4. Statistical analysis
    summary, pairwise = compute_statistics(df)

    # Save analysis results
    summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)
    if not pairwise.empty:
        pairwise.to_csv(os.path.join(output_dir, "pairwise_tests.csv"), index=False)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir", default="./figures")
    args = parser.parse_args()

    generate_report_figures(args.results_dir, args.output_dir)
