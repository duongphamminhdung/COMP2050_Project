# [OWN WORK] Visualization utilities for segmentation experiments
# Generates all figures needed for the research report

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def plot_training_curves(results_dir, architectures, loss_name, output_path=None):
    """Plot training loss and validation Dice curves for multiple architectures.

    Shows individual seed runs as thin lines with mean as thick line.

    Args:
        results_dir: Path to results directory containing run subdirectories.
        architectures: List of architecture names to plot.
        loss_name: Which loss function runs to plot.
        output_path: If provided, save figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"unet": "#1f77b4", "attention_unet": "#ff7f0e", "resunet": "#2ca02c", "smp_resnet18": "#d62728"}

    for arch in architectures:
        color = colors.get(arch, None)
        runs_dir = glob.glob(os.path.join(results_dir, f"{arch}_{loss_name}_seed*"))
        all_losses = []
        all_dices = []

        for run_dir in sorted(runs_dir):
            history_path = os.path.join(run_dir, "history.json")
            if not os.path.exists(history_path):
                continue
            with open(history_path) as f:
                h = json.load(f)
            all_losses.append(h["train_loss"])
            all_dices.append(h["val_dice"])

            # Plot individual run as thin line
            ax1.plot(h["train_loss"], alpha=0.3, color=color, linewidth=0.8)
            ax2.plot(h["val_dice"], alpha=0.3, color=color, linewidth=0.8)

        if all_losses:
            min_len = min(len(r) for r in all_losses)
            mean_loss = np.mean([r[:min_len] for r in all_losses], axis=0)
            mean_dice = np.mean([r[:min_len] for r in all_dices], axis=0)
            ax1.plot(mean_loss, color=color, linewidth=2, label=arch)
            ax2.plot(mean_dice, color=color, linewidth=2, label=arch)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"Training Loss ({loss_name})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Dice")
    ax2.set_title(f"Validation Dice ({loss_name})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_segmentation_results(images, masks, predictions, model_names, num_examples=4, output_path=None):
    """Plot side-by-side segmentation results.

    Args:
        images: Tensor or array of input images, shape (N, C, H, W).
        masks: Tensor or array of ground truth masks, shape (N, 1, H, W).
        predictions: List of tensors/arrays, one per model, each shape (N, 1, H, W).
        model_names: List of model names for column headers.
        num_examples: Number of example rows to show.
        output_path: If provided, save figure.
    """
    num_models = len(predictions)
    ncols = 2 + num_models  # input + GT + predictions
    nrows = num_examples

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    for row in range(nrows):
        # Input image
        img = images[row] if isinstance(images, np.ndarray) else images[row].cpu().numpy()
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)
        # Denormalize ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Input" if row == 0 else "")
        axes[row, 0].axis("off")

        # Ground truth
        gt = masks[row].squeeze() if isinstance(masks, np.ndarray) else masks[row].cpu().squeeze().numpy()
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground Truth" if row == 0 else "")
        axes[row, 1].axis("off")

        # Predictions
        for col, (pred, name) in enumerate(zip(predictions, model_names)):
            p = pred[row].squeeze() if isinstance(pred, np.ndarray) else pred[row].cpu().squeeze().numpy()
            axes[row, 2 + col].imshow(p, cmap="gray", vmin=0, vmax=1)
            axes[row, 2 + col].set_title(name if row == 0 else "")
            axes[row, 2 + col].axis("off")

    plt.suptitle("Segmentation Results Comparison", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_attention_maps(image, attention_maps, output_path=None):
    """Visualize attention gate outputs as heatmaps.

    Args:
        image: Input image tensor, shape (C, H, W).
        attention_maps: List of 4 attention coefficient tensors from AttentionGate,
                        shapes (1, 1, H, W) from deepest to shallowest.
        output_path: If provided, save figure.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Original image
    img = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    level_names = ["Level 4 (deep)", "Level 3", "Level 2", "Level 1 (shallow)"]

    for i, att_map in enumerate(attention_maps):
        am = att_map.squeeze().cpu().numpy()  # (H, W)
        # Resize to match input if needed
        if am.shape != img.shape[:2]:
            from PIL import Image as PILImage
            am = np.array(PILImage.fromarray(am).resize((img.shape[1], img.shape[0]), PILImage.BILINEAR))

        axes[i + 1].imshow(img)
        axes[i + 1].imshow(am, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[i + 1].set_title(f"Attention {level_names[i]}")
        axes[i + 1].axis("off")

    plt.suptitle("Attention Gate Activation Maps", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_ablation_bar_chart(results_df, output_path=None):
    """Plot grouped bar chart: Dice per architecture, grouped by loss function.

    Args:
        results_df: DataFrame with columns: architecture, loss, test_dice.
        output_path: If provided, save figure.
    """
    architectures = results_df["architecture"].unique()
    losses = results_df["loss"].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(architectures))
    width = 0.8 / len(losses)
    colors = plt.cm.Set2(np.linspace(0, 1, len(losses)))

    for i, loss in enumerate(losses):
        subset = results_df[results_df["loss"] == loss]
        means = []
        stds = []
        for arch in architectures:
            arch_data = subset[subset["architecture"] == arch]["test_dice"]
            means.append(arch_data.mean())
            stds.append(arch_data.std())

        offset = (i - len(losses) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=loss,
                      color=colors[i], capsize=3, edgecolor="black", linewidth=0.5)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Architecture")
    ax.set_ylabel("Test Dice Coefficient")
    ax.set_title("Segmentation Performance: Architecture x Loss Function")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, rotation=15, ha="right")
    ax.legend(title="Loss Function")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_loss_comparison(results_dir, architecture, losses, output_path=None):
    """Plot validation Dice curves for different loss functions on the same architecture.

    Args:
        results_dir: Path to results directory.
        architecture: Architecture name to compare.
        losses: List of loss function names to compare.
        output_path: If provided, save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(losses)))

    for i, loss in enumerate(losses):
        runs_dir = glob.glob(os.path.join(results_dir, f"{architecture}_{loss}_seed*"))
        all_dices = []

        for run_dir in sorted(runs_dir):
            history_path = os.path.join(run_dir, "history.json")
            if not os.path.exists(history_path):
                continue
            with open(history_path) as f:
                h = json.load(f)
            all_dices.append(h["val_dice"])
            ax.plot(h["val_dice"], alpha=0.2, color=colors[i], linewidth=0.8)

        if all_dices:
            min_len = min(len(r) for r in all_dices)
            mean_dice = np.mean([r[:min_len] for r in all_dices], axis=0)
            std_dice = np.std([r[:min_len] for r in all_dices], axis=0)
            epochs = range(1, min_len + 1)

            ax.plot(epochs, mean_dice, color=colors[i], linewidth=2, label=loss)
            ax.fill_between(epochs, mean_dice - std_dice, mean_dice + std_dice,
                           color=colors[i], alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice Coefficient")
    ax.set_title(f"Loss Function Comparison ({architecture})")
    ax.legend(title="Loss Function")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
