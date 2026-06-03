# [OWN WORK] Config-driven training script for segmentation experiments
# Supports all architecture and loss function combinations via CLI args

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import get_model
from losses import get_loss
from data.dataset import get_dataloaders
from utils.metrics import compute_all_metrics


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate on a dataset. Returns avg loss and all metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        num_batches += 1

        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(all_preds, all_targets, threshold)
    metrics["loss"] = total_loss / num_batches

    return metrics


def train(config):
    """Main training loop with early stopping.

    Args:
        config: Namespace or dict with keys: architecture, loss, seed, epochs,
                lr, batch_size, data_root, patience, output_dir.

    Returns:
        Dictionary with best validation metrics and test metrics.
    """
    if isinstance(config, dict):
        config = argparse.Namespace(**config)

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dataloaders = get_dataloaders(config.data_root, batch_size=config.batch_size, seed=config.seed)

    # Model
    model = get_model(config.architecture).to(device)

    # Loss and optimizer
    criterion = get_loss(config.loss)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Tracking
    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    # Output directory for this run
    run_name = f"{config.architecture}_{config.loss}_seed{config.seed}"
    run_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_metrics = evaluate(model, dataloaders["val"], criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        # Early stopping on validation Dice
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"Val IoU: {val_metrics['iou']:.4f} | "
                  f"Best: {best_val_dice:.4f} @ ep{best_epoch} | "
                  f"Time: {elapsed:.0f}s")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best val Dice: {best_val_dice:.4f} @ epoch {best_epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=True))
    test_metrics = evaluate(model, dataloaders["test"], criterion, device)

    total_time = time.time() - start_time

    # Save results
    results = {
        "architecture": config.architecture,
        "loss": config.loss,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "total_time_sec": round(total_time, 1),
        **{f"test_{k}": round(v, 6) if isinstance(v, float) else v for k, v in test_metrics.items()},
        "best_val_dice": round(best_val_dice, 6),
    }

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTest Results ({run_name}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Total time: {total_time:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--architecture", type=str, default="unet",
                        choices=["unet", "attention_unet", "resunet", "smp_resnet18"])
    parser.add_argument("--loss", type=str, default="bce_dice",
                        choices=["bce", "dice", "focal", "bce_dice", "tversky"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
