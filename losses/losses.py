# [OWN WORK] Loss functions for medical image segmentation
# Implementations: BCE, Dice, Focal, BCE+Dice Combined, Tversky

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Computes 1 - Dice coefficient between predictions and targets.
    The Dice coefficient measures overlap: 2*|A∩B| / (|A|+|B|).

    Args:
        smooth: Smoothing factor to avoid division by zero.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation.

    Down-weights well-classified examples to focus on hard ones.
    FL(p_t) = alpha * (1 - p_t)^gamma * BCE(p_t)

    Reference: Lin et al., 2017 (arXiv:1708.02002)

    Args:
        alpha: Weighting factor for positive class (default: 0.25).
        gamma: Focusing parameter; higher values focus more on hard examples (default: 2.0).
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class CombinedBCELoss(nn.Module):
    """Combined BCE + Dice loss.

    BCE provides stable gradients, Dice handles class imbalance.
    Loss = BCE(logits, targets) + lambda * DiceLoss(logits, targets)

    Args:
        lambda_dice: Weight for Dice loss component (default: 1.0).
        smooth: Smoothing factor for Dice computation.
    """

    def __init__(self, lambda_dice=1.0, smooth=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.lambda_dice * self.dice(logits, targets)


class TverskyLoss(nn.Module):
    """Tversky loss for binary segmentation.

    Generalization of Dice loss that allows separate weighting of
    false positives and false negatives.

    TL = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    Setting alpha > beta penalizes false negatives more (useful for
    small target structures).

    Reference: Abraham & Khan, 2019 (ISBI 2019)

    Args:
        alpha: Weight for false positives (default: 0.7).
        beta: Weight for false negatives (default: 0.3).
        smooth: Smoothing factor.
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


def get_loss(name, **kwargs):
    """Factory function to get loss function by name.

    Args:
        name: One of 'bce', 'dice', 'focal', 'bce_dice', 'tversky'.
        **kwargs: Additional arguments passed to the loss constructor.

    Returns:
        Loss module.
    """
    losses = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "bce_dice": CombinedBCELoss,
        "tversky": TverskyLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name](**kwargs)
