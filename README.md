# Attention U-Net for Medical Image Segmentation

COMP2050 Programming Project — VinUniversity Artificial Intelligence Program

## Overview

This project studies the U-Net convolutional neural network architecture for biomedical image segmentation. We implement the baseline U-Net (Ronneberger et al., 2015), propose variations including Attention U-Net and Residual U-Net, and evaluate them on the Kvasir-SEG polyp segmentation dataset through a rigorous ablation study across architecture and loss function dimensions.

## Dataset

**Kvasir-SEG** — 1,000 gastrointestinal polyp images with binary segmentation masks.

| Split | Images |
|-------|--------|
| Train | 700 |
| Validation | 150 |
| Test | 150 |

Downloaded via Kaggle. Split is deterministic (controlled by random seed).

Reference: Jha et al., "Kvasir-SEG: A Segmentation Polyp Dataset", MMM 2020.

## Models

All models are implemented from scratch in PyTorch.

| Model | Description |
|-------|-------------|
| **U-Net** | Baseline encoder-decoder with skip connections (Ronneberger et al., 2015) |
| **Attention U-Net** | Adds attention gates on skip connections to suppress irrelevant features (Oktay et al., 2018) |
| **ResUNet** | Replaces double-conv blocks with residual blocks for better gradient flow |
| **U-Net + ResNet18** | Uses pretrained ImageNet encoder via segmentation_models_pytorch |

## Loss Functions

| Loss | Purpose |
|------|---------|
| BCE | Standard binary cross-entropy baseline |
| Dice | Handles class imbalance, optimizes overlap directly |
| BCE + Dice | Combined: stable gradients + overlap optimization |

## Experiments

Full ablation: **4 architectures x 3 losses x 3 seeds = 36 training runs**

Each run: 50 epochs, AdamW (lr=1e-4), CosineAnnealing scheduler, early stopping (patience=10 on val Dice).

**Metrics:** Dice Coefficient, IoU, Pixel Accuracy, Sensitivity, Specificity, Hausdorff Distance (HD95)

**Statistical analysis:** Wilcoxon signed-rank test, Cohen's d effect size, Bonferroni correction.

## Project Structure

```
models/
  unet.py              # Baseline U-Net
  attention_modules.py  # Attention gate mechanism
  attention_unet.py     # Attention U-Net
  resunet.py            # Residual U-Net
losses/
  losses.py            # BCE, Dice, Focal, BCE+Dice, Tversky losses
data/
  dataset.py           # Kvasir-SEG dataset class
utils/
  metrics.py           # Evaluation metrics (Dice, IoU, HD95, etc.)
  visualization.py     # Training curves, attention maps, ablation charts
train.py               # Config-driven training with early stopping
run_experiments.py     # Full ablation runner (36 experiments)
evaluate.py            # Statistical analysis and figure generation
notebooks/
  01_setup_data.ipynb  # Install deps, download dataset
  02_train.ipynb       # Train single config, visualize results
  03_run_all_experiments.ipynb  # Run full ablation
  04_analysis.ipynb    # Statistical tests, report figures
```

## Quick Start (Google Colab)

1. Create a new Colab notebook with **T4 GPU** runtime
2. Clone and install:

```python
!git clone https://github.com/duongphamminhdung/COMP2050_Project.git
%cd COMP2050_Project
!pip install -q segmentation-models-pytorch scipy scikit-learn pandas tqdm
```

3. Setup Kaggle credentials and download data (see notebook 01)
4. Run training:

```python
from train import train

results = train({
    "architecture": "attention_unet",
    "loss": "bce_dice",
    "seed": 42,
    "epochs": 50,
    "lr": 1e-4,
    "batch_size": 16,
    "patience": 10,
    "data_root": "./data",
    "output_dir": "./results",
})
```

## References

1. Ronneberger, O., Fischer, P., Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
2. Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." MIDL.
3. Zhou, Z., et al. (2018). "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
5. Milletari, F., et al. (2016). "V-Net: Fully Convolutional 3D MR Image Segmentation." 3DV.
6. Lin, T.Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
7. Jha, D., et al. (2020). "Kvasir-SEG: A Segmentation Polyp Dataset." MMM.
