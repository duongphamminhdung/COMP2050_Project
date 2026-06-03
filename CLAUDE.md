# COMP2050 Programming Project - Attention U-Net Medical Segmentation

## Project
University assignment (VinUniversity, AI program): study U-Net, propose
variations, evaluate experimentally. See Project-Alignment.md for full
assignment mapping and grading criteria.

## Stack
Python, PyTorch, segmentation_models_pytorch (pretrained encoders only).
Runs on Google Colab (T4 GPU). Not a git repo.

## Dataset
Kvasir-SEG (1,000 polyp images). Downloaded via Kaggle in Colab.
Kaggle auth uses Colab secret KAGGLE_API_TOKEN (see global CLAUDE.md).

## Code Conventions
All source files start with a comment tag: [OWN WORK], [EXISTING SOURCE],
or [AI ASSISTED]. Required by the assignment (statement.pdf must identify
code origins). Models built from scratch. smp used for pretrained encoder only.

## Experiments
4 architectures x 3 losses x 3 seeds = 36 runs. See run_experiments.py.
Results go in ./results/. Figures in ./figures/.

## Deliverables
report.pdf (LaTeX research paper), code.zip, statement.pdf
Grading: Writing 20%, Creativity 20%, Implementation 30%, Experiments 30%
