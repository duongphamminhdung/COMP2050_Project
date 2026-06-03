# Attention U-Net for Medical Image Segmentation

## Alignment with COMP2050 Programming Project Description

---

## 1. Requirement: "Study in detail one AI algorithm of your choice"

**Chosen algorithm:** U-Net (Ronneberger, Fischer & Brox, 2015)

U-Net is a convolutional neural network architecture originally designed for biomedical image segmentation. It uses an encoder-decoder structure with skip connections that preserve spatial information from earlier layers. The encoder extracts features at increasing levels of abstraction through repeated convolution and downsampling. The decoder reconstructs the spatial resolution through upsampling and concatenation with encoder features via skip connections.

This algorithm fits the assignment because:

- U-Net is a well-defined single algorithm with clear architectural principles (encoder-decoder, skip connections, double convolution blocks) that can be studied in depth.
- The architecture is simple enough to understand every component, yet powerful enough to produce state-of-the-art segmentation results.
- It has a rich ecosystem of published variations (Attention U-Net, U-Net++, ResUNet, etc.), providing a strong foundation for proposing meaningful modifications.
- The assignment states: "You are free to choose which algorithm you want to study. It does not have to be an algorithm that we have discussed in class." U-Net qualifies as a legitimate choice.

---

## 2. Requirement: "Propose your own variations of the algorithm"

**Primary variation:** Attention U-Net

The Attention U-Net introduces attention gates into the skip connections of the standard U-Net. Each attention gate uses a gating signal from the decoder to compute attention coefficients that weight the encoder features before they are concatenated in the decoder. This allows the network to focus on relevant structures and suppress irrelevant regions.

The attention gate mechanism works as follows:
1. The gating signal g (from the decoder, lower resolution) and skip connection feature x (from the encoder, higher resolution) are each passed through 1x1 convolutions to an intermediate dimension.
2. They are summed and passed through ReLU activation.
3. A 1x1 convolution followed by sigmoid produces attention coefficients in [0, 1].
4. The output is the element-wise product of the original skip features and the attention coefficients.

This variation is my own implementation (built from scratch in PyTorch) based on the concept from Oktay et al. (2018). While the idea originates from that paper, the specific implementation details, integration choices, and experimental evaluation are my own work.

**Additional variations:**

1. **ResUNet** -- Replaces the standard double-convolution blocks with residual blocks (Conv-BN-ReLU-Conv-BN + identity shortcut). This improves gradient flow and is inspired by He et al. (2016).

2. **Loss function ablation** -- Tests five loss functions: BCE, Dice Loss, Focal Loss, BCE+Dice combined, and Tversky Loss. Each loss function addresses different challenges (class imbalance, boundary accuracy, false negative penalty). This is algorithmic variation, not just hyperparameter tuning.

3. **Encoder backbone comparison** -- Compares training from scratch vs. using pretrained ImageNet encoders (ResNet18) via segmentation_models_pytorch. Tests whether transfer learning helps for medical images.

The assignment notes: "Note that your variations do not need to lead to better experimental results than the original algorithm. Your grade will be based on the quality of your work, not the numerical difference in the results." This project follows that spirit -- the variations are studied for understanding, not just for improvement.

---

## 3. Requirement: "Study the impact experimentally"

### Experimental Design

**Dataset:** Kvasir-SEG (Jha et al., 2020) -- 1,000 gastrointestinal polyp images with binary segmentation masks. Split: 700 train / 150 validation / 150 test.

**Ablation matrix:** 4 architectures x 3 loss functions = 12 experimental configurations. Each configuration is run with 3 different random seeds (42, 123, 456), producing 36 total training runs.

| # | Architecture | Loss Function | Seeds |
|---|---|---|---|
| 1 | U-Net (baseline) | BCE | 42, 123, 456 |
| 2 | U-Net (baseline) | Dice | 42, 123, 456 |
| 3 | U-Net (baseline) | BCE + Dice | 42, 123, 456 |
| 4 | Attention U-Net | BCE | 42, 123, 456 |
| 5 | Attention U-Net | Dice | 42, 123, 456 |
| 6 | Attention U-Net | BCE + Dice | 42, 123, 456 |
| 7 | ResUNet | BCE | 42, 123, 456 |
| 8 | ResUNet | Dice | 42, 123, 456 |
| 9 | ResUNet | BCE + Dice | 42, 123, 456 |
| 10 | U-Net + ResNet18 encoder | BCE | 42, 123, 456 |
| 11 | U-Net + ResNet18 encoder | Dice | 42, 123, 456 |
| 12 | U-Net + ResNet18 encoder | BCE + Dice | 42, 123, 456 |

**Metrics:** Dice Coefficient, IoU, Pixel Accuracy, Sensitivity, Specificity, Hausdorff Distance (HD95).

**What is being studied (per the assignment's suggestions):**

The assignment suggests studying the impact of:
- (i) own variations of the algorithm -- covered by the 4 architecture variants
- (ii) hyper-parameters -- learning rate, batch size, augmentation strategy can be added
- (iii) problem settings -- the loss function ablation covers this dimension
- (iv) different environments -- the encoder backbone comparison (scratch vs pretrained) and the dataset choice address this

---

## 4. Requirement: "Perform a statistical analysis"

Each configuration is run with 3 random seeds. For each metric, I report mean and standard deviation across the 3 runs.

For pairwise comparisons (e.g., Attention U-Net vs. baseline U-Net), I use the Wilcoxon signed-rank test, a non-parametric test appropriate for small sample sizes. I also report Cohen's d as an effect size measure.

Where multiple pairwise comparisons are made (e.g., comparing all 4 architectures), I apply Bonferroni correction to control for Type I error.

The assignment states: "You are strongly recommended to perform a statistical analysis to study your experimental results. At a bare minimum, you should consider multiple runs, so that the mean and the variance can be studied." This project exceeds the minimum by including significance testing and effect sizes alongside mean/variance reporting.

---

## 5. Requirement: "Use plots and tables, as in a research paper"

The report includes the following figures and tables:

**Figures:**
1. Architecture diagram comparing U-Net and Attention U-Net (shows the attention gate mechanism)
2. Training curves (loss and validation Dice vs. epoch for each architecture, with mean line and individual run lines)
3. Qualitative segmentation results (input image | ground truth | U-Net prediction | Attention U-Net prediction, for 4-6 examples)
4. Attention map visualizations (heatmap of attention gate outputs at each decoder level, showing where the network focuses)
5. Ablation bar chart (grouped bars showing Dice score per architecture, grouped by loss function, with error bars)
6. Loss function comparison plot (validation Dice curves for same architecture with different losses)

**Tables:**
1. Full results table -- rows = 12 configurations, columns = Dice, IoU, Accuracy, Sensitivity, Specificity, HD95 (mean +/- std)
2. Pairwise comparison table -- p-values from Wilcoxon tests between architectures
3. Dataset statistics table

Tools: Matplotlib for plots, LaTeX tables in the report.

---

## 6. Requirement: Research paper structure

The report follows the suggested structure:

| Section | Content |
|---|---|
| Title and name | "Attention U-Net for Medical Image Segmentation: An Ablation Study on Architecture and Loss Functions" |
| Abstract | One paragraph summarizing the study, methods, and key findings |
| Introduction | Motivation for medical image segmentation, why U-Net, why attention mechanisms, research questions |
| Related Work | U-Net, Attention U-Net, U-Net++, ResNet, loss functions for segmentation, Kvasir-SEG benchmark |
| Methodology | Dataset description, U-Net baseline, Attention U-Net architecture details, ResUNet, loss functions, training protocol, evaluation metrics, statistical analysis plan |
| Results | Quantitative results (tables), statistical tests, training curves, ablation analysis |
| Discussion | Which variations helped most and why, attention gate analysis, loss function insights, limitations, when attention might not help |
| Conclusion | Summary of findings, key takeaways, future work |
| Acknowledgements | Credit for any help received |
| References | Full bibliography |

Advised to use LaTeX (the assignment recommends it: "learning LaTeX would be helpful for your career").

---

## 7. Requirement: Implementation quality and effort

**Approach:** Hybrid -- build U-Net and Attention U-Net from scratch in PyTorch. Use segmentation_models_pytorch only for the pretrained encoder experiment.

The from-scratch implementation covers:
- Dataset class with augmentation pipeline
- U-Net model (~100 lines)
- AttentionGate module (~30 lines)
- Attention U-Net (~120 lines)
- ResUNet (~110 lines)
- Five loss functions (BCE, Dice, Focal, BCE+Dice, Tversky)
- Metrics computation (Dice, IoU, pixel accuracy, sensitivity, specificity, HD95)
- Attention map visualization (hooking into attention gate outputs)
- Config-driven training script (architecture, loss, seed as arguments)
- Experiment runner script

The assignment states three valid approaches:
1. Start with existing implementation, then work on variations -- valid
2. Use AI for initial standard algorithm, then implement variations yourself -- valid
3. Fully implement from scratch -- valid

This project primarily follows approach 3 (from scratch) for the core models and approach 1 for the pretrained encoder experiment.

---

## 8. Requirement: Statement file

A statement.pdf will be included detailing:
- Which parts of the code were written from scratch (U-Net, Attention U-Net, ResUNet, loss functions, training loop)
- Which parts use existing libraries (segmentation_models_pytorch for pretrained encoder, torchvision for augmentation, scipy for Hausdorff distance)
- Which parts were assisted by AI tools (if applicable)
- Which parts are entirely my own work (attention gate implementation, experiment design, analysis)

Code comments will clearly mark each category.

---

## 9. Grading Alignment

### Writing and Presentation (20%)

- LaTeX-formatted research paper with proper structure
- Architecture diagrams (U-Net and Attention U-Net schematics)
- Attention map visualizations (compelling visual evidence)
- Side-by-side segmentation result comparisons
- Properly formatted tables with captions
- Clear, technical writing with appropriate citations

### Creativity (20%)

- Attention U-Net as primary architectural variation
- Two-factor ablation study (architecture x loss function) rather than single-dimension
- Attention gate visualization to explain WHY improvements occur
- Cross-architecture comparison (scratch vs. pretrained encoder)
- Choice of medical imaging domain (practical application)

### Implementation Quality and Effort (30%)

- U-Net and Attention U-Net built from scratch in PyTorch
- Clean modular code structure (models, losses, data, utils, train)
- Three distinct architecture implementations (U-Net, Attention U-Net, ResUNet)
- Five loss function implementations
- Config-driven experiment runner supporting all ablation combinations
- Attention map extraction and visualization pipeline

### Experimental Evaluation (30%)

- 36 training runs (12 configurations x 3 seeds)
- 6 evaluation metrics
- Statistical significance testing (Wilcoxon signed-rank)
- Effect size reporting (Cohen's d)
- Multiple comparison correction (Bonferroni)
- Ablation study across two dimensions (architecture and loss function)
- Training curves with confidence bands

---

## 10. Framework and Environment Justification

The assignment suggests frameworks like Gymnasium, OpenSpiel, Neural MMO, etc. These are primarily reinforcement learning environments.

The assignment also states: "you can use frameworks that are not in this list, as well as build your own environment" and "you can do your project in the programming language of your choice."

For this computer vision project, the relevant frameworks are:

- **PyTorch** -- deep learning framework (the industry standard for CV research)
- **segmentation_models_pytorch** -- for pretrained encoder comparison only
- **Gymnasium** -- not applicable (this is a supervised learning project, not RL)
- **Google Colab** -- compute environment with free T4 GPU

The "environment" in this project is the Kvasir-SEG dataset and the segmentation task itself, analogous to how Gymnasium provides environments for RL agents.

---

## 11. Key References

1. Ronneberger, O., Fischer, P., Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI. arXiv:1505.04597
2. Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." MIDL. arXiv:1804.03999
3. Zhou, Z., et al. (2018). "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." DLMIA. arXiv:1807.10165
4. Milletari, F., et al. (2016). "V-Net: Fully Convolutional 3D MR Image Segmentation." 3DV.
5. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
6. Lin, T.Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
7. Abraham, N., Khan, N.M. (2019). "A Novel Focal Tversky Loss Function for Lesion Segmentation." ISBI.
8. Sudre, C.H., et al. (2017). "Generalised Dice Overlap as a Deep Learning Loss Function." DLMIA.
9. Jha, D., et al. (2020). "Kvasir-SEG: A Segmentation Polyp Dataset." MMM.
10. Demsar, J. (2006). "Statistical Comparisons of Classifiers over Multiple Data Sets." JMLR.
