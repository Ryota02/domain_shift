# Domain Shift Experiments for Chest X-ray Classification

This repository contains experiments for binary chest X-ray classification under cross-domain shift.

The main task is **NORMAL vs PNEUMONIA** classification using multiple chest X-ray datasets collected from different domains.

## Project Overview

The purpose of this project is to evaluate how well chest X-ray classifiers generalize across datasets from different domains, and to investigate whether domain adaptation can reduce cross-domain performance degradation.

The main experimental settings are:

1. **Source-only baseline**
   - Train on source domain data.
   - Test on target domain data.

2. **Adversarial Domain Adaptation, ADA / DANN-style**
   - Train using source data and target adaptation images.
   - Use a Domain Discriminator with Gradient Reversal Layer, GRL.

3. **DALN: Discriminator-free Adversarial Learning Network**
   - Reuse the task-specific disease classifier as an implicit discriminator.
   - Train with source classification loss and Nuclear-norm Wasserstein Discrepancy, NWD.
   - No additional domain discriminator is used.

4. **Domain Discriminator analysis**
   - Check whether feature representations still contain domain-specific information.
   - Train only the Domain Discriminator while freezing the Feature Extractor.

5. **Synthetic domain shift experiment**
   - Apply artificial image transformations to the same images.
   - Check whether the Domain Discriminator reacts to visual style shifts.

6. **CNN-based domain classification**
   - Train CNN models such as ResNet18, ResNet50, and VGG16 to classify the dataset origin directly from images.
   - Visualize learned features using t-SNE.

7. **Yoshiken domain classification**
   - Analyze domain differences among KM, NIHCC, and NIOSH datasets.

---

## Main Workflow

```text
source datasets
  China / Doha / KM / NIHCC

target datasets
  Nigeria / NIOSH

        images
          |
          v
  CNN / DenseNet201 / ResNet
          |
          +--> disease classifier
          |
          +--> domain discriminator / domain classifier
          |
          +--> DALN classifier-based implicit discriminator
```

Example output files are saved under `outputs/`, such as:

```text
outputs/
├── source_only_china_to_nigeria_densenet201/
├── ada_china_doha_to_nigeria_densenet201/
├── daln/china_to_nigeria_resnet50/
├── cnn_domain_classifier/
├── yoshiken_domain_classifier/
└── figures/
```

---

## Features

- Train source-only baseline models for `NORMAL` vs `PNEUMONIA` classification.
- Train ADA models using a feature extractor, disease classifier, gradient reversal layer, and domain discriminator.
- Train DALN models without an additional domain discriminator.
- Evaluate target-domain performance using Accuracy, Recall, Sensitivity, Specificity, Precision, PPV, NPV, F1, ROC-AUC, and confusion matrix.
- Check whether learned features still contain domain information.
- Train CNN domain classifiers using ResNet18, ResNet50, and VGG16.
- Visualize learned feature spaces with t-SNE.
- Support Yoshiken domain analysis for `KM`, `NIHCC`, and `NIOSH`.

---

## Datasets

### Main Domain Adaptation Data

Expected dataset root:

```text
/media/share/Member/ueki/datasets/
├── ZhangLabData_binary_dataset/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── COVID-19_Radiography_binary_dataset_clean/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
└── nigerian_pneumonia_binary_dataset/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

### Domain names used in scripts

| Domain name | Dataset directory |
|---|---|
| `china` | `ZhangLabData_binary_dataset` |
| `doha` | `COVID-19_Radiography_binary_dataset_clean` |
| `nigeria` | `nigerian_pneumonia_binary_dataset` |

### Classification task

```text
NORMAL     → class 0
PNEUMONIA  → class 1
```

For disease classification experiments, the labels are:

```text
NORMAL     → 0
PNEUMONIA  → 1
```

For domain classification experiments, the labels are not disease labels.  
They are dataset-origin labels such as:

```text
china   → domain label 0
doha    → domain label 1
nigeria → domain label 2
```

The exact domain label mapping depends on the order passed to the script or config.

---

## Yoshiken Data

The Yoshiken Data domains are:

```text
km
nihcc
niosh
```

Expected raw dataset root:

```text
/media/share/Member/ueki/datasets/
├── KM_dicom_dataset/
├── nih/
└── NIOSH_practice/
    └── 画像/
```

The current known data summary is:

| Dataset | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| NIHCC | 90 | 0 | 90 |
| KM | 4 | 91 | 95 |
| NIOSH | 23 | 28 | 51 |
| Total | 117 | 119 | 236 |

If using the prepared balanced dataset, the expected structure is:

```text
/media/share/Member/ueki/datasets/Yoshiken_50_50/
├── train/
│   ├── km/
│   ├── nihcc/
│   └── niosh/
├── val/
│   ├── km/
│   ├── nihcc/
│   └── niosh/
└── test/
    ├── km/
    ├── nihcc/
    └── niosh/
```

---

## Environment

Tested environment:

```text
Python >= 3.10
PyTorch
Torchvision
scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
Pillow
PyYAML
```

GPU is recommended.

Known working GPU environment:

```text
GPU: Tesla V100-SXM2-32GB
CUDA: 11.8 compatible PyTorch environment
PyTorch: 2.7.1+cu118
```

Example installation:

```bash
pip install torch torchvision torchaudio
pip install scikit-learn numpy pandas matplotlib seaborn pillow pyyaml
```

If using a V100 GPU, make sure the installed PyTorch CUDA version is compatible with the GPU.  
For example, PyTorch with CUDA 11.8 is safer than newer CUDA builds that may not support V100 properly.

---

## Repository Structure

```text
domain_shift/
├── configs/
│   ├── source_only_china_to_nigeria_densenet201.yaml
│   ├── source_only_china_doha_to_nigeria_densenet201.yaml
│   ├── ada_china_to_nigeria_densenet201.yaml
│   ├── ada_china_doha_to_nigeria_densenet201.yaml
│   ├── ada_china_doha_to_nigeria_densenet201_convdisc.yaml
│   ├── daln_china_to_nigeria_resnet50.yaml
│   └── daln_china_doha_to_nigeria_resnet50.yaml
│
├── scripts/
│   ├── train_source_only.py
│   ├── train_ada.py
│   ├── train_daln.py
│   ├── check_real_domain_discriminator.py
│   ├── check_pairwise_domain_discriminator.py
│   ├── check_domain_discriminator_synthetic.py
│   ├── train_cnn_domain_classifier.py
│   ├── visualize_cnn_domain_tsne.py
│   ├── train_yoshiken_domain_classifier.py
│   └── visualize_yoshiken_domain_tsne.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── plots.py
│   ├── utils.py
│   ├── ada_model.py
│   ├── ada_train.py
│   ├── daln.py
│   ├── daln_train.py
│   ├── domain_data.py
│   ├── domain_models.py
│   ├── domain_train.py
│   ├── domain_evaluate.py
│   ├── domain_visualize.py
│   └── yoshiken_domain_data.py
│
├── outputs/
├── README.md
└── requirements.txt
```

---

## Usage

## 1. Source-only Baseline

The source-only setting trains only on source data and evaluates on the Nigeria target test set.

### China → Nigeria

```bash
python scripts/train_source_only.py \
  --config configs/source_only_china_to_nigeria_densenet201.yaml
```

### China + Doha → Nigeria

```bash
python scripts/train_source_only.py \
  --config configs/source_only_china_doha_to_nigeria_densenet201.yaml
```

---

## 2. ADA / DANN-style Training

ADA uses:

```text
Feature Extractor
Disease Classifier
Gradient Reversal Layer
Domain Discriminator
```

The training loss is composed of:

```text
Source classification loss
Target classification loss if supervised target labels are used
Domain adversarial loss
```

### China → Nigeria

```bash
python scripts/train_ada.py \
  --config configs/ada_china_to_nigeria_densenet201.yaml
```

### China + Doha → Nigeria

```bash
python scripts/train_ada.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml
```

### China + Doha → Nigeria with CNN-type domain discriminator

```bash
python scripts/train_ada.py \
  --config configs/ada_china_doha_to_nigeria_densenet201_convdisc.yaml
```

For two-domain ADA, ideal domain discriminator behavior after adaptation is:

```text
Overall Domain Accuracy ≈ 0.5
Source Domain Accuracy  ≈ 0.5
Target Domain Accuracy  ≈ 0.5
```

If the overall domain accuracy is close to 0.5 but source and target accuracies are extremely imbalanced, domain alignment should not be considered successful.

---

## 3. DALN Training

DALN stands for Discriminator-free Adversarial Learning Network.

Unlike DANN-style ADA, DALN does not use an additional domain discriminator.  
Instead, the task-specific disease classifier is reused as an implicit discriminator through Nuclear-norm Wasserstein Discrepancy, NWD.

DALN uses:

```text
Feature Extractor
Task-specific Disease Classifier
Gradient Reversal Layer
Nuclear-norm Wasserstein Discrepancy
```

The training loss is:

```text
Total Loss = Source Classification Loss + lambda * DALN Transfer Loss
```

### China → Nigeria

```bash
python scripts/train_daln.py \
  --config configs/daln_china_to_nigeria_resnet50.yaml
```

### China + Doha → Nigeria

```bash
python scripts/train_daln.py \
  --config configs/daln_china_doha_to_nigeria_resnet50.yaml
```

Typical DALN output files:

```text
outputs/daln/china_to_nigeria_resnet50/
├── daln_resnet50.json
├── daln_resnet50.pth
├── final_test_metrics.json
├── final_test_confusion_matrix.csv
├── daln_loss_curves.png
├── daln_accuracy_curves.png
└── daln_auc_curve.png
```

---

## 4. Domain Discriminator Evaluation After ADA

After ADA training, this experiment evaluates how well the Domain Discriminator can distinguish source and target features.

Example output:

```text
Domain Acc
Source Domain Acc
Target Domain Acc
Pred Source Ratio
Pred Target Ratio
Confusion Matrix
```

Interpretation:

| Metric | Meaning |
|---|---|
| Domain Acc | Overall source/target domain classification accuracy |
| Source Domain Acc | Accuracy for classifying source images as source |
| Target Domain Acc | Accuracy for classifying target images as target |
| Pred Source Ratio | Ratio of predictions assigned to source domain |
| Pred Target Ratio | Ratio of predictions assigned to target domain |

In ADA, a Domain Accuracy close to 0.5 can indicate domain-invariant features.  
However, it must be interpreted together with Source Domain Acc and Target Domain Acc.  
A value close to 0.5 is not always good if predictions are biased toward one domain.

---

## 5. Real Domain Discriminator Probe

This experiment checks whether fixed features still contain real domain information.

The Feature Extractor is frozen.  
A new Domain Discriminator is trained to classify source and target features.

```text
Feature Extractor: frozen
Classifier: frozen
Domain Discriminator: newly initialized and trained
GRL: not used
```

### ADA-before features

```bash
python scripts/check_real_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/source_pretrained_model.pth \
  --epochs 20 \
  --lr 1e-4 \
  --output_name real_domain_before_ada.json
```

### ADA-after features

```bash
python scripts/check_real_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth \
  --epochs 20 \
  --lr 1e-4 \
  --output_name real_domain_after_ada.json
```

If Domain Acc remains high after ADA, then domain-specific information is still present in the learned features.

---

## 6. Pairwise Domain Discriminator Probe

This experiment trains only the Domain Discriminator on fixed features to classify pairs or groups of domains.

Supported settings:

```text
Doha vs Nigeria
Doha vs China
China vs Nigeria
China vs Doha vs Nigeria
```

### Doha vs Nigeria

```bash
python scripts/check_pairwise_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth \
  --domains doha nigeria \
  --epochs 20 \
  --output_name domain_probe_after_doha_vs_nigeria.json
```

### Doha vs China

```bash
python scripts/check_pairwise_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth \
  --domains doha china \
  --epochs 20 \
  --output_name domain_probe_after_doha_vs_china.json
```

### China vs Nigeria

```bash
python scripts/check_pairwise_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth \
  --domains china nigeria \
  --epochs 20 \
  --output_name domain_probe_after_china_vs_nigeria.json
```

### China vs Doha vs Nigeria

```bash
python scripts/check_pairwise_domain_discriminator.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth \
  --domains china doha nigeria \
  --epochs 20 \
  --output_name domain_probe_after_china_doha_nigeria.json
```

This experiment helps identify which dataset pairs have the strongest domain gap.

---

## 7. Synthetic Domain Shift Experiment

This experiment checks whether the Domain Discriminator responds to artificial image style shifts.

The same image is used twice:

```text
original image → domain label 0
shifted image  → domain label 1
```

The shifted image may include:

```text
Brightness change
Contrast change
Gaussian blur
Gaussian noise
```

Example execution:

```bash
python scripts/check_domain_discriminator_synthetic.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml \
  --checkpoint outputs/ada_china_doha_to_nigeria_densenet201/checkpoints/best_model.pth
```

This experiment is a behavior check for the Domain Discriminator.  
It does not directly measure real China/Doha/Nigeria domain shift.

---

## 8. CNN Domain Classifier

This experiment trains a CNN to classify the dataset origin directly from images.

Input:

```text
Chest X-ray image
```

Output:

```text
Domain label: china / doha / nigeria
```

Disease labels are not used for training.

### ResNet18, three-domain classification

```bash
python scripts/train_cnn_domain_classifier.py \
  --domains china doha nigeria \
  --model resnet18 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --output_dir outputs/cnn_domain_classifier/resnet18_china_doha_nigeria
```

### ResNet50, three-domain classification

```bash
python scripts/train_cnn_domain_classifier.py \
  --domains china doha nigeria \
  --model resnet50 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --output_dir outputs/cnn_domain_classifier/resnet50_china_doha_nigeria
```

### VGG16, three-domain classification

```bash
python scripts/train_cnn_domain_classifier.py \
  --domains china doha nigeria \
  --model vgg16 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --output_dir outputs/cnn_domain_classifier/vgg16_china_doha_nigeria
```

The training data is created by splitting each domain's `train` folder into train and validation sets.

```text
train: 80% of each domain's train folder
val:   20% of each domain's train folder
test:  each domain's test folder
```

---

## 9. t-SNE Visualization for CNN Domain Classifier

After training a CNN domain classifier, use t-SNE to visualize the learned feature space.

### ResNet18 t-SNE

```bash
python scripts/visualize_cnn_domain_tsne.py \
  --domains china doha nigeria \
  --model resnet18 \
  --checkpoint outputs/cnn_domain_classifier/resnet18_china_doha_nigeria/cnn_domain_resnet18_china_vs_doha_vs_nigeria.pth \
  --split test \
  --output_dir outputs/cnn_domain_classifier/resnet18_china_doha_nigeria/tsne
```

Output:

```text
tsne_by_domain.png
tsne_by_disease.png
features.npy
domain_labels.npy
disease_labels.npy
tsne_features.npy
```

Interpretation:

- `tsne_by_domain.png` shows whether the model separates China, Doha, and Nigeria.
- `tsne_by_disease.png` shows whether the same feature space also separates NORMAL and PNEUMONIA.

---

## 10. Yoshiken Domain Classifier

This experiment analyzes whether the dataset origin can be predicted among Yoshiken-related datasets.

The Yoshiken Data domains are:

```text
km
nihcc
niosh
```

### ResNet18

```bash
python scripts/train_yoshiken_domain_classifier.py \
  --workdir /media/share/Member/ueki/datasets \
  --model resnet18 \
  --domains km nihcc niosh \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --output_dir outputs/yoshiken_domain_classifier/resnet18_3domain
```

### ResNet50

```bash
python scripts/train_yoshiken_domain_classifier.py \
  --workdir /media/share/Member/ueki/datasets \
  --model resnet50 \
  --domains km nihcc niosh \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --output_dir outputs/yoshiken_domain_classifier/resnet50_3domain
```

### VGG16

```bash
python scripts/train_yoshiken_domain_classifier.py \
  --workdir /media/share/Member/ueki/datasets \
  --model vgg16 \
  --domains km nihcc niosh \
  --epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --output_dir outputs/yoshiken_domain_classifier/vgg16_3domain
```

---

## 11. Yoshiken t-SNE

### ResNet18

```bash
python scripts/visualize_yoshiken_domain_tsne.py \
  --workdir /media/share/Member/ueki/datasets \
  --model resnet18 \
  --domains km nihcc niosh \
  --checkpoint outputs/yoshiken_domain_classifier/resnet18_3domain/yoshiken_domain_resnet18_km_vs_nihcc_vs_niosh.pth \
  --split test \
  --output_dir outputs/yoshiken_domain_classifier/resnet18_3domain/tsne
```

Use `--split all` if you want to visualize all images, including training images.

---

## Output Files

Typical output directories:

```text
outputs/
├── source_only_china_to_nigeria_densenet201/
│   ├── checkpoints/
│   ├── results/
│   └── figures/
│
├── ada_china_doha_to_nigeria_densenet201/
│   ├── checkpoints/
│   ├── results/
│   └── figures/
│
├── daln/
│   └── china_to_nigeria_resnet50/
│       ├── daln_resnet50.json
│       ├── daln_resnet50.pth
│       ├── final_test_metrics.json
│       ├── final_test_confusion_matrix.csv
│       ├── daln_loss_curves.png
│       └── daln_accuracy_curves.png
│
└── cnn_domain_classifier/
    └── resnet18_china_doha_nigeria/
        ├── cnn_domain_resnet18_china_vs_doha_vs_nigeria.json
        ├── cnn_domain_resnet18_china_vs_doha_vs_nigeria.pth
        └── tsne/
            ├── tsne_by_domain.png
            └── tsne_by_disease.png
```

---

## Evaluation Metrics

For disease classification, the following metrics are reported:

```text
Accuracy
Recall / Sensitivity
Specificity
Precision / PPV
NPV
F1-score
ROC-AUC
Confusion Matrix
```

For binary disease classification:

```text
NORMAL     → negative class
PNEUMONIA  → positive class
```

Confusion matrix format:

```text
                 Pred NORMAL    Pred PNEUMONIA
True NORMAL           TN              FP
True PNEUMONIA        FN              TP
```

---

## Important Notes

### Domain Accuracy is not disease classification accuracy

Domain Accuracy measures whether the model can identify dataset origin.  
It does not measure NORMAL/PNEUMONIA classification performance.

### Domain Accuracy near 0.5 is not always good

For binary domain classification, 0.5 can mean that domains are indistinguishable.  
However, it can also mean that the classifier is biased or undertrained.  
Always check:

```text
Domain Acc
Source Domain Acc
Target Domain Acc
Pred Source Ratio
Pred Target Ratio
Confusion Matrix
```

### CNN domain classification and ADA are different

The CNN domain classifier directly learns to classify dataset origin from images.  
ADA aims to remove domain-specific information while preserving disease classification ability.

### DALN and ADA are different

ADA / DANN-style training uses an additional Domain Discriminator.  
DALN does not use an additional Domain Discriminator.  
DALN reuses the task-specific classifier as an implicit discriminator through NWD.

### High disease classification performance does not always mean successful domain alignment

A model can achieve high disease classification performance while still retaining strong domain-specific information in its features.  
Therefore, disease classification metrics and domain analysis metrics should both be checked.

### Test data should not be used for model selection

Use validation data for best epoch selection.  
Use test data only for final evaluation.

### Yoshiken domain classification needs careful interpretation

For Yoshiken Data, NIHCC is mostly normal and KM is mostly pneumonia.  
Therefore, domain classification results may partly reflect disease distribution differences, not only acquisition-domain differences.

### Checkpoint compatibility

Existing MLP-based ADA checkpoints are not directly compatible with CNN-type domain discriminator checkpoints.  
If model architecture changes, old checkpoints may not load with `strict=True`.

---
# ChestXray8 One-vs-Rest Classification

This project performs one-vs-rest binary classification using a 12-class ChestXray8 image dataset.

For example, when Pneumonia is selected as the target class, the labels are converted as follows:

Pneumonia              -> 1
All other 11 classes   -> 0

When Nodule is selected:

Nodule                 -> 1
All other 11 classes   -> 0

You do not need to copy the images into new positive and negative folders. The existing 12-class ImageFolder structure is used directly, and the labels are converted to binary labels inside the dataset class.

## Expected Dataset Structure

This implementation assumes that each image belongs to exactly one class folder.

ChestXray8/
├── source/
│   ├── train/
│   │   ├── Atelectasis/
│   │   ├── Cardiomegaly/
│   │   ├── Consolidation/
│   │   ├── Edema/
│   │   ├── Effusion/
│   │   ├── Emphysema/
│   │   ├── Fibrosis/
│   │   ├── Infiltration/
│   │   ├── Mass/
│   │   ├── Nodule/
│   │   ├── Pneumonia/
│   │   └── Pneumothorax/
│   │
│   └── val/                  # Optional
│       └── ...
│
└── target/
    └── test/
        ├── Atelectasis/
        ├── Cardiomegaly/
        ├── Consolidation/
        ├── Edema/
        ├── Effusion/
        ├── Emphysema/
        ├── Fibrosis/
        ├── Infiltration/
        ├── Mass/
        ├── Nodule/
        ├── Pneumonia/
        └── Pneumothorax/

---
## What This Code Does

The overall workflow is:

12-class ImageFolder
        |
Select one target class
        |
Target class       -> 1
Other 11 classes   -> 0
        |
Train a CNN or ViT binary classifier
        |
Select the best model and threshold on validation data
        |
Evaluate on target/test

The code performs the following steps:

Loads the 12-class dataset with torchvision.datasets.ImageFolder.

Converts the selected target class to label 1.

Converts all other classes to label 0.

Trains the model using source/train.

Selects the best model using validation performance.

Selects the classification threshold using validation data.

Evaluates the final model on target/test.

Saves checkpoints, metrics, ROC curves, PR curves, training curves, and confusion matrices.

---
## Data Usage

When source/val is not specified, source/train is split into training and validation subsets using stratified sampling.

85% of source/train   -> Training
15% of source/train   -> Validation
100% of target/test   -> Final test

Example:

dataset:
  train: source/train
  test: target/test
  val_ratio: 0.15

When a separate validation folder exists, specify it directly:

dataset:
  train: source/train
  val: source/val
  test: target/test

Then the data usage becomes:

source/train   -> Training
source/val     -> Validation
target/test    -> Final test

source/test and target/train are not used unless the code is explicitly modified to include them.

---
## Project Structure

domain_shift/
├── configs/
│   ├── one_vs_rest_pneumonia_densenet201.yaml
│   └── one_vs_rest_pneumonia_vit_b16.yaml
│
├── scripts/
│   └── train_one_vs_rest.py
│
└── src/
    ├── data.py
    ├── plots.py
    ├── utils.py
    ├── one_vs_rest_data.py
    ├── one_vs_rest_model.py
    └── one_vs_rest_train.py

---
## How to use
```bash
python scripts/train_one_vs_rest.py --config configs/one_vs_rest_{model name}.yaml --target_class {target class}
```
---

## Author

- Ryota Ueki
- Project: Chest X-ray domain shift and domain adaptation experiments

---

## License

This repository is intended for internal research use.

---

## Updated Date

- 2026/07/30
