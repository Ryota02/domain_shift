# Domain Shift / Domain Adaptation for Chest X-ray Classification

This repository contains experiments for binary chest X-ray classification under cross-domain shift.
The main task is **NORMAL vs PNEUMONIA** classification, using multiple CXR datasets collected from different domains.

## Project Overview

The purpose of this project is to evaluate how well chest X-ray classifiers generalize across datasets from different domains, and to investigate whether adversarial domain adaptation can reduce cross-domain performance degradation.

The main experimental settings are:

1. **Source-only baseline**
   - Train on source domain data.
   - Test on target domain data.

2. **Supervised Adversarial Domain Adaptation (ADA)**
   - Train using source data and a small labeled target adaptation set.
   - Use a Domain Discriminator with Gradient Reversal Layer (GRL).

3. **Domain Discriminator analysis**
   - Check whether feature representations still contain domain-specific information.
   - Train only the Domain Discriminator while freezing the Feature Extractor.

4. **Synthetic domain shift experiment**
   - Apply artificial image transformations to the same images.
   - Check whether the Domain Discriminator reacts to visual style shifts.

5. **CNN-based domain classification**
   - Train CNN models such as ResNet18 and VGG16 to classify the dataset origin directly from images.
   - Visualize learned features using t-SNE.

---

## Datasets

The project assumes the following dataset directory structure:

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

For domain classification experiments, the labels are not disease labels. They are domain labels such as:

```text
china   → domain label 0
doha    → domain label 1
nigeria → domain label 2
```

The exact domain label mapping depends on the order passed to `--domains`.

---

## Repository Structure

```text
domain_shift/
├── configs/
│   ├── source_only_china_to_nigeria_densenet201.yaml
│   ├── source_only_china_doha_to_nigeria_densenet201.yaml
│   ├── ada_china_to_nigeria_densenet201.yaml
│   └── ada_china_doha_to_nigeria_densenet201.yaml
│
├── scripts/
│   ├── train_source_only.py
│   ├── train_ada.py
│   ├── check_real_domain_discriminator.py
│   ├── check_pairwise_domain_discriminator.py
│   ├── check_domain_discriminator_synthetic.py
│   ├── train_cnn_domain_classifier.py
│   └── visualize_cnn_domain_tsne.py
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
│   ├── domain_data.py
│   ├── domain_models.py
│   ├── domain_train.py
│   ├── domain_evaluate.py
│   └── domain_visualize.py
│
├── outputs/
├── docs/
│   └── experiment_log.md
├── README.md
└── requirements.txt
```

---

## Environment

Recommended environment:

```text
Python >= 3.10
PyTorch
Torchvision
scikit-learn
matplotlib
seaborn
PyYAML
numpy
```

Example installation:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pyyaml numpy
```

If using a V100 GPU, make sure the installed PyTorch CUDA version is compatible with the GPU.
For example, PyTorch with CUDA 11.8 is safer than newer CUDA builds that may not support V100 properly.

---

## 1. Source-only Baseline

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

The source-only setting trains only on source data and evaluates on the Nigeria target test set.

---

## 2. Supervised ADA Training

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

ADA uses:

```text
Feature Extractor
Classifier
Gradient Reversal Layer
Domain Discriminator
```

The training loss is composed of:

```text
Source classification loss
Target classification loss
Domain adversarial loss
```

---

## 3. Domain Discriminator Evaluation After ADA

After ADA training, the script evaluates how well the Domain Discriminator can distinguish source and target features.

Example output:

```text
Domain Acc
Source Domain Acc
Target Domain Acc
```

Interpretation:


| Metric | Meaning |
|---|---|
| Domain Acc | Overall source/target domain classification accuracy |
| Source Domain Acc | Accuracy for classifying source images as source |
| Target Domain Acc | Accuracy for classifying target images as target |

In ADA, a Domain Accuracy close to 0.5 can indicate domain-invariant features. However, it must be interpreted together with Source Domain Acc and Target Domain Acc. A value close to 0.5 is not always good if predictions are biased toward one domain.

---

## 4. Real Domain Discriminator Probe

This experiment checks whether fixed features still contain real domain information.

The Feature Extractor is frozen. A new Domain Discriminator is trained to classify:

```text
Source = China / Doha
Target = Nigeria
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

This is not ADA training. It is a probe experiment.

```text
Feature Extractor: frozen
Classifier: frozen
Domain Discriminator: newly initialized and trained
GRL: not used
```

If Domain Acc remains high after ADA, then domain-specific information is still present in the learned features.

---

## 5. Pairwise Domain Discriminator Probe

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

## 6. Synthetic Domain Shift Experiment

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

This experiment is a behavior check for the Domain Discriminator. It does not directly measure real China/Doha/Nigeria domain shift.

---

## 7. CNN Domain Classifier

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
The `test` folder is used only for final evaluation.

```text
train: 80% of each domain's train folder
val:   20% of each domain's train folder
test:  each domain's test folder
```

---

## 8. t-SNE Visualization for CNN Domain Classifier

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

## Output Files

Typical output directories:

```text
outputs/
├── ada_china_doha_to_nigeria_densenet201/
│   ├── checkpoints/
│   ├── results/
│   └── figures/
│
└── cnn_domain_classifier/
    ├── resnet18_china_doha_nigeria/
    │   ├── cnn_domain_resnet18_china_vs_doha_vs_nigeria.json
    │   ├── cnn_domain_resnet18_china_vs_doha_vs_nigeria.pth
    │   └── tsne/
    │       ├── tsne_by_domain.png
    │       └── tsne_by_disease.png
```

---

## Important Notes

### Domain Accuracy is not disease classification accuracy

Domain Accuracy measures whether the model can identify dataset origin.
It does not measure NORMAL/PNEUMONIA classification performance.

### Domain Accuracy near 0.5 is not always good

For binary domain classification, 0.5 can mean that domains are indistinguishable. However, it can also mean that the classifier is biased or undertrained. Always check:

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

---

## Recommended Experiment Order

1. Run source-only baseline.
2. Run supervised ADA.
3. Evaluate Domain Discriminator behavior.
4. Run pairwise domain probes.
5. Run synthetic domain shift experiment.
6. Train CNN domain classifier.
7. Visualize CNN domain features with t-SNE.
8. Compare whether domain information remains before and after ADA.

## Updated Date
* 2026/7/10 