# Domain Shift Experiments for Chest X-ray Classification

Chest X-ray binary classification and domain adaptation experiments for evaluating dataset shift across multiple medical image domains.

This repository contains code for:

- source-only baseline training
- adversarial domain adaptation, ADA / DANN-style training
- domain discriminator analysis
- CNN-based domain classification with ResNet18, ResNet50, and VGG16
- t-SNE visualization of learned domain features
- Yoshiken domain classification experiments

## DEMO

The main workflow is:

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
```

Example output files are saved under `outputs/`, such as:

```text
outputs/
├── cnn_domain_classifier/
├── yoshiken_domain_classifier/
├── ada_china_doha_to_nigeria_densenet201/
└── figures/
```

## Features

- Train source-only baseline models for `NORMAL` vs `PNEUMONIA` classification.
- Train ADA models using a feature extractor, disease classifier, gradient reversal layer, and domain discriminator.
- Evaluate target-domain performance using Accuracy, Recall, Specificity, Precision, NPV, F1, and ROC-AUC.
- Check whether learned features still contain domain information.
- Train CNN domain classifiers using ResNet18, ResNet50, and VGG16.
- Visualize learned feature spaces with t-SNE.
- Support Yoshiken Data domain analysis for `KM`, `NIHCC`, and `NIOSH`.

## Dataset Structure

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

### Yoshiken Data

Expected dataset root:

```text
/media/share/Member/ueki/datasets/
├── KM_dicom_dataset/
├── nih/
└── NIOSH_practice/
    └── 画像/
```

For Yoshiken domain classification, the images do not need to be split into `train`, `validation`, and `test` directories beforehand. The script internally splits each domain into:

```text
train : 70%
val   : 15%
test  : 15%
```

## Requirement

Tested environment:

```text
Python 3.x
PyTorch
TorchVision
scikit-learn
NumPy
Pandas
Matplotlib
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

## Installation

Create and activate your Python environment, then install the required packages.

```bash
pip install torch torchvision torchaudio
pip install scikit-learn numpy pandas matplotlib pillow pyyaml
```

If you use a CUDA environment, install the PyTorch build matching your CUDA version.

## Usage

### 1. Train Source-only Baseline

Example:

```bash
python scripts/train_source_only.py \
  --config configs/source_only_china_to_nigeria_densenet201.yaml
```

For China + Doha to Nigeria:

```bash
python scripts/train_source_only.py \
  --config configs/source_only_china_doha_to_nigeria_densenet201.yaml
```

### 2. Train ADA Model

Example:

```bash
python scripts/train_ada.py \
  --config configs/ada_china_doha_to_nigeria_densenet201.yaml
```

If using a CNN-type domain discriminator:

```bash
python scripts/train_ada.py \
  --config configs/ada_china_doha_to_nigeria_densenet201_convdisc.yaml
```

The ADA model is based on:

```text
Feature Extractor
Disease Classifier
Gradient Reversal Layer
Domain Discriminator
```

For two-domain ADA, ideal domain discriminator behavior after adaptation is:

```text
Overall Domain Accuracy ≈ 0.5
Source Domain Accuracy  ≈ 0.5
Target Domain Accuracy  ≈ 0.5
```

If the overall domain accuracy is close to 0.5 but source and target accuracies are extremely imbalanced, domain alignment should not be considered successful.

### 3. CNN Domain Classifier

This experiment checks whether the dataset origin can be predicted directly from images.

#### ResNet18

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

#### ResNet50

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

#### VGG16

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

### 4. t-SNE for CNN Domain Classifier

Example:

```bash
python scripts/visualize_cnn_domain_tsne.py \
  --domains china doha nigeria \
  --model resnet18 \
  --checkpoint outputs/cnn_domain_classifier/resnet18_china_doha_nigeria/cnn_domain_resnet18_china_vs_doha_vs_nigeria.pth \
  --split test \
  --output_dir outputs/cnn_domain_classifier/resnet18_china_doha_nigeria/tsne
```

### 5. Yoshiken Domain Classifier

The Yoshiken Data domains are:

```text
km
nihcc
niosh
```

The current known data summary is:

| Dataset | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| NIHCC | 90 | 0 | 90 |
| KM | 4 | 91 | 95 |
| NIOSH | 23 | 28 | 51 |
| Total | 117 | 119 | 236 |

Since the data are not pre-split into train, validation, and test, the script internally splits each domain.

#### ResNet18

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

#### ResNet50

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

#### VGG16

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

### 6. Yoshiken t-SNE

#### ResNet18

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

## Directory Structure

```text
.
├── configs/
│   ├── source_only_china_to_nigeria_densenet201.yaml
│   ├── source_only_china_doha_to_nigeria_densenet201.yaml
│   ├── ada_china_doha_to_nigeria_densenet201.yaml
│   └── ada_china_doha_to_nigeria_densenet201_convdisc.yaml
│
├── scripts/
│   ├── train_source_only.py
│   ├── train_ada.py
│   ├── train_cnn_domain_classifier.py
│   ├── visualize_cnn_domain_tsne.py
│   ├── train_yoshiken_domain_classifier.py
│   └── visualize_yoshiken_domain_tsne.py
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── utils.py
│   ├── ada_model.py
│   ├── domain_data.py
│   ├── domain_models.py
│   ├── domain_train.py
│   ├── domain_evaluate.py
│   ├── domain_visualize.py
│   └── yoshiken_domain_data.py
│
├── outputs/
└── README.md
```

## Notes

- Domain classifier accuracy can be high because of true acquisition-domain differences, disease-label imbalance, or both.
- For Yoshiken Data, NIHCC is mostly normal and KM is mostly pneumonia. Therefore, domain classification results may partly reflect disease distribution differences.
- In ADA, high disease classification performance does not automatically mean successful domain alignment.
- For two-domain ADA, domain discriminator accuracy should be interpreted together with source-domain accuracy and target-domain accuracy.
- Test data should not be used for model selection. Use validation data for best epoch selection and test data only for final evaluation.
- Existing MLP-based ADA checkpoints are not directly compatible with CNN-type domain discriminator checkpoints.

## Author

- Ryota Ueki
- Project: Chest X-ray domain shift and domain adaptation experiments

## License

This repository is intended for internal research use.