# 🌿 Crop Weed Recognition for Smart Farming
### CNN-Based Multiclass Weed Species Classification — DeepWeeds Dataset

**Course:** DSC01 – Machine Learning 120A | Phase II Submission  
**Author:** Kuchi Harika (Matriculation: 61119426)  
**Date:** June 7, 2026

---

## Overview

Weeds are one of the most significant threats to agricultural productivity worldwide, competing with crops for sunlight, water, and nutrients. This project develops deep learning models to automatically classify weed species from field images, supporting targeted robotic weed control in smart farming systems and reducing reliance on manual labour and broad-spectrum herbicides.

Three CNN architectures are trained and compared on the **DeepWeeds** dataset:

| Model | Strategy |
|-------|----------|
| Custom CNN | Trained from scratch |
| ResNet50 | Transfer learning + fine-tuning |
| MobileNetV2 | Transfer learning + fine-tuning |

---

## Dataset

**Name:** DeepWeeds (Weed Dataset)  
**Source:** [Kaggle – nasimrajlaskar/weed-dataset](https://www.kaggle.com/datasets/nasimrajlaskar/weed-dataset)  
**Original Paper:** Olsen et al. (2019), *Scientific Reports*

| Property | Details |
|----------|---------|
| Total Images | 17,509 labelled images |
| Number of Classes | 9 (8 weed species + 1 negative class) |
| Image Type | JPEG – in-situ field photography (RGB) |
| Image Resolution | 256 × 256 × 3 pixels |
| Collection Sites | 8 rangeland locations across Queensland, Australia |
| Camera | FLIR Blackfly 23S6C |

### Classes

1. Chinee apple
2. Lantana
3. Parkinsonia
4. Parthenium
5. Prickly acacia
6. Rubber vine
7. Siam weed
8. Snake weed
9. Negatives (non-weed / background vegetation)

> **Note:** The dataset exhibits some class imbalance, with the Negative class being the largest. Data augmentation and class-weighted training are applied to address this.

### Train / Validation / Test Split

| Split | Ratio |
|-------|-------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

---

## Requirements

```
tensorflow >= 2.x
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
kaggle
```

Install with:

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy pandas opencv-python kaggle
```

---

## Project Structure

```
crop-weed-cnn/
├── KUCHI_HARIKA_Phase_II_CropWeed_CNN_Notebook.ipynb   # Main notebook
├── data/
│   └── crop_weed_dataset/
│       └── data/
│           ├── train/
│           ├── val/
│           └── test/
├── best_model.h5                  # Saved Custom CNN weights
├── best_resnet50.keras            # Saved ResNet50 weights
├── best_mobilenet.keras           # Saved MobileNetV2 weights
├── custom_cnn_final.keras
├── resnet50_final.keras
├── mobilenetv2_final.keras
└── *.png                          # Training curves, confusion matrices, etc.
```

---

## Setup & Data Download

**Option A – Kaggle API:**

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d nasimrajlaskar/weed-dataset --unzip -p ./data/
```

**Option B – Google Colab (Google Drive):**

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = '/content/drive/MyDrive/crop weed dataset.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/crop_weed_dataset')
```

Update `DATASET_PATH` in the notebook to point to the extracted directory:

```python
DATASET_PATH = '/content/crop_weed_dataset/data/data'
```

---

## Methodology

### Data Preprocessing

- Images resized to **128×128** (Custom CNN) or **224×224** (transfer learning models)
- Pixel values normalized to [0, 1]
- Directory-based `ImageDataGenerator` used for efficient loading

### Data Augmentation (training set only)

- Random horizontal and vertical flipping
- Random rotation (±30°)
- Zoom in/out (factor 0.2)
- Random brightness adjustments ([0.75, 1.25])
- Width and height shifts (±20%)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs (Custom CNN) | 50 (with EarlyStopping) |
| Epochs (Transfer Learning) | 30 (with EarlyStopping) |
| Initial Learning Rate | 1e-3 |
| Fine-Tune Learning Rate | 1e-5 |

Callbacks used: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`

---

## Model Architectures

### Custom CNN

| Layer | Config | Output Shape |
|-------|--------|--------------|
| Input | 128×128×3 | 128×128×3 |
| Conv Block 1 | Conv2D(32) + BN + ReLU | 128×128×32 |
| MaxPooling 1 | 2×2 | 64×64×32 |
| Conv Block 2 | Conv2D(64) + BN + ReLU | 64×64×64 |
| MaxPooling 2 | 2×2 | 32×32×64 |
| Conv Block 3 | Conv2D(128) + BN + ReLU | 32×32×128 |
| MaxPooling 3 | 2×2 | 16×16×128 |
| Flatten | — | — |
| Dense 1 | Dense(256) + ReLU + Dropout(0.5) | 256 |
| Output | Dense(9) + Softmax | 9 |

### ResNet50 (Transfer Learning)

- Base: ResNet50 pretrained on ImageNet (top removed)
- Custom head: `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(9, Softmax)`
- **Phase 1:** Freeze base, train head only (15 epochs)
- **Phase 2:** Unfreeze last 20 layers, fine-tune with lr=1e-5

### MobileNetV2 (Transfer Learning)

- Base: MobileNetV2 pretrained on ImageNet (top removed)
- Custom head: `GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(9, Softmax)`
- Same two-phase fine-tuning strategy as ResNet50
- Lightweight architecture suitable for edge/embedded deployment

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (per class and macro/weighted averages)
- Confusion Matrix
- Training/Validation accuracy and loss curves
- ROC curves with per-class AUC scores
- Grad-CAM visualizations (highlighting discriminative image regions)

---

## Key Findings

- Transfer learning models significantly outperform the custom CNN on this complex real-world dataset.
- ResNet50 achieves performance close to the published benchmark of 95.7%.
- Data augmentation reduces overfitting and improves generalization.
- Grad-CAM confirms that models focus on biologically meaningful plant features.
- The highest inter-class confusion occurs between visually similar species (e.g., Siam weed vs. Lantana).

### Expected Performance

| Model | Expected Test Accuracy |
|-------|----------------------|
| Custom CNN | 80–88% |
| ResNet50 | 88–95% |
| MobileNetV2 | 88–95% |

---

## Research Questions Addressed

1. Can a custom CNN reliably classify 9 weed/non-weed categories from in-situ field images?
2. How does the custom CNN compare to ResNet50 and MobileNetV2 in accuracy and F1-score?
3. Which weed species are most frequently misclassified?
4. How effectively does data augmentation improve generalization?
5. Can the model achieve real-time inference speeds suitable for robotic deployment?

---

## References

- Olsen, A., et al. (2019). *DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning*. Scientific Reports, 9, 2058. https://doi.org/10.1038/s41598-018-38343-3
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv:1704.04861.
- Kaggle Dataset: https://www.kaggle.com/datasets/nasimrajlaskar/weed-dataset
