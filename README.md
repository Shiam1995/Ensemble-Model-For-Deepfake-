# Ensemble Model for Deepfake Detection

**MSc Artificial Intelligence — University of Leeds (2024)**

An ensemble approach to deepfake face-image detection using three independently trained deep learning architectures: **ConvNeXt-Base**, **FastViT-S12**, and **YOLOv8L-cls**. Each model is fine-tuned on face-swap and AI-generated face imagery, then evaluated individually and as a combined system. The best individual model (YOLOv8L-cls) achieves **98.22% accuracy** and **98.61% F1** on the Celeb-DF test set.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Citation](#citation)

---

## Overview

Deepfake detection is framed as a binary classification problem: given a cropped face image, predict whether it is **real** or **fake**. This project explores whether combining architectures with different inductive biases — a modernised CNN (ConvNeXt), a hybrid vision transformer (FastViT), and a detection-backbone classifier (YOLOv8) — can produce more robust detection than any single model.

All models take 224×224 face crops as input, normalised with ImageNet statistics, and output a single sigmoid-activated probability score.

**Two training configurations** are compared:
1. **Celeb-DF only** — trained and evaluated on the Celeb-DF face-swap dataset
2. **Celeb-DF + Leonardo** — trained on Celeb-DF combined with AI-generated faces from Leonardo.ai

---

## Results

### Individual Model Performance (Celeb-DF Test Set)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| ConvNeXt-Base | 85.87% | 96.98% | 80.68% | 88.08% |
| FastViT-S12 | 71.93% | 93.65% | 60.73% | 73.68% |
| **YOLOv8L-cls** | **98.22%** | **99.88%** | **97.36%** | **98.61%** |

**Key findings:**
- All three models achieve >93% precision — when they flag a deepfake, they are almost always correct.
- The accuracy gap is driven by recall: YOLOv8 catches 97% of deepfakes, while FastViT misses nearly 40%.
- YOLOv8's strong performance likely stems from its aggressive data augmentation pipeline (RandAugment, mosaic, erasing) and large backbone.

### YOLOv8 Training Metrics (Celeb-DF + Leonardo)

| Epoch | Train Loss | Top-1 Accuracy | Top-5 Accuracy | Val Loss |
|---|---|---|---|---|
| 1 | 0.175 | 97.31% | 100% | 0.344 |
| 2 | 0.041 | 98.26% | 100% | 0.332 |
| 5 | 0.051 | 97.95% | 100% | 0.335 |
| 10 | 0.012 | 98.22% | 100% | 0.330 |

### ConvNeXt Training Metrics (Celeb-DF + Leonardo)

| Epoch | Train Loss | Validation Loss |
|---|---|---|
| 1 | 0.0374 | 0.5357 |
| 5 | 0.0018 | 0.9600 |
| 9 | 0.0012 | 0.4566 |
| 10 | 0.0010 | 0.7400 |

---

## Architecture

### ConvNeXt-Base
- **Source:** `timm` library, pretrained on ImageNet
- **Fine-tuning:** Classification head + final convolutional block (`stages.3.blocks.2`) unfrozen; all other layers frozen
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam, lr=1e-4
- **Output:** Single neuron with sigmoid activation

### FastViT-S12
- **Source:** `timm` library (Apple's hybrid architecture), pretrained on ImageNet
- **Fine-tuning:** Full model fine-tuned
- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** Adam, lr=1e-4
- **Output:** Single neuron with sigmoid activation

### YOLOv8L-cls
- **Source:** Ultralytics, pretrained classification backbone
- **Training:** End-to-end via `yolo task=classify` CLI
- **Input:** 224×224
- **Epochs:** 10, Batch size: 16
- **Augmentation:** RandAugment, horizontal flip, scale, erasing (0.4)

### Shared Preprocessing

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Datasets

### Celeb-DF
A large-scale celebrity deepfake dataset containing real and face-swapped video frames. Data is pre-split into `train/`, `val/`, and `test/` partitions, each containing `real/` and `fake/` subdirectories of extracted face crops.

### Leonardo AI-Generated
AI-generated face images from Leonardo.ai, preprocessed and merged into the `fake/` class. Used as supplementary training data in the combined configuration to test generalisation to diffusion-based generation.

**Expected data layout:**

```
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## Repository Structure

```
Ensemble-Model-For-Deepfake-/
│
├── celebdf_convnext.ipynb                        # ConvNeXt training on Celeb-DF only
├── convnext_on_celebdf_n_leonardo.ipynb           # ConvNeXt training on Celeb-DF + Leonardo
├── yolo_on_celebdf_n_leonardo.ipynb               # YOLOv8 training on Celeb-DF + Leonardo
│
├── Prediction_trained_on_celebdf_only_*.ipynb     # Evaluation: models trained on Celeb-DF only
├── Prediction_trained_on_celebdf_n_leonardo_*.ipynb # Evaluation: models trained on combined data
│
├── args.yaml                                      # YOLOv8 training configuration
├── results.csv                                    # YOLOv8 per-epoch training metrics
├── train_report.txt                               # ConvNeXt per-epoch train/val loss
│
├── convnext_celebdf_only.txt                      # ConvNeXt per-image predictions (Celeb-DF)
├── fastvit_celebdf_only.txt                       # FastViT per-image predictions (Celeb-DF)
├── fastvit_celebdf_only.json                      # FastViT evaluation metrics (JSON)
├── yolo_leonardo_only.txt                         # YOLOv8 per-image predictions (Leonardo)
├── convnext_trained_celebdf_leonardo_eval_*.txt   # ConvNeXt predictions (combined model)
├── fastvit_trained_celebdf_leonardo_eval_*.txt    # FastViT predictions (combined model)
│
├── nn.py                                          # Neural network architecture visualization
├── humanface.py                                   # Face matrix visualization utility
├── imagetoarray.py                                # Image to array conversion utility
├── boundingbox.py                                 # Bounding box utility (stub)
│
├── results.png                                    # YOLOv8 training curves plot
├── convnext_leonardo_only.png                     # ConvNeXt confusion matrix
├── val_batch1_labels.jpg                          # Sample validation batch
├── rgbarray.png                                   # RGB array visualization
├── greyscalearray.png                             # Greyscale array visualization
│
└── project.zip                                    # Archived project files
```

---

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (trained on Google Colab)

### Install Dependencies

```bash
pip install torch torchvision timm ultralytics
pip install numpy matplotlib seaborn scikit-learn tqdm pillow
```

### Download Model Checkpoints

The trained model weights are not included in this repository. To reproduce results, train the models using the provided notebooks, or contact the author for checkpoint files.

- `convnext_epoch_9_step_20000.pth` — ConvNeXt fine-tuned weights
- `fastvit_epoch_9_step_20000.pth` — FastViT fine-tuned weights
- `best.pt` — YOLOv8 best classification weights

### Download Datasets

1. **Celeb-DF:** Download from the [official Celeb-DF repository](https://github.com/yuezunli/celeb-deepfakeforensics) and extract face crops into the `data/` structure shown above.
2. **Leonardo:** Generate or obtain AI-generated face images and place them in the `fake/` directories.

---

## How to Run

### 1. Training

All training notebooks are designed for **Google Colab** with GPU runtime. To run locally, update the data paths from `/content/...` to your local paths.

**Train ConvNeXt (Celeb-DF only):**
```
Open celebdf_convnext.ipynb and run all cells.
```

**Train ConvNeXt (Celeb-DF + Leonardo):**
```
Open convnext_on_celebdf_n_leonardo.ipynb and run all cells.
```

**Train YOLOv8 (Celeb-DF + Leonardo):**
```bash
# Via CLI (as used in the notebook):
yolo task=classify mode=train model=yolov8l-cls.pt data="path/to/data" epochs=10 imgsz=224

# Or open yolo_on_celebdf_n_leonardo.ipynb and run all cells.
```

### 2. Inference / Prediction

**Run predictions with ConvNeXt or FastViT:**

```python
import torch
import timm
from torchvision import transforms
from PIL import Image

# Load model
model = timm.create_model('convnext_base', pretrained=False, num_classes=1)
# Or: model = timm.create_model('fastvit_s12', pretrained=False, num_classes=1)
model.load_state_dict(torch.load('convnext_epoch_9_step_20000.pth'))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = transform(Image.open('face.png').convert('RGB')).unsqueeze(0)

# Predict
with torch.no_grad():
    prob = torch.sigmoid(model(img)).item()
    label = "FAKE" if prob > 0.5 else "REAL"
    print(f"{label} (confidence: {prob:.4f})")
```

**Run predictions with YOLOv8:**

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('face.png', verbose=False)

pred_class = results[0].probs.top1        # 0 = fake, 1 = real
confidence = results[0].probs.top1conf.item()
print(f"Class: {pred_class}, Confidence: {confidence:.4f}")
```

### 3. Evaluation

Open either prediction notebook and run all cells to reproduce the full evaluation:

```
Prediction_trained_on_celebdf_only_convnext_fastvit_yolo.ipynb
Prediction_trained_on_celebdf_n_leonardo_convnext_fastvit_yolo.ipynb
```

These notebooks load per-image prediction text files, compute accuracy/precision/recall/F1, and generate confusion matrix plots.

### 4. Prediction File Format

Each prediction `.txt` file follows this format:

```
filename.png  predicted_label  true_label  probability
```

For example:
```
id0_id26_0009__5564a565.png 0.0 0 7.296e-07
```

Where `0 = fake` and `1 = real`, and the probability is the sigmoid output for the positive (real) class.

---

## Citation

If you use this work, please cite:

```
Chuttoo, S. (2024). Ensemble Model for Deepfake Detection.
MSc Artificial Intelligence, University of Leeds.
```

---

## License

This project was developed as part of an MSc dissertation. Please contact the author for usage permissions.
