# Sign Language Recognition: Static ASL to Word-Level Classification

This repository contains a ASL recognition system:  
- static alphabet classification using ResNet-18, and  
- dynamic word-level recognition using temporal models.

---

## Overview

### Static ASL Alphabet Classification
- ResNet-18 (ImageNet pretrained)
- Freezing/unfreezing strategies compared:
  - T-A: Head only  
  - T-B: Last block + head  
  - T-C: Progressive unfreezing  
  - S-A: Train from scratch  
- Metrics: accuracy, macro-F1, confusion matrix  
- Evaluated on:
  - Official 28-image test set  
  - Custom 20-image test set (new conditions)

### Dynamic Word Recognition
- Best static model used as frame-level feature extractor  
- Global average–pooled embeddings fed into:
  - LSTM/GRU  
  - Transformer encoder  
- Compared:
  - 2A: Frozen CNN + temporal head  
  - 2B: Light unfreezing (layer4 + temporal head)

---

## Datasets

**ASL Alphabet (Static):**  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet  

**WLASL100 (Dynamic):**  
https://www.kaggle.com/datasets/thtrnphc/wlasl100-new  

---

## Repo Structure

```text
asl_static/         # Static model training
wlasl_dynamic/      # Temporal modeling
models/             # Checkpoints
utils/              # Preprocessing & dataloaders
notebooks/          # Training logs
new_test_set/       # Custom images
```

## Evaluation

- Reports accuracy, macro-F1, confusion matrices, and training curves  
- Includes ablation comparison of T-A, T-B, T-C, and S-A  
- Dynamic word recgonition evaluated on the official WLASL100 test split  

---

## Requirements

Python 3.x
PyTorch, torchvision
numpy, pandas
opencv-python
scikit-learn
matplotlib, seaborn
