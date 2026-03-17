# MedMNIST Skin Lesion Classification

This project implements deep learning models to classify skin lesion images using the DermaMNIST dataset from the MedMNIST collection.

Two models are trained and compared:

- A custom Convolutional Neural Network (SimpleCNN)
- A pretrained ResNet18 using transfer learning

The goal is to evaluate how transfer learning improves performance in medical image classification tasks.

---

# Dataset

This project uses the **DermaMNIST** dataset.

Dataset characteristics:

- Image size: 224 × 224
- Number of classes: 7
- Task: Skin lesion classification

Classes:

1. actinic keratoses and intraepithelial carcinoma  
2. basal cell carcinoma  
3. benign keratosis-like lesions  
4. dermatofibroma  
5. melanoma  
6. melanocytic nevi  
7. vascular lesions  

---

# Models

## 1. SimpleCNN

A custom convolutional neural network trained from scratch.

Architecture includes:

- Convolution layers
- ReLU activation
- MaxPooling layers
- Fully connected classifier

---

## 2. ResNet18

A pretrained **ResNet18** model using transfer learning.

- Pretrained on ImageNet
- Final classification layer modified to output 7 classes

---

# Training Setup

Training configuration:

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Image size: 224 × 224
- Data augmentation:
  - Random horizontal flip
  - Random rotation

Training was performed on GPU using the HiPerGator HPC cluster.

---

# Results

## Validation Accuracy

| Model | Accuracy |
|------|------|
| SimpleCNN | 75.67% |
| ResNet18 | 87.84% |

---

## Test Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|------|------|------|------|------|
| SimpleCNN | 0.7471 | 0.7178 | 0.7471 | 0.7185 |
| ResNet18 | **0.8863** | **0.8856** | **0.8863** | **0.8844** |

The pretrained ResNet18 significantly outperforms the custom CNN, demonstrating the effectiveness of transfer learning for biomedical image classification.

---

# Confusion Matrix

Example confusion matrix generated during evaluation.

![Confusion Matrix](results/resnet18_confusion_matrix.png)

---

# Project Structure
project2/
│
├── code/
│ ├── train.py
│ ├── evaluate.py
│ ├── models.py
│ └── plot_history.py
│
├── results/
│
└── README.md

---

# How to Run

### 1. Install dependencies
pip install torch torchvision medmnist scikit-learn matplotlib

### 2. Train models
python train.py

### 3. Evaluate models
python evaluate.py

### 4. Plot training curves
python plot_history.py

---

# Key Takeaway

Transfer learning using pretrained convolutional neural networks significantly improves performance for medical image classification tasks compared to training CNN models from scratch.

---

# Author

Haibin Fan  
Master's Student in Biomedical Engineering  
University of Florida
