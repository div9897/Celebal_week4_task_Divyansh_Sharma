# CIFAR-10 Image Classification: ANN vs CNN

A learning project comparing **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** on the CIFAR-10 image classification dataset, exploring how architecture and training strategies affect model performance.

## 📌 Overview

This notebook walks through the full deep learning pipeline — from data loading to model evaluation — using CIFAR-10, a dataset of 60,000 32×32 color images across 10 classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The goal is to understand **why CNNs outperform ANNs** on image tasks, and how architectural depth, regularization, and data augmentation influence results.

## 🗂️ Dataset

- **Source:** `tf.keras.datasets.cifar10`
- **Training set:** 50,000 images
- **Test set:** 10,000 images
- **Image size:** 32×32×3 (RGB)

## 🧠 Models Implemented

| Model | Description |
|---|---|
| **ANN** | Baseline fully-connected network on flattened pixel vectors |
| **ANN v2** | Deeper ANN (more Dense layers) with EarlyStopping, 20 epochs |
| **CNN** | Baseline CNN with Conv2D + BatchNorm + MaxPooling layers |
| **CNN v2** | Deeper CNN with increased filters (64→128→256), EarlyStopping, 20 epochs |
| **CNN + Augmentation** | CNN trained with RandomFlip, RandomRotation, and RandomZoom for better generalization |

## 📊 Results

| Model | Test Accuracy |
|---|---|
| ANN | 42.10% |
| ANN v2 | 44.98% |
| CNN | 71.44% |
| CNN v2 | 74.96% |
| CNN + Augmentation | 66.85% |

**Key takeaway:** CNNs significantly outperform ANNs on image data because they preserve spatial relationships between pixels through convolution and pooling, whereas ANNs treat images as flat, unstructured vectors. Increasing CNN filter depth (CNN v2) gave the best overall result.

## 🛠️ Techniques Covered

- Data normalization (pixel scaling 0–255 → 0–1)
- Dense (ANN) vs Convolutional (CNN) architectures
- Dropout and Batch Normalization for regularization
- `EarlyStopping` callback to prevent overfitting
- Data augmentation (`RandomFlip`, `RandomRotation`, `RandomZoom`)
- Validation accuracy comparison via learning curves
- Final model comparison using a summary table

## 📁 Notebook Structure

1. Problem statement & dataset overview
2. Data loading and visualization
3. Preprocessing
4. ANN baseline + ANN v2 (deeper network)
5. CNN baseline + CNN v2 (more filters)
6. Learning curve comparison (ANN vs CNN)
7. Data augmentation strategy + training
8. Final comparison table across all 5 models
9. Student learning tasks
10. Conclusion

## ⚙️ Requirements

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

Designed to run in **Google Colab** with GPU acceleration enabled (Runtime → Change runtime type → GPU) for faster training.

## ▶️ How to Run

1. Open the notebook in Google Colab
2. Run all cells sequentially (Runtime → Run all)
3. Training the CNN models will take the longest — using a GPU runtime is recommended

## ✅ Conclusion

- ANNs work but ignore spatial image structure, capping their performance
- CNNs extract hierarchical spatial features, leading to significantly better accuracy
- Architectural depth (more filters/layers) improves results, with diminishing returns
- Data augmentation increases training-data diversity to improve generalization, though it may need more epochs to outperform a well-tuned baseline CNN
- This project builds strong fundamentals for computer vision interviews and deep learning practice

---
*Author: Divyansh Sharma — Week 4 Learning Project*
