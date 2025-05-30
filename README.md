# 🧠 Assignment 1 – MLP vs CNN on Rock-Paper-Scissors

This deep learning assignment compares the performance of a **Multilayer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)** on the classic image classification task of identifying Rock, Paper, or Scissors hand signs.

---

## 🗂 Dataset

- **Rock-Paper-Scissors** dataset
- Loaded and preprocessed using TensorFlow/Keras
- Images are resized and normalized before training

---

## 🧪 Models Implemented

### 🔹 MLP (Multilayer Perceptron)
- Flattened input layer from 2D image
- Multiple Dense layers with ReLU activations
- Softmax output for 3-class classification

### 🔸 CNN (Convolutional Neural Network)
- Stacked Conv2D + MaxPooling layers
- Deeper architecture with fewer parameters due to weight sharing
- Outperforms MLP in accuracy and generalization

---

## 📊 Evaluation

- Accuracy and loss tracked over epochs
- Plots and training history saved for both models
- Saved models organized under `/Rock_Paper_Scissors_Models/mlp` and `/cnn`

---

## 📁 File Structure

📓 Assignment#1.ipynb
📁 Rock_Paper_Scissors_Models/
├── mlp/
└── cnn/



---

## 🚀 How to Run

> ✅ This project was developed on **Google Colab** using Drive for model saving.

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `Assignment#1.ipynb`
3. Mount your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')




📈 Why CNN Outperforms MLP in Image Tasks
This project provides a clear demonstration of why CNNs are better suited than MLPs for image classification:

MLP Limitations
MLPs treat images as flat vectors, destroying the 2D spatial relationships between pixels. This results in:

A massive number of weights

Poor generalization

Random guessing behavior across all tested optimizers

A fixed ~0.33 validation accuracy (equivalent to guessing among 3 classes)

CNN Advantages
CNNs preserve spatial hierarchies by using convolution and pooling operations. This allows them to:

Extract low- and high-level spatial features (edges, shapes, patterns)

Share weights and reduce parameters

Show steady improvement in validation accuracy

Generalize well across optimizers (especially with RMSProp and Adam)

🔍 Conclusion:
MLPs are not suitable for raw image input due to their inability to detect and learn spatial features. CNNs, by design, are tailored for image data and demonstrate clear learning progression and high accuracy — making them the standard for visual tasks.

