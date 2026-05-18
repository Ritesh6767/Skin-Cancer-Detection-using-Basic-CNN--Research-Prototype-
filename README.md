# 🔬 Skin Cancer Detection using Basic CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-brightgreen)

## 📌 Overview
This research prototype focuses on detecting and classifying skin cancer from dermoscopic images using a Convolutional Neural Network (CNN). Early detection of skin cancer is crucial for successful treatment, and this deep learning model aims to assist in the initial screening process.

## 🚀 Features
- **Custom CNN Architecture:** A deep learning model built from scratch to classify skin lesions.
- **Image Preprocessing:** Data augmentation and resizing pipelines for robust training.
- **Evaluation:** Accuracy and loss tracking across training and validation sets.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib

## 📂 Project Structure
```text
Skin-Cancer-Detection/
├── Model.ipynb          # Jupyter notebook with CNN architecture, training, and evaluation
├── GroundTruth.csv      # Labels for the dataset
├── train/               # Directory containing training images
└── val/                 # Directory containing validation images
```

## 💻 How to Run
1. Clone the repository: `git clone <your-repo-url>`
2. Ensure you have TensorFlow installed: `pip install tensorflow opencv-python numpy matplotlib pandas`
3. Open `Model.ipynb` in Jupyter Notebook or Google Colab and run the cells to train or evaluate the model.
