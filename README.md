Skin Cancer Detection using Basic CNN — Research Prototype

A deep learning prototype for binary classification of skin lesions (benign vs malignant) using a custom CNN architecture. This project demonstrates end-to-end medical image processing, model training, evaluation, and interpretability via Grad-CAM.

###  Dataset
- Source: [Kaggle Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Images: 10,015 dermatoscopic images
- Labels: Derived from `GroundTruth.csv` using MEL, BCC, AKIEC indicators
- Preprocessing: Resized to 128×128, RGB conversion, stratified train/val split

###  Approach
- **Model**: Custom CNN with 3 Conv2D layers, MaxPooling, Dense, Dropout
- **Augmentation**: Rotation, horizontal/vertical flips via `ImageDataGenerator`
- **Loss**: Binary Crossentropy | **Optimizer**: Adam
- **Training**: 10 epochs on 8012 training images, validated on 2003 images

###  Results
- Final Accuracy: ~80.6% (val)
- Evaluation: Confusion Matrix, Classification Report, ROC Curve (AUC)
- Interpretability: Grad-CAM heatmaps for visualizing model attention


###  Highlights
-  Automated dataset preparation with label mapping and resizing
-  Augmented training pipeline for robust learning
-  Grad-CAM overlay for model interpretability
-  Saved model (`.h5`) for deployment or further tuning

###  Future Work
- Integrate transfer learning (e.g., ResNet50, EfficientNet)
- Deploy via Streamlit or Flask for real-time predictions
- Add model card and explainability dashboard
