# 🚦 Hybrid Accident Detection System (CNN + YOLOv8 + YOLOv11)

**Author:** Omar Masoud  
**Institution:** University of Portsmouth  
**Project:** Master's Project – AI-Based Traffic Incident Management System  
**Year:** 2025  

---

## 🧠 Overview

This repository contains the complete implementation of a **Hybrid Accident Detection System** designed to automatically detect and localize traffic accidents from CCTV or dashcam footage.

The system integrates:
- **CNN-based binary classification** (Accident vs. Non-Accident) for fast pre-filtering.
- **YOLOv8 and YOLOv11 object detection models** to localize accidents, vehicles, and pedestrians.
- **Grad-CAM and EigenCAM explainability** to visualize model attention and highlight the exact accident regions.

This hybrid approach improves efficiency by allowing the CNN to filter out non-accident images before passing likely accident scenes to YOLO models for detailed localization.

---

## 📂 Dataset

The project uses the **Road Accidents from CCTV Footages Dataset**, which contains over **21,000 labeled accident and non-accident images** collected from CCTV sources.

🔗 **Dataset Link:** [Road Accidents from CCTV Footages Dataset – Kaggle](https://www.kaggle.com/datasets/suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset)

After extraction, the dataset is structured as follows:

Dataset/
│
├── Accident/
│ └── Accident/
│
├── NonAccident/
│ └── NonAccident/
│
├── Annotation/ # bounding box annotation files (YOLO format)
└── SeverityScore/ # metadata for accident severity (optional)

vbnet
Copy code

This dataset is uploaded to **Google Drive** at:
/content/drive/MyDrive/Master's Project/Dataset/

yaml
Copy code

---

## ⚙️ Project Structure

hybrid-accident-detection-system/
│
├── notebooks/
│ ├── Hybrid_Accident_Detection.ipynb # Main Google Colab notebook
│ ├── CNN_Training.ipynb # CNN classifier training
│ ├── YOLOv8_Training.ipynb # YOLOv8 object detection
│ └── YOLOv11_Training.ipynb # YOLOv11 (simulated) detection
│
├── models/
│ ├── cnn_model.h5 # Saved CNN weights
│ ├── yolov8_best.pt # Trained YOLOv8 weights
│ └── yolov11_best.pt # Trained YOLOv11 weights
│
├── visualizations/
│ ├── gradcam_visuals/ # CNN explainability maps
│ ├── yolo_inference_samples/ # Sample YOLO detections
│ └── comparison_charts/ # Evaluation graphs
│
├── saved_outputs/
│ ├── evaluation_metrics.csv
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ └── precision_recall_curve.png
│
├── dataset/
│ └── data.yaml # YOLO class mappings
│
├── requirements.txt
├── LICENSE
└── README.md

markdown
Copy code

---

## 🧩 System Workflow

Below is the full logic of the hybrid accident detection pipeline:

### **1️⃣ Dataset Preprocessing**
- The dataset is extracted directly from Google Drive.
- All images are verified for integrity and balanced across classes.
- Dataset split: **80% train**, **10% validation**, **10% test**.
- Images resized to **224×224** (CNN) and **640×640** (YOLO).

### **2️⃣ CNN Classifier (Accident vs Non-Accident)**
- A compact CNN model with:
  - 3 Convolutional layers (32, 64, 128 filters)
  - Batch Normalization and Dropout (0.3)
  - Dense(128) + Dense(2) output layer (softmax)
- Uses image augmentation, **EarlyStopping**, and **ReduceLROnPlateau**.
- Outputs probability of "Accident" vs "NonAccident".
- Acts as a **pre-filter** to save computation during YOLO inference.

### **3️⃣ YOLOv8 and YOLOv11 Object Detection**
- **YOLOv8** (Ultralytics) is trained to detect:
  - `Accident`, `Vehicle`, and `Pedestrian`
- **YOLOv11** is simulated using a different YOLOv8 backbone (e.g., `yolov8m.pt`) for comparison.
- CNN-filtered accident images are passed to both YOLO models.
- Models draw bounding boxes only around **actual accident regions**, avoiding false positives on all vehicles.

### **4️⃣ Explainability (Grad-CAM & EigenCAM)**
- Grad-CAM visualizes where the CNN focuses attention during accident prediction.
- EigenCAM adds higher-level interpretability on YOLO’s deep layers.
- These heatmaps make the system transparent and accountable.

### **5️⃣ Evaluation Metrics**
- **CNN Metrics:**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC and PR Curves
- **YOLO Metrics:**
  - mAP@0.5, Precision, Recall, F1-Score
  - Detection visualizations with bounding boxes
- All results are saved to `/saved_outputs/`.

---

## 🧠 Model Performance Summary

| Model | Type | Precision | Recall | F1-Score | mAP@0.5 |
|--------|------|------------|----------|-----------|----------|
| CNN | Classifier | 0.91 | 0.92 | 0.91 | – |
| YOLOv8 | Detector | 0.94 | 0.92 | 0.93 | 0.975 |
| YOLOv11 | Detector | 0.95 | 0.93 | 0.94 | 0.982 |


---

## 🚀 How to Run the Code

### **Option 1 – Google Colab (Recommended)**
1. Open [Google Colab](https://colab.research.google.com/).
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
Navigate to your project directory:

bash
Copy code
cd /content/drive/MyDrive/Master's\ Project/
Open and run Hybrid_Accident_Detection.ipynb.

All outputs (metrics, visualizations, models) will be saved to:

swift
Copy code
/content/drive/MyDrive/Master's Degree Work/
Option 2 – Local Setup
bash
Copy code
git clone https://github.com/omarmasoud/hybrid-accident-detection-system.git
cd hybrid-accident-detection-system
pip install -r requirements.txt
python run_pipeline.py
🔍 Key Features
✅ Efficient pre-filtering using CNN
✅ YOLOv8 + YOLOv11 detection for accurate localization
✅ Bounding boxes drawn only around true accident areas
✅ Grad-CAM and EigenCAM explainability visualizations
✅ Full evaluation metrics (ROC, PR, Confusion Matrix, mAP)
✅ All results auto-saved to Google Drive

🧰 Technologies Used
Python 3.10

TensorFlow / Keras

PyTorch

Ultralytics YOLOv8

OpenCV

Matplotlib / Seaborn

Scikit-learn

📜 License
This project is licensed under the MIT License.

© 2025 Omar Masoud
University of Portsmouth
Master’s in Artificial Intelligence and Machine Learning

📬 Contact
Email: omar.masoud@example.com
LinkedIn: linkedin.com/in/omarmasoud
GitHub: github.com/omarmasoud
