# 🩺Enhancing  Eye Disease Classification using Vision Transformer (ViT)

## 📌 Overview

This project is an AI-powered medical image classification system developed as my **M.Sc. Computer Science Major Project**.

The model uses a **Vision Transformer (ViT-B/16)** to classify retinal fundus images into four categories with **96% classification accuracy**. To improve transparency and interpretability, **Grad-CAM visualization** is integrated to highlight the image regions influencing the model's predictions.

A **Streamlit web application** was also developed to allow users to upload retinal images and obtain real-time disease predictions.

---

## 🚀 Features

- Vision Transformer (ViT-B/16)
- Transfer Learning using ImageNet pretrained weights
- Four-class Eye Disease Classification
- Streamlit Web Application
- Grad-CAM Explainability
- Image Preprocessing & Data Augmentation
- High Accuracy (96%)

---

## 👁️ Diseases Classified

- ✅ Normal
- ✅ Diabetic Retinopathy
- ✅ Cataract
- ✅ Glaucoma

---

## 🧠 Model

- Architecture: Vision Transformer (ViT-B/16)
- Framework: PyTorch
- Transfer Learning
- Image Size: 224 × 224
- Optimizer: AdamW
- Loss Function: CrossEntropy Loss
- Learning Rate Scheduler: CosineAnnealingWarmRestarts

---

## 📂 Dataset

Dataset: **Eye Diseases Classification Dataset**

Source:
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

The dataset contains retinal fundus images belonging to four eye disease classes.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Grad-CAM

---

## 📊 Results

| Metric | Value |
|---------|--------|
| Model | Vision Transformer (ViT-B/16) |
| Accuracy | **96%** |
| Classes | 4 |

---

## 🔥 Grad-CAM Visualization

Grad-CAM is used to visualize the regions of the retinal image that contributed to the model's prediction, improving model interpretability and trustworthiness.

---

## 💻 Streamlit Application

The project includes a Streamlit web interface where users can:

- Upload retinal fundus images
- Predict eye diseases
- View prediction confidence

---

## 📁 Project Structure

```
Eye_Disease_Project/
│
├── app.py
├── inference.py
├── eye-disease-classification.ipynb
├── requirements.txt
├── README.md
├── images/
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/amayamohan/eye-disease-classification-vit.git

cd eye-disease-classification-vit

pip install -r requirements.txt

streamlit run app.py
```

---

## 📌 Note

The trained model (`best_model.pth`) is not included in this repository because it exceeds GitHub's file size limit. The notebook contains the complete training pipeline, and the model can be retrained using the provided code.

---

## 🎓 Academic Project

This project was developed as part of my **M.Sc. Computer Science Major Project**.

---

## 👩‍💻 Author

**Amaya M**

LinkedIn: *https://www.linkedin.com/in/amaya-mohan/*

GitHub: https://github.com/amayamohan
