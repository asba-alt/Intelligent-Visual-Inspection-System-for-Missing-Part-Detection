# Intelligent Visual Inspection System 🔍  
**AI-Based Visual Inspection for Assembly Integrity Verification**

The Intelligent Visual Inspection System is a deep learning–based solution designed to automatically verify the completeness of mechanical assemblies. It detects missing, misplaced, or extra components using computer vision techniques, enabling reliable and scalable quality inspection for industrial and aerospace manufacturing environments.

---

## 🔍 Problem Statement

In mechanical and aerospace manufacturing, assemblies must strictly adhere to design specifications. Even a single missing or extra component—such as a screw, fastener, or electronic part—can compromise safety, reliability, and performance.

Traditional manual inspection is:
- Time-consuming
- Error-prone
- Difficult to scale

There is a strong need for an **automated, consistent, and explainable visual inspection system** that can assist quality engineers in identifying assembly defects early in the production pipeline.

---

## 💡 Solution Overview

This project proposes an **AI-powered visual inspection system** that analyzes images (and later videos) of mechanical assemblies to determine assembly integrity.

The system is built in **phases**, gradually increasing capability while maintaining reliability and explainability.

At its core, the system uses **Convolutional Neural Networks (CNNs)** trained on industrial inspection datasets to learn visual patterns of correct and incorrect assemblies.

---

## ✨ Key Features

- 📷 Image-based inspection of mechanical assemblies
- 🧠 CNN-based defect and anomaly detection
- 📊 Confidence-driven inspection results
- 🧩 Modular, phase-wise system design
- 🔍 Explainability through visual localization (future phases)
- 🧱 Engineering-focused, production-ready architecture

---

## 🧱 Phased Development Roadmap

### **Phase 1 – Missing Part Detection (Baseline)**
- Binary classification:
  - **Complete Assembly**
  - **Missing Part Detected**
- Image-level CNN classification
- High recall for defect detection
- Confidence score output

---

### **Phase 2 – Missing Part Localization**
- Identify the approximate location of missing components
- Use explainability techniques such as **Grad-CAM**
- Visual heatmaps overlaid on input images

---

### **Phase 3 – Extra / Incorrect Part Detection**
- Detect:
  - Extra components
  - Incorrect placements
  - Unauthorized parts
- Expand classification beyond binary labels
- Hybrid classification + anomaly detection approaches

---

### **Phase 4 – Video-Based Inspection**
- Support video input from inspection cameras
- Frame extraction and aggregation logic
- Video-level inspection summary

---

### **Phase 5 – Web-Based Inspection Platform**
- User-friendly web interface
- Image/video upload
- Visual inspection reports
- Designed for quality engineers and inspectors

---

## 🛠 Tech Stack

### Core Technologies
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**

### Datasets
- **MVTec Anomaly Detection Dataset**
  - Screw
  - Transistor

### Backend (Future Phase)
- Flask (REST-based inference API)

### Frontend (Future Phase)
- HTML / CSS
- Minimal JavaScript

---

## 🧠 System Architecture

Input Image / Video
↓
Preprocessing (Resize, Normalize)
↓
CNN-Based Inspection Models
↓
Inspection Logic (Phase-based)
↓
Visual & Textual Output


## 🚦 How It Works (Phase 1)

1. Input image of a mechanical assembly is provided
2. Image is preprocessed (resize, normalization)
3. CNN model predicts:
   - Complete
   - Missing Part
4. Confidence score is generated
5. Inspection result is displayed

---

## 📊 Evaluation Criteria

The system is evaluated using:
- Accuracy
- Precision
- Recall
- Confusion Matrix

Special emphasis is placed on **minimizing false negatives**, which is critical in safety-sensitive manufacturing environments.

---

## 📦 Project Structure

Intelligent_Visual_Inspection_System/
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
├── models/
│ └── phase1_missing_part_classifier.h5
├── src/
│ ├── preprocess.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── config.py
├── results/
├── logs/
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🔐 Assumptions & Constraints

- Images are captured from a fixed or limited range of viewpoints
- Lighting conditions are reasonably consistent
- Dataset is a proxy for gearbox assemblies
- Offline inference only (Phase 1)

---

## ⚠️ Limitations

- Localization is not included in Phase 1
- Dataset is not gearbox-specific
- Real-time performance not evaluated

---

## 🚀 Future Enhancements

- Gearbox-specific dataset integration
- Edge-device deployment
- Automated inspection reports
- Integration with manufacturing execution systems (MES)

---

## 🧠 Learnings & Reflection

This project emphasizes:
- Applied deep learning in industrial contexts
- Engineering-first design over experimental complexity
- Explainability and reliability in AI systems
- Phase-wise development for production readiness

---

## 📄 License

MIT License

---

## 🤝 Acknowledgements

- MVTec Anomaly Detection Dataset
- Open-source deep learning community

---
