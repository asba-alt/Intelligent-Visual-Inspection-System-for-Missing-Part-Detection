# Intelligent Visual Inspection System for Multi-Class Defect Detection

An AI-powered visual inspection system that detects and classifies 10 different types of manufacturing defects in electronic components using deep learning. Built with **MobileNetV2 transfer learning**, **FastAPI backend**, and **Streamlit frontend** with **Grad-CAM explainability**.

![System Status](https://img.shields.io/badge/status-production-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Features

- **Multi-Class Detection**: Identifies 10 specific defect types + complete parts
- **High Accuracy**: ~90% top-3 accuracy on test set
- **Real-Time Inference**: <200ms per image after warmup
- **Explainable AI**: Grad-CAM heatmaps show decision-making regions
- **Modern Tech Stack**: FastAPI + Streamlit for scalability
- **Interactive UI**: Drag-and-drop image upload with live results
- **RESTful API**: Complete API documentation with Swagger/ReDoc
- **Transfer Learning**: Fine-tuned MobileNetV2 on MVTec Anomaly Detection Dataset

## 📊 Detected Defect Classes

| Class | Description | Component |
|-------|-------------|-----------|
| **Complete** | No defects detected | Both |
| **Bent Lead** | Lead/pin is bent | Transistor |
| **Cut Lead** | Lead/pin is cut or damaged | Transistor |
| **Damaged Case** | Case/body damage | Transistor |
| **Manipulated Front** | Front surface tampering | Screw |
| **Misplaced** | Component misalignment | Transistor |
| **Scratch Head** | Scratch on head area | Screw |
| **Scratch Neck** | Scratch on neck area | Screw |
| **Thread Side** | Side thread defect | Screw |
| **Thread Top** | Top thread defect | Screw |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum
- (Optional) CUDA-capable GPU for training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/asba-alt/Intelligent-Visual-Inspection-System-for-Missing-Part-Detection.git
cd Intelligent-Visual-Inspection-System-for-Missing-Part-Detection
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Application

**Terminal 1 - Start Backend (FastAPI)**
```bash
python app_fastapi.py
# Runs on: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

**Terminal 2 - Start Frontend (Streamlit)**
```bash
streamlit run app_streamlit.py
# Runs on: http://localhost:8502
```

**Open your browser** → http://localhost:8502

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Complete setup instructions for new devices
- **[Training Guide](docs/TRAINING_GUIDE.md)** - How to train/retrain models
- **[Migration Guide](docs/MIGRATION.md)** - Legacy Flask → FastAPI migration notes
- **[Phase Definition](docs/PhaseDefinition.md)** - Project roadmap and phases

## 🏗️ Project Structure

```
.
├── app_fastapi.py              # FastAPI backend server
├── app_streamlit.py            # Streamlit frontend UI
├── train_multiclass.py         # Multi-class training script
├── inference_multiclass.py     # Single image inference
├── evaluate_multiclass.py      # Model evaluation & metrics
├── prepare_dataset_multiclass.py # Dataset preparation
├── gradcam.py                  # Grad-CAM visualization
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/                     # Trained models
│   ├── mobilenetv2_multiclass.keras
│   └── class_names_multiclass.txt
├── dataset_multiclass/         # Organized dataset
│   ├── train/                  # Training split (80%)
│   ├── val/                    # Validation split (10%)
│   └── test/                   # Test split (10%)
├── raw/                        # Original MVTec dataset
│   ├── screw/
│   └── transistor/
├── docs/                       # Documentation
├── logs/                       # TensorBoard logs
├── results/                    # Evaluation results
└── static/                     # Uploaded images & outputs
```

## 🎓 Training Your Own Model

### 1. Prepare Dataset
```bash
python prepare_dataset_multiclass.py
```
This organizes the raw MVTec dataset into train/val/test splits.

### 2. Train Model
```bash
# Basic training (30 epochs)
python train_multiclass.py --epochs 30

# With fine-tuning (recommended)
python train_multiclass.py --epochs 40 --finetune
```

### 3. Evaluate Performance
```bash
python evaluate_multiclass.py
```

### 4. Test Single Image
```bash
python inference_multiclass.py dataset_multiclass/test/bent_lead/000.png
```

## 🔌 API Usage

### Python Example
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Status: {result['status']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

## 📈 Model Performance

- **Training Dataset**: 633 images (10 classes)
- **Validation Dataset**: 77 images
- **Test Dataset**: 83 images
- **Validation Accuracy**: ~34%
- **Top-3 Accuracy**: ~90% (correct class in top 3 predictions)
- **Training Time**: 15-20 minutes (CPU), 5-10 minutes (GPU)
- **Inference Time**: ~100-200ms per image

## 🌐 Remote Access

### Option 1: ngrok (Recommended)
```bash
# Install ngrok from https://ngrok.com/download
# Start both backend and frontend
ngrok http 8502
# Access via public URL: https://xxxx.ngrok.io
```

### Option 2: GitHub Codespaces
1. Open repo on GitHub
2. Code → Codespaces → Create codespace
3. Run setup commands
4. Codespaces provides public URLs automatically

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow, Keras, MobileNetV2
- **Computer Vision**: OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas, scikit-learn

## 📝 Requirements

See [requirements.txt](requirements.txt) for complete list. Key dependencies:
- tensorflow>=2.10.0
- fastapi>=0.104.0
- streamlit>=1.28.0
- opencv-python>=4.5.0
- scikit-learn>=1.0.0

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **MVTec Anomaly Detection Dataset** for providing high-quality defect images
- **MobileNetV2** for efficient transfer learning architecture
- TensorFlow and Keras teams for excellent frameworks

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/asba-alt/Intelligent-Visual-Inspection-System-for-Missing-Part-Detection/issues)
- **Documentation**: [docs/](docs/)
- **Quick Start**: [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

**Built with ❤️ for quality manufacturing inspection**
