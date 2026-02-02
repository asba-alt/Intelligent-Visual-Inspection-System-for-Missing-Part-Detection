# Training Guide - Intelligent Visual Inspection System
## Phase 1: Missing Part Detection (Binary Classification)

This guide explains how to train the CNN-based model for detecting missing parts in mechanical assemblies.

---

## 📋 Prerequisites

### 1. Dataset Prepared
Your dataset should be organized as:
```
dataset/
├── train/
│   ├── complete/    (good assemblies)
│   └── missing/     (assemblies with defects/missing parts)
├── val/
│   ├── complete/
│   └── missing/
└── test/
    ├── complete/
    └── missing/
```

✅ **Status**: Dataset already prepared with images from MVTec (screw + transistor)

### 2. Python Environment
Make sure your virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 🚀 Training Steps

### Step 1: Verify Dataset
Check that images are properly distributed:

```powershell
# Count training images
Get-ChildItem -Path "dataset\train\complete" | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -Path "dataset\train\missing" | Measure-Object | Select-Object -ExpandProperty Count

# Count validation images
Get-ChildItem -Path "dataset\val\complete" | Measure-Object | Select-Object -ExpandProperty Count
Get-ChildItem -Path "dataset\val\missing" | Measure-Object | Select-Object -ExpandProperty Count
```

### Step 2: Train the Model

**Basic Training** (25 epochs, default settings):
```powershell
python train.py
```

**Custom Training**:
```powershell
# Custom epochs
python train.py --epochs 30

# Custom dataset location
python train.py --dataset path/to/dataset --epochs 30
```

**With TensorFlow warnings suppressed**:
```powershell
$env:TF_ENABLE_ONEDNN_OPTS='0'; $env:TF_CPP_MIN_LOG_LEVEL='2'; python train.py --epochs 25
```

### Step 3: Monitor Training

The training script includes:
- **Early Stopping**: Stops if validation loss doesn't improve for 6 epochs
- **Model Checkpoint**: Saves best model to `models/mobilenetv2_inspection.keras`
- **TensorBoard Logging**: Logs saved to `logs/fit/`

To view TensorBoard during/after training:
```powershell
tensorboard --logdir logs/fit
```
Then open: http://localhost:6006

---

## 📊 What Happens During Training

### 1. Model Architecture
- **Base**: MobileNetV2 (pretrained on ImageNet)
- **Transfer Learning**: Base frozen, only training top layers
- **Output**: Binary classification (sigmoid activation)
  - 0 = complete assembly
  - 1 = missing part/defect

### 2. Training Configuration
- **Input Size**: 224x224 RGB images
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy

### 3. Data Augmentation
Currently uses basic normalization (rescaling to [0,1]). 
For improved robustness, consider adding augmentation in future iterations.

---

## 📈 Expected Training Time

- **CPU**: ~15-30 minutes per epoch (depends on dataset size)
- **GPU**: ~2-5 minutes per epoch

With early stopping, training typically completes in 10-20 epochs.

---

## ✅ After Training

### 1. Check Model File
```powershell
Get-ChildItem models\
```
You should see: `mobilenetv2_inspection.keras`

### 2. Evaluate Model
```powershell
python evaluate.py --dataset dataset
```

This will:
- Load the trained model
- Run predictions on test set
- Generate classification report
- Create confusion matrix
- Save results to `results/`

### 3. Test Single Image
```powershell
python inference.py --image path/to/test/image.png
```

---

## 🎯 Phase 1 Evaluation Criteria

As per PhaseDefinition.md, the model is evaluated on:

1. **Accuracy**: Overall correctness
2. **Precision**: Of defect predictions, how many are correct
3. **Recall**: Of actual defects, how many are caught (CRITICAL)
4. **Confusion Matrix**: Detailed breakdown

**Priority**: Minimize **False Negatives** (missing actual defects)
- Better to flag a good assembly for review than miss a defect
- Phase 1 emphasizes high recall for safety-critical applications

---

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"
```powershell
pip install tensorflow>=2.10.0
```

### Issue: "Model file not found"
Run training first:
```powershell
python train.py
```

### Issue: Out of Memory
Reduce batch size in [config.py](config.py):
```python
BATCH_SIZE = 16  # or 8
```

### Issue: Training too slow
- Ensure GPU is available (if present)
- Reduce epochs for quick testing
- Use smaller subset of data for experimentation

---

## 📁 Output Files

After training, you'll have:

```
models/
└── mobilenetv2_inspection.keras    (Best model weights)

logs/
└── fit/                            (TensorBoard logs)
    └── events.out.tfevents...

results/                            (After evaluation)
├── confusion_matrix.png
├── classification_report.txt
└── test_predictions.csv
```

---

## 🔄 Next Steps (Future Phases)

### Phase 2: Localization
- Implement Grad-CAM visualization ✅ (already available)
- Identify where the missing part is located

### Phase 3: Multi-class Detection
- Detect extra/incorrect parts
- Expand beyond binary classification

### Phase 4: Video Support
- Process video streams
- Frame-by-frame analysis

### Phase 5: Web Deployment
- FastAPI backend ✅ (already implemented)
- Streamlit frontend ✅ (already implemented)

---

## 📝 Training Tips

1. **Start Small**: Train with fewer epochs (e.g., 5) to verify pipeline
2. **Monitor Loss**: Validation loss should decrease steadily
3. **Check Overfitting**: If train accuracy >> val accuracy, add:
   - Dropout (already included: 0.3)
   - Data augmentation
   - More training data
4. **Fine-tuning**: After initial training, unfreeze some base layers for better performance

---

## 🎓 Understanding the Output

### During Training:
```
Epoch 1/25
32/32 [==============================] - 45s 1s/step - loss: 0.4521 - accuracy: 0.7812 - val_loss: 0.3456 - val_accuracy: 0.8500
```

- **loss**: Training loss (lower is better)
- **accuracy**: Training accuracy
- **val_loss**: Validation loss (used for early stopping)
- **val_accuracy**: Validation accuracy (model's generalization)

### Target Metrics (Phase 1):
- **Accuracy**: > 90%
- **Recall (Defect)**: > 95% (critical for safety)
- **Precision (Defect)**: > 85%

---

## 📞 Quick Reference Commands

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Train model (basic)
python train.py

# Train with custom epochs
python train.py --epochs 30

# Evaluate model
python evaluate.py

# Test single image
python inference.py --image test.png

# View training logs
tensorboard --logdir logs/fit

# Run web application (FastAPI + Streamlit)
# Terminal 1:
python app_fastapi.py

# Terminal 2:
streamlit run app_streamlit.py
```

---

## 📚 References

- **Dataset**: MVTec Anomaly Detection Dataset (Screw, Transistor)
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Framework**: TensorFlow/Keras
- **Phase**: Phase 1 - Binary Classification (Complete vs Missing)

---

**Ready to train? Run:**
```powershell
python train.py
```
