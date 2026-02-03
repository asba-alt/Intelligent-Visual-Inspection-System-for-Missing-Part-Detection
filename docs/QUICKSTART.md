# Quick Start Guide - Setup on New Device

## Prerequisites
- Python 3.8 or higher
- Git (for cloning repository)
- 4GB RAM minimum
- Windows/Linux/macOS

## Step 1: Clone Repository
```bash
git clone https://github.com/asba-alt/Intelligent-Visual-Inspection-System-for-Missing-Part-Detection.git
cd Intelligent-Visual-Inspection-System-for-Missing-Part-Detection
```

## Step 2: Create Virtual Environment

### Windows (PowerShell)
```powershell
# Enable script execution if needed
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Linux/macOS
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Verify Models Exist
Check that these files exist in the `models/` folder:
- `mobilenetv2_multiclass.keras` (multi-class model - 10 defect types)
- `class_names_multiclass.txt` (class labels)

**If models are missing**, you need to train them (see Training section below).

## Step 5: Run the Application

### Option A: Run Backend + Frontend Separately (Recommended)

**Terminal 1 - Backend (FastAPI)**
```powershell
# Windows
.\venv\Scripts\Activate.ps1
python app_fastapi.py
```

```bash
# Linux/macOS
source venv/bin/activate
python app_fastapi.py
```

**Terminal 2 - Frontend (Streamlit)**
```powershell
# Windows
.\venv\Scripts\Activate.ps1
streamlit run app_streamlit.py
```

```bash
# Linux/macOS
source venv/bin/activate
streamlit run app_streamlit.py
```

**Access the application:**
- Frontend: http://localhost:8502
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option B: Quick Test with Single Image
```bash
# Activate venv first
python inference_multiclass.py dataset_multiclass/test/complete/000.png
```

## Training Models (If Needed)

### Prepare Dataset (First Time Only)
```bash
# This organizes the raw MVTec dataset into train/val/test splits
python prepare_dataset_multiclass.py
```

### Train Multi-Class Model
```bash
# Training with 40 epochs and fine-tuning (takes 15-20 minutes)
python train_multiclass.py --epochs 40 --finetune
```

### Evaluate Model
```bash
# Test model performance on test set
python evaluate_multiclass.py
```

## Quick Command Reference

### Activate Virtual Environment
```powershell
# Windows
.\venv\Scripts\Activate.ps1
```
```bash
# Linux/macOS
source venv/bin/activate
```

### Start Backend Server
```bash
python app_fastapi.py
# Runs on: http://localhost:8000
```

### Start Frontend UI
```bash
streamlit run app_streamlit.py
# Runs on: http://localhost:8502
```

### Single Image Inference
```bash
python inference_multiclass.py path/to/image.png
```

### Train Model
```bash
python train_multiclass.py --epochs 40 --finetune
```

### Evaluate Model
```bash
python evaluate_multiclass.py
```

## Remote Access Setup (Optional)

### Using ngrok (Easiest)
1. Download ngrok: https://ngrok.com/download
2. Extract and add to PATH
3. Start both backend and frontend
4. Run ngrok:
```bash
# Expose Streamlit frontend
ngrok http 8502
```
5. Copy the public URL (e.g., `https://xxxx.ngrok.io`)
6. Access from any device using that URL

### Using GitHub Codespaces (Cloud-based)
1. Open repository on GitHub
2. Click "Code" → "Codespaces" → "Create codespace"
3. Install dependencies: `pip install -r requirements.txt`
4. Run application commands as normal
5. Codespaces will provide public URLs automatically

## Troubleshooting

### "Model not found" Error
```bash
# Train the model first
python train_multiclass.py --epochs 40 --finetune
```

### Port Already in Use
```bash
# Kill existing process
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:8000 | xargs kill -9
```

### Virtual Environment Not Activating
```powershell
# Windows - Set execution policy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### TensorFlow Warnings
```bash
# Suppress TensorFlow logging
# Windows PowerShell
$env:TF_CPP_MIN_LOG_LEVEL='2'

# Linux/macOS
export TF_CPP_MIN_LOG_LEVEL=2
```

## Project Structure
```
.
├── app_fastapi.py           # Backend API server
├── app_streamlit.py         # Frontend UI
├── train_multiclass.py      # Training script
├── inference_multiclass.py  # Single image inference
├── evaluate_multiclass.py   # Model evaluation
├── prepare_dataset_multiclass.py  # Dataset preparation
├── requirements.txt         # Python dependencies
├── models/                  # Trained models
│   ├── mobilenetv2_multiclass.keras
│   └── class_names_multiclass.txt
├── dataset_multiclass/      # Organized dataset
│   ├── train/
│   ├── val/
│   └── test/
└── raw/                     # Original MVTec dataset
    ├── screw/
    └── transistor/
```

## Supported Defect Classes (10 Total)
1. **complete** - No defects
2. **bent_lead** - Bent component lead
3. **cut_lead** - Cut/damaged lead
4. **damaged_case** - Case/body damage
5. **manipulated_front** - Front tampering
6. **misplaced** - Component misalignment
7. **scratch_head** - Scratch on head area
8. **scratch_neck** - Scratch on neck area
9. **thread_side** - Side thread defect
10. **thread_top** - Top thread defect

## Performance Notes
- Model loads once at startup (fast predictions)
- First prediction may take 2-3 seconds (model warmup)
- Subsequent predictions: ~100-200ms
- Training on CPU: 15-20 minutes
- Training on GPU: 5-10 minutes

## Support
For issues, visit: https://github.com/asba-alt/Intelligent-Visual-Inspection-System-for-Missing-Part-Detection/issues
