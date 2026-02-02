import os

# Configuration for Intelligent Visual Inspection System - Phase 1

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
STATIC_UPLOADS = os.path.join(ROOT_DIR, "static", "uploads")
STATIC_OUTPUTS = os.path.join(ROOT_DIR, "static", "outputs")

# Image / training config (Phase 1: Binary Classification)
IMG_SIZE = 224
IMAGE_SIZE = 224  # Alias for compatibility
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42

# Model
MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_inspection.keras')

# Classes (Phase 1: Binary)
CLASSES = ['complete', 'missing']

# Inference thresholds (probability of defect/missing)
THRESH_FAIL = 0.80   # High confidence defect -> FAIL
THRESH_PASS = 0.55   # Low confidence defect -> PASS

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_UPLOADS, exist_ok=True)
os.makedirs(STATIC_OUTPUTS, exist_ok=True)
