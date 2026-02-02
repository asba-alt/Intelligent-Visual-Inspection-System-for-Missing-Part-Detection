import os

# Configuration for Intelligent Visual Inspection System

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, 'dataset')
MODEL_DIR = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
LOGS_DIR = os.path.join(ROOT, 'logs')

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 1234

MODEL_PATH = os.path.join(MODEL_DIR, 'mobilenetv2_inspection.keras')

CLASSES = ['complete', 'defect']

# Inference thresholds
THRESH_FAIL = 0.80
THRESH_PASS = 0.55

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
import os

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
STATIC_UPLOADS = os.path.join(ROOT_DIR, "static", "uploads")
STATIC_OUTPUTS = os.path.join(ROOT_DIR, "static", "outputs")

# Image / training config
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

# Reproducibility
PYTHON_HASH_SEED = "0"

# Thresholds for PASS/FAIL/REVIEW (probability of DEFECT)
THRESH_FAIL = 0.80
THRESH_PASS = 0.55

# Model filename
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")

# Ensure directories exist
for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR, STATIC_UPLOADS, STATIC_OUTPUTS]:
    os.makedirs(d, exist_ok=True)
