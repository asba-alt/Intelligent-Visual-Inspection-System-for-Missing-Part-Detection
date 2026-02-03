import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import shutil
from inference_multiclass import predict_multiclass, load_class_names
import tensorflow as tf
from gradcam import run_gradcam

MODEL_PATH = 'models/mobilenetv2_multiclass.keras'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = FastAPI(title="Intelligent Visual Inspection System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model once at startup
model = None
class_names = []

@app.on_event("startup")
async def startup_event():
    global model, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = load_class_names()
        print(f"✅ Multi-class model loaded: {len(class_names)} classes")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}")


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.get("/")
async def root():
    return {
        "message": "Intelligent Visual Inspection System API",
        "model_exists": os.path.exists(MODEL_PATH),
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)) -> Dict:
    """
    Upload an image and get prediction with Grad-CAM visualization
    """
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}"
        )
    
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please train a model first."
        )
    
    # Save uploaded file
    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run inference
    try:
        result = predict_multiclass(save_path, model, class_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    # Run Grad-CAM
    out_name = f"gradcam_{filename}"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    overlay_saved = None
    
    try:
        overlay_saved = run_gradcam(save_path, output_path=out_path, model_path=MODEL_PATH)
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
    
    return {
        "status": result['status'],
        "predicted_class": result['predicted_class'],
        "confidence": float(result['confidence']),
        "top3_predictions": result['top3_predictions'],
        "image_url": f"/static/uploads/{filename}",
        "overlay_url": f"/static/outputs/{out_name}" if overlay_saved else None
    }


@app.get("/image/{folder}/{filename}")
async def get_image(folder: str, filename: str):
    """
    Serve uploaded or processed images
    """
    file_path = os.path.join("static", folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
