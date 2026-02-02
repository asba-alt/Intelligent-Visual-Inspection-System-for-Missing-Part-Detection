# Intelligent Visual Inspection System

This project provides an end-to-end pipeline for inspecting assembly images and classifying them as `complete` or `defect` using transfer learning (MobileNetV2). Phase 2 includes Grad-CAM heatmaps to highlight suspicious regions.

Project layout (root):

- `dataset/` - prepared dataset with `train/`, `val/`, `test/` and class subfolders `complete/`, `defect/`
- `models/` - saved `.keras` models
- `results/` - internal outputs
- `logs/` - training logs
- `templates/`, `static/` - Flask web app assets
- `data.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, `gradcam.py`, `app.py`

Requirements

Install with:

```bash
python -m pip install -r requirements.txt
```

Preparing your data

You mentioned you have two archives: `tranistor.tar.xz` and `screw.tar.xz`.

1. Extract the archives into a `raw/` folder or leave them as-is. Example (Windows PowerShell):

```powershell
mkdir raw
tar -xvf tranistor.tar.xz -C raw
tar -xvf screw.tar.xz -C raw
```

2. Decide which extracted folder corresponds to which class label (`complete` or `defect`). Then run the helper to prepare the dataset splits:

```bash
python data.py --archives tranistor.tar.xz screw.tar.xz --map tranistor:defect screw:complete
```

The `--map` entries match a substring of the extracted folder name to the desired class. If you already extracted and renamed folders, you can map absolute paths too:

```bash
python data.py --map C:/path/to/tranistor:defect C:/path/to/screw:complete
```

This will create `dataset/train/complete`, `dataset/train/defect`, `dataset/val/...`, `dataset/test/...`.

Training

Train a model (uses MobileNetV2 with ImageNet weights, EarlyStopping, ModelCheckpoint):

```bash
python train.py --dataset dataset --epochs 25
```

The best model will be saved to `models/mobilenetv2_inspection.keras`.

Evaluation

Run evaluation on test split:

```bash
python evaluate.py --test_dir dataset/test
```

Inference (CLI)

```bash
python inference.py path/to/image.jpg
```

This prints probabilities and decision (PASS/FAIL/REVIEW). Decision logic:
- `prob_defect >= 0.80` -> `FAIL`
- `prob_defect <= 0.55` -> `PASS`
- otherwise -> `REVIEW`

Grad-CAM

Generate an overlay image for a single file (saved to `results/` by default):

```bash
python gradcam.py path/to/image.jpg --out static/outputs/gradcam_image.jpg
```

Web app (Flask)

Start web UI (runs on http://0.0.0.0:5000):

```bash
python app.py
```

Open the browser, upload an image, and see the prediction, confidence, decision, and Grad-CAM overlay.

Deployment notes

- For production, run Flask behind a WSGI server such as `gunicorn` (Linux) or use IIS/WSGI on Windows.
- Use a GPU-enabled TensorFlow build for faster training and inference.
- Ensure `models/mobilenetv2_inspection.keras` is present; otherwise the app will show a helpful message.

Troubleshooting

- If Keras raises errors loading the model, remove custom objects or re-save the model using `model.save('models/..', save_format='keras')`.
- If dataset classes are flipped, check the alphabetical order of class subfolders when using `image_dataset_from_directory`.

License and safety

This project is a template — adapt it to your data and validate before using in production.
# Intelligent Visual Inspection System

End-to-end project (Phase 1 & Phase 2) for image assembly inspection using
MobileNetV2 transfer learning, Grad-CAM visualization, and a Flask web UI.

Quick start

1. Create a Python virtual env and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare dataset folder structure under `dataset/` with `train/`, `val/`, `test/` each containing `complete/` and `defect/` subfolders.

3. Train model (optional if you have pre-trained model in `models/`):

```bash
python -m src.train
```

4. Run the web app (serves upload UI):

```bash
python -m src.app
# then open http://127.0.0.1:5000
```

If model file is missing the app will show a helpful message.

Project layout

- dataset/ (train/ val/ test/ with class subfolders)
- models/ (saved .keras model)
- results/ (evaluation outputs)
- logs/
- static/uploads/ (uploaded images)
- static/outputs/ (grad-cam overlays)
- src/ (code)
- config.py
