"""Flask web app for uploading images and showing inference + Grad-CAM."""
import os
import time
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from config import STATIC_UPLOADS, STATIC_OUTPUTS, BEST_MODEL_PATH
from src.inference import predict, load_model
from src.gradcam import make_gradcam
import tensorflow as tf


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = STATIC_UPLOADS
app.secret_key = 'replace-this-with-secure-key'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        ts = int(time.time())
        out_name = f"{ts}_{fname}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
        file.save(save_path)

        try:
            model = load_model(BEST_MODEL_PATH)
        except Exception as e:
            msg = f"Model not found. Train model first. Error: {e}"
            return render_template('index.html', error=msg)

        try:
            res = predict(save_path, model=model)
            prob = res['prob_defect']
            label = res['label']
            decision = res['decision']

            # Grad-CAM
            out_filename = f"gradcam_{out_name}"
            out_path = os.path.join(STATIC_OUTPUTS, out_filename)
            make_gradcam(save_path, model, out_path)

            # URLs for template (relative)
            upload_url = os.path.join('static', 'uploads', out_name)
            out_url = os.path.join('static', 'outputs', out_filename)

            return render_template('result.html',
                                   image_path=upload_url,
                                   overlay_path=out_url,
                                   prob=prob,
                                   label=label,
                                   decision=decision)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        flash('Unsupported file type')
        return redirect(request.url)


if __name__ == '__main__':
    # Run with: python -m src.app
    app.run(debug=True)
