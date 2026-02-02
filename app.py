import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from inference import predict, load_model
from gradcam import run_gradcam
from config import MODEL_PATH

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'change-me-in-production'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@app.route('/')
def index():
    model_exists = os.path.exists(MODEL_PATH)
    return render_template('index.html', model_exists=model_exists)


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Run inference
        try:
            model = load_model()
        except Exception as e:
            flash(str(e))
            return redirect(url_for('index'))

        try:
            res = predict(save_path, model=model)
        except Exception as e:
            flash(f'Inference failed: {e}')
            return redirect(url_for('index'))

        # Run Grad-CAM
        out_name = f"gradcam_{filename}"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_name)
        try:
            saved = run_gradcam(save_path, output_path=out_path, model_path=MODEL_PATH)
        except Exception as e:
            flash(f'Grad-CAM failed: {e}')
            saved = None

        return render_template('result.html',
                               image_url='/' + save_path.replace('\\', '/'),
                               overlay_url='/' + saved.replace('\\', '/') if saved else None,
                               predicted=res['predicted_label'],
                               prob_defect=f"{res['prob_defect']:.4f}",
                               decision=res['decision'])
    else:
        flash('Unsupported file type')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
