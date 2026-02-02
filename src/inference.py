"""Inference helper: single-image prediction and decision logic."""
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from config import IMAGE_SIZE, BEST_MODEL_PATH, THRESH_FAIL, THRESH_PASS


def load_model(path=BEST_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Please train the model first.")
    return tf.keras.models.load_model(path)


def preprocess_image(image_path, img_size=IMAGE_SIZE):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(image_path, model=None):
    """Return probability of DEFECT, predicted label, and decision string."""
    if model is None:
        model = load_model()

    x = preprocess_image(image_path)
    prob = float(model.predict(x)[0][0])

    # Sigmoid output -> probability of class index 1 (defect)
    if prob >= THRESH_FAIL:
        decision = 'FAIL'
    elif prob <= THRESH_PASS:
        decision = 'PASS'
    else:
        decision = 'REVIEW'

    # label map: 0 -> complete, 1 -> defect
    predicted_label = 'defect' if prob >= 0.5 else 'complete'

    return {'prob_defect': prob, 'label': predicted_label, 'decision': decision}


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m src.inference PATH_TO_IMAGE')
        sys.exit(1)
    r = predict(sys.argv[1])
    print(r)
