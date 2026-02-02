import os
import cv2
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MODEL_PATH, CLASSES, THRESH_FAIL, THRESH_PASS


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Train model first.")
    model = tf.keras.models.load_model(path)
    return model


def preprocess_image(image_path, target_size=IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    img_norm = img_resized.astype('float32') / 255.0
    return img_norm, img_rgb


def predict(image_path, model=None):
    if model is None:
        model = load_model()
    img_norm, img_rgb = preprocess_image(image_path)
    inp = np.expand_dims(img_norm, axis=0)
    prob_missing = float(model.predict(inp)[0][0])
    # Keras orders labels alphabetically: 'complete' (0), 'missing' (1)
    # Model output: probability of 'missing' class
    prob_complete = 1.0 - prob_missing
    
    # Decision logic based on missing probability
    if prob_missing >= THRESH_FAIL:
        decision = 'FAIL'
    elif prob_missing <= (1.0 - THRESH_PASS):  # If prob_complete >= THRESH_PASS
        decision = 'PASS'
    else:
        decision = 'REVIEW'

    result = {
        'prob_defect': prob_missing,
        'prob_complete': prob_complete,
        'predicted_label': 'missing' if prob_missing >= 0.5 else 'complete',
        'decision': decision,
    }
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to image')
    args = parser.parse_args()
    print(predict(args.image))
