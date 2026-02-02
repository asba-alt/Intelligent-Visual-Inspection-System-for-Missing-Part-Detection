import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from config import IMG_SIZE, MODEL_PATH


def evaluate(test_dir):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train a model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        shuffle=False
    )
    norm = tf.keras.layers.Rescaling(1./255)
    y_true = []
    y_pred = []
    for x, y in test_ds:
        x = norm(x)
        probs = model.predict(x)
        preds = (probs.flatten() >= 0.5).astype(int)
        y_true.extend(y.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='dataset/test', help='Path to test dataset directory')
    args = parser.parse_args()
    evaluate(args.test_dir)
