"""Evaluation utilities and script.

Run with: python -m src.evaluate
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from config import BEST_MODEL_PATH
from src.data import load_datasets


def evaluate():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}. Train and save a model first.")

    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    _, _, test_ds = load_datasets()

    y_true = []
    y_pred = []
    y_prob = []
    for imgs, labels in test_ds:
        probs = model.predict(imgs)
        preds = (probs >= 0.5).astype(int)
        y_true.extend(labels.numpy().astype(int).flatten().tolist())
        y_pred.extend(preds.flatten().tolist())
        y_prob.extend(probs.flatten().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion Matrix:\n", cm)

    return dict(accuracy=acc, precision=prec, recall=rec, confusion_matrix=cm)


if __name__ == '__main__':
    evaluate()
