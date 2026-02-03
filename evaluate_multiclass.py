import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
DATASET_DIR = 'dataset_multiclass'
MODEL_PATH = 'models/mobilenetv2_multiclass.keras'
RESULTS_DIR = 'results/multiclass'

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_dataset():
    """Load test dataset."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATASET_DIR, 'test'),
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED
    )
    return test_ds

def evaluate_multiclass_model(model, test_ds, class_names):
    """Evaluate multi-class model."""
    # Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds_norm = test_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds_norm:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification report
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Multi-Class Defect Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to {confusion_path}")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), per_class_acc)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # Color bars
    for i, bar in enumerate(bars):
        if per_class_acc[i] >= 0.9:
            bar.set_color('green')
        elif per_class_acc[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{per_class_acc[i]:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    acc_path = os.path.join(RESULTS_DIR, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f"✅ Per-class accuracy plot saved to {acc_path}")
    
    # Overall metrics
    overall_acc = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\n{'='*70}")
    print(f"OVERALL ACCURACY: {overall_acc:.2%}")
    print(f"{'='*70}\n")
    
    return y_true, y_pred, cm

def main():
    print("Loading test dataset...")
    test_ds = load_test_dataset()
    class_names = test_ds.class_names
    
    print(f"Classes: {class_names}")
    print(f"\nLoading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print("\nEvaluating model...")
    evaluate_multiclass_model(model, test_ds, class_names)

if __name__ == '__main__':
    main()
