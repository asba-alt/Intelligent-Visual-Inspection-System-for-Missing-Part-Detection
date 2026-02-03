import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'models/mobilenetv2_multiclass.keras'
CLASS_NAMES_PATH = 'models/class_names_multiclass.txt'

def load_class_names():
    """Load class names from file."""
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return [line.strip() for line in f]
    else:
        # Default classes in alphabetical order
        return [
            'bent_lead', 'complete', 'cut_lead', 'damaged_case', 
            'manipulated_front', 'misplaced', 'scratch_head', 
            'scratch_neck', 'thread_side', 'thread_top'
        ]

def preprocess_image(image_path):
    """Load and preprocess image for prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_multiclass(image_path, model, class_names):
    """Predict defect class for a single image."""
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[predicted_class_idx])
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3_predictions = [
        {
            'class': class_names[idx],
            'confidence': float(predictions[idx])
        }
        for idx in top3_indices
    ]
    
    # Determine status
    if predicted_class == 'complete':
        status = 'PASS'
    else:
        status = 'FAIL'
        if confidence < 0.7:
            status = 'REVIEW'
    
    result = {
        'status': status,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top3_predictions': top3_predictions,
        'all_probabilities': {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    }
    
    return result

def main(args):
    # Load model and class names
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print(f"Loading class names from {CLASS_NAMES_PATH}...")
    class_names = load_class_names()
    print(f"Classes: {class_names}")
    
    # Predict
    print(f"\nPredicting: {args.image}")
    result = predict_multiclass(args.image, model, class_names)
    
    # Display results
    print("\n" + "="*50)
    print(f"STATUS: {result['status']}")
    print(f"PREDICTED CLASS: {result['predicted_class']}")
    print(f"CONFIDENCE: {result['confidence']:.2%}")
    print("\nTOP 3 PREDICTIONS:")
    for i, pred in enumerate(result['top3_predictions'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.2%}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict defect class for an image')
    parser.add_argument('image', help='Path to image file')
    args = parser.parse_args()
    
    main(args)
