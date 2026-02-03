import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
DATASET_DIR = 'dataset_multiclass'
MODEL_PATH = 'models/mobilenetv2_multiclass.keras'
LOGS_DIR = 'logs/multiclass'

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# All classes
CLASSES = [
    'bent_lead', 'complete', 'cut_lead', 'damaged_case', 
    'manipulated_front', 'misplaced', 'scratch_head', 
    'scratch_neck', 'thread_side', 'thread_top'
]

def set_seeds(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_multiclass_model(num_classes=10, img_size=IMG_SIZE):
    """Build MobileNetV2-based multi-class classifier."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def get_datasets(data_dir):
    """Load datasets with categorical labels."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    
    return train_ds, val_ds

def calculate_class_weights(train_ds):
    """Calculate class weights for imbalanced dataset."""
    labels = []
    for _, y_batch in train_ds:
        labels.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
    
    labels = np.array(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weight_dict

def main(args):
    set_seeds()
    
    print(f"Loading dataset from {args.dataset}...")
    train_ds, val_ds = get_datasets(args.dataset)
    
    # Get class names
    class_names = train_ds.class_names
    print(f"\nClasses ({len(class_names)}): {class_names}")
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(train_ds)
    print(f"Class weights: {class_weights}")
    
    # Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Build model
    print(f"\nBuilding multi-class model...")
    model = build_multiclass_model(num_classes=len(class_names))
    
    # Unfreeze last layers for fine-tuning
    if args.finetune:
        print("Fine-tuning: Unfreezing last 20 layers...")
        for layer in model.layers[1].layers[-20:]:
            layer.trainable = True
    
    # Compile
    lr = 1e-5 if args.finetune else 1e-4
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    print(f"\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=os.path.join(LOGS_DIR, 'fit'))
    ]
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    print(f"\n✅ Training complete! Model saved to {MODEL_PATH}")
    
    # Save class names
    with open('models/class_names_multiclass.txt', 'w') as f:
        f.write('\n'.join(class_names))
    print(f"✅ Class names saved to models/class_names_multiclass.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-class defect classifier')
    parser.add_argument('--dataset', default='dataset_multiclass', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning of base model')
    args = parser.parse_args()
    
    main(args)
