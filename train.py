import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image
from config import IMG_SIZE, BATCH_SIZE, SEED, MODEL_PATH, LOGS_DIR
from model import build_model, compile_model


def set_seeds(seed=SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_datasets(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),
        labels='inferred',
        label_mode='binary',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        labels='inferred',
        label_mode='binary',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED
    )
    return train_ds, val_ds


def main(args):
    set_seeds()
    train_ds, val_ds = get_datasets(args.dataset)

    # Simple data normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    model = build_model(img_size=IMG_SIZE, pretrained=True)
    model = compile_model(model)

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True))
    callbacks.append(ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, save_format='keras'))
    tb_logdir = os.path.join(LOGS_DIR, 'fit')
    callbacks.append(TensorBoard(log_dir=tb_logdir))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print(f"Training finished. Best model saved to {MODEL_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset', help='Root dataset folder with train/val/test subfolders')
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    main(args)
